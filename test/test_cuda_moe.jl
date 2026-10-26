# CUDA MoE expert offload: budgeted to_device, HostOffload embedding/lm_head,
# config-agnostic top-k SwiGLU vs the CPU oracle. Named skip without a device.

using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

using .GessoTestHelpers: approx_eq, deterministic_rng

function _pack_f32_slice(A3, ne0, ne1)
    parts = UInt8[]
    for e in 1:size(A3, 1)
        W = A3[e, :, :]
        g = vec(permutedims(W, (2, 1)))
        append!(parts, reinterpret(UInt8, g))
    end
    return Gesso.GgufQuantSlice(
        parts,
        0,
        Int32(0),
        size(A3, 1),
        ne0,
        ne1,
        ne0 * ne1 * 4,
        C_NULL,
        Int64(1),
    )
end

# Runs without a device: only the extension has to load.
mutable struct _SvHotMock
    layer::Int
    t::Int
    retry_t::Int
end

if CUDA_LOADED
    @testset "cuda hot-expert pool fallback is logged once per episode" begin
        ext = Base.get_extension(Gesso, :GessoCUDAExt)
        hot = _SvHotMock(7, 100, 0)
        cfg = Gesso.current_config()
        prev_io = cfg.io
        buf = IOBuffer()
        cfg.io = buf
        try
            delete!(ext._HOT_FALLBACK, 7)
            @test ext._hot_fallback!(hot, "test") === false
            @test hot.retry_t == 100 + ext._HOT_RETRY_TOKENS
            hot.t = 200
            @test ext._hot_fallback!(hot, "test") === false
            @test hot.retry_t == 200 + ext._HOT_RETRY_TOKENS
            lines = filter(!isempty, split(String(take!(buf)), '\n'))
            @test length(lines) == 1
            @test occursin("fallback", lines[1]) && occursin("layer 7", lines[1])
            # a pool that fits ends the episode; the next failure is logged again
            delete!(ext._HOT_FALLBACK, 7)
            ext._hot_fallback!(hot, "test")
            @test count(==('\n'), String(take!(buf))) == 1
        finally
            delete!(ext._HOT_FALLBACK, 7)
            cfg.io = prev_io
        end
    end
end

if !CUDA_LOADED || !CUDA.functional()
    _skip("no NVIDIA device — CUDA MoE offload tests skipped (§LXXVII skip law)")
else
    cuda = Gesso.CUDABackend()
    cpu = Gesso.CPUBackend()
    Inf_ = Gesso.Inference

    @testset "cuda to_device uploads tiny GgufQuantSlice as GpuQuantSlice" begin
        slice = _pack_f32_slice(ones(Float32, 2, 4, 8), 8, 4)
        emb = Gesso.EmbeddingTable(; shape=(4, 2), storage=ones(Float32, 4, 2))
        bt = (;
            mixer=:full_attention,
            n_expert_used=2,
            wexp_gate=Gesso.ExpertWeight(; shape=size(slice), storage=slice),
        )
        ts = (; embedding=emb, blocks=[bt], lm_head=emb)
        gpu = Gesso.to_device(cuda, ts)
        @test gpu.blocks[1].wexp_gate.storage isa Gesso.GpuQuantSlice
        @test gpu.embedding.storage isa CUDA.CuArray{Float32}
        gpu0 = withenv("GESSO_GPU_BUDGET_MIB" => "0") do
            Gesso.to_device(cuda, ts)
        end
        @test gpu0.blocks[1].wexp_gate.storage isa Gesso.GgufQuantSlice
        @test gpu0.embedding.storage isa Gesso.HostOffload
    end

    @testset "cuda to_device places an expert triple whole or not at all" begin
        # 392 KiB slabs: two fit a 1 MiB budget, three do not.
        mk() = _pack_f32_slice(ones(Float32, 2, 224, 224), 224, 224)
        emb = Gesso.EmbeddingTable(; shape=(4, 2), storage=ones(Float32, 4, 2))
        ew(s) = Gesso.ExpertWeight(; shape=size(s), storage=s)
        bt = (;
            mixer=:full_attention,
            n_expert_used=2,
            wexp_gate=ew(mk()),
            wexp_up=ew(mk()),
            wexp_down=ew(mk()),
        )
        ts = (; embedding=emb, blocks=[bt], lm_head=emb)
        slabs(g) = map(k -> g.blocks[1][k].storage, (:wexp_gate, :wexp_up, :wexp_down))
        g1 = withenv("GESSO_GPU_BUDGET_MIB" => "1") do
            Gesso.to_device(cuda, ts)
        end
        @test all(x -> x isa Gesso.GgufQuantSlice, slabs(g1))
        g2 = withenv("GESSO_GPU_BUDGET_MIB" => "2") do
            Gesso.to_device(cuda, ts)
        end
        @test all(x -> x isa Gesso.GpuQuantSlice, slabs(g2))
    end

    @testset "cuda to_device keeps expert slabs without a qgemv kernel on the host" begin
        # Relabelled bytes: placement must decide on type alone, before any
        # kernel reads them. Q6_K has no CUDA expert qgemv.
        mk() = _pack_f32_slice(ones(Float32, 2, 8, 8), 8, 8)
        emb = Gesso.EmbeddingTable(; shape=(4, 2), storage=ones(Float32, 4, 2))
        ew(s) = Gesso.ExpertWeight(; shape=size(s), storage=s)
        ts_of(ty) = begin
            sl = (mk(), mk(), mk())
            foreach(s -> s.ggml_type = ty, sl)
            bt = (;
                mixer=:full_attention,
                n_expert_used=2,
                wexp_gate=ew(sl[1]),
                wexp_up=ew(sl[2]),
                wexp_down=ew(sl[3]),
            )
            (; embedding=emb, blocks=[bt], lm_head=emb)
        end
        slabs(g) = map(k -> g.blocks[1][k].storage, (:wexp_gate, :wexp_up, :wexp_down))
        @test all(
            x -> x isa Gesso.GgufQuantSlice,
            slabs(Gesso.to_device(cuda, ts_of(Inf_.GGML_TYPE_Q6_K))),
        )
        @test all(
            x -> x isa Gesso.GpuQuantSlice,
            slabs(Gesso.to_device(cuda, ts_of(Inf_.GGML_TYPE_Q2_K))),
        )
    end

    @testset "cuda to_device wraps overflow as HostOffload, small-first" begin
        small = ones(Float32, 8, 8)            # 256 B
        large = ones(Float32, 1024, 1024)      # 4 MiB
        emb = Gesso.EmbeddingTable(; shape=size(large), storage=large)
        rms = Gesso.FrozenParameter(; shape=(8,), storage=ones(Float32, 8))
        w = Gesso.ProjectionWeight(; shape=size(small), storage=small)
        bt = (; mixer=:full_attention, attn_rms=rms, wq=w)
        ts = (; embedding=emb, blocks=[bt], lm_head=emb)
        gpu = withenv("GESSO_GPU_BUDGET_MIB" => "0") do
            Gesso.to_device(cuda, ts)
        end
        @test gpu.embedding.storage isa Gesso.HostOffload
        @test gpu.blocks[1].wq.storage isa Gesso.HostOffload
        @test gpu.blocks[1].attn_rms.storage isa Gesso.HostOffload
        @test gpu.lm_head.storage === gpu.embedding.storage

        # 1 MiB fits the 256 B projection + 32 B rms; 4 MiB embedding overflows.
        gpu2 = withenv("GESSO_GPU_BUDGET_MIB" => "1") do
            Gesso.to_device(cuda, ts)
        end
        @test gpu2.blocks[1].wq.storage isa CUDA.CuArray{Float32}
        @test gpu2.blocks[1].attn_rms.storage isa CUDA.CuArray{Float32}
        @test gpu2.embedding.storage isa Gesso.HostOffload
        @test gpu2.lm_head.storage === gpu2.embedding.storage
    end

    @testset "cuda check allows HostOffload and GgufQuantSlice; raw Array fails" begin
        ho = Gesso.HostOffload(ones(Float32, 2, 2))
        @test Inf_._check_device_storage(
            :probe,
            cuda,
            Gesso.Activation(; shape=(2, 2), storage=ho),
        ) === nothing
        slice = _pack_f32_slice(ones(Float32, 2, 2, 2), 2, 2)
        @test Inf_._check_device_storage(
            :probe,
            cuda,
            Gesso.ExpertWeight(; shape=size(slice), storage=slice),
        ) === nothing
        gq = Gesso.Inference.gpu_quant_slice(
            slice,
            Vector{UInt8}(Gesso.Inference.packed_quant_bytes(slice)),
        )
        @test Inf_._check_device_storage(
            :probe,
            cuda,
            Gesso.ExpertWeight(; shape=size(gq), storage=gq),
        ) === nothing
        err = try
            Inf_._check_device_storage(
                :probe,
                cuda,
                Gesso.Activation(; shape=(2, 2), storage=ones(Float32, 2, 2)),
            )
            nothing
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.code == Gesso.ERR_INVALID_PLAN
    end

    @testset "cuda HostOffload embedding_lookup and matmul match on-device" begin
        wl = Gesso.PrefillWorkload()
        tab = Float32[10 11 12; 20 21 22; 30 31 32; 40 41 42]
        dst = mat(Gesso.Activation, CUDA.zeros(Float32, 2, 3))
        Gesso.embedding_lookup!(
            cuda,
            dst,
            mat(Gesso.EmbeddingTable, Gesso.HostOffload(tab)),
            [2, 0],
            wl,
        )
        @test Array(dst.storage) ≈ Float32[30 31 32; 10 11 12] atol = 1e-5

        x0 = reshape(Float32[1, 2, 3, 4, 5, 6], (2, 3))
        W0 = reshape(Float32[1, 0, 0, 1, 0, 0], (2, 3))  # (out, in)
        dstm = mat(Gesso.Activation, CUDA.zeros(Float32, 2, 2))
        Gesso.matmul!(
            cuda,
            dstm,
            mat(Gesso.Activation, CuArray{Float32}(x0)),
            mat(Gesso.ProjectionWeight, Gesso.HostOffload(W0)),
            wl,
        )
        want = x0 * transpose(W0)
        @test Array(dstm.storage) ≈ want atol = 1e-5
    end

    @testset "cuda MoE top-k SwiGLU matches CPU across configs" begin
        configs = (
            (n_exp=4, n_used=2, h=8, ff=8),
            (n_exp=64, n_used=4, h=16, ff=16),
            (n_exp=128, n_used=8, h=16, ff=16),
            (n_exp=256, n_used=8, h=16, ff=8),
        )
        for (i, c) in enumerate(configs)
            rng = deterministic_rng(10 + i)
            seq = 2
            x = randn(rng, Float32, seq, c.h)
            router = randn(rng, Float32, c.n_exp, c.h)
            gate3 = randn(rng, Float32, c.n_exp, c.ff, c.h)
            up3 = randn(rng, Float32, c.n_exp, c.ff, c.h)
            down3 = randn(rng, Float32, c.n_exp, c.h, c.ff)
            y_cpu = zeros(Float32, seq, c.h)
            Inf_._moe_topk_swiglu!(cpu, y_cpu, x, router, gate3, up3, down3, c.n_used)

            y_gpu = CUDA.zeros(Float32, seq, c.h)
            Inf_._moe_topk_swiglu!(
                cuda,
                y_gpu,
                CuArray{Float32}(x),
                CuArray{Float32}(router),
                CuArray{Float32}(gate3),
                CuArray{Float32}(up3),
                CuArray{Float32}(down3),
                c.n_used,
            )
            @test approx_eq(Array(y_gpu), y_cpu; atol=1e-3)

            y_off = CUDA.zeros(Float32, seq, c.h)
            Inf_._moe_topk_swiglu!(
                cuda,
                y_off,
                CuArray{Float32}(x),
                Gesso.HostOffload(router),
                Gesso.HostOffload(gate3),
                Gesso.HostOffload(up3),
                Gesso.HostOffload(down3),
                c.n_used,
            )
            @test approx_eq(Array(y_off), y_cpu; atol=1e-3)
        end
    end

    @testset "cuda qgemv matches CPU on F32 pack and live IQ2_XS" begin
        rng = deterministic_rng(9)
        ne0, ne1, n_exp = 8, 4, 3
        A3 = randn(rng, Float32, n_exp, ne1, ne0)
        slice = _pack_f32_slice(A3, ne0, ne1)
        x = randn(rng, Float32, ne0)
        y_cpu = zeros(Float32, ne1)
        Inf_._qgemv_expert!(y_cpu, x, slice, 2)
        y_gpu = CUDA.zeros(Float32, ne1)
        Inf_._qgemv_expert!(y_gpu, CuArray{Float32}(x), slice, 2)
        @test approx_eq(Array(y_gpu), y_cpu; atol=1e-5)

        packed = Inf_.gpu_quant_slice(slice, CuArray{UInt8}(Inf_.packed_quant_bytes(slice)))
        y_gpu2 = CUDA.zeros(Float32, ne1)
        Inf_._qgemv_expert!(y_gpu2, CuArray{Float32}(x), packed, 2)
        @test approx_eq(Array(y_gpu2), y_cpu; atol=1e-5)

        gguf = get(ENV, "GESSO_ORNITH35_GGUF", "")
        if isfile(gguf)
            hdr = Gesso.gguf_header(gguf)
            einfo = only(t for t in hdr.tensors if t.name == "blk.0.ffn_gate_exps.weight")
            live = Gesso.Inference._expert_slice(hdr, einfo)
            xl = randn(rng, Float32, live.ne0)
            yg = CUDA.zeros(Float32, live.ne1)
            Inf_._qgemv_expert!(yg, CuArray{Float32}(xl), live, 1)
            # The device kernel quantizes x to Q8_1 (32-value blocks), so it is
            # held to the derived activation bound of test_cuda_quant_dense.jl
            # against an F64 reference, not to the host's F32 result.
            bytes = Vector{UInt8}(Inf_._quant_expert_bytes(live, 1))
            W = Float64.(
                permutedims(
                    reshape(
                        Inf_._dequant_f32(bytes, 0, live.ggml_type, live.ne0 * live.ne1),
                        live.ne0,
                        live.ne1,
                    ),
                ),
            )
            x64 = Float64.(xl)
            ref = W * x64
            nb = div(live.ne0, 32)
            bound = zeros(Float64, live.ne1)
            for b in 1:nb
                cols = (32*(b-1)+1):(32*b)
                bound .+=
                    maximum(abs, view(x64, cols)) / 254.0 .*
                    vec(sum(abs, view(W, :, cols); dims=2))
            end
            tol = bound .+ 5.0e-5 .* (abs.(W) * abs.(x64)) .+ 1.0e-9
            @test all(abs.(Float64.(Array(yg)) .- ref) .<= tol)
        else
            _skip("Ornith 35B GGUF not on disk — live IQ2_XS CUDA qgemv skipped")
        end
    end

    @testset "cuda K-quant expert qgemv matches CPU dequantize reference" begin
        # Q2_K, Q3_K and Q4_K expert blocks on the device, against libggml's F32
        # dequantize followed by an F64 matvec. The device quantizes activations
        # to Q8_1, so each row is held to the bound that rounding implies: half
        # a quantization step per 32-value block, times that row's weight mass.
        ext = Base.get_extension(Gesso, :GessoCUDAExt)
        rng = deterministic_rng(73)

        # Payload bytes are random; only the super-scales are written, as valid
        # fp16, because a random fp16 can be NaN or Inf and the check would then
        # say nothing about the kernel.
        function kq_synth_slice(ggml_type, bsz::Int, dpos, ne0::Int, ne1::Int, n_exp::Int)
            nsb = div(ne0, 256)
            bpe = ne1 * nsb * bsz
            data = rand(rng, UInt8, n_exp * bpe)
            for e in 0:(n_exp-1), r in 0:(ne1-1), s in 0:(nsb-1)
                base = e * bpe + (r * nsb + s) * bsz
                for p in dpos
                    v = Float16((rand(rng) < 0.5 ? -1 : 1) * (0.01 + 0.05 * rand(rng)))
                    data[(base+p+1):(base+p+2)] .= reinterpret(UInt8, [v])
                end
            end
            return Gesso.GgufQuantSlice(
                data,
                0,
                ggml_type,
                n_exp,
                ne0,
                ne1,
                bpe,
                C_NULL,
                Int64(1),
            )
        end

        # `host` is the slice the reference is decoded from; `dev` is the slice
        # the kernel reads (host or packed device form of the same bytes).
        function kq_expert_check(host, dev, e::Int, x::Vector{Float32})
            ne0, ne1 = host.ne0, host.ne1
            bytes = Vector{UInt8}(Inf_._quant_expert_bytes(host, e))
            W = Float64.(
                permutedims(
                    reshape(
                        Inf_._dequant_f32(bytes, 0, host.ggml_type, ne0 * ne1),
                        ne0,
                        ne1,
                    ),
                ),
            )
            x64 = Float64.(x)
            ref = W * x64
            bound = zeros(Float64, ne1)
            for b in 1:div(ne0, 32)
                cols = (32*(b-1)+1):(32*b)
                bound .+=
                    maximum(abs, view(x64, cols)) / 254.0 .*
                    vec(sum(abs, view(W, :, cols); dims=2))
            end
            yg = CUDA.zeros(Float32, ne1)
            Inf_._qgemv_expert!(yg, CuArray{Float32}(x), dev, e)
            err = abs.(Float64.(Array(yg)) .- ref)
            tol = bound .+ 5.0e-5 .* (abs.(W) * abs.(x64)) .+ 1.0e-9
            return all(err .<= tol), maximum(err) / maximum(abs, ref)
        end

        # block bytes, and the offsets of the fp16 super-scales inside a block
        kq_types = (
            (Inf_.GGML_TYPE_Q2_K, 84, (80, 82), "Q2_K"),
            (Inf_.GGML_TYPE_Q3_K, 110, (108,), "Q3_K"),
            (Inf_.GGML_TYPE_Q4_K, 144, (0, 2), "Q4_K"),
        )
        ne0, ne1, n_exp = 512, 6, 3
        for (ggml_type, bsz, dpos, label) in kq_types
            slice = kq_synth_slice(ggml_type, bsz, dpos, ne0, ne1, n_exp)
            packed =
                Inf_.gpu_quant_slice(slice, CuArray{UInt8}(Inf_.packed_quant_bytes(slice)))
            worst = 0.0
            for e in 1:n_exp
                x = randn(rng, Float32, ne0)
                ok, rel = kq_expert_check(slice, slice, e, x)
                @test ok
                worst = max(worst, rel)
            end
            ok_p, _ = kq_expert_check(slice, packed, 2, randn(rng, Float32, ne0))
            @test ok_p
            @info "cuda K-quant synthetic qgemv" type = label max_rel_err = worst
        end

        # Multi-column: pairs read their own token's column through the table,
        # so each must match a one-column launch bit for bit.
        for (ggml_type, bsz, dpos, label) in kq_types
            slice = kq_synth_slice(ggml_type, bsz, dpos, ne0, ne1, n_exp)
            bpe = slice.bytes_per_expert
            q = CuArray{UInt8}(slice.data)
            experts = [2, 1, 3, 2]
            tokens = [3, 1, 2, 3]
            n = length(experts)
            x = CuArray(randn(rng, Float32, ne0, 3))
            meta = CuArray(Int32[(experts .- 1) .* bpe; 1:n; tokens])
            y = CUDA.fill(NaN32, ne1, n)
            ext._cuda_launch_qgemv!(
                y,
                x,
                q,
                ggml_type,
                meta,
                n,
                n,
                ne0,
                ne1,
                1.0f0,
                0.0f0;
                xcoloff=2n,
            )
            ref = similar(Array(y))
            for i in 1:n
                yi = CUDA.fill(NaN32, ne1, 1)
                mi = CuArray(Int32[(experts[i]-1)*bpe, 1])
                ext._cuda_launch_qgemv!(
                    yi,
                    x[:, tokens[i]:tokens[i]],
                    q,
                    ggml_type,
                    mi,
                    1,
                    1,
                    ne0,
                    ne1,
                    1.0f0,
                    0.0f0,
                )
                ref[:, i] .= vec(Array(yi))
            end
            @test Array(y) == ref
        end

        live_gguf = get(ENV, "GESSO_GSQ_GGUF", "")
        if isempty(live_gguf) || !isfile(live_gguf)
            _skip(
                "GESSO_GSQ_GGUF not set to a GGUF file — live Q2_K/Q3_K expert qgemv skipped",
            )
        else
            hdr = Gesso.gguf_header(live_gguf)
            for (ggml_type, label) in
                ((Inf_.GGML_TYPE_Q2_K, "Q2_K"), (Inf_.GGML_TYPE_Q3_K, "Q3_K"))
                infos = [
                    t for
                    t in hdr.tensors if length(t.dims) == 3 && t.ggml_type == ggml_type
                ]
                if isempty(infos)
                    _skip(
                        "GSQ GGUF has no 3D $label expert tensor — live $label check skipped",
                    )
                    continue
                end
                live = Inf_._expert_slice(hdr, first(infos))
                ok, rel = kq_expert_check(
                    live,
                    live,
                    1,
                    randn(deterministic_rng(79), Float32, live.ne0),
                )
                @test ok
                @info "cuda K-quant live qgemv" type = label max_rel_err = rel
            end
        end
    end

    @testset "cuda hot experts accept a GSQ mixed K-quant triple" begin
        # GSQ-RCO picks a type per projection, so one layer's gate, up and down
        # can be Q2_K, Q3_K and Q4_K at once; the hot pool must still take it.
        ext = Base.get_extension(Gesso, :GessoCUDAExt)
        I_ = Gesso.Inference
        sl(t, ne0, ne1) =
            Gesso.GgufQuantSlice(UInt8[], 0, t, 4, ne0, ne1, 1024, C_NULL, Int64(1))
        @test ext._hot_supported(
            sl(I_.GGML_TYPE_Q2_K, 256, 64),
            sl(I_.GGML_TYPE_Q3_K, 256, 64),
            sl(I_.GGML_TYPE_Q4_K, 64, 256),
        )
        @test ext._hot_supported(
            sl(I_.GGML_TYPE_IQ2_XS, 256, 64),
            sl(I_.GGML_TYPE_Q2_K, 256, 64),
            sl(I_.GGML_TYPE_IQ4_XS, 64, 256),
        )
        @test !ext._hot_supported(
            sl(I_.GGML_TYPE_Q8_0, 256, 64),
            sl(I_.GGML_TYPE_Q2_K, 256, 64),
            sl(I_.GGML_TYPE_Q2_K, 64, 256),
        )
    end

    @testset "cuda qgemv activation-column table gathers each pair's token" begin
        # Hot-layer prefill runs (token, expert) pairs in one launch, each pair
        # reading its token's column of x. Activations quantize per column in
        # 32-value blocks, so a pair must match a one-column launch bit for bit.
        ext = Base.get_extension(Gesso, :GessoCUDAExt)
        gguf = get(ENV, "GESSO_ORNITH35_GGUF", "")
        if isfile(gguf)
            rng = deterministic_rng(57)
            hdr = Gesso.gguf_header(gguf)
            einfo = only(t for t in hdr.tensors if t.name == "blk.0.ffn_gate_exps.weight")
            live = Gesso.Inference._expert_slice(hdr, einfo)
            nb = live.bytes_per_expert
            q = CuArray{UInt8}(
                vcat((Vector{UInt8}(Inf_._quant_expert_bytes(live, e)) for e in 1:3)...),
            )
            ne0, ne1 = live.ne0, live.ne1
            seq = 4
            x = CuArray(randn(rng, Float32, ne0, seq))
            experts = [2, 1, 3, 2, 1]
            tokens = [3, 1, 2, 4, 3]
            n = length(experts)
            meta = CuArray(Int32[(experts .- 1) .* nb; 1:n; tokens])
            y = CUDA.fill(NaN32, ne1, n)
            ext._cuda_launch_qgemv!(
                y,
                x,
                q,
                live.ggml_type,
                meta,
                n,
                n,
                ne0,
                ne1,
                1.0f0,
                0.0f0;
                xcoloff=2n,
            )
            ref = similar(Array(y))
            for i in 1:n
                yi = CUDA.fill(NaN32, ne1, 1)
                mi = CuArray(Int32[(experts[i]-1)*nb, 1])
                ext._cuda_launch_qgemv!(
                    yi,
                    x[:, tokens[i]:tokens[i]],
                    q,
                    live.ggml_type,
                    mi,
                    1,
                    1,
                    ne0,
                    ne1,
                    1.0f0,
                    0.0f0,
                )
                ref[:, i] .= vec(Array(yi))
            end
            @test Array(y) == ref
            # red control: without the table, pair i reads column i
            y0 = CUDA.fill(NaN32, ne1, n)
            x5 = CuArray(hcat(Array(x), randn(rng, Float32, ne0)))
            ext._cuda_launch_qgemv!(
                y0,
                x5,
                q,
                live.ggml_type,
                meta,
                n,
                n,
                ne0,
                ne1,
                1.0f0,
                0.0f0,
            )
            @test Array(y0)[:, 4] == ref[:, 4]
            @test Array(y0)[:, 1] != ref[:, 1]
        else
            _skip("Ornith 35B GGUF not on disk — qgemv activation-column table skipped")
        end
    end

    @testset "cuda MoE GgufQuantSlice F32 pack matches dense" begin
        rng = deterministic_rng(21)
        seq, h, ff, n_exp, n_used = 2, 8, 4, 6, 2
        x = randn(rng, Float32, seq, h)
        router = randn(rng, Float32, n_exp, h)
        gate3 = randn(rng, Float32, n_exp, ff, h)
        up3 = randn(rng, Float32, n_exp, ff, h)
        down3 = randn(rng, Float32, n_exp, h, ff)
        y_cpu = zeros(Float32, seq, h)
        Inf_._moe_topk_swiglu!(cpu, y_cpu, x, router, gate3, up3, down3, n_used)
        y_gpu = CUDA.zeros(Float32, seq, h)
        Inf_._moe_topk_swiglu!(
            cuda,
            y_gpu,
            CuArray{Float32}(x),
            CuArray{Float32}(router),
            _pack_f32_slice(gate3, h, ff),
            _pack_f32_slice(up3, h, ff),
            _pack_f32_slice(down3, ff, h),
            n_used,
        )
        @test approx_eq(Array(y_gpu), y_cpu; atol=1e-3)
    end

    @testset "cuda host-expert MoE: router ties and staging regrowth" begin
        ext = Base.get_extension(Gesso, :GessoCUDAExt)
        # partialsort! must keep the full sort's tie order: higher probability
        # first, lower expert index among equals.
        rng = deterministic_rng(31)
        row = randn(rng, Float32, 64)
        m0 = maximum(row)
        row[[5, 9, 40]] .= m0 + 1
        row[[12, 13]] .= m0 + 0.5f0
        order = collect(1:64)
        experts = zeros(Int, 6)
        wbuf = zeros(Float32, 6)
        p = copy(row)
        ext._moe_route_row!(order, experts, wbuf, p, 64, 6)
        m = maximum(row)
        pr = exp.(row .- m)
        pr ./= sum(pr)
        @test experts == sort(1:64; by=e -> (-pr[e], e))[1:6]
        @test experts[1:3] == [5, 9, 40]
        @test sum(wbuf) ≈ 1

        # Pinned staging is sized by seq; grow, shrink back and grow again.
        h, ff, n_exp, n_used = 8, 4, 6, 2
        router = randn(rng, Float32, n_exp, h)
        gate3 = randn(rng, Float32, n_exp, ff, h)
        up3 = randn(rng, Float32, n_exp, ff, h)
        down3 = randn(rng, Float32, n_exp, h, ff)
        gs = _pack_f32_slice(gate3, h, ff)
        us = _pack_f32_slice(up3, h, ff)
        ds = _pack_f32_slice(down3, ff, h)
        for seq in (1, 3, 1, 5)
            x = randn(rng, Float32, seq, h)
            y_cpu = zeros(Float32, seq, h)
            Inf_._moe_topk_swiglu!(cpu, y_cpu, x, router, gate3, up3, down3, n_used)
            # the overlap is queued between the copies and the event wait in
            # one mode and after it in the other; either way it runs once and
            # its GPU work lands before the caller reads it
            for ov in ("1", "0")
                y_gpu = CUDA.fill(NaN32, seq, h)
                xd = CuArray{Float32}(x)
                side = CUDA.zeros(Float32, seq, h)
                ran = Ref(0)
                withenv("GESSO_MOE_OVERLAP" => ov) do
                    Inf_._moe_topk_swiglu!(
                        cuda,
                        y_gpu,
                        xd,
                        CuArray{Float32}(router),
                        gs,
                        us,
                        ds,
                        n_used;
                        overlap=() -> (ran[] += 1; side.=2 .* xd; nothing),
                    )
                end
                @test approx_eq(Array(y_gpu), y_cpu; atol=1e-3)
                @test ran[] == 1
                @test Array(side) == 2 .* x
            end
        end
    end

    @testset "cuda prefill route table matches the host route bit for bit" begin
        ext = Base.get_extension(Gesso, :GessoCUDAExt)
        rng = deterministic_rng(43)
        for (n_exp, n_used, seq) in ((256, 8, 37), (64, 6, 5), (256, 8, 300))
            L = randn(rng, Float32, seq, n_exp) .* 4
            # quantized rows tie often; equal logits must keep the lower index
            L[1:2:end, :] .= Float32.(rand(rng, -3:3, length(1:2:seq), n_exp))
            L[2, :] .= 0
            b = ext._moe_host_xfer(seq, n_exp, 8, n_used)
            ext._moe_route_rows_launch!(b, CuArray(L), seq, n_exp, n_used, CUDA.stream())
            CUDA.synchronize()
            ext._moe_route_rows_unpack!(b, seq, n_used)
            order = collect(1:n_exp)
            experts = zeros(Int, n_used)
            wbuf = zeros(Float32, n_used)
            ok = true
            for r in 1:seq
                wsum = ext._moe_route_row!(order, experts, wbuf, L[r, :], n_exp, n_used)
                ok &= b.E[:, r] == experts && b.W[:, r] == wbuf && b.live[r] == (wsum > 0)
            end
            @test ok
        end
    end

    @testset "cuda device MoE route matches the host route bit for bit" begin
        ext = Base.get_extension(Gesso, :GessoCUDAExt)
        rng = deterministic_rng(41)
        h, ff, n_exp, n_used = 16, 8, 256, 8
        gpu_slice(A3, ne0, ne1) = (
            s=_pack_f32_slice(A3, ne0, ne1);
            Gesso.GpuQuantSlice(
                CuArray(s.data),
                s.ggml_type,
                s.n_exp,
                ne0,
                ne1,
                s.bytes_per_expert,
                s.blck_size,
            )
        )
        gate3 = randn(rng, Float32, n_exp, ff, h)
        up3 = randn(rng, Float32, n_exp, ff, h)
        down3 = randn(rng, Float32, n_exp, h, ff)
        gs, us, ds = gpu_slice(gate3, h, ff), gpu_slice(up3, h, ff), gpu_slice(down3, ff, h)
        tied = [200, 7, 130, 64, 9, 251, 33, 180, 2, 99]
        for trial in 1:6
            x = randn(rng, Float32, 1, h)
            router = randn(rng, Float32, n_exp, h)
            if trial > 3
                # identical router rows give identical logits: the tie order
                # (lower index first) decides which of them make the top k
                lead = vec(x) .* (30 / sum(abs2, x))
                for e in tied
                    router[e, :] .= lead
                end
            end
            rd = CuArray(router)
            xd = CuArray(x)
            y_host = CUDA.fill(NaN32, 1, h)
            withenv("GESSO_MOE_DEVICE_ROUTE" => "0") do
                Inf_._moe_topk_swiglu!(cuda, y_host, xd, rd, gs, us, ds, n_used)
            end
            y_dev = CUDA.fill(NaN32, 1, h)
            ran = Ref(0)
            Inf_._moe_topk_swiglu!(
                cuda,
                y_dev,
                xd,
                rd,
                gs,
                us,
                ds,
                n_used;
                overlap=() -> (ran[] += 1; nothing),
            )
            @test Array(y_dev) == Array(y_host)
            @test ran[] == 1

            logits = Array(ext._dev_scratch(:moe_logits, 1, n_exp))
            order = collect(1:n_exp)
            experts = zeros(Int, n_used)
            wbuf = zeros(Float32, n_used)
            ext._moe_route_row!(order, experts, wbuf, vec(logits), n_exp, n_used)
            meta = Array(ext._moe_route_meta(n_used).g)
            @test meta[1:n_used] .÷ gs.bytes_per_expert .+ 1 == experts
            @test meta[(n_used+1):end] == 1:n_used
            @test Array(ext._dev_scratch(:moe_ws, n_used)) == wbuf
            trial > 3 && @test experts == sort(tied)[1:n_used]
        end
    end

    @testset "cuda device MoE prefill matches the token loop bit for bit" begin
        # A resident layer's prefill runs each slab as one launch over every
        # (token, expert) pair; the token loop it replaces is the reference.
        ext = Base.get_extension(Gesso, :GessoCUDAExt)
        gguf = get(ENV, "GESSO_ORNITH35_GGUF", "")
        if isfile(gguf)
            rng = deterministic_rng(73)
            hdr = Gesso.gguf_header(gguf)
            n_exp, n_used = 16, 8
            function resident(name)
                info = only(t for t in hdr.tensors if t.name == name)
                sl = Gesso.Inference._expert_slice(hdr, info)
                data = CuArray{UInt8}(
                    vcat(
                        (
                            Vector{UInt8}(Inf_._quant_expert_bytes(sl, e)) for e in 1:n_exp
                        )...,
                    ),
                )
                return Gesso.GpuQuantSlice(
                    data,
                    sl.ggml_type,
                    n_exp,
                    sl.ne0,
                    sl.ne1,
                    sl.bytes_per_expert,
                    sl.blck_size,
                )
            end
            gs = resident("blk.0.ffn_gate_exps.weight")
            us = resident("blk.0.ffn_up_exps.weight")
            ds = resident("blk.0.ffn_down_exps.weight")
            h = gs.ne0
            router = CuArray(0.05f0 .* randn(rng, Float32, n_exp, h))
            prefill(y, x, chunk) = withenv("GESSO_MOE_DEVICE_PREFILL_CHUNK" => chunk) do
                Inf_._moe_topk_swiglu!(cuda, y, x, router, gs, us, ds, n_used)
            end
            # chunk 16 over 37 tokens leaves a short last chunk; chunk 1 gives
            # one-column launches
            for (seq, chunk) in ((37, "16"), (5, "512"), (3, "1"))
                x = CuArray(randn(rng, Float32, seq, h))
                y_loop = CUDA.fill(NaN32, seq, h)
                withenv("GESSO_MOE_DEVICE_PREFILL" => "0") do
                    Inf_._moe_topk_swiglu!(cuda, y_loop, x, router, gs, us, ds, n_used)
                end
                y_pf = CUDA.fill(NaN32, seq, h)
                # the loop reference would pass this comparison too, so check
                # the batched path is the one that filled y_pf
                ext._MOE_PAIR_BATCH.meta_d = nothing
                prefill(y_pf, x, chunk)
                @test ext._MOE_PAIR_BATCH.meta_d !== nothing
                @test ext._moe_device_prefill_ok(y_pf, x, router, gs, us, ds, n_used)
                @test Array(y_pf) == Array(y_loop)
            end
            # red control: changing one token's activation moves its row alone
            seq = 9
            xh = randn(rng, Float32, seq, h)
            y1 = CUDA.fill(NaN32, seq, h)
            prefill(y1, CuArray(xh), "4")
            xh[4, :] .= randn(rng, Float32, h)
            y2 = CUDA.fill(NaN32, seq, h)
            prefill(y2, CuArray(xh), "4")
            a1, a2 = Array(y1), Array(y2)
            @test a1[[1:3; 5:seq], :] == a2[[1:3; 5:seq], :]
            @test a1[4, :] != a2[4, :]
        else
            _skip("Ornith 35B GGUF not on disk — device MoE prefill skipped")
        end
    end

    @testset "cuda device MoE layer with mixed Flash-Next types matches CPU" begin
        # Gate and up share one IQ2/IQ3 type, down is IQ4_NL or Q2_0, and every
        # expert is routed so each expert's scales reach the output. The CPU
        # reference decodes the same bytes; the device rounds activations to
        # int8, so the tolerance is relative to the output's largest magnitude.
        ext = Base.get_extension(Gesso, :GessoCUDAExt)
        rng = deterministic_rng(91)
        n_exp, n_used, h, ff = 4, 4, 256, 512
        tol = 2.0e-2
        layer_cases = (
            (
                Inf_.GGML_TYPE_IQ2_XXS,
                66,
                256,
                Inf_.GGML_TYPE_IQ4_NL,
                18,
                32,
                "IQ2_XXS gate/up, IQ4_NL down",
            ),
            (
                Inf_.GGML_TYPE_IQ2_S,
                82,
                256,
                Inf_.GGML_TYPE_Q2_0,
                18,
                64,
                "IQ2_S gate/up, Q2_0 down",
            ),
            (
                Inf_.GGML_TYPE_IQ3_S,
                110,
                256,
                Inf_.GGML_TYPE_IQ4_NL,
                18,
                32,
                "IQ3_S gate/up, IQ4_NL down",
            ),
        )

        # Payload bytes are random; each block's fp16 scale is written as a
        # valid value so the reference is finite and meaningful.
        function synth_host(ggml_type, bsz::Int, blck::Int, ne0::Int, ne1::Int)
            nb = div(ne0, blck)
            bpe = ne1 * nb * bsz
            data = rand(rng, UInt8, n_exp * bpe)
            for e in 0:(n_exp-1), r in 0:(ne1-1), b in 0:(nb-1)
                base = e * bpe + (r * nb + b) * bsz
                v = Float16((rand(rng) < 0.5 ? -1 : 1) * (0.01 + 0.05 * rand(rng)))
                data[(base+1):(base+2)] .= reinterpret(UInt8, [v])
            end
            return Gesso.GgufQuantSlice(
                data,
                0,
                ggml_type,
                n_exp,
                ne0,
                ne1,
                bpe,
                C_NULL,
                Int64(blck),
            )
        end

        # Device copy of a host layer. With `factor` != 1 the block scales of
        # expert `expert` are multiplied, which is the negative control.
        function to_device(host, bsz::Int; expert::Int=0, factor::Float32=1.0f0)
            parts = Vector{UInt8}[]
            for e in 1:host.n_exp
                b = copy(Vector{UInt8}(Inf_._quant_expert_bytes(host, e)))
                if e == expert
                    for s in 0:bsz:(length(b)-bsz)
                        bits = UInt16(b[s+1]) | (UInt16(b[s+2]) << 8)
                        nbits = reinterpret(
                            UInt16,
                            Float16(Float32(reinterpret(Float16, bits)) * factor),
                        )
                        b[s+1] = UInt8(nbits & 0xff)
                        b[s+2] = UInt8(nbits >> 8)
                    end
                end
                push!(parts, b)
            end
            return Gesso.GpuQuantSlice(
                CuArray{UInt8}(vcat(parts...)),
                host.ggml_type,
                host.n_exp,
                host.ne0,
                host.ne1,
                host.bytes_per_expert,
                host.blck_size,
            )
        end

        function cpu_layer(x, router, gate, up, down)
            y = zeros(Float32, size(x, 1), h)
            Inf_._moe_topk_swiglu!(cpu, y, x, router, gate, up, down, n_used)
            return y
        end

        function rel_err(y_dev, y_ref)
            return maximum(abs, Array(y_dev) .- y_ref) / maximum(abs, y_ref)
        end

        for (gtype, gbsz, gblck, dtype, dbsz, dblck, label) in layer_cases
            gate = synth_host(gtype, gbsz, gblck, h, ff)
            up = synth_host(gtype, gbsz, gblck, h, ff)
            down = synth_host(dtype, dbsz, dblck, ff, h)
            router = 0.05f0 .* randn(rng, Float32, n_exp, h)
            router_d = CuArray(router)
            gd = to_device(gate, gbsz)
            ud = to_device(up, gbsz)
            dd = to_device(down, dbsz)
            @test ext._hot_supported(gd, ud, dd)

            for seq in (1, 5)
                x = randn(rng, Float32, seq, h)
                y_ref = cpu_layer(x, router, gate, up, down)
                y_dev = CUDA.fill(NaN32, seq, h)
                Inf_._moe_topk_swiglu!(
                    cuda,
                    y_dev,
                    CuArray(x),
                    router_d,
                    gd,
                    ud,
                    dd,
                    n_used,
                )
                rel = rel_err(y_dev, y_ref)
                @test rel <= tol
                seq > 1 && @test ext._moe_device_prefill_ok(
                    y_dev,
                    CuArray(x),
                    router_d,
                    gd,
                    ud,
                    dd,
                    n_used,
                )
                @info "cuda mixed-type MoE layer" case = label seq max_rel_err = rel
            end

            # negative control: one expert's scale off by 8x must leave tolerance
            x = randn(rng, Float32, 5, h)
            y_ref = cpu_layer(x, router, gate, up, down)
            for (gb, db, which) in (
                (to_device(gate, gbsz; expert=2, factor=8.0f0), dd, "gate expert 2"),
                (gd, to_device(down, dbsz; expert=3, factor=8.0f0), "down expert 3"),
            )
                y_bad = CUDA.fill(NaN32, 5, h)
                Inf_._moe_topk_swiglu!(
                    cuda,
                    y_bad,
                    CuArray(x),
                    router_d,
                    gb,
                    ud,
                    db,
                    n_used,
                )
                rel_bad = rel_err(y_bad, y_ref)
                @test rel_bad > tol
                @info "cuda mixed-type MoE negative control" case = label corrupted = which max_rel_err =
                    rel_bad
            end
        end
    end

    @testset "cuda synthetic MoE block greedy ids match CPU F32" begin
        h, nh, nkv, hd, vocab = 8, 2, 2, 4, 16
        n_exp, k, ff = 4, 2, 8
        spec = Gesso.ArchitectureSpec(;
            family=:probe,
            hidden_size=h,
            num_layers=1,
            n_heads=nh,
            n_kv_heads=nkv,
            vocab_size=vocab,
            intermediate_size=ff,
            head_dim=hd,
            moe=Gesso.MoeSpec(;
                n_experts=n_exp,
                n_expert_used=k,
                expert_ff=ff,
                shared_ff=ff,
            ),
        )
        model = Gesso.spec_to_model(spec)
        rng = deterministic_rng(3)
        emb = randn(rng, Float32, vocab, h)
        function pw(sh)
            return Gesso.ProjectionWeight(; shape=sh, storage=randn(rng, Float32, sh...))
        end
        function fr(sh)
            return Gesso.FrozenParameter(; shape=sh, storage=ones(Float32, sh...))
        end
        bt = (
            mixer=:full_attention,
            n_expert_used=k,
            attn_rms=fr((h,)),
            wq=pw((nh * hd, h)),
            wk=pw((nkv * hd, h)),
            wv=pw((nkv * hd, h)),
            wo=pw((h, nh * hd)),
            ffn_rms=fr((h,)),
            wrouter=pw((n_exp, h)),
            wexp_gate=Gesso.ExpertWeight(;
                shape=(n_exp, ff, h),
                storage=randn(rng, Float32, n_exp, ff, h),
            ),
            wexp_up=Gesso.ExpertWeight(;
                shape=(n_exp, ff, h),
                storage=randn(rng, Float32, n_exp, ff, h),
            ),
            wexp_down=Gesso.ExpertWeight(;
                shape=(n_exp, h, ff),
                storage=randn(rng, Float32, n_exp, h, ff),
            ),
            wshared_gate=pw((ff, h)),
            wshared_up=pw((ff, h)),
            wshared_down=pw((h, ff)),
            wshared_gate_inp=fr((h,)),
        )
        tensors = (
            embedding=Gesso.EmbeddingTable(; shape=(vocab, h), storage=emb),
            blocks=[bt],
            lm_head=Gesso.EmbeddingTable(; shape=(vocab, h), storage=emb),
            final_rms=fr((h,)),
            rope=spec.rope,
            spec=spec,
        )
        ids_cpu = Gesso.reference_generate(
            model,
            tensors,
            [1];
            max_new_tokens=3,
            eps=spec.rms_norm_eps,
        )
        # Exact id equality with the CPU path assumes no hot-expert pool, which
        # is on by default now.
        gpu = withenv("GESSO_GPU_HOT_EXPERTS" => "0") do
            Gesso.to_device(cuda, tensors)
        end
        ids_gpu = Gesso.reference_generate(
            model,
            gpu,
            [1];
            max_new_tokens=3,
            eps=spec.rms_norm_eps,
            backend=cuda,
        )
        @test ids_gpu == ids_cpu

        gpu_off = withenv("GESSO_GPU_BUDGET_MIB" => "0", "GESSO_GPU_HOT_EXPERTS" => "0") do
            Gesso.to_device(cuda, tensors)
        end
        @test gpu_off.embedding.storage isa Gesso.HostOffload
        @test gpu_off.blocks[1].wexp_gate.storage isa Gesso.HostOffload
        ids_off = Gesso.reference_generate(
            model,
            gpu_off,
            [1];
            max_new_tokens=3,
            eps=spec.rms_norm_eps,
            backend=cuda,
        )
        @test ids_off == ids_cpu
    end
end
