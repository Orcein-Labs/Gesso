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

if !CUDA_LOADED || !CUDA.functional()
    _skip(
        "no NVIDIA device — CUDA MoE offload tests skipped (§LXXVII skip law)",
    )
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
            :probe, cuda, Gesso.Activation(; shape=(2, 2), storage=ho),
        ) === nothing
        slice = _pack_f32_slice(ones(Float32, 2, 2, 2), 2, 2)
        @test Inf_._check_device_storage(
            :probe, cuda, Gesso.ExpertWeight(; shape=size(slice), storage=slice),
        ) === nothing
        gq = Gesso.Inference.gpu_quant_slice(slice, Vector{UInt8}(Gesso.Inference.packed_quant_bytes(slice)))
        @test Inf_._check_device_storage(
            :probe, cuda, Gesso.ExpertWeight(; shape=size(gq), storage=gq),
        ) === nothing
        err = try
            Inf_._check_device_storage(
                :probe, cuda, Gesso.Activation(; shape=(2, 2), storage=ones(Float32, 2, 2)),
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

        gguf = get(
            ENV,
            "GESSO_ORNITH35_GGUF",
            "",
        )
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
            W = Float64.(permutedims(reshape(
                Inf_._dequant_f32(bytes, 0, live.ggml_type, live.ne0 * live.ne1),
                live.ne0, live.ne1)))
            x64 = Float64.(xl)
            ref = W * x64
            nb = div(live.ne0, 32)
            bound = zeros(Float64, live.ne1)
            for b in 1:nb
                cols = (32 * (b - 1) + 1):(32 * b)
                bound .+= maximum(abs, view(x64, cols)) / 254.0 .* vec(sum(abs, view(W, :, cols); dims=2))
            end
            tol = bound .+ 5.0e-5 .* (abs.(W) * abs.(x64)) .+ 1.0e-9
            @test all(abs.(Float64.(Array(yg)) .- ref) .<= tol)
        else
            _skip("Ornith 35B GGUF not on disk — live IQ2_XS CUDA qgemv skipped")
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
                        cuda, y_gpu, xd, CuArray{Float32}(router), gs, us, ds, n_used;
                        overlap=() -> (ran[] += 1; side .= 2 .* xd; nothing),
                    )
                end
                @test approx_eq(Array(y_gpu), y_cpu; atol=1e-3)
                @test ran[] == 1
                @test Array(side) == 2 .* x
            end
        end
    end

    @testset "cuda device MoE route matches the host route bit for bit" begin
        ext = Base.get_extension(Gesso, :GessoCUDAExt)
        rng = deterministic_rng(41)
        h, ff, n_exp, n_used = 16, 8, 256, 8
        gpu_slice(A3, ne0, ne1) = (s = _pack_f32_slice(A3, ne0, ne1);
            Gesso.GpuQuantSlice(CuArray(s.data), s.ggml_type, s.n_exp, ne0, ne1, s.bytes_per_expert, s.blck_size))
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
            Inf_._moe_topk_swiglu!(cuda, y_dev, xd, rd, gs, us, ds, n_used;
                overlap=() -> (ran[] += 1; nothing))
            @test Array(y_dev) == Array(y_host)
            @test ran[] == 1

            logits = Array(ext._dev_scratch(:moe_logits, 1, n_exp))
            order = collect(1:n_exp)
            experts = zeros(Int, n_used)
            wbuf = zeros(Float32, n_used)
            ext._moe_route_row!(order, experts, wbuf, vec(logits), n_exp, n_used)
            meta = Array(ext._moe_route_meta(n_used).g)
            @test meta[1:n_used] .÷ gs.bytes_per_expert .+ 1 == experts
            @test meta[(n_used + 1):end] == 1:n_used
            @test Array(ext._dev_scratch(:moe_ws, n_used)) == wbuf
            trial > 3 && @test experts == sort(tied)[1:n_used]
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
            moe=Gesso.MoeSpec(; n_experts=n_exp, n_expert_used=k, expert_ff=ff, shared_ff=ff),
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
            model, tensors, [1]; max_new_tokens=3, eps=spec.rms_norm_eps,
        )
        gpu = Gesso.to_device(cuda, tensors)
        ids_gpu = Gesso.reference_generate(
            model, gpu, [1]; max_new_tokens=3, eps=spec.rms_norm_eps, backend=cuda,
        )
        @test ids_gpu == ids_cpu

        gpu_off = withenv("GESSO_GPU_BUDGET_MIB" => "0") do
            Gesso.to_device(cuda, tensors)
        end
        @test gpu_off.embedding.storage isa Gesso.HostOffload
        @test gpu_off.blocks[1].wexp_gate.storage isa Gesso.HostOffload
        ids_off = Gesso.reference_generate(
            model, gpu_off, [1]; max_new_tokens=3, eps=spec.rms_norm_eps, backend=cuda,
        )
        @test ids_off == ids_cpu
    end
end
