# Lava expert GEMV on K-quant expert banks (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K). Each
# device result is checked against the F64 product of the host expert matrix
# (gguf_dequant_expert) with x, plus alpha and beta. Without a Vulkan device the
# whole file is a named skip. The real-model check reads four experts
# of two layers from GESSO_ORNITH35_GGUF and is skipped when that is unset.
#
# Tolerance: |dev - ref| <= 1e-4 * max(1, max|ref|). The weight side decodes the
# same bytes on both sides, so the only error is F32 accumulation.

using Test: AbstractTestSet, Broken, get_testset, record
using Random: MersenneTwister

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

if !LAVA_LOADED || !VULKAN_OK
    _skip("no usable Vulkan device — Lava K-quant expert tests skipped")
else
    const _KX_INF = Gesso.Inference
    const _KX_LAVA = Gesso.LavaBackend()

    # (block bytes, byte offset of d, byte offset of dmin or -1) per K-quant type.
    # Q3_K and Q6_K have no dmin.
    const _KX_LAYOUT = Dict(
        :q2_k => (84, 80, 82),
        :q3_k => (110, 108, -1),
        :q4_k => (144, 0, 2),
        :q5_k => (176, 0, 2),
        :q6_k => (210, 208, -1),
    )
    const _KX_TYPE = Dict(
        :q2_k => _KX_INF.GGML_TYPE_Q2_K,
        :q3_k => _KX_INF.GGML_TYPE_Q3_K,
        :q4_k => _KX_INF.GGML_TYPE_Q4_K,
        :q5_k => _KX_INF.GGML_TYPE_Q5_K,
        :q6_k => _KX_INF.GGML_TYPE_Q6_K,
    )

    _kx_put_f16!(data::Vector{UInt8}, o::Int, v::Float64) =
        data[(o+1):(o+2)] .= reinterpret(UInt8, [Float16(v)])

    # Random superblock bytes with finite f16 scales: every other byte is random,
    # so every nibble and high bit is exercised while the scales stay in range.
    function _kx_random(rng, kind::Symbol, n_exp::Int, ne0::Int, ne1::Int)
        bbytes, doff, moff = _KX_LAYOUT[kind]
        nblk = div(ne0, 256)
        bpe = nblk * bbytes * ne1
        data = rand(rng, UInt8, n_exp * bpe)
        for b in 0:(n_exp*ne1*nblk-1)
            o = b * bbytes
            _kx_put_f16!(data, o + doff, (rand(rng) * 0.04 + 0.002) * (rand(rng) < 0.5 ? -1 : 1))
            moff >= 0 && _kx_put_f16!(data, o + moff, rand(rng) * 0.04 + 0.002)
        end
        return data, bpe
    end

    # Host slice over the bytes, with the ggml traits the reference dequant uses.
    function _kx_slice(data::Vector{UInt8}, kind::Symbol, n_exp::Int, ne0::Int, ne1::Int, bpe::Int)
        t = _KX_TYPE[kind]
        tr = _KX_INF._ggml_traits(t)
        return _KX_INF.GgufQuantSlice(
            data, 0, t, n_exp, ne0, ne1, bpe, Int64(tr.blck_size),
        )
    end

    function _kx_bank(s)
        return _KX_INF.GpuQuantSlice(
            Lava.LavaArray{UInt8}(collect(_KX_INF.packed_quant_bytes(s))),
            s.ggml_type,
            s.n_exp,
            s.ne0,
            s.ne1,
            s.bytes_per_expert,
            s.blck_size,
        )
    end

    function _kx_gemv(bank, e::Int, xv::Vector{Float32}, y0::Vector{Float32}, alpha, beta)
        x = Lava.LavaArray{Float32}(xv)
        y = Lava.LavaArray{Float32}(y0)
        _KX_INF._qgemv_expert!(y, x, bank, e; alpha=alpha, beta=beta)
        return Array(y)
    end

    # Largest absolute error against the F64 reference, and the tolerance it must
    # stay under.
    function _kx_bound(s, e::Int, xv, y0, alpha, beta, dev)
        W = Float64.(_KX_INF.gguf_dequant_expert(s, e))
        ref = Float64(alpha) .* (W * Float64.(xv)) .+ Float64(beta) .* Float64.(y0)
        tol = 1.0e-4 * max(1.0, maximum(abs, ref))
        return maximum(abs.(Float64.(dev) .- ref)), tol
    end

    @testset "Lava K-quant expert GEMV: synthetic banks" begin
        for kind in (:q2_k, :q3_k, :q4_k, :q5_k, :q6_k), ne0 in (512, 2048)
            @testset "$kind ne0=$ne0" begin
                rng = MersenneTwister(20261027 + ne0)
                ne1 = 64
                data, bpe = _kx_random(rng, kind, 3, ne0, ne1)
                s = _kx_slice(data, kind, 3, ne0, ne1, bpe)
                bank = _kx_bank(s)
                xv = randn(rng, Float32, ne0)
                for (e, alpha, beta) in ((1, 1.0f0, 0.0f0), (2, 1.5f0, 0.5f0), (3, -0.75f0, 1.0f0))
                    y0 = randn(rng, Float32, ne1)
                    dev = _kx_gemv(bank, e, xv, y0, alpha, beta)
                    err, tol = _kx_bound(s, e, xv, y0, alpha, beta, dev)
                    @test err <= tol
                end
            end
        end
    end

    # Negative controls: a bank with the field perturbed must miss the same bound.
    # Each one is the device result on the perturbed bytes against the reference of
    # the unperturbed bytes, so the check fires only if the decoder reads that field.
    @testset "Lava K-quant expert GEMV: negative controls fire" begin
        ne0, ne1 = 512, 64
        rng = MersenneTwister(20261028)
        xv = randn(rng, Float32, ne0)
        y0 = zeros(Float32, ne1)

        # Q2_K: swap the scale and min nibbles of every scales byte (bytes 0..15).
        data, bpe = _kx_random(rng, :q2_k, 3, ne0, ne1)
        s = _kx_slice(data, :q2_k, 3, ne0, ne1, bpe)
        bad = copy(data)
        for b in 0:(3*ne1*(ne0 ÷ 256)-1), i in 0:15
            o = b * 84 + i + 1
            bad[o] = (bad[o] << 4) | (bad[o] >> 4)
        end
        dev = _kx_gemv(_kx_bank(_kx_slice(bad, :q2_k, 3, ne0, ne1, bpe)), 1, xv, y0, 1.0f0, 0.0f0)
        err, tol = _kx_bound(s, 1, xv, y0, 1.0f0, 0.0f0, dev)
        println("negative control q2_k nibble swap: err=$err tol=$tol")
        @test err > tol

        # Q3_K: invert the hmask bit sense (bytes 0..31 of each superblock).
        data, bpe = _kx_random(rng, :q3_k, 3, ne0, ne1)
        s = _kx_slice(data, :q3_k, 3, ne0, ne1, bpe)
        bad = copy(data)
        for b in 0:(3*ne1*(ne0 ÷ 256)-1), l in 0:31
            bad[b * 110 + l + 1] = ~bad[b * 110 + l + 1]
        end
        dev = _kx_gemv(_kx_bank(_kx_slice(bad, :q3_k, 3, ne0, ne1, bpe)), 1, xv, y0, 1.0f0, 0.0f0)
        err, tol = _kx_bound(s, 1, xv, y0, 1.0f0, 0.0f0, dev)
        println("negative control q3_k hmask inversion: err=$err tol=$tol")
        @test err > tol
    end

    # One expert slice per layer for the Q4_K up bank and a Q3_K bank, read from the
    # real header. The Q3_K bank comes from a different layer than the Q4_K one when
    # the file has one.
    _kx_layer(name::AbstractString) =
        (m = match(r"blk\.(\d+)\.", name)) === nothing ? -1 : parse(Int, m.captures[1])

    @testset "Lava K-quant expert GEMV: real Ornith-35B expert banks" begin
        path = get(ENV, "GESSO_ORNITH35_GGUF", "")
        if isempty(path)
            _skip("GESSO_ORNITH35_GGUF not set — Ornith-35B K-quant expert check skipped")
        else
            hdr = _KX_INF.gguf_header(path)
            expert(info, t) =
                info.ggml_type == t && length(info.dims) == 3 && _KX_INF._is_expert_tensor(info.name)
            iq4 = findfirst(info -> expert(info, _KX_INF.GGML_TYPE_Q4_K) && occursin("up", lowercase(info.name)), hdr.tensors)
            iq4 === nothing && error("no Q4_K up expert bank in $path")
            layer4 = _kx_layer(hdr.tensors[iq4].name)
            iq3 = findfirst(
                info -> expert(info, _KX_INF.GGML_TYPE_Q3_K) && _kx_layer(info.name) != layer4,
                hdr.tensors,
            )
            iq3 === nothing && (iq3 = findfirst(info -> expert(info, _KX_INF.GGML_TYPE_Q3_K), hdr.tensors))
            iq3 === nothing && error("no Q3_K expert bank in $path")
            rng = MersenneTwister(20261029)
            for info in (hdr.tensors[iq4], hdr.tensors[iq3])
                @testset "$(info.name)" begin
                    t = info.ggml_type
                    ne0, ne1, n_exp = info.dims
                    tr = _KX_INF._ggml_traits(t)
                    rb = div(ne0, Int(tr.blck_size)) * Int(tr.type_size)
                    bpe = rb * ne1
                    n = min(4, n_exp)
                    s = _KX_INF.GgufQuantSlice(
                        hdr.data,
                        Int(hdr.data_start) + Int(info.offset),
                        t,
                        n,
                        ne0,
                        ne1,
                        bpe,
                        Int64(tr.blck_size),
                    )
                    bank = _kx_bank(s)
                    xv = randn(rng, Float32, ne0)
                    y0 = randn(rng, Float32, ne1)
                    for e in 1:n
                        dev = _kx_gemv(bank, e, xv, y0, 1.5f0, 0.5f0)
                        err, tol = _kx_bound(s, e, xv, y0, 1.5f0, 0.5f0, dev)
                        @test err <= tol
                    end
                end
            end
        end
    end
end
