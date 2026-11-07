# Shared-input quantized projections (CPU decode): one Q8_K quantization of the
# input, one threaded pass over the concatenated rows of several weights.
#
#   julia --project=test -t 8 test/test_cpu_matmul_group.jl
#
# Every output must be bit-identical to the separate `_quant_matmul!` of the
# same weight, which quantizes the same input the same way.

using Gesso
using Test
using Random: Xoshiro

const _GI = Gesso.Inference

# Random valid rows for a weight whose block starts with an f16 super-scale
# (IQ4_XS and Q4_K both do). The scale is finite, so every output is finite.
function _group_random_qm(rng, ty::Int32, nrows::Int, ne0::Int)
    rb = _GI._qrow_bytes(ty, ne0)
    data = rand(rng, UInt8, nrows * rb)
    bsz = _GI._iq_block_bytes(ty)
    for blk in 0:(nrows*div(rb, bsz)-1)
        bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
        data[blk*bsz+1] = UInt8(bits & 0xff)
        data[blk*bsz+2] = UInt8(bits >> 8)
    end
    return _GI.QuantMatrix(data, 0, ty, ne0, nrows, rb, Int64(_GI.QK_K))
end

_bits(a::AbstractArray{Float32}) = reinterpret(UInt32, vec(a))

@testset "shared-input group matmul" begin
    rng = Xoshiro(20261009)
    ne0 = 1024
    for ty in (_GI.GGML_TYPE_IQ4_XS, _GI.GGML_TYPE_Q4_K)
        # uneven row counts, including a weight shorter than one chunk
        ws = (
            _group_random_qm(rng, ty, 37, ne0),
            _group_random_qm(rng, ty, 5, ne0),
            _group_random_qm(rng, ty, 21, ne0),
        )
        x = randn(rng, Float32, 1, ne0)
        xq = _GI.Q8KVec(div(ne0, _GI.QK_K))
        _GI.quantize_q8_K!(xq, vec(x))
        dsts = map(w -> zeros(Float32, 1, w.nrows), ws)
        _GI._quant_matmul_group!(dsts, xq, ws)
        for (d, w) in zip(dsts, ws)
            ref = zeros(Float32, 1, w.nrows)
            _GI._quant_matmul!(ref, x, w)
            @test _bits(d) == _bits(ref)
        end
        # a single weight is the degenerate group
        d1 = (zeros(Float32, 1, ws[1].nrows),)
        _GI._quant_matmul_group!(d1, xq, (ws[1],))
        ref1 = zeros(Float32, 1, ws[1].nrows)
        _GI._quant_matmul!(ref1, x, ws[1])
        @test _bits(d1[1]) == _bits(ref1)
    end

    @testset "decode helper: group when every weight is Q8_K kind, matmul! otherwise" begin
        ne0 = 512
        x = Gesso.Activation(; shape=(1, ne0), storage=randn(rng, Float32, 1, ne0))
        ws = (
            _group_random_qm(rng, _GI.GGML_TYPE_IQ4_XS, 19, ne0),
            _group_random_qm(rng, _GI.GGML_TYPE_IQ4_XS, 8, ne0),
        )
        pw = map(w -> Gesso.ProjectionWeight((w.nrows, ne0), w), ws)
        xq = _GI.Q8KVec(div(ne0, _GI.QK_K))
        outs = map(
            w -> Gesso.Activation(; shape=(1, w.nrows), storage=zeros(Float32, 1, w.nrows)),
            ws,
        )
        _GI._dec_matmul_group!(
            _GI.CPUBackend(),
            outs,
            xq,
            x,
            pw,
            Gesso.Semantics.DecodeWorkload(),
        )
        for (o, w) in zip(outs, ws)
            ref = zeros(Float32, 1, w.nrows)
            _GI._quant_matmul!(ref, x.storage, w)
            @test _bits(o.storage) == _bits(ref)
        end

        # a Q8_0 weight has no Q8_K kernel: the helper falls back, same values
        w80 = _group_random_qm(rng, _GI.GGML_TYPE_Q8_0, 6, ne0)
        pw80 = (Gesso.ProjectionWeight((6, ne0), w80),)
        out80 = (Gesso.Activation(; shape=(1, 6), storage=zeros(Float32, 1, 6)),)
        _GI._dec_matmul_group!(
            _GI.CPUBackend(),
            out80,
            xq,
            x,
            pw80,
            Gesso.Semantics.DecodeWorkload(),
        )
        ref80 = zeros(Float32, 1, 6)
        _GI._quant_matmul!(ref80, x.storage, w80)
        @test _bits(out80[1].storage) == _bits(ref80)
        @test_throws ErrorException _GI._quant_matmul_group!(
            (zeros(Float32, 1, 6),),
            xq,
            (w80,),
        )

        # a Q8_K weight of another width cannot share `xq`: the helper falls
        # back to the per-weight matmul instead of erroring
        x2 = Gesso.Activation(; shape=(1, 2ne0), storage=randn(rng, Float32, 1, 2ne0))
        w2 = _group_random_qm(rng, _GI.GGML_TYPE_IQ4_XS, 7, 2ne0)
        out2 = (Gesso.Activation(; shape=(1, 7), storage=zeros(Float32, 1, 7)),)
        _GI._dec_matmul_group!(
            _GI.CPUBackend(),
            out2,
            xq,
            x2,
            (Gesso.ProjectionWeight((7, 2ne0), w2),),
            Gesso.Semantics.DecodeWorkload(),
        )
        ref2 = zeros(Float32, 1, 7)
        _GI._quant_matmul!(ref2, x2.storage, w2)
        @test _bits(out2[1].storage) == _bits(ref2)
    end
end

# Random K-quant rows with finite Float16 d (and dmin, where the type has one)
# at each type's own block offsets.
function _group_random_kq(rng, ty::Int32, nrows::Int, ne0::Int)
    rb = _GI._qrow_bytes(ty, ne0)
    data = rand(rng, UInt8, nrows * rb)
    bsz = _GI._ggml_traits(ty).type_size
    offs = filter(>=(0), [_GI._q8k_d_offset(ty), _GI._q8k_dmin_offset(ty)])
    for blk in 0:(nrows*div(rb, bsz)-1), off in offs
        bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
        data[blk*bsz+off+1] = UInt8(bits & 0xff)
        data[blk*bsz+off+2] = UInt8(bits >> 8)
    end
    return _GI.QuantMatrix(data, 0, ty, ne0, nrows, rb, Int64(_GI.QK_K))
end

# Dense decode now groups its projections too. A Q4_K_M or Q3_K_S GGUF mixes
# these types within one group (QKV, gate+up), so the bits must hold for each
# type and for a mixed group.
@testset "shared-input group matmul: every K-quant type, mixed groups" begin
    rng = Xoshiro(20261101)
    ne0 = 768
    tys = (
        _GI.GGML_TYPE_Q2_K,
        _GI.GGML_TYPE_Q3_K,
        _GI.GGML_TYPE_Q4_K,
        _GI.GGML_TYPE_Q5_K,
        _GI.GGML_TYPE_Q6_K,
    )
    x = randn(rng, Float32, 1, ne0)
    xq = _GI.Q8KVec(div(ne0, _GI.QK_K))
    _GI.quantize_q8_K!(xq, vec(x))
    for ty in tys
        @test _GI._quant_q8k_kind(_group_random_kq(rng, ty, 1, ne0))
    end
    ws = map((ty, n) -> _group_random_kq(rng, ty, n, ne0), tys, (29, 3, 64, 17, 40))
    dsts = map(w -> zeros(Float32, 1, w.nrows), ws)
    _GI._quant_matmul_group!(dsts, xq, ws)
    for (d, w) in zip(dsts, ws)
        ref = zeros(Float32, 1, w.nrows)
        _GI._quant_matmul!(ref, x, w)
        @test all(isfinite, ref)
        @test _bits(d) == _bits(ref)
    end
    # negative control: a group over a different input must not match
    xo = randn(rng, Float32, 1, ne0)
    xqo = _GI.Q8KVec(div(ne0, _GI.QK_K))
    _GI.quantize_q8_K!(xqo, vec(xo))
    do_ = map(w -> zeros(Float32, 1, w.nrows), ws)
    _GI._quant_matmul_group!(do_, xqo, ws)
    @test all(_bits(a) != _bits(b) for (a, b) in zip(do_, dsts))
end

# Decode groups read a prepped column: Q3_K..Q6_K rows run the one-column multi
# kernel, other types the plain dot. Either way each row must keep the bits of
# the plain group.
@testset "shared-input group matmul: prepped column, K = 1 multi kernel" begin
    rng = Xoshiro(20261031)
    ne0 = 768
    tys = (
        _GI.GGML_TYPE_Q2_K,
        _GI.GGML_TYPE_Q3_K,
        _GI.GGML_TYPE_Q4_K,
        _GI.GGML_TYPE_Q5_K,
        _GI.GGML_TYPE_Q6_K,
    )
    ws = map((ty, n) -> _group_random_kq(rng, ty, n, ne0), tys, (11, 33, 64, 17, 40))
    x = randn(rng, Float32, 1, ne0)
    xq = _GI.Q8KVec(div(ne0, _GI.QK_K))
    _GI.quantize_q8_K!(xq, vec(x))
    p = _GI._q6k_prep(_GI.Q8KVec(div(ne0, _GI.QK_K)))
    _GI.quantize_q8_K!(p.q, vec(x))
    _GI._q6k_prep!(p)
    plain = map(w -> zeros(Float32, 1, w.nrows), ws)
    prep = map(w -> zeros(Float32, 1, w.nrows), ws)
    _GI._quant_matmul_group!(plain, xq, ws)
    _GI._quant_matmul_group!(prep, p, ws)
    for (a, b) in zip(plain, prep)
        @test _bits(a) == _bits(b)
    end
    # each type alone, so a chunk split inside one weight is covered too
    for w in ws
        a = zeros(Float32, 1, w.nrows)
        b = zeros(Float32, 1, w.nrows)
        _GI._quant_matmul_group!((a,), xq, (w,))
        _GI._quant_matmul_group!((b,), p, (w,))
        @test _bits(a) == _bits(b)
    end

    # negative control: a prep left stale after requantizing must change the
    # prep-type rows, so those rows do read xp and mo
    _GI.quantize_q8_K!(p.q, randn(rng, Float32, ne0))
    stale = map(w -> zeros(Float32, 1, w.nrows), ws)
    _GI._quant_matmul_group!(stale, p, ws)
    _GI._q6k_prep!(p)
    fresh = map(w -> zeros(Float32, 1, w.nrows), ws)
    _GI._quant_matmul_group!(fresh, p, ws)
    if _GI.q8k_simd_enabled()
        for (ty, a, b) in zip(tys, stale, fresh)
            @test (_bits(a) == _bits(b)) == !_GI._q6k_prep_type(ty)
        end
    end

    # the decode helper quantizes into p.q and preps once per group
    xa = Gesso.Activation(; shape=(1, ne0), storage=randn(rng, Float32, 1, ne0))
    pw = map(w -> Gesso.ProjectionWeight((w.nrows, ne0), w), ws)
    outs = map(
        w -> Gesso.Activation(; shape=(1, w.nrows), storage=zeros(Float32, 1, w.nrows)),
        ws,
    )
    _GI._dec_matmul_group!(
        _GI.CPUBackend(),
        outs,
        _GI._q6k_prep(_GI.Q8KVec(div(ne0, _GI.QK_K))),
        xa,
        pw,
        Gesso.Semantics.DecodeWorkload(),
    )
    for (o, w) in zip(outs, ws)
        ref = zeros(Float32, 1, w.nrows)
        _GI._quant_matmul!(ref, xa.storage, w)
        @test all(isfinite, ref)
        @test _bits(o.storage) == _bits(ref)
    end
end

# The rows kernel runs a row range in one call: every row keeps the bits of
# its own K = 1 tile, and rows outside the range are left alone.
@testset "K = 1 rows kernel: row ranges match the per-row tile" begin
    if _GI.q8k_simd_enabled()
        rng = Xoshiro(20261101)
        ne0 = 768
        nb = div(ne0, _GI.QK_K)
        x = randn(rng, Float32, ne0)
        p = _GI._q6k_prep(_GI.Q8KVec(nb))
        _GI.quantize_q8_K!(p.q, x)
        _GI._q6k_prep!(p)
        for ty in (
            _GI.GGML_TYPE_Q3_K,
            _GI.GGML_TYPE_Q4_K,
            _GI.GGML_TYPE_Q5_K,
            _GI.GGML_TYPE_Q6_K,
        )
            n = 23
            w = _group_random_kq(rng, ty, n, ne0)
            rb = _GI._qrow_bytes(ty, ne0)
            ref = fill(NaN32, 1, n)
            out = zeros(Float32, 1)
            for j in 2:(n-1)
                _GI._dot_q8k_multi_kernel!(out, ty, w.data, w.offset0 + (j - 1) * rb, [p], 1, Val(1))
                ref[1, j] = 1.0f0 * out[1] + 0.0f0
                @test ref[1, j] == _GI._dot_q8k(ty, w.data, w.offset0 + (j - 1) * rb, p.q, true)
            end
            got = fill(NaN32, 1, n)
            _GI._dot_q8k_rows_k1!(got, ty, w.data, w.offset0, rb, p, 2, 9)
            _GI._dot_q8k_rows_k1!(got, ty, w.data, w.offset0, rb, p, 10, n - 1)
            _GI._dot_q8k_rows_k1!(got, ty, w.data, w.offset0, rb, p, 5, 4)
            @test all(isfinite, got[1, 2:(n-1)])
            @test isnan(got[1, 1]) && isnan(got[1, n])
            @test _bits(got) == _bits(ref)
            # past the weight's bytes, past dst, a row size that is not nb blocks
            big = zeros(Float32, 1, n + 1)
            @test_throws ErrorException _GI._dot_q8k_rows_k1!(big, ty, w.data, w.offset0, rb, p, 1, n + 1)
            @test_throws BoundsError _GI._dot_q8k_rows_k1!(got, ty, w.data, w.offset0, rb, p, 1, n + 1)
            @test_throws ErrorException _GI._dot_q8k_rows_k1!(got, ty, w.data, w.offset0, rb + 1, p, 1, 1)
        end
        w2 = _group_random_kq(rng, _GI.GGML_TYPE_Q2_K, 2, ne0)
        @test_throws ErrorException _GI._dot_q8k_rows_k1!(
            zeros(Float32, 1, 2),
            _GI.GGML_TYPE_Q2_K,
            w2.data,
            w2.offset0,
            _GI._qrow_bytes(_GI.GGML_TYPE_Q2_K, ne0),
            p,
            1,
            2,
        )
    end
end

# Decode (K = 1) Q6_K rows take the offset-word (mo) kernel when the prep has
# mo. It must give the biased kernel's bits and _dot_q8k's bits, on random and
# integer-extreme rows; a prep without mo (the prefill view) and
# GESSO_Q6K_K1_MO=0 take the biased kernel; a corrupted mo must change the bits.
@testset "K = 1 Q6_K mo kernel matches the biased kernel" begin
    if _GI.q8k_simd_enabled()
        rng = Xoshiro(20261107)
        ty = _GI.GGML_TYPE_Q6_K
        bsz = _GI._ggml_traits(ty).type_size
        for ne0 in (256, 768, 2816), xkind in (:randn, :extreme)
            nb = div(ne0, _GI.QK_K)
            rb = _GI._qrow_bytes(ty, ne0)
            n = 37
            x = xkind === :randn ? randn(rng, Float32, ne0) :
                rand(rng, (-1000.0f0, 1000.0f0), ne0)
            p = _GI._q6k_prep(_GI.Q8KVec(nb))
            _GI.quantize_q8_K!(p.q, x)
            _GI._q6k_prep!(p)
            @test _GI._q6k_has_mo(p)
            p0 = _GI.Q6KPrepVec(p.d, p.q, p.xp, p.xb)
            @test !_GI._q6k_has_mo(p0)
            w = _group_random_kq(rng, ty, n, ne0)
            # integer extremes: rows 3-5 all-ones / all-zero / mixed quants,
            # scales -128 / 127, d = 1.0
            for (r, qb, sc) in ((3, 0xff, 0x80), (4, 0x00, 0x7f), (5, 0x0f, 0x80))
                for blk in 0:(nb-1)
                    o = (r - 1) * rb + blk * bsz
                    w.data[o+1:o+192] .= qb == 0x0f ? rand(rng, (0x00, 0xff), 192) : qb
                    w.data[o+193:o+208] .= sc
                    w.data[o+209] = 0x00
                    w.data[o+210] = 0x3c
                end
            end
            ref = fill(NaN32, 1, n)
            _GI._dot_q8k_rows_k1!(ref, ty, w.data, w.offset0, rb, p0, 1, n)
            for j in 1:n
                @test ref[1, j] == _GI._dot_q8k(ty, w.data, w.offset0 + (j - 1) * rb, p.q, true)
            end
            got = fill(NaN32, 1, n)
            _GI._dot_q8k_rows_k1!(got, ty, w.data, w.offset0, rb, p, 2, 20)
            _GI._dot_q8k_rows_k1!(got, ty, w.data, w.offset0, rb, p, 21, n - 1)
            @test isnan(got[1, 1]) && isnan(got[1, n])
            @test _bits(got[1, 2:(n-1)]) == _bits(ref[1, 2:(n-1)])
            off = fill(NaN32, 1, n)
            withenv("GESSO_Q6K_K1_MO" => "0") do
                _GI._dot_q8k_rows_k1!(off, ty, w.data, w.offset0, rb, p, 1, n)
            end
            @test _bits(off) == _bits(ref)
            # refill: a stale mo is rewritten by _q6k_prep!
            fill!(p.mo, Int16(0))
            _GI._q6k_prep!(p)
            all1 = fill(NaN32, 1, n)
            _GI._dot_q8k_rows_k1!(all1, ty, w.data, w.offset0, rb, p, 1, n)
            @test _bits(all1) == _bits(ref)
            # negative control: one corrupted offset word must change the bits
            p.mo[1] += Int16(32)
            bad = fill(NaN32, 1, n)
            _GI._dot_q8k_rows_k1!(bad, ty, w.data, w.offset0, rb, p, 1, n)
            @test _bits(bad) != _bits(ref)
        end
    end
end

@testset "decode FFN activation on the pool matches the serial loop" begin
    rng = Xoshiro(20261101)
    cpu = _GI.CPUBackend()
    wl = Gesso.Semantics.DecodeWorkload()
    for n in (4096, 6656, 9729), geglu in (false, true)
        g = 4.0f0 .* randn(rng, Float32, 1, n)
        g[1:4] .= (-0.0f0, 0.0f0, 30.0f0, -30.0f0)
        u = randn(rng, Float32, 1, n)
        ref = zeros(Float32, 1, n)
        geglu ? Gesso._cpu_geglu!(ref, g, u) : Gesso._cpu_swiglu_range!(ref, g, u, 1:n)
        act = Gesso.Activation(; shape=(1, n), storage=fill(NaN32, 1, n))
        _GI._dec_ffn_act!(
            cpu,
            act,
            Gesso.Activation(; shape=(1, n), storage=g),
            Gesso.Activation(; shape=(1, n), storage=u),
            geglu,
            wl,
        )
        @test _bits(act.storage) == _bits(ref)

        # the chunks tile 1:n exactly, in any order
        nc = 7
        d = fill(NaN32, 1, n)
        f = _GI._DecActChunk{geglu, Matrix{Float32}}(d, g, u, n, nc)
        foreach(f, nc:-1:1)
        @test _bits(d) == _bits(ref)

        # negative control: a skipped chunk leaves its range unwritten
        d2 = fill(NaN32, 1, n)
        foreach(_GI._DecActChunk{geglu, Matrix{Float32}}(d2, g, u, n, nc), 1:(nc - 1))
        @test _bits(d2) != _bits(ref)
    end
end

@testset "decode KV append on the pool matches the broadcast" begin
    rng = Xoshiro(20261102)
    for (ctx, nkv, d) in ((104, 32, 64), (37, 8, 128), (9, 3, 50)), row in (1, 5, 9)
        kh = randn(rng, Float32, 1, nkv, d)
        vh = randn(rng, Float32, 1, nkv, d)
        kc0 = randn(rng, Float32, ctx, nkv, d)
        vc0 = randn(rng, Float32, ctx, nkv, d)
        kref, vref = copy(kc0), copy(vc0)
        @views kref[row, :, :] .= kh[1, :, :]
        @views vref[row, :, :] .= vh[1, :, :]
        kc, vc = copy(kc0), copy(vc0)
        _GI._kv_append_storage!(kc, vc, kh, vh, row)
        @test _bits(kc) == _bits(kref)
        @test _bits(vc) == _bits(vref)

        # the chunks tile the row exactly, in any order, and touch no other row
        n, nc = nkv * d, 7
        kc, vc = copy(kc0), copy(vc0)
        foreach(_GI._KVAppendChunk{Float32}(kc, vc, kh, vh, row, n, nc), nc:-1:1)
        @test _bits(kc) == _bits(kref)
        @test _bits(vc) == _bits(vref)

        # negative control: a skipped chunk leaves its part of the row unwritten
        kc, vc = copy(kc0), copy(vc0)
        foreach(_GI._KVAppendChunk{Float32}(kc, vc, kh, vh, row, n, nc), 1:(nc - 1))
        @test _bits(kc) != _bits(kref)
    end
    @test_throws BoundsError _GI._kv_append_storage!(
        zeros(Float32, 4, 32, 64), zeros(Float32, 4, 32, 64),
        zeros(Float32, 1, 32, 64), zeros(Float32, 1, 32, 64), 5)
end

# Random Q8_0 rows: a finite Float16 d at the head of each 34-byte block.
function _group_random_q80(rng, nrows::Int, ne0::Int)
    ty = _GI.GGML_TYPE_Q8_0
    rb = _GI._qrow_bytes(ty, ne0)
    data = rand(rng, UInt8, nrows * rb)
    for blk in 0:(nrows*div(rb, 34)-1)
        bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
        data[blk*34+1] = UInt8(bits & 0xff)
        data[blk*34+2] = UInt8(bits >> 8)
    end
    return _GI.QuantMatrix(data, 0, ty, ne0, nrows, rb, Int64(32))
end

# Gemma4 prefill projections (Q/K/V, dense gate/up) quantize x once when every
# weight takes the same activation; the 26B-A4B Q2_K_XL mixes a Q8_0 attn_k into
# some layers, where the shared path declines. Either way each output keeps the
# bits of its own _g4_gemm!.
@testset "Gemma4 prefill projections share one quantized x" begin
    rng = Xoshiro(20261105)
    ne0, seq = 512, 37
    x = randn(rng, Float32, seq, ne0)
    q5, q6 = _GI.GGML_TYPE_Q5_K, _GI.GGML_TYPE_Q6_K
    kq = (_group_random_kq(rng, q5, 48, ne0), _group_random_kq(rng, q6, 16, ne0), _group_random_kq(rng, q6, 16, ne0))
    mixed = (_group_random_kq(rng, q5, 48, ne0), _group_random_q80(rng, 16, ne0), _group_random_kq(rng, q6, 16, ne0))
    gu = (_group_random_kq(rng, q5, 40, ne0), _group_random_kq(rng, q5, 40, ne0))
    zs(ws) = map(w -> zeros(Float32, seq, w.nrows), ws)
    @test _GI._quant_matmul_shared!(zs(kq), x, kq)
    @test !_GI._quant_matmul_shared!(zs(mixed), x, mixed)
    refs = Dict{Any,Any}()
    for ws in (kq, mixed, gu)
        dsts = _GI._g4_gemms!(zs(ws), x, ws)
        refs[ws] = dsts
        for (d, w) in zip(dsts, ws)
            ref = _GI._g4_gemm!(zeros(Float32, seq, w.nrows), x, w)
            @test all(isfinite, ref)
            @test _bits(d) == _bits(ref)
        end
    end
    # negative control: the shared path over a perturbed x must not match
    neg = _GI._g4_gemms!(zs(kq), x .* 1.01f0, kq)
    @test all(_bits(a) != _bits(b) for (a, b) in zip(neg, refs[kq]))
end

# Qwen-style window prefill: projections of one normed x (GDN qkv/z, Q/K/V/gate,
# FFN gate/up) quantize x once through _window_matmul_shared!, and each output
# keeps the bits of its own _window_matmul!. A mixed group declines and runs
# each weight alone.
@testset "window prefill projections share one quantized x" begin
    rng = Xoshiro(20261106)
    ne0, seq = 512, 29
    x = randn(rng, Float32, seq, ne0)
    q4, q5, q6 = _GI.GGML_TYPE_Q4_K, _GI.GGML_TYPE_Q5_K, _GI.GGML_TYPE_Q6_K
    groups = (
        (_group_random_kq(rng, q4, 64, ne0), _group_random_kq(rng, q4, 32, ne0)),
        (_group_random_kq(rng, q4, 48, ne0), _group_random_kq(rng, q5, 16, ne0),
         _group_random_kq(rng, q6, 16, ne0), _group_random_kq(rng, q4, 48, ne0)),
        (_group_random_kq(rng, q5, 40, ne0), _group_random_q80(rng, 16, ne0)),
    )
    zs(qs) = map(q -> zeros(Float32, seq, q.nrows), qs)
    for qs in groups
        ws = map(q -> (storage=q,), qs)
        got = _GI._window_matmul_shared!(nothing, zs(qs), x, ws, nothing, nothing)
        for (d, w) in zip(got, ws)
            ref = _GI._window_matmul!(nothing, zeros(Float32, seq, w.storage.nrows), x, w, nothing, nothing)
            @test all(isfinite, ref)
            @test _bits(d) == _bits(ref)
        end
    end
    # negative control: a perturbed x through the shared path must not match
    qs = groups[2]
    ws = map(q -> (storage=q,), qs)
    neg = _GI._window_matmul_shared!(nothing, zs(qs), x .* 1.01f0, ws, nothing, nothing)
    for (d, w) in zip(neg, ws)
        ref = _GI._window_matmul!(nothing, zeros(Float32, seq, w.storage.nrows), x, w, nothing, nothing)
        @test _bits(d) != _bits(ref)
    end
end

# The MoE shared expert at decode takes the dense FFN's route (gate and up as
# one group over the prepped residual column, down over the prepped FFN
# column) in the decode scratch's FFN rows. Its output must keep the bits of
# the per-weight route that prefill and the device backends still take.
@testset "shared expert decode route: bit-identical to the per-weight route" begin
    rng = Xoshiro(20261107)
    dim, ff = 2048, 512
    wl = Gesso.Semantics.DecodeWorkload()
    zl = sz -> zeros(Float32, sz)
    pw(w) = Gesso.ProjectionWeight((w.nrows, w.ne0), w)
    bt = (;
        wshared_gate=pw(_group_random_kq(rng, _GI.GGML_TYPE_Q5_K, ff, dim)),
        wshared_up=pw(_group_random_kq(rng, _GI.GGML_TYPE_Q5_K, ff, dim)),
        wshared_down=pw(_group_random_kq(rng, _GI.GGML_TYPE_Q6_K, dim, ff)),
    )
    act(n) = Gesso.Activation(; shape=(1, n), storage=fill(NaN32, 1, n))
    xq_in = _GI._q6k_prep(_GI.Q8KVec(div(dim, _GI.QK_K)))
    xq_ff = _GI._q6k_prep(_GI.Q8KVec(div(ff, _GI.QK_K)))
    cpu = _GI.CPUBackend()
    for trial in 1:3
        # These random rows put |gate| near 1e3 for a unit-scale x, where
        # silu and gelu agree to more digits than Q8_K keeps of the act row;
        # at this scale gate is O(1) and the activation shows in the output.
        x = Gesso.Activation(;
            shape=(1, dim), storage=1f-3 .* randn(rng, Float32, 1, dim),
        )
        ref = Gesso.Activation(; shape=(1, dim), storage=zeros(Float32, 1, dim))
        _GI._shared_expert_thunk(cpu, ref, x, bt, wl, zl)()
        @test all(isfinite, ref.storage)
        sd = Gesso.Activation(; shape=(1, dim), storage=zeros(Float32, 1, dim))
        bufs = (act(ff), act(ff), act(ff))
        _GI._shared_expert_thunk(
            cpu, sd, x, bt, wl, zl; xq_in, xq_ff, ffn_bufs=bufs,
        )()
        @test _bits(sd.storage) == _bits(ref.storage)
        # each scratch row holds the per-weight route's gate, up and swiglu
        g = Gesso.Activation(; shape=(1, ff), storage=zeros(Float32, 1, ff))
        u = Gesso.Activation(; shape=(1, ff), storage=zeros(Float32, 1, ff))
        a = Gesso.Activation(; shape=(1, ff), storage=zeros(Float32, 1, ff))
        _GI.matmul!(cpu, g, x, bt.wshared_gate, wl)
        _GI.matmul!(cpu, u, x, bt.wshared_up, wl)
        _GI.swiglu!(cpu, a, g, u, wl)
        @test maximum(abs, g.storage) < 100
        @test _bits(bufs[1].storage) == _bits(g.storage)
        @test _bits(bufs[2].storage) == _bits(u.storage)
        @test _bits(bufs[3].storage) == _bits(a.storage)
        # rows of another width are not used, and the result is unchanged
        wide = (act(ff + 256), act(ff + 256), act(ff + 256))
        sw = Gesso.Activation(; shape=(1, dim), storage=zeros(Float32, 1, dim))
        _GI._shared_expert_thunk(
            cpu, sw, x, bt, wl, zl; xq_in, xq_ff, ffn_bufs=wide,
        )()
        @test _bits(sw.storage) == _bits(ref.storage)
        @test all(b -> all(isnan, b.storage), wide)
    end
end
