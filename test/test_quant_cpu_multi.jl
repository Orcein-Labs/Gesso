# Multi-column Q8_K dot products (src/Inference/quant_cpu_multi.jl):
#   * every tile size 1..8, at several activation offsets, on random rows: each
#     column bit-equal (===) to the single-column _dot_q8k, for simd on and off
#   * the same on real rows when GESSO_SALUKI_GGUF is set (skipped otherwise)
#   * the prefill path through _quant_matmul! equals the per-token path bit for
#     bit, with the GESSO_PREFILL_MULTI switch on and off
#   * the tile kernels do not allocate after warm-up

using Random: Xoshiro
using Test: AbstractTestSet, Broken, get_testset, record

if !isdefined(@__MODULE__, :_skip)
    function _skip(msg::AbstractString)
        ts = get_testset()
        ts isa AbstractTestSet && record(ts, Broken(:skipped, String(msg)))
        return true
    end
end

const _MQ = Gesso.Inference

const _MQ_TYPES = (
    ("IQ2_XXS", _MQ.GGML_TYPE_IQ2_XXS),
    ("IQ3_S", _MQ.GGML_TYPE_IQ3_S),
    ("IQ2_S", _MQ.GGML_TYPE_IQ2_S),
    ("IQ1_S", _MQ.GGML_TYPE_IQ1_S),
    ("IQ1_M", _MQ.GGML_TYPE_IQ1_M),
    ("IQ2_XS", _MQ.GGML_TYPE_IQ2_XS),
    ("IQ3_XXS", _MQ.GGML_TYPE_IQ3_XXS),
    ("IQ4_XS", _MQ.GGML_TYPE_IQ4_XS),
    ("Q2_K", _MQ.GGML_TYPE_Q2_K),
    ("Q4_K", _MQ.GGML_TYPE_Q4_K),
)

# Random valid rows: arbitrary bytes everywhere, with finite small f16 scales at
# each format's super-scale offsets (random f16 bits are often NaN or Inf).
function _mq_scale_offsets(ggml_type::Int32)
    ggml_type == _MQ.GGML_TYPE_Q2_K && return (80, 82)
    ggml_type == _MQ.GGML_TYPE_Q4_K && return (0, 2)
    return (0,)
end

function _mq_random_rows(rng, ggml_type::Int32, nrows::Int, nb::Int)
    bsz = _MQ._iq_block_bytes(ggml_type)
    w = rand(rng, UInt8, nrows * nb * bsz)
    for blk in 0:(nrows * nb - 1)
        if ggml_type == _MQ.GGML_TYPE_IQ1_M
            _MQ._iq1m_write_scale!(w, blk * bsz, 0.005f0 + 0.05f0 * rand(rng, Float32))
            continue
        end
        for off in _mq_scale_offsets(ggml_type)
            bits = reinterpret(UInt16, Float16(0.005f0 + 0.05f0 * rand(rng, Float32)))
            w[blk * bsz + off + 1] = UInt8(bits & 0xff)
            w[blk * bsz + off + 2] = UInt8(bits >> 8)
        end
    end
    return w
end

_mq_acts(rng, ne0::Int, n::Int) = [_MQ.quantize_q8_K!(_MQ.Q8KVec(div(ne0, 256)), randn(rng, Float32, ne0)) for _ in 1:n]

# Checks every column of a tile of `ntile` activations starting at t0 against
# the single-column kernel, for both simd modes, on the rows of w.
function _mq_check_tiles(ggml_type::Int32, w, nrows::Int, rb::Int, xqs, t0::Int, ntile::Int)
    nmis = 0
    out = zeros(Float32, length(xqs))
    for simd in (true, false), j in 1:nrows
        woff = (j - 1) * rb
        _MQ._dot_q8k_multi!(out, ggml_type, w, woff, xqs, t0, ntile, simd)
        for c in 1:ntile
            ref = _MQ._dot_q8k(ggml_type, w, woff, xqs[t0 + c - 1], simd)
            nmis += !(out[t0 + c - 1] === ref)
        end
    end
    return nmis
end

@testset "multi-column dot: every type, every tile size, bit-equal to _dot_q8k" begin
    rng = Xoshiro(0x51ce)
    nb = 3
    ne0 = 256 * nb
    nrows = 4
    for (name, ty) in _MQ_TYPES
        rb = _MQ._qrow_bytes(ty, ne0)
        w = _mq_random_rows(rng, ty, nrows, nb)
        xqs = _mq_acts(rng, ne0, 9)
        @testset "$name" begin
            for ntile in 1:8, t0 in (1, 2)
                nmis = _mq_check_tiles(ty, w, nrows, rb, xqs, t0, ntile)
                @test nmis == 0
            end
        end
    end
end

@testset "multi-column dot: the tile kernels do not allocate" begin
    rng = Xoshiro(7)
    nb = 4
    ne0 = 256 * nb
    for ty in (_MQ.GGML_TYPE_IQ2_XXS, _MQ.GGML_TYPE_IQ3_S, _MQ.GGML_TYPE_Q4_K)
        rb = _MQ._qrow_bytes(ty, ne0)
        w = _mq_random_rows(rng, ty, 1, nb)
        xqs = _mq_acts(rng, ne0, 4)
        out = zeros(Float32, 4)
        _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 4, true)
        a = @allocated _MQ._dot_q8k_multi!(out, ty, w, 0, xqs, 1, 4, true)
        @test a == 0
    end
end

@testset "prefill: _quant_matmul! equals the per-token path bit for bit" begin
    rng = Xoshiro(0xbeef)
    nb = 3
    ne0 = 256 * nb
    nrows = 37
    seq = 7
    for (name, ty) in _MQ_TYPES
        rb = _MQ._qrow_bytes(ty, ne0)
        data = _mq_random_rows(rng, ty, nrows, nb)
        q = _MQ.QuantMatrix(data, 0, ty, ne0, nrows, rb, C_NULL, Int64(256))
        x = randn(rng, Float32, seq, ne0)
        saved = _MQ._PREFILL_MULTI[]
        dst_multi = zeros(Float32, seq, nrows)
        dst_per_token = zeros(Float32, seq, nrows)
        try
            _MQ._PREFILL_MULTI[] = Int8(1)
            _MQ._quant_matmul!(dst_multi, x, q)
            _MQ._PREFILL_MULTI[] = Int8(0)
            _MQ._quant_matmul!(dst_per_token, x, q)
        finally
            _MQ._PREFILL_MULTI[] = saved
        end
        @testset "$name" begin
            @test all(reinterpret.(UInt32, dst_multi) .== reinterpret.(UInt32, dst_per_token))
        end
    end
end

@testset "GESSO_PREFILL_MULTI switch is read as a flag" begin
    @test _MQ._prefill_multi_enabled() isa Bool
end

@testset "real tensors: multi-column vs single-column (Saluki GGUF)" begin
    path = get(ENV, "GESSO_SALUKI_GGUF", "")
    if !isfile(path)
        _skip("set GESSO_SALUKI_GGUF to the Saluki 27B GGUF — real multi-column rows skipped")
    else
        hdr = _MQ.gguf_header(path)
        rng = Xoshiro(31)
        for (name, ty) in _MQ_TYPES
            idx = findfirst(t -> t.ggml_type == ty, hdr.tensors)
            if idx === nothing
                _skip("no $name tensor in $(basename(path))")
                continue
            end
            info = hdr.tensors[idx]
            ne0, ne1 = info.dims[1], info.dims[2]
            rb = _MQ._qrow_bytes(ty, ne0)
            base = Int(hdr.data_start) + Int(info.offset)
            xqs = _mq_acts(rng, ne0, 8)
            rows = unique(round.(Int, range(1, ne1; length=min(8, ne1))))
            nmis = 0
            out = zeros(Float32, 8)
            for simd in (true, false), j in rows, ntile in (1, 4, 8)
                _MQ._dot_q8k_multi!(out, ty, hdr.data, base + (j - 1) * rb, xqs, 1, ntile, simd)
                for c in 1:ntile
                    ref = _MQ._dot_q8k(ty, hdr.data, base + (j - 1) * rb, xqs[c], simd)
                    nmis += !(out[c] === ref)
                end
            end
            @testset "$name" begin
                @test nmis == 0
            end
        end
    end
end
