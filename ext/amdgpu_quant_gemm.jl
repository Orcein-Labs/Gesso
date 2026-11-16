# Prefill GEMM for device quant weights.
#
# The quant GEMV kernels launch one workgroup per (token, output row), so a
# prefill of n tokens decodes every weight n times. From GESSO_TUNE=dq_gemm_min
# tokens on (the CUDA and Lava default, 32), W is instead dequantized once into
# row chunks of an F32 scratch, capped by GESSO_TUNE=dq_gemm_scratch_mib, and
# the vendor BLAS (rocBLAS on device, OpenBLAS on KA CPU) multiplies X by each
# chunk. The scratch is F32, not the F16 that CUDA and Lava use: rocBLAS takes
# an all-F32 SGEMM directly, and the result keeps the GEMV's F32 weights.
# Element decode is the shared gguf_elem_decode.jl, the same code Lava runs.

_amdq_gemm_min() = env_tune(:dq_gemm_min, 32)
_amdq_gemm_scratch_mib() = env_tune(:dq_gemm_scratch_mib, 32)

# Element e of row r0 + r goes to Wt[r * ne0 + e + 1], so Wt is the (ne0 x rc)
# column-major matrix whose columns are the chunk's rows.
@kernel function _amdq_dequant_f32_kernel!(
    Wt,
    @Const(W),
    @Const(kv),
    @Const(g2),
    @Const(g3),
    @Const(ks),
    kind::Int,
    ne0::Int,
    row_bytes::Int,
    r0::Int,
    total::Int,
)
    idx = @index(Global, Linear)
    if idx <= total
        q = idx - 1
        r = q ÷ ne0
        e = q - r * ne0
        rowoff = (r0 + r) * row_bytes
        @inbounds Wt[idx] = _lq_elem(kind, W, kv, g2, g3, ks, rowoff, e)
    end
end

# Copies a chunk's columns into Y at flat offset `off`.
@kernel function _amdq_scatter_kernel!(Y, @Const(C), off::Int, total::Int)
    idx = @index(Global, Linear)
    if idx <= total
        @inbounds Y[off+idx] = C[idx]
    end
end

# Scratch outlives the call, so a prefill does not allocate per projection and
# leave finalizers to run inside a later step. Emptied once it holds
# _AMDQ_SCRATCH_MAX shapes, so many prompt lengths cannot grow VRAM unbounded.
const _AMDQ_SCRATCH = Dict{Tuple{Any, Symbol, Tuple{Vararg{Int}}}, Any}()
const _AMDQ_SCRATCH_MAX = 24
const _AMDQ_SCRATCH_LOCK = ReentrantLock()

function _amdq_scratch(ka, tag::Symbol, dims::Tuple{Vararg{Int}})
    lock(_AMDQ_SCRATCH_LOCK) do
        key = (ka, tag, dims)
        a = get(_AMDQ_SCRATCH, key, nothing)
        if a === nothing
            length(_AMDQ_SCRATCH) >= _AMDQ_SCRATCH_MAX && empty!(_AMDQ_SCRATCH)
            a = KA.allocate(ka, Float32, dims)
            _AMDQ_SCRATCH[key] = a
        end
        return a
    end
end

# Dequantizes rows r0 .. r0 + rc - 1 of q into the flat (ne0 * rc) buffer wtf.
function _amdq_dequant_rows!(wtf, q::AmdDenseQuant, r0::Int, rc::Int)
    kind = _lq_kind(q.ggml_type)
    kind === nothing && throw(
        gesso_error(
            ERR_INVALID_PLAN,
            "amdgpu quant GEMM: no element decoder for ggml type $(q.ggml_type)";
            ggml_type=q.ggml_type,
        ),
    )
    ka = KA.get_backend(wtf)
    tb = _amdq_iq_device_tables(ka)
    total = q.ne0 * rc
    _amdq_dequant_f32_kernel!(ka, 256)(
        wtf, q.data, tb.kv, tb.g2, tb.g3, tb.ks, kind, q.ne0, q.row_bytes, r0, total; ndrange=total,
    )
    return wtf
end

# Y[t, r] = sum_e X[t, e] * W[r, e], with W dequantized once per call.
function _amdq_gemm_prefill!(Y, X, q::AmdDenseQuant)
    ntok, k = size(X)
    nrows = q.nrows
    ka = KA.get_backend(X)
    rc_max = min(nrows, max(1, (_amdq_gemm_scratch_mib() << 20) ÷ (4 * k)))
    r0 = 0
    while r0 < nrows
        rc = min(rc_max, nrows - r0)
        # The dequant writes flat indices into the flat buffer; mul! takes the
        # reshape as a dense (k x rc) operand.
        wtf = _amdq_scratch(ka, :wt, (k * rc,))
        _amdq_dequant_rows!(wtf, q, r0, rc)
        wt = reshape(wtf, k, rc)
        if rc == nrows
            mul!(Y, X, wt, one(Float32), zero(Float32))
        else
            yc = _amdq_scratch(ka, :yc, (ntok, rc))
            mul!(yc, X, wt, one(Float32), zero(Float32))
            _amdq_scatter_kernel!(ka, 256)(Y, yc, r0 * ntok, ntok * rc; ndrange=ntok * rc)
        end
        r0 += rc
    end
    return Y
end
