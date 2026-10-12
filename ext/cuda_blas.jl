# A pool allocation that fails walks CUDACore's reclaim ladder up to
# RECLAIM_DROP, which clears cuBLAS's task-local handle and destroys the idle
# ones. The next CUBLAS.handle() then runs cublasCreate, a driver allocation
# outside the pool with no retry, and with hot-expert pools pinned across the
# device it fails mid-prefill. Gesso keeps one handle of its own, created
# before the weights and pools are placed, that no reclaim can reach. Its
# state follows CUBLAS.handle(): device pointer mode, the active stream and
# the task's math mode, so results are the same as through CUDA.jl's handle.
#
# The handle belongs to the task that placed the weights. cuBLAS handles must
# not be shared by concurrent tasks, so any other task gets CUDA.jl's own.
mutable struct _GessoBlas
    handle::CUDA.CUBLAS.cublasHandle_t
    ctx::Union{Nothing,CUDA.CuContext}
    task::Union{Nothing,Task}
    stream::Union{Nothing,CUDA.CuStream}
    math_mode::CUDA.CUDACore.MathMode
    math_precision::Symbol
end

const _BLAS = _GessoBlas(CUDA.CUBLAS.cublasHandle_t(C_NULL), nothing, nothing, nothing,
    CUDA.CUDACore.DEFAULT_MATH, :none)

function _blas_reserve!()
    C = CUDA.CUBLAS
    B = _BLAS
    st = CUDA.CUDACore.active_state()
    if B.ctx != st.context
        # a handle of a context that is gone is not destroyed: its memory went with it
        B.handle = C.cublasCreate()
        C.cublasSetPointerMode_v2(B.handle, C.CUBLAS_POINTER_MODE_DEVICE)
        B.ctx = st.context
        B.stream = nothing
        B.math_precision = :none
    end
    B.task = current_task()
    return
end

function _blas_handle()
    C = CUDA.CUBLAS
    B = _BLAS
    st = CUDA.CUDACore.active_state()
    (B.task === current_task() && B.ctx == st.context) || return C.handle()
    if B.stream != st.stream
        C.cublasSetStream_v2(B.handle, st.stream)
        B.stream = st.stream
    end
    if B.math_mode != st.math_mode || B.math_precision != st.math_precision
        C.math_mode!(B.handle, st.math_mode, st.math_precision)
        B.math_mode = st.math_mode
        B.math_precision = st.math_precision
    end
    return B.handle
end

# CUBLAS.gemm! and gemv! on the Gesso handle, same argument checks and
# leading dimensions. Element types without a case here go to CUDA.jl.
for (gemm, gemv, T) in ((:cublasSgemm_v2_64, :cublasSgemv_v2_64, :Float32),
                        (:cublasDgemm_v2_64, :cublasDgemv_v2_64, :Float64))
    @eval begin
        function _blas_gemm!(tA::Char, tB::Char, alpha, A::CUDA.StridedCuVecOrMat{$T},
                             B::CUDA.StridedCuVecOrMat{$T}, beta, Cm::CUDA.StridedCuVecOrMat{$T})
            CUDA.CUBLAS.version() >= v"12.0" || return CUDA.CUBLAS.gemm!(tA, tB, alpha, A, B, beta, Cm)
            m = size(A, tA == 'N' ? 1 : 2)
            k = size(A, tA == 'N' ? 2 : 1)
            n = size(B, tB == 'N' ? 2 : 1)
            (m == size(Cm, 1) && n == size(Cm, 2) && k == size(B, tB == 'N' ? 1 : 2)) ||
                throw(DimensionMismatch("gemm: A $(size(A)), B $(size(B)), C $(size(Cm))"))
            CUDA.CUBLAS.$gemm(_blas_handle(), tA, tB, m, n, k, alpha, A, max(1, stride(A, 2)),
                B, max(1, stride(B, 2)), beta, Cm, max(1, stride(Cm, 2)))
            return Cm
        end
        function _blas_gemv!(tr::Char, alpha, A::CUDA.StridedCuMatrix{$T},
                             x::CUDA.StridedCuVector{$T}, beta, y::CUDA.StridedCuVector{$T})
            CUDA.CUBLAS.version() >= v"12.0" || return CUDA.CUBLAS.gemv!(tr, alpha, A, x, beta, y)
            m, n = size(A)
            (length(x) == (tr == 'N' ? n : m) && length(y) == (tr == 'N' ? m : n)) ||
                throw(DimensionMismatch("gemv: A $(size(A)), x $(length(x)), y $(length(y))"))
            CUDA.CUBLAS.$gemv(_blas_handle(), tr, m, n, alpha, A, max(1, stride(A, 2)),
                x, stride(x, 1), beta, y, stride(y, 1))
            return y
        end
    end
end
_blas_gemm!(tA, tB, alpha, A, B, beta, Cm) = CUDA.CUBLAS.gemm!(tA, tB, alpha, A, B, beta, Cm)
_blas_gemv!(tr, alpha, A, x, beta, y) = CUDA.CUBLAS.gemv!(tr, alpha, A, x, beta, y)
