# Native F32 dequant for every ggml type in _GGML_KNOWN_BLOCKS: ports of
# ggml's dequantize_row_* for the types no other file decodes, and one
# dispatcher over all of them. Loading a GGUF needs no libggml; the oracle
# tests still compare these against ggml's to_float when GESSO_GGML_DIR is set.

const GGML_TYPE_Q4_1 = Int32(3)
const GGML_TYPE_Q5_0 = Int32(6)

function _dq_half!(y, w, off0::Int, n::Int, bf16::Bool)
    @inbounds for i in 1:n
        u = _ld_u16(w, off0 + 2(i - 1))
        y[i] = bf16 ? reinterpret(Float32, UInt32(u) << 16) : Float32(reinterpret(Float16, u))
    end
    return y
end

# Q4_1, Q5_0, Q5_1: 32-value blocks; low nibbles fill 0..15, high nibbles
# 16..31, and the 5-bit types take bit 4 of value j from bit j of qh.
function _dq_q4_1!(y, w, off0::Int, n::Int)
    @inbounds for ib in 0:(div(n, 32)-1)
        b = off0 + 20ib
        d = _ld_f16(w, b)
        m = _ld_f16(w, b + 2)
        for j in 0:15
            q = _ld_u8(w, b + 4 + j)
            y[32ib+j+1] = Float32(q & 0x0f) * d + m
            y[32ib+j+17] = Float32(q >> 4) * d + m
        end
    end
    return y
end

function _dq_q5!(y, w, off0::Int, n::Int, with_min::Bool)
    bsz = with_min ? 24 : 22
    hoff = with_min ? 4 : 2
    @inbounds for ib in 0:(div(n, 32)-1)
        b = off0 + bsz * ib
        d = _ld_f16(w, b)
        m = with_min ? _ld_f16(w, b + 2) : 0.0f0
        qh = _ld_u32(w, b + hoff)
        for j in 0:15
            q = _ld_u8(w, b + hoff + 4 + j)
            x0 = Int32(q & 0x0f) | Int32(((qh >> j) << 4) & 0x10)
            x1 = Int32(q >> 4) | Int32((qh >> (j + 12)) & 0x10)
            if with_min
                y[32ib+j+1] = Float32(x0) * d + m
                y[32ib+j+17] = Float32(x1) * d + m
            else
                y[32ib+j+1] = Float32(x0 - Int32(16)) * d
                y[32ib+j+17] = Float32(x1 - Int32(16)) * d
            end
        end
    end
    return y
end

# The IQ and Q2_0 decoders below unfuse the dot kernels in gguf_quants.jl.
function _dq_iq2_xs!(y, w, off0::Int, n::Int)
    @inbounds for ib in 0:(div(n, QK_K)-1)
        b = off0 + 74ib
        d = _ld_f16(w, b)
        for ib32 in 0:7
            sc = _ld_u8(w, b + 66 + ib32)
            db0 = d * (0.5f0 + Float32(sc & 0x0f)) * 0.25f0
            db1 = d * (0.5f0 + Float32(sc >> 4)) * 0.25f0
            for l in 0:3
                qv = _ld_u16(w, b + 2 + 2(4ib32 + l))
                grid = IQ2XS_GRID[(qv&0x01ff)+0x0001]
                signs = KSIGNS_IQ2XS[(qv>>9)+0x0001]
                db = l < 2 ? db0 : db1
                o = QK_K * ib + 32ib32 + 8l
                for t in 0:7
                    g = Float32((grid >> 8t) & 0xff)
                    y[o+t+1] = (signs & KMASK_IQ2XS[t+1]) != 0x00 ? -db * g : db * g
                end
            end
        end
    end
    return y
end

function _dq_iq3_xxs!(y, w, off0::Int, n::Int)
    @inbounds for ib in 0:(div(n, QK_K)-1)
        b = off0 + 98ib
        d = _ld_f16(w, b)
        for ib32 in 0:7
            aux = _ld_u32(w, b + 66 + 4ib32)
            db = d * (0.5f0 + Float32(aux >> 28)) * 0.5f0
            qb = b + 2 + 8ib32
            for l in 0:3
                signs = KSIGNS_IQ2XS[((aux>>7l)&UInt32(127))+UInt32(1)]
                g1 = IQ3XXS_GRID[Int(_ld_u8(w, qb + 2l))+1]
                g2 = IQ3XXS_GRID[Int(_ld_u8(w, qb + 2l + 1))+1]
                o = QK_K * ib + 32ib32 + 8l
                for t in 0:3
                    v1 = db * Float32((g1 >> 8t) & 0xff)
                    v2 = db * Float32((g2 >> 8t) & 0xff)
                    y[o+t+1] = (signs & KMASK_IQ2XS[t+1]) != 0x00 ? -v1 : v1
                    y[o+t+5] = (signs & KMASK_IQ2XS[t+5]) != 0x00 ? -v2 : v2
                end
            end
        end
    end
    return y
end

function _dq_iq4_xs!(y, w, off0::Int, n::Int)
    @inbounds for ibk in 0:(div(n, QK_K)-1)
        b = off0 + 136ibk
        d = _ld_f16(w, b)
        scales_h = _ld_u16(w, b + 2)
        for ib in 0:7
            sl = _ld_u8(w, b + 4 + div(ib, 2))
            ls = Int((sl >> 4(ib % 2)) & 0x0f) | Int((scales_h >> 2ib) & 0x0003) << 4
            dl = d * Float32(ls - 32)
            o = QK_K * ibk + 32ib
            for t in 0:15
                q = _ld_u8(w, b + 8 + 16ib + t)
                y[o+t+1] = dl * Float32(KVALUES_IQ4NL[(q&0x0f)+0x01])
                y[o+t+17] = dl * Float32(KVALUES_IQ4NL[(q>>4)+0x01])
            end
        end
    end
    return y
end

function _dq_iq4_nl!(y, w, off0::Int, n::Int)
    @inbounds for ib in 0:(div(n, 32)-1)
        b = off0 + 18ib
        d = _ld_f16(w, b)
        for t in 0:15
            q = _ld_u8(w, b + 2 + t)
            y[32ib+t+1] = d * Float32(KVALUES_IQ4NL[(q&0x0f)+0x01])
            y[32ib+t+17] = d * Float32(KVALUES_IQ4NL[(q>>4)+0x01])
        end
    end
    return y
end

function _dq_q2_0!(y, w, off0::Int, n::Int)
    @inbounds for ib in 0:(div(n, 64)-1)
        b = off0 + 18ib
        d = _ld_f16(w, b)
        for t in 0:63
            code = (Int(_ld_u8(w, b + 2 + (t >> 2))) >> 2(t & 3)) & 0x3
            y[64ib+t+1] = d * Float32(code - 1)
        end
    end
    return y
end

const _KQ_DEQUANT_TYPES = (GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K)
const _IQS_DEQUANT_TYPES = (GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_S, GGML_TYPE_IQ1_M)

"""
    _ggml_dequant!(y, w, off0, ggml_type, n)

Decode the first `n` values of a `ggml_type` payload starting at 0-based byte
`off0` of `w` into `y[1:n]`. `n` must be a whole number of blocks; the caller
checks that the bytes are in range.
"""
function _ggml_dequant!(y::AbstractVector{Float32}, w::AbstractVector{UInt8}, off0::Int, ggml_type::Int32, n::Int)
    length(y) >= n || error("ggml dequant: output holds $(length(y)) values, need $n")
    out = view(y, 1:n)
    if ggml_type == GGML_TYPE_F32
        @inbounds for i in 1:n
            out[i] = _ld_f32(w, off0 + 4(i - 1))
        end
    elseif ggml_type == GGML_TYPE_F16 || ggml_type == GGML_TYPE_BF16
        _dq_half!(out, w, off0, n, ggml_type == GGML_TYPE_BF16)
    elseif ggml_type in _KQ_DEQUANT_TYPES
        _kq_dequant_row!(out, w, off0, ggml_type, n)
    elseif ggml_type in _IQS_DEQUANT_TYPES
        _iqs_dequant_row!(out, w, off0, ggml_type, n)
    elseif ggml_type == GGML_TYPE_Q4_1
        _dq_q4_1!(out, w, off0, n)
    elseif ggml_type == GGML_TYPE_Q5_0 || ggml_type == GGML_TYPE_Q5_1
        _dq_q5!(out, w, off0, n, ggml_type == GGML_TYPE_Q5_1)
    elseif ggml_type == GGML_TYPE_IQ2_XS
        _dq_iq2_xs!(out, w, off0, n)
    elseif ggml_type == GGML_TYPE_IQ3_XXS
        _dq_iq3_xxs!(out, w, off0, n)
    elseif ggml_type == GGML_TYPE_IQ4_XS
        _dq_iq4_xs!(out, w, off0, n)
    elseif ggml_type == GGML_TYPE_IQ4_NL
        _dq_iq4_nl!(out, w, off0, n)
    elseif ggml_type == GGML_TYPE_Q2_0
        _dq_q2_0!(out, w, off0, n)
    else
        _ggml_traits(ggml_type)  # throws :unsupported_ggml_type for a type outside the table
        error("ggml dequant: type $ggml_type is in the block table but has no decoder")
    end
    return y
end
