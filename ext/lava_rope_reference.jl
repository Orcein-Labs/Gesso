# Imported Float64 frequencies use the same coefficient and rounding contract as CPU.
# Only position metadata crosses the host boundary; Q/K remain on Vulkan.
@kernel cpu=false function _lava_rope_reference_kernel!(
    x,
    @Const(C),
    @Const(S),
    seq::Int,
    nh::Int,
    rd::Int,
    interleaved::Bool,
)
    lin=@index(Global, Linear)
    if lin<=seq*nh
        t=(lin-1)%seq+1
        h=(lin-1)÷seq+1
        T=eltype(C)
        half=rd÷2
        @inbounds for i in 1:half
            c=C[t, i]
            s=S[t, i]
            a=interleaved ? 2i-1 : i
            b=interleaved ? 2i : i+half
            x1=T(x[t, h, a])
            x2=T(x[t, h, b])
            x[t, h, a]=Float32(x1*c-x2*s)
            x[t, h, b]=Float32(x1*s+x2*c)
        end
    end
end

# Every layer of one token rotates by the same positions, but a host-to-device
# upload is a blocking flush (Lava copy_buffer! waits for the queue), so a fresh
# table per call stalled the queue twice per layer. Recent uploaded tables are
# kept and reused while the context, positions, frequencies and rd match (under a
# lock, so concurrent callers never see a half-updated entry); the device arrays
# are read-only once uploaded and are replaced, never written. Several entries are
# kept because families with per-layer RoPE (sliding-window layers on a local
# base, full layers on the global base) interleave two frequency sets, and a
# single entry was replaced at every switch, missing on every such layer.
# GESSO_LAVA_ROPE_TABLE_CACHE=0 restores the per-call upload.
struct _LavaRopeTableEntry
    positions::Vector{Int}
    freq::Vector{Float64}
    rd::Int
    ctx::Any
    cd::Any
    sd::Any
end
const _LAVA_ROPE_TABLE_SLOTS = 4
const _LAVA_ROPE_TABLE = _LavaRopeTableEntry[]   # most recently used first
const _LAVA_ROPE_TABLE_LOCK = ReentrantLock()
_lava_rope_table_cache() = get(ENV, "GESSO_LAVA_ROPE_TABLE_CACHE", "1") != "0"

_lava_rope_entry_hit(e, ctx, positions, freq, rd) =
    e.ctx === ctx &&
    e.rd == rd &&
    length(e.positions) == length(positions) &&
    length(e.freq) >= rd ÷ 2 &&
    all(i -> e.freq[i] == Float64(freq[i]), 1:(rd÷2)) &&
    all(((a, b),) -> a == b, zip(e.positions, positions))

function _lava_rope_reference_table(positions, freq, rd)
    seq = length(positions)
    c = Matrix{Float64}(undef, seq, rd ÷ 2)
    s = similar(c)
    for t in eachindex(positions), i in 1:(rd÷2)
        phase = Float64(positions[t]) * Float64(freq[i])
        c[t, i] = cos(phase)
        s[t, i] = sin(phase)
    end
    return Lava.LavaArray{Float64}(c), Lava.LavaArray{Float64}(s)
end

function _lava_rope_reference_cached(positions, freq, rd)
    _lava_rope_table_cache() || return _lava_rope_reference_table(positions, freq, rd)
    ctx = Lava.vk_context()
    lock(_LAVA_ROPE_TABLE_LOCK) do
        tbl = _LAVA_ROPE_TABLE
        for j in eachindex(tbl)
            e = tbl[j]
            _lava_rope_entry_hit(e, ctx, positions, freq, rd) || continue
            j > 1 && (deleteat!(tbl, j); pushfirst!(tbl, e))
            return e.cd, e.sd
        end
        cd, sd = _lava_rope_reference_table(positions, freq, rd)
        e = _LavaRopeTableEntry(
            collect(Int, positions), Float64[freq[i] for i in 1:(rd÷2)], rd, ctx, cd, sd)
        pushfirst!(tbl, e)
        length(tbl) > _LAVA_ROPE_TABLE_SLOTS && resize!(tbl, _LAVA_ROPE_TABLE_SLOTS)
        return cd, sd
    end
end

function _lava_rope_reference!(q, k, positions, freq, rd, interleaved)
    seq = length(positions)
    seq == 0 && return nothing
    cd, sd = _lava_rope_reference_cached(positions, freq, rd)
    for x in (q, k)
        size(x, 2) == 0 && continue
        kernel = Lava.with_no_contraction(
            Lava.with_float_controls(
                _lava_rope_reference_kernel!(KA.get_backend(x)),
                (32, 64),
            ),
        )
        kernel(
            x,
            cd,
            sd,
            size(x, 1),
            size(x, 2),
            rd,
            interleaved;
            ndrange=size(x, 1) * size(x, 2),
        )
    end
    return nothing
end
