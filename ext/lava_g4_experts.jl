# Gemma 4 routed-expert prefill on Lava, for N >= g4_expert_prefill_min tokens.
# Routing stays on the host, as the CPU route does. Per expert with tokens:
# its gate_up and down bytes are staged into one device buffer, its tokens are
# gathered on device, and the Lava quant GEMV runs it with ntok = the expert's
# token count (X[e*ntok+t], Y[(r-1)*ntok+t]). GELU*mul is a kernel, and the
# weighted rows are scatter-added into moe on device. Decode (N = 1) never
# reaches this file. The staging and scratch sizes come from the hot-expert
# pool budget, which already keeps the GPU reserve, so nothing loosens here.

const _G4X_CHUNK = 256
const _G4X_STAGE_MIB = 64

# Device-path counters: calls that ran the experts on the device, and expert
# chunks launched. Tests read them to show the path fired; the cost is one
# atomic add per call and per chunk.
const _G4X_CALLS = Threads.Atomic{Int}(0)
const _G4X_CHUNKS = Threads.Atomic{Int}(0)

_g4x_enabled(N::Int) =
    N >= env_tune(:g4_expert_prefill_min, 32) && env_tune(:lava_g4_expert_prefill, false)

@kernel cpu=false function _g4x_zero_kernel!(A)
    i = @index(Global)
    if i <= length(A)
        A[i] = 0.0f0
    end
end

# X[(f-1)*c + t] = Xsrc[tok[base+t], f] for the c slots of one expert chunk.
@kernel cpu=false function _g4x_gather_kernel!(X, Xsrc, tok, base::Int, c::Int, E::Int)
    i = @index(Global)
    if i <= c * E
        f = (i - 1) ÷ c + 1
        t = (i - 1) % c + 1
        X[i] = Xsrc[Int(tok[base + t]), f]
    end
end

# A[i] = gelu(G[i]) * G[i + n], n = ff*c: the gate rows come first, then the up rows.
@kernel cpu=false function _g4x_act_kernel!(A, G, n::Int)
    i = @index(Global)
    if i <= n
        A[i] = Gesso.Inference._g4_gelu(G[i]) * G[i + n]
    end
end

# moe[tok, r] += w * D[(r-1)*c + t]. Tokens are unique within one expert, so
# the writes of one launch never collide; experts are launched one after another.
@kernel cpu=false function _g4x_scatter_kernel!(moe, D, tok, wts, base::Int, c::Int, E::Int)
    i = @index(Global)
    if i <= c * E
        r = (i - 1) ÷ c + 1
        t = (i - 1) % c + 1
        s = base + t
        moe[Int(tok[s]), r] += wts[s] * D[i]
    end
end

function _g4x_eligible(gu, dn)
    gu isa Gesso.Inference.GgufQuantSlice && dn isa Gesso.Inference.GgufQuantSlice || return false
    kg = _lq_expert_kind(gu.ggml_type)
    kd = _lq_expert_kind(dn.ggml_type)
    (kg === nothing || kd === nothing) && return false
    ff = dn.ne0
    return gu.ne1 == 2ff && dn.ne1 == gu.ne0 && gu.n_exp == dn.n_exp &&
           gu.ne0 % _lq_block_values(kg) == 0 && ff % _lq_block_values(kd) == 0
end

# Slots sorted by expert: expert e owns slots start[e]+1:start[e+1], in token order.
function _g4x_slots(experts::Matrix{Int}, weights::Matrix{Float32}, n_exp::Int)
    n_used, N = size(experts)
    start = zeros(Int, n_exp + 1)
    for t in 1:N, k in 1:n_used
        start[experts[k, t] + 1] += 1
    end
    cumsum!(start, start)
    nxt = start[1:n_exp]
    tok = Vector{Int32}(undef, N * n_used)
    wts = Vector{Float32}(undef, N * n_used)
    for t in 1:N, k in 1:n_used
        e = experts[k, t]
        nxt[e] += 1
        tok[nxt[e]] = t
        wts[nxt[e]] = weights[k, t]
    end
    return start, tok, wts
end

# Host bytes of one expert straight into the staging buffer at byte offset off.
function _g4x_upload!(stage, off::Int, bank, e::Int)
    I = Gesso.Inference
    src = I._quant_expert_off0(bank, e)
    GC.@preserve bank begin
        Lava.copy_buffer!(:upload, stage.buf[], pointer(bank.data, src + 1), bank.bytes_per_expert;
                          offset=stage.offset + off)
    end
    return nothing
end

# Routed branch on the device. Returns the device moe (N x E, before its post
# norm), or nothing when this layer takes the CPU path.
function _g4x_routed(rin_h::Matrix{Float32}, x2::Lava.LavaArray, hbt)
    N, E = size(rin_h)
    N >= 1 || return nothing
    gu = hbt.wexp_gate_up isa Gesso.ExpertWeight ? hbt.wexp_gate_up.storage : hbt.wexp_gate_up
    dn = hbt.wexp_down isa Gesso.ExpertWeight ? hbt.wexp_down.storage : hbt.wexp_down
    _g4x_eligible(gu, dn) || return nothing
    n_used = hbt.n_expert_used
    n_exp = gu.n_exp
    ff = dn.ne0
    kg = _lq_expert_kind(gu.ggml_type)
    kd = _lq_expert_kind(dn.ggml_type)
    slotb = gu.bytes_per_expert + dn.bytes_per_expert

    # Scratch per chunk: G (2ff), A (ff), Xg (E) and D (E) floats per token.
    chunk_bytes = (3ff + 2E) * _G4X_CHUNK * 4
    moe_bytes = N * E * 4 + N * n_used * 8
    avail = Int(_lhot_budget_bytes())
    stage_cap = min(_G4X_STAGE_MIB << 20, avail - chunk_bytes - moe_bytes)
    stage_cap >= slotb || return nothing

    r_stor = hbt.wrouter.storage
    down_scale = hbt.wexp_down_scale isa FrozenParameter ? hbt.wexp_down_scale.storage : hbt.wexp_down_scale
    experts = Matrix{Int}(undef, n_used, N)
    weights = Matrix{Float32}(undef, n_used, N)
    Gesso.Inference._g4_route_rows!(experts, weights, rin_h, r_stor, n_used, down_scale)
    start, tokh, wsh = _g4x_slots(experts, weights, n_exp)
    tokd = Lava.LavaArray{Int32}(tokh)
    wsd = Lava.LavaArray{Float32}(wsh)

    moe = _g4l_new(N, E)
    _g4l_launch!(_g4x_zero_kernel!, moe, length(moe), moe)
    Xg = _g4l_new(E * _G4X_CHUNK)
    G = _g4l_new(2ff * _G4X_CHUNK)
    A = _g4l_new(ff * _G4X_CHUNK)
    D = _g4l_new(E * _G4X_CHUNK)
    stage = Lava.LavaArray{UInt8}(undef, stage_cap)

    active = [e for e in 1:n_exp if start[e + 1] > start[e]]
    per_round = max(1, stage_cap ÷ slotb)
    row_gu = gu.bytes_per_expert ÷ gu.ne1
    row_dn = dn.bytes_per_expert ÷ dn.ne1
    i = 1
    while i <= length(active)
        batch = active[i:min(length(active), i + per_round - 1)]
        i += length(batch)
        for (j, e) in enumerate(batch)
            off = (j - 1) * slotb
            _g4x_upload!(stage, off, gu, e)
            _g4x_upload!(stage, off + gu.bytes_per_expert, dn, e)
        end
        for (j, e) in enumerate(batch)
            off = (j - 1) * slotb
            for c0 in (start[e] + 1):_G4X_CHUNK:start[e + 1]
                c1 = min(start[e + 1], c0 + _G4X_CHUNK - 1)
                c = c1 - c0 + 1
                base = c0 - 1
                Threads.atomic_add!(_G4X_CHUNKS, 1)
                _g4l_launch!(_g4x_gather_kernel!, Xg, c * E, Xg, x2, tokd, base, c, E)
                _lq_launch!(G, Xg, stage, off, kg, c, E, 2ff, row_gu, 1.0f0, 0.0f0)
                _g4l_launch!(_g4x_act_kernel!, A, ff * c, A, G, ff * c)
                _lq_launch!(D, A, stage, off + gu.bytes_per_expert, kd, c, ff, E, row_dn, 1.0f0, 0.0f0)
                _g4l_launch!(_g4x_scatter_kernel!, D, c * E, moe, D, tokd, wsd, base, c, E)
            end
        end
    end
    _g4l_free!(stage, Xg, G, A, D, tokd, wsd)
    Threads.atomic_add!(_G4X_CALLS, 1)
    return moe
end
