# scripts/p1_storage_any_cost.jl — the measurement behind DECISION PACKET 3
# (`docs/DECISION_PACKETS.md`, "storage::Any and what it costs").
#
# Run: julia --project=. scripts/p1_storage_any_cost.jl
#
# THIS IS EVIDENCE, NOT A GATE. It changes no repository file, asserts
# nothing, and is not wired into `scripts/test.jl`. Its only job is to make
# the numbers quoted in the packet re-derivable on demand instead of being
# agent confidence. The gates that guard the packet's invariant are the
# three `@test_broken` in `test/test_type_stability.jl` and they stay broken.
#
# Prints five things:
#   1. the warmed `decode!` baseline (toy2 CPU)
#   2. what one `::Any` read costs, microbenchmarked
#   3. the counterfactual: same read, different FIELD TYPE
#   4. `Profile.Allocs` attribution of everything the warmed call allocates
#   5. how many `::Any` field declarations the decode path actually carries

using Gesso
using Test
using Profile

const TD = joinpath(@__DIR__, "..", "test")
include(joinpath(TD, "testhelpers.jl"))
include(joinpath(TD, "test_helpers.jl"))
include(joinpath(TD, "toyfixtures.jl"))
include(joinpath(TD, "test_toyfixtures.jl"))
include(joinpath(TD, "test_modelir.jl"))
include(joinpath(TD, "test_reference_prefill.jl"))

println("="^72)
println("1. BASELINE — warmed decode! host allocation, THIS tree")
println("="^72)

function warmed_alloc(s, prompt, n=2)
    Gesso.prefill!(s, prompt)
    for _ in 1:n
        Gesso.decode!(s)
    end
    return @allocated Gesso.decode!(s)
end

ts = toy2_tensors()
model = toy2_modelir()
s = Gesso.Session(model, ts; context_length=128, eos_token_id=2)
base = warmed_alloc(s, [1, 3, 4, 5])
println("  toy2 CPU warmed decode! @allocated = ", base, " B")

# repeatability: 5 independent sessions, report the floor (least noise)
reps = Int[]
for _ in 1:5
    ss = Gesso.Session(model, ts; context_length=128, eos_token_id=2)
    push!(reps, warmed_alloc(ss, [1, 3, 4, 5]))
end
println("  5-session reps = ", reps, "  min = ", minimum(reps))

println()
println("="^72)
println("2. WHAT ONE ::Any READ COSTS (the per-read primitive)")
println("="^72)

struct AnyBox
    x::Any
end
struct TypedBox
    x::Vector{Float64}
end
struct ArrBox
    a::Any
end
struct ArrBoxT{T}
    a::T
end

# a CALL on an ::Any field is a dynamic dispatch; the same call on a typed
# field is free. This is the shape the engine has on every weight.
any_loop(b, n) = (acc=0; for _ in 1:n
    acc += length(b.x)
end; acc)
function arr_loop(b, n)
    t = 0.0
    for i in 1:n
        t += b.a[i]          # b.a is ::Any -> getindex is a dynamic call
    end
    return t
end
function arr_loop_t(b::ArrBoxT, n)
    t = 0.0
    for i in 1:n
        t += b.a[i]          # b.a is T -> inferred, no dispatch
    end
    return t
end

const NREAD = 1000
v = ones(1024)
ab = AnyBox(v);
tb = TypedBox(v)
any_loop(ab, 1);
any_loop(tb, 1)                        # warm
arr_loop(ArrBox(v), 1);
arr_loop_t(ArrBoxT(v), 1)       # warm

a_any_scalar = @allocated any_loop(ab, NREAD)
a_typed_scalar = @allocated any_loop(tb, NREAD)
a_any_arr = @allocated arr_loop(ArrBox(v), NREAD)
a_typed_arr = @allocated arr_loop_t(ArrBoxT(v), NREAD)

println("  scalar field read, ::Any field   : ", a_any_scalar / NREAD, " B/read")
println("  scalar field read, typed field   : ", a_typed_scalar / NREAD, " B/read")
println("  a[i] on ::Any field (engine's use): ", a_any_arr / NREAD, " B/read")
println("  a[i] on typed field              : ", a_typed_arr / NREAD, " B/read")
println(
    "  → boxing cost of ONE ::Any array read = ",
    (a_any_arr - a_typed_arr) / NREAD,
    " B",
)

println()
println("="^72)
println("3. INFERENCE — does the field type decide it?")
println("="^72)

# The COUNTERFACTUAL: byte-identical read patterns, one field-type
# difference. Nothing else about rows 1 and 2 differs.
struct FakeAct{S}
    shape::Tuple{Vararg{Int}}
    storage::S
end
struct FakeActAny
    shape::Tuple{Vararg{Int}}
    storage::Any
end

real_read(x) = x.storage[1, 1]
fake_read(x::FakeAct) = x.storage[1, 1]
fake_read(x::FakeActAny) = x.storage[1, 1]

verdict(f) =
    try
        f()
        "INFERRED"
    catch
        "NOT INFERRED (boxed)"
    end

a_real = Gesso.Activation(shape=(2, 2), storage=rand(2, 2))
a_fake = FakeAct{Matrix{Float64}}((2, 2), rand(2, 2))
a_any = FakeActAny((2, 2), rand(2, 2))

println("  real  Activation (storage::Any) : ", verdict(() -> @inferred real_read(a_real)))
println("  fake  FakeAct{Matrix{Float64}}  : ", verdict(() -> @inferred fake_read(a_fake)))
println("  fake  FakeActAny (storage::Any) : ", verdict(() -> @inferred fake_read(a_any)))

# and what the SAME read costs, both ways
struct Store1{T}
    s::T
end
struct StoreAny
    s::Any
end
store_loop(x, n) = (t=0.0; for _ in 1:n
    t += x.s[1]
end; t)
store_loop(Store1(v), 1);
store_loop(StoreAny(v), 1)       # warm
println(
    "  read `s[1]` alloc, typed field   : ",
    (@allocated store_loop(Store1(v), NREAD)) / NREAD,
    " B/call",
)
println(
    "  read `s[1]` alloc, ::Any field   : ",
    (@allocated store_loop(StoreAny(v), NREAD)) / NREAD,
    " B/call",
)

println()
println("="^72)
println("4. THE RESIDUAL — where every byte the warmed call allocates goes")
println("="^72)

s2 = Gesso.Session(model, ts; context_length=128, eos_token_id=2)
Gesso.prefill!(s2, [1, 3, 4, 5])
Gesso.decode!(s2)
Gesso.decode!(s2)
gid = @allocated Gesso.Inference._session_greedy_id!(
    s2,
    Gesso.CPUBackend(),
    Gesso.DecodeWorkload(),
)
println("  _session_greedy_id! @allocated = ", gid, " B")

Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 1.0 Gesso.decode!(s2)
allocs = Profile.Allocs.fetch().allocs
println(
    "  total decode! allocs            = ",
    length(allocs),
    " in ",
    sum(a.size for a in allocs),
    " B",
)
sites = Dict{String, Tuple{Int, Int}}()
for a in allocs
    fr = filter(f -> occursin("Gesso/src", String(f.file)), a.stacktrace)
    top = isempty(fr) ? nothing : first(fr)
    k =
        top === nothing ? "(outside src/)" :
        string(basename(String(top.file)), ":", top.line, " ", top.func)
    c, b = get(sites, k, (0, 0))
    sites[k] = (c + 1, b + a.size)
end
for (k, (c, b)) in sort(collect(sites), by=x -> -x[2][2])[1:min(24, length(sites))]
    println("    ", lpad(b, 8), " B /", lpad(c, 5), "  ", k)
end

println()
println("="^72)
println("5. FIELD COUNT — ::Any declarations on the decode path")
println("="^72)
src = joinpath(@__DIR__, "..", "src")
fam = filter(
    l -> occursin("storage::Any", l),
    readlines(joinpath(src, "Parameters", "Parameters.jl")),
)
println("  §XI families with storage::Any      : ", length(fam))
kvp = filter(
    l -> occursin(r"^\s+(storage|prototype)::Any", l),
    readlines(joinpath(src, "Inference", "kv_manager.jl")),
)
println("  KVPage/PagedKVManager ::Any fields  : ", length(kvp))
sfields = filter(
    l -> occursin(r"^\s+\w+::Any", l),
    readlines(joinpath(src, "Inference", "session.jl")),
)
println("  Session + DecodeWorkspace ::Any     : ", length(sfields))
println(
    "  TOTAL ::Any fields on the decode path: ",
    length(fam) + length(kvp) + length(sfields),
)
