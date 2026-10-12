# Turn a routing trace (GESSO_ROUTING_TRACE) into a ranked expert profile.
#
#   julia scripts/routing_profile.jl <trace> <profile-out>
#
# Trace lines: `layer token id_1 ... id_k` (one per layer per token row).
# Profile lines: `layer expert count`, sorted by layer, then count descending,
# then expert id. Feed the profile back with GESSO_EXPERT_PROFILE=<profile-out>.
# Pure Julia, no packages, no GPU.

length(ARGS) == 2 || error("usage: julia scripts/routing_profile.jl <trace> <profile-out>")
trace, out = ARGS
isfile(trace) || error("routing_profile: no trace at $trace")

counts = Dict{Tuple{Int,Int},Int}()
rows = Dict{Int,Int}()
for (ln, raw) in enumerate(eachline(trace))
    line = strip(raw)
    isempty(line) && continue
    f = split(line)
    length(f) >= 3 || error("routing_profile: $trace:$ln needs layer, token and at least one id")
    layer = parse(Int, f[1])
    rows[layer] = get(rows, layer, 0) + 1
    for s in f[3:end]
        key = (layer, parse(Int, s))
        counts[key] = get(counts, key, 0) + 1
    end
end
isempty(rows) && error("routing_profile: trace $trace has no routing rows")

ranked = sort(collect(counts); by=p -> (p[1][1], -p[2], p[1][2]))
open(out, "w") do io
    println(io, "# layer expert count (from $(basename(trace)))")
    for ((layer, e), c) in ranked
        println(io, layer, '\t', e, '\t', c)
    end
end

for layer in sort(collect(keys(rows)))
    println(stderr, "layer $layer rows $(rows[layer]) distinct experts $(count(p -> p[1] == layer, keys(counts)))")
end
println(stderr, "wrote $(length(ranked)) (layer, expert) rows to $out")
