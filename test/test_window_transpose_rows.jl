using Test
using Gesso
using Random

# The blocked window transpose against the per-row column copy it replaced, over
# row ranges and widths that cut through the 16-row blocks and 64-column tiles.
@testset "window transpose rows equals the per-row copy" begin
    inf = Gesso.Inference
    rng = Xoshiro(11)
    for (T, n) in ((1, 1), (5, 63), (16, 64), (17, 65), (40, 130), (512, 200))
        W = randn(rng, Float32, T, n)
        for rows in (1:T, 1:cld(T, 2), (cld(T, 3) + 1):T)
            isempty(rows) && continue
            ref = fill(NaN32, n, T)
            got = fill(NaN32, n, T)
            for t in rows
                copyto!(view(ref, :, t), view(W, t, :))
            end
            inf._window_transpose_rows!(got, W, rows)
            @test isequal(got, ref)
        end
    end
end
