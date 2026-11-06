using Test
using Gesso

# The blocked delta-rule slab against the plain per-head recurrence, bit for bit,
# across head counts that hit every block width (16, 8, 4, 1).
@testset "GDN delta slab equals the per-head recurrence (bit-exact)" begin
    inf = Gesso.Inference
    function ref_step!(st, core, qn, kn, v, decay, beta, nv, dk, dv)
        @inbounds for j in 1:dv, h in 1:nv
            acc = 0.0f0
            for i in 1:dk
                acc = acc + (st[h, i, j] * decay[h]) * kn[nv*(i-1)+h]
            end
            d = (v[nv*(j-1)+h] - acc) * beta[h]
            out = 0.0f0
            for i in 1:dk
                s = st[h, i, j] * decay[h] + kn[nv*(i-1)+h] * d
                st[h, i, j] = s
                out = out + s * qn[nv*(i-1)+h]
            end
            core[nv*(j-1)+h] = out
        end
        return st
    end
    for nv in (1, 3, 4, 8, 13, 16, 24, 29, 32), (dk, dv) in ((5, 3), (16, 7))
        st0 = randn(Float32, nv, dk, dv) .* 0.5f0
        a = copy(st0)
        b = copy(st0)
        ca = zeros(Float32, nv * dv)
        cb = zeros(Float32, nv * dv)
        for t in 1:3
            qn = randn(Float32, nv * dk)
            kn = randn(Float32, nv * dk)
            v = randn(Float32, nv * dv)
            decay = rand(Float32, nv)
            beta = rand(Float32, nv)
            # Two slabs over the value columns, writing in place as decode does.
            mid = div(dv, 2)
            inf._gdn_delta_slab!(a, a, ca, qn, kn, v, decay, beta, 1, mid, nv, dk)
            inf._gdn_delta_slab!(a, a, ca, qn, kn, v, decay, beta, mid + 1, dv, nv, dk)
            ref_step!(b, cb, qn, kn, v, decay, beta, nv, dk, dv)
            @test reinterpret(UInt32, vec(a)) == reinterpret(UInt32, vec(b))
            @test reinterpret(UInt32, ca) == reinterpret(UInt32, cb)
        end
    end
end
