# Independent format conformance: no use of the round-trip fixture writer.
using JSON
using Test

@testset "Safetensors C-order coordinates (Regime I arm 1 repair)" begin
    mktempdir() do dir
        function raw_tensor(name, shape, values)
            path = joinpath(dir, name * ".safetensors")
            header = JSON.json(
                Dict(
                    name => Dict(
                        "dtype" => "F32",
                        "shape" => shape,
                        "data_offsets" => [0, 4 * length(values)],
                    ),
                ),
            )
            open(path, "w") do io
                write(io, UInt64(sizeof(header)))
                write(io, header)
                write(io, reinterpret(UInt8, Float32.(values)))
            end
            return Gesso.load_safetensors(path)[name]
        end
        matrix = raw_tensor("matrix", [2, 3], [1, 2, 3, 4, 5, 6])
        @test matrix == [1.0 2.0 3.0; 4.0 5.0 6.0]
        # C ordering: last axis varies fastest, then the middle, then first.
        cube = raw_tensor(
            "cube",
            [2, 3, 4],
            [100i + 10j + k for i in 1:2 for j in 1:3 for k in 1:4],
        )
        @test size(cube) == (2, 3, 4)
        @test all(cube[i, j, k] == 100i + 10j + k for i in 1:2, j in 1:3, k in 1:4)
        @test raw_tensor("vector", [3], [2, 4, 8]) == [2.0, 4.0, 8.0]
        scalar = raw_tensor("scalar", Int[], [7])
        @test size(scalar) == ()
        @test scalar[] == 7.0
        @test size(raw_tensor("empty", [2, 0, 3], Float32[])) == (2, 0, 3)
    end
end
