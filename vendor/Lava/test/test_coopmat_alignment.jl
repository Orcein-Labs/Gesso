using Lava, Test

# Compile real physical-pointer cooperative loads/stores without device dispatch.
# Vulkan VUID 04708 requires the Aligned operand; 06314 requires scalar-size
# multiples. This checks the emitted contract directly because spirv-val v2026.4
# misses the absent optional memory-operand branch for cooperative instructions.
function coop_physical_alignment_roundtrip(out::UInt64, inp::UInt64, ::Val{T}) where T
    tile = Lava.AcceleratedMatrix{T,16,16,Lava.MatrixA}(Ptr{T}(inp), 1, 16)
    copyto!(Ptr{T}(out), 1, 16, tile)
    return nothing
end

@testset "physical cooperative matrix memory alignment" begin
    for T in (Float16, Float32)
        compiled = Lava.lava_compile_gpu(coop_physical_alignment_roundtrip,
            Tuple{UInt64,UInt64,Val{T}}; workgroup_size=(32,1,1), validate=true)
        dis = Lava.disassemble_spirv(compiled.spirv_bytes)
        @test occursin("PhysicalStorageBuffer", dis)
        for opname in ("OpCooperativeMatrixLoadKHR", "OpCooperativeMatrixStoreKHR")
            ops = filter(line -> occursin(opname,line), split(dis,'\n'))
            @test length(ops) == 1
            for line in ops
                aligned = match(r"\bAligned (\d+)",line)
                @test aligned !== nothing
                if aligned !== nothing
                    bytes = parse(Int,aligned.captures[1])
                    @test bytes >= sizeof(T) && bytes % sizeof(T) == 0
                    @test ispow2(bytes)
                end
            end
        end
    end
end
