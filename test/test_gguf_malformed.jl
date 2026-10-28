# Damaged GGUF files: a truncated download, an impossible count or length, a
# bad alignment or a tensor running past the end must throw a GessoError with
# reason :malformed_gguf from `gguf_header`, never hang, crash or hand a bad
# offset to a kernel. Each file is built in memory by a writer that can break
# one field at a time; the unbroken file must still parse.

using Test
using Gesso

const _mg_inf = Gesso.Inference

_mg_str(io, s) = (write(io, UInt64(sizeof(s))); write(io, s))

# tensors: (name, ggml dims, ggml type, payload bytes). Offsets are laid out
# at `align` unless `offsets` overrides them.
function _mg_bytes(;
    # The tokenizer arrays and the chat template take the skip path, not the
    # value reader, so the good file carries one of each.
    kv=Pair{String, Any}[
        "general.architecture" => "llama",
        "tokenizer.ggml.tokens" => ["a", "bc", "def"],
        "tokenizer.ggml.token_type" => Int32[1, 1, 3],
        "tokenizer.chat_template" => "{{ messages }}",
        "general.name" => "after the skipped keys",
    ],
    tensors=[
        ("a.weight", [8, 2], Int32(0), zeros(UInt8, 64)),
        ("b.weight", [64, 2], Int32(8), zeros(UInt8, 4 * 34)),
    ],
    align=32,
    offsets=nothing,
    n_tensors=nothing,
    n_kv=nothing,
)
    io = IOBuffer()
    write(io, UInt8['G', 'G', 'U', 'F'])
    write(io, UInt32(3))
    write(io, UInt64(something(n_tensors, length(tensors))))
    write(io, UInt64(something(n_kv, length(kv))))
    for (k, v) in kv
        _mg_str(io, k)
        if v isa AbstractString
            write(io, UInt32(8))
            _mg_str(io, v)
        elseif v isa UInt32
            write(io, UInt32(4))
            write(io, v)
        elseif v isa Vector{String}
            write(io, UInt32(9), UInt32(8), UInt64(length(v)))
            foreach(s -> _mg_str(io, s), v)
        elseif v isa Vector{Int32}
            write(io, UInt32(9), UInt32(5), UInt64(length(v)))
            write(io, v)
        else
            error("writer: unsupported value $(typeof(v))")
        end
    end
    data = IOBuffer()
    for (i, (name, dims, t, payload)) in enumerate(tensors)
        write(data, zeros(UInt8, (align - position(data) % align) % align))
        off = offsets === nothing ? position(data) : offsets[i]
        write(data, payload)
        _mg_str(io, name)
        write(io, UInt32(length(dims)))
        foreach(d -> write(io, UInt64(d)), dims)
        write(io, UInt32(t))
        write(io, UInt64(off))
    end
    write(io, zeros(UInt8, (align - position(io) % align) % align))
    write(io, take!(data))
    return take!(io)
end

function _mg_file(bytes::Vector{UInt8})
    path = tempname() * ".gguf"
    write(path, bytes)
    return path
end

function _mg_rejects(bytes::Vector{UInt8})
    path = _mg_file(bytes)
    try
        # A count the file cannot hold must fail fast, not loop over it.
        t = @elapsed err = try
            _mg_inf.gguf_header(path)
            nothing
        catch e
            e
        end
        return err isa Gesso.GessoError && err.detail[:reason] == :malformed_gguf && t < 10
    finally
        rm(path; force=true)
    end
end

@testset "gguf_header: damaged files throw :malformed_gguf" begin
    good = _mg_bytes()

    @testset "the unbroken file parses" begin
        path = _mg_file(good)
        hdr = _mg_inf.gguf_header(path)
        @test [t.name for t in hdr.tensors] == ["a.weight", "b.weight"]
        @test hdr.kv["general.architecture"] == "llama"
        @test hdr.kv["tokenizer.ggml.tokens"] == 3
        @test hdr.kv["tokenizer.ggml.token_type"] == 3
        @test hdr.kv["general.name"] == "after the skipped keys"
        @test hdr.path == path
        rm(path)
    end

    @testset "truncated at $n of $(length(good)) bytes" for n in
        (0, 3, 16, 23, 40, 60, length(good) ÷ 2, length(good) - 34, length(good) - 1)
        if n < 4
            path = _mg_file(good[1:n])
            @test_throws Exception _mg_inf.gguf_header(path)
            rm(path)
        else
            @test _mg_rejects(good[1:n])
        end
    end

    @testset "impossible counts and lengths" begin
        @test _mg_rejects(_mg_bytes(; n_tensors=typemax(UInt64) >> 1))
        @test _mg_rejects(_mg_bytes(; n_kv=typemax(UInt64) >> 1))
        @test _mg_rejects(_mg_bytes(; n_kv=6))
        # A string length far past the end of the file.
        b = copy(good)
        b[25:32] .= reinterpret(UInt8, [UInt64(1) << 50])
        @test _mg_rejects(b)
        # A fixed-width array claiming 2^60 elements: skipped by arithmetic,
        # so it must fail on the length check rather than walk the count.
        io = IOBuffer()
        write(io, UInt8['G', 'G', 'U', 'F'], UInt32(3), UInt64(0), UInt64(1))
        _mg_str(io, "tokenizer.ggml.token_type")
        write(io, UInt32(9), UInt32(5), UInt64(1) << 60)
        @test _mg_rejects(take!(io))
        # The same count on an array of strings.
        io = IOBuffer()
        write(io, UInt8['G', 'G', 'U', 'F'], UInt32(3), UInt64(0), UInt64(1))
        _mg_str(io, "tokenizer.ggml.tokens")
        write(io, UInt32(9), UInt32(8), UInt64(1) << 60)
        @test _mg_rejects(take!(io))
    end

    @testset "bad tensor records" begin
        @test _mg_rejects(_mg_bytes(; tensors=[("a", Int[], Int32(0), UInt8[])]))
        @test _mg_rejects(_mg_bytes(; tensors=[("a", [1, 1, 1, 1, 1], Int32(0), zeros(UInt8, 4))]))
        @test _mg_rejects(_mg_bytes(; kv=Pair{String, Any}["general.alignment" => UInt32(24)]))
        @test _mg_rejects(_mg_bytes(; kv=Pair{String, Any}["general.alignment" => UInt32(0)]))
        @test _mg_rejects(_mg_bytes(; offsets=[0, 68]))
        # Q8_0 rows are whole blocks of 32.
        @test _mg_rejects(_mg_bytes(; tensors=[("b", [48, 2], Int32(8), zeros(UInt8, 4 * 34))]))
        # Dims whose product overflows Int64.
        @test _mg_rejects(_mg_bytes(; tensors=[("a", [1 << 32, 1 << 32], Int32(0), zeros(UInt8, 4))]))
        # A tensor that runs one byte past the end of the file.
        @test _mg_rejects(_mg_bytes(; tensors=[("a", [17], Int32(0), zeros(UInt8, 64))]))
        @test _mg_rejects(_mg_bytes(; offsets=[0, 1 << 40]))
    end

    @testset "error names the tensor" begin
        path = _mg_file(_mg_bytes(; offsets=[0, 1 << 40]))
        err = try
            _mg_inf.gguf_header(path)
        catch e
            e
        end
        @test err isa Gesso.GessoError
        @test err.detail[:tensor] == "b.weight"
        @test err.code == Gesso.ERR_RUNTIME
        rm(path)
    end
end
