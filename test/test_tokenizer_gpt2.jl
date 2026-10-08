# Phase 3 item C tests (§LXXVI): GPT-2 byte-level BPE encode.
#
# Golden ids are DERIVED test-side from the published algorithm (pre-tokenize
# RAW text → map UTF-8 bytes through bytes_to_unicode → merge by rank →
# vocab lookup) and the tiny fixture's merge table is small enough to trace
# by hand:
#
#   ranks: ("l","l")=0  ("e","ll")=1  ("ell","o")=2  ("H","ello")=3
#          ("a","b")=4  ("Ġ","a")=5
#
# The regex-order law is pinned by " Hello": the mapped space 'Ġ' (U+0120)
# is a letter category, so byte-mapping before pre-tokenization would
# corrupt the split — mapping AFTER is what makes this golden id reachable.

using .GessoTestHelpers: approx_eq

const TKDIR = joinpath(@__DIR__, "fixtures", "gpt2_tiny")

@testset "bytes_to_unicode: the published 256-byte map" begin
    tk = Gesso.load_gpt2_tokenizer(TKDIR)
    enc = tk.byte_encoder
    @test length(enc) == 256                       # every byte mapped
    @test length(Set(values(enc))) == 256          # injectively
    @test enc[UInt8('H')] == 'H'                   # printable ASCII maps to itself
    @test enc[UInt8(' ')] == '\u0120'              # space → Ġ (the GPT-2 trick)
    @test enc[0x00] == '\u0100'                    # first non-printable byte
    @test enc[UInt8('\n')] == '\u010a'             # 11th non-printable byte → Ċ
    @test enc[0xff] == 'ÿ'                         # latin-1 printable tail maps to itself
    # 0xa0 sits after 0x7f in the remap order: 33 bytes (0x00..0x20) then
    # 33 more (0x7f..0xa0) precede it ⇒ 0x100 + 66 = 0x0142
    @test enc[0xa0] == '\u0142'
end

@testset "load_gpt2_tokenizer: fixture loads with ranks in file order" begin
    tk = Gesso.load_gpt2_tokenizer(TKDIR)
    @test length(tk.vocab) == 13
    @test length(tk.merges) == 6
    @test tk.merges[1] == ("l" => "l")
    @test tk.merge_rank[("e"=>"ll")] == 1
    @test tk.bos_token_id == 0                     # no config: GPT-2 default
    @test tk.eos_token_id == 0
end

@testset "encode: golden ids from the traced merge table" begin
    tk = Gesso.load_gpt2_tokenizer(TKDIR)
    @test Gesso.encode(tk, "ab") == [11]           # a+b (rank 4) → "ab"
    @test Gesso.encode(tk, "Hello") == [10]        # ll → e+ll → ell+o → H+ello
    @test Gesso.encode(tk, " Hello") == [6, 10]    # space pre-token "Ġ", then "Hello"
    @test Gesso.encode(tk, "") == Int[]
    @test Gesso.encode(tk, "ab ") == [11, 6]       # trailing whitespace is its own chunk
    @test Gesso.encode(tk, "  a") == [6, 12]       # first space splits (\s+(?!\S)), then ` ?\p{L}+` re-absorbs " a" → Ġa
    @test Gesso.encode(tk, "ale") == [4, 2, 1]     # no adjacent merge pair: a,l,e stay split
    @test Gesso.encode(tk, "aab") == [4, 11]       # best pair merges ALL occurrences
    @test Gesso.encode(tk, " Hello") == Gesso.encode(tk, " Hello")  # deterministic
end

@testset "encode: out-of-vocab symbol is loud (§LXX)" begin
    tk = Gesso.load_gpt2_tokenizer(TKDIR)
    err = try
        Gesso.encode(tk, "hello")                  # lowercase h is not in the vocab
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("h", sprint(showerror, err))
    @test occursin("vocab", sprint(showerror, err))
end

@testset "load refusals name the offender (§LXX)" begin
    # empty vocab
    dir = mktempdir()
    cp(TKDIR, dir; force=true)
    open(joinpath(dir, "vocab.json"), "w") do io
        JSON.print(io, Dict{String, Any}())
    end
    err = try
        Gesso.load_gpt2_tokenizer(dir)
        nothing
    catch e
        e
    end
    @test err isa Gesso.GessoError
    @test occursin("empty", sprint(showerror, err))

    # merge part not in vocab
    dir2 = mktempdir()
    cp(TKDIR, dir2; force=true)
    open(joinpath(dir2, "merges.txt"), "w") do io
        println(io, "#version: 0.2")
        println(io, "z y")
    end
    err2 = try
        Gesso.load_gpt2_tokenizer(dir2)
        nothing
    catch e
        e
    end
    @test err2 isa Gesso.GessoError
    @test occursin("z", sprint(showerror, err2))

    # unknown special-token id in the config
    dir3 = mktempdir()
    cp(TKDIR, dir3; force=true)
    open(joinpath(dir3, "tokenizer_config.json"), "w") do io
        JSON.print(io, Dict{String, Any}("bos_token_id" => 99))
    end
    err3 = try
        Gesso.load_gpt2_tokenizer(dir3)
        nothing
    catch e
        e
    end
    @test err3 isa Gesso.GessoError
    @test occursin("bos_token_id", sprint(showerror, err3))
end
