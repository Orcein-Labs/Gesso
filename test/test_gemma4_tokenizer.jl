using Test
using JSON
using Gesso

const _G4_TK = Gesso.Inference
const _G4_FIXTURE = joinpath(@__DIR__, "fixtures", "gemma4_tokenizer_fixture.json")

@testset "gemma4 tokenizer vs llama-server" begin
    gguf = get(ENV, "GESSO_GEMMA4_GGUF", "")
    if isempty(gguf) || !isfile(gguf)
        @info "skipping gemma4 tokenizer test: GESSO_GEMMA4_GGUF unset or missing"
        @test_skip false
    else
        fx = JSON.parsefile(_G4_FIXTURE)
        tk = _G4_TK.load_gguf_tokenizer(gguf)
        @test tk isa _G4_TK.GGUFGemma4BPE
        for c in fx["cases"]
            ids = _G4_TK.encode(tk, c["text"];
                                parse_special=c["parse_special"],
                                add_special=c["add_special"])
            @test collect(ids) == Int.(c["ids"])
        end
        for d in fx["decodes"]
            ids = Int.(d["ids"])
            @test _G4_TK.decode(tk, ids) == d["text"]
        end
    end
end

@testset "gemma4 tokenizer synthetic unit tests" begin
    # 1. Construct a synthetic vocabulary with specials, byte fallback tokens, and BPE merges
    tokens = String["<unk>", "<pad>", "<bos>", "<eos>", "<|turn>", "<turn|>", "<user_sp>"]
    types = Int32[
        _G4_TK._GGUF_TOK_UNKNOWN,
        _G4_TK._GGUF_TOK_CONTROL,
        _G4_TK._GGUF_TOK_CONTROL,
        _G4_TK._GGUF_TOK_CONTROL,
        _G4_TK._GGUF_TOK_CONTROL,
        _G4_TK._GGUF_TOK_CONTROL,
        _G4_TK._GGUF_TOK_USER,
    ]

    byte_id = fill(-1, 256)
    for b in 0:255
        h = uppercase(string(b, base=16, pad=2))
        push!(tokens, "<0x" * h * ">")
        push!(types, _G4_TK._GGUF_TOK_BYTE)
        byte_id[b+1] = length(tokens) - 1
    end

    normal_words = [
        _G4_TK._GEMMA4_SPACE, "h", "e", "l", "o", "w", "r", "d",
        "ll", "he", "hell", "hello", _G4_TK._GEMMA4_SPACE * "world", "\n",
    ]
    for w in normal_words
        push!(tokens, w)
        push!(types, _G4_TK._GGUF_TOK_NORMAL)
    end

    vocab = Dict{String, Int}()
    for (i, t) in enumerate(tokens)
        vocab[t] = i - 1
    end

    specials = _G4_TK._GGUFSpecial[
        _G4_TK._GGUFSpecial("<user_sp>", 6, false),
        _G4_TK._GGUFSpecial("<turn|>", 5, true),
        _G4_TK._GGUFSpecial("<|turn>", 4, true),
        _G4_TK._GGUFSpecial("<eos>", 3, true),
        _G4_TK._GGUFSpecial("<bos>", 2, true),
        _G4_TK._GGUFSpecial("<pad>", 1, true),
        _G4_TK._GGUFSpecial("<unk>", 0, true),
    ]

    merge_rank = Dict{Tuple{String, String}, Int}(
        ("l", "l") => 1,
        ("h", "e") => 2,
        ("he", "ll") => 3,
        ("hell", "o") => 4,
        ("w", "o") => 5,
        ("r", "l") => 6,
        ("wo", "rl") => 7,
        ("worl", "d") => 8,
        (_G4_TK._GEMMA4_SPACE, "world") => 9,
    )

    tk = _G4_TK.GGUFGemma4BPE(
        tokens, types, vocab, byte_id, merge_rank, specials,
        2, 3, 1, nothing, true
    )

    # Metadata
    meta = _G4_TK.tokenizer_metadata(tk)
    @test meta.kind === :gguf_gemma4_spm_bpe
    @test meta.vocab_size == length(tokens)
    @test meta.bos_token_id == 2
    @test meta.eos_token_id == 3
    @test meta.pad_token_id == 1

    # Basic encode / decode
    id_hello = vocab["hello"]
    id_world = vocab[_G4_TK._GEMMA4_SPACE * "world"]
    @test _G4_TK.encode(tk, "hello") == [id_hello]
    @test _G4_TK.decode(tk, [id_hello]) == "hello"

    # Space handling with U+2581 (▁)
    ids_hw = _G4_TK.encode(tk, "hello world")
    @test ids_hw == [id_hello, id_world]
    @test _G4_TK.decode(tk, ids_hw) == "hello world"

    # Newlines
    id_nl = vocab["\n"]
    @test _G4_TK.encode(tk, "hello\nworld") != ids_hw
    @test _G4_TK.decode(tk, _G4_TK.encode(tk, "hello\n")) == "hello\n"

    # <0xXX> byte fallback for unknown characters
    id_A = byte_id[0x41 + 1]
    id_B = byte_id[0x42 + 1]
    @test _G4_TK.encode(tk, "AB") == [id_A, id_B]
    @test _G4_TK.decode(tk, [id_A, id_B]) == "AB"

    # Multi-byte UTF-8 fallback (e.g. 'é' = 0xc3, 0xa9)
    id_c3 = byte_id[0xc3 + 1]
    id_a9 = byte_id[0xa9 + 1]
    @test _G4_TK.encode(tk, "é") == [id_c3, id_a9]
    @test _G4_TK.decode(tk, [id_c3, id_a9]) == "é"

    # parse_special and add_special
    @test _G4_TK.encode(tk, "<turn|>hello"; parse_special=true) == [5, id_hello]
    @test _G4_TK.encode(tk, "<turn|>hello"; parse_special=false) != [5, id_hello]
    # USER token is parsed even with parse_special=false
    @test _G4_TK.encode(tk, "<user_sp>hello"; parse_special=false)[1] == 6

    # add_special prepends BOS
    @test _G4_TK.encode(tk, "hello"; add_special=true) == [2, id_hello]
    @test _G4_TK.encode(tk, "hello"; add_special=false) == [id_hello]

    # _prompt_ids
    @test _G4_TK._prompt_ids(tk, "hello") == [2, id_hello]
    # Does not duplicate BOS if already present
    @test _G4_TK._prompt_ids(tk, "<bos>hello") == [2, id_hello]

    # add_bos = false does not prepend BOS
    tk_nobos = _G4_TK.GGUFGemma4BPE(
        tokens, types, vocab, byte_id, merge_rank, specials,
        2, 3, 1, nothing, false
    )
    @test _G4_TK._prompt_ids(tk_nobos, "hello") == [id_hello]

    # StreamDetokenizer
    sdec = _G4_TK.StreamDetokenizer(tk)
    @test Base.push!(sdec, id_hello) == "hello"
    @test Base.push!(sdec, id_world) == " world"
    @test _G4_TK.finish!(sdec) == ""

    # StreamDetokenizer with multi-byte UTF-8 split across byte tokens ('€' = 0xe2, 0x82, 0xac)
    sdec2 = _G4_TK.StreamDetokenizer(tk)
    b1 = byte_id[0xe2 + 1]
    b2 = byte_id[0x82 + 1]
    b3 = byte_id[0xac + 1]
    @test Base.push!(sdec2, b1) == ""
    @test Base.push!(sdec2, b2) == ""
    @test Base.push!(sdec2, b3) == "€"
    @test _G4_TK.finish!(sdec2) == ""
end
