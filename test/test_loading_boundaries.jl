using Test, JSON

@testset "Regime I strict checkpoint boundaries" begin
    mktempdir() do dir
        function raw(header, payload=UInt8[]; length_override=nothing)
            text = header isa AbstractString ? header : JSON.json(header)
            path = joinpath(dir, "boundary.safetensors")
            open(path, "w") do io
                write(
                    io,
                    htol(
                        UInt64(
                            length_override === nothing ? sizeof(text) : length_override,
                        ),
                    ),
                )
                write(io, text)
                write(io, payload)
            end
            return path
        end
        meta(; shape=[1], offsets=[0, 4], dtype="F32") =
            Dict("dtype"=>dtype, "shape"=>shape, "data_offsets"=>offsets)
        bytes = collect(reinterpret(UInt8, Float32[1, 2]))
        @test Gesso.load_safetensors(raw(Dict("x"=>meta()), bytes[1:4]))["x"] == [1.0]
        bad_headers = [
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
            "{\"x\":{\"dtype\":\"F32\",\"dtype\":\"F64\",\"shape\":[1],\"data_offsets\":[0,4]}}",
            Dict("x"=>meta(shape=[-1])),
            Dict("x"=>meta(shape=[1.0])),
            Dict("x"=>meta(shape=[true])),
            Dict("x"=>meta(shape=[typemax(Int), 2])),
            Dict("x"=>meta(offsets=[0])),
            Dict("x"=>meta(offsets=[0, 4, 8])),
            Dict("x"=>meta(offsets=[-1, 3])),
            Dict("x"=>meta(offsets=[4, 0])),
            Dict("x"=>meta(offsets=[0, 12])),
            Dict("x"=>meta(dtype="I32")),
            Dict("x"=>meta(), "y"=>meta()), # overlap
            Dict("x"=>meta(offsets=[4, 8])), # hole
            Dict("x"=>meta()), # unindexed trailing payload
            Dict("__metadata__"=>Dict("k"=>1), "x"=>meta(offsets=[0, 8], shape=[2])),
            Dict("x"=>Dict("shape"=>[2])),
        ]
        for header in bad_headers
            @test_throws Gesso.GessoError Gesso.load_safetensors(raw(header, bytes))
        end
        @test_throws Gesso.GessoError Gesso.load_safetensors(
            raw("{}"; length_override=1000),
        )
        @test_throws Gesso.GessoError Gesso.load_safetensors(
            raw("{}"; length_override=100_000_001),
        )
        @test Gesso.load_safetensors(
            raw(Dict("empty"=>meta(shape=[0], offsets=[0, 0]))),
        )["empty"] == Float64[]
        @test isempty(Gesso.load_safetensors(raw("{}")))
        # Format permits nonfinite floating values; execution policy is separate.
        @test isnan(
            Gesso.load_safetensors(
                raw(Dict("x"=>meta()), collect(reinterpret(UInt8, Float32[NaN]))),
            )["x"][1],
        )
        for field in (
            "num_hidden_layers",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
        )
            checkpoint = joinpath(dir, field)
            make_micro_checkpoint(checkpoint)
            configpath=joinpath(checkpoint, "config.json")
            cfg=JSON.parsefile(configpath)
            cfg[field]=0
            write(configpath, JSON.json(cfg))
            @test_throws Gesso.GessoError Gesso.load_llama_config(configpath)
        end
        sharddir=joinpath(dir, "shards")
        make_micro_checkpoint(sharddir; shards=true)
        idxpath=joinpath(sharddir, "model.safetensors.index.json")
        idx=JSON.parsefile(idxpath)
        wm=idx["weight_map"]
        names=collect(keys(wm))
        a=first(names)
        b=first(filter(n->wm[n]!=wm[a], names))
        wm[a], wm[b]=wm[b], wm[a]
        write(idxpath, JSON.json(idx))
        @test_throws Gesso.GessoError Gesso.load_llama(sharddir)
        wm[a]="../escape.safetensors"
        write(idxpath, JSON.json(idx))
        @test_throws Gesso.GessoError Gesso.load_llama(sharddir)
        for kw in (
            (; theta=Inf),
            (; theta=0),
            (; factor=Inf),
            (; low_freq_factor=0),
            (; high_freq_factor=-1),
            (;
                kind=:llama3,
                original_max_position_embeddings=8192,
                high_freq_factor=1,
                low_freq_factor=1,
            ),
        )
            @test_throws ErrorException Gesso.RoPEPolicy(; kw...)
        end
        tkdir=joinpath(dir, "tokenizer")
        mkpath(tkdir)
        write(joinpath(tkdir, "merges.txt"), "#version: 0.2\n")
        for vocab in
            ("{\"a\":0,\"a\":1}", "{\"a\":0,\"b\":0}", "{\"a\":0,\"b\":2}", "{\"a\":-1}")
            write(joinpath(tkdir, "vocab.json"), vocab)
            @test_throws Gesso.GessoError Gesso.load_gpt2_tokenizer(tkdir)
        end
    end
end
