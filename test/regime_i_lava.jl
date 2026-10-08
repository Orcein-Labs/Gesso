# Primary device acceptance: no unavailable-device skip in campaign invocation.
using Gesso, Lava, JSON, Test
backend=Gesso.LavaBackend()
const ext=Base.get_extension(Gesso, :GessoLavaExt)
sync() = ext._lava_sync!()
function owned_size(s)
    return Base.summarysize(
        s;
        exclude=Union{
            DataType,
            Core.TypeName,
            Core.MethodInstance,
            Lava.VkContext,
            Lava.BatchQueue,
            Lava.PoolBlock,
        },
    )
end
function owned_gpu_bytes(s)
    arrays=IdDict{Any, Nothing}()
    add(a) = (a isa Lava.LavaArray && (arrays[a]=nothing); nothing)
    add(s.h)
    for name in propertynames(s.ws)
        x=getproperty(s.ws, name)
        x===nothing && continue
        add(hasproperty(x, :storage) ? x.storage : x)
    end
    for x in (s.tensors.embedding, s.tensors.lm_head, s.tensors.final_rms)
        x===nothing || add(x.storage)
    end
    for block in s.tensors.blocks, name in propertynames(block)
        add(getproperty(block, name).storage)
    end
    for lists in (s.mgr.k_pages, s.mgr.v_pages), pages in lists, page in pages
        add(page.storage)
    end
    return sum(length(a)*sizeof(eltype(a)) for a in keys(arrays))
end
function clean_device()
    sync()
    GC.gc(true)
    Lava.trim_gpu_pool!()
    sync()
    GC.gc(true)
end
checkpoint=ENV["GESSO_SMOLLM2_DIR"];
ref=JSON.parsefile(ENV["GESSO_HF_REFERENCE"])
model, ts, cfg=Gesso.load_llama(checkpoint);
gpu=Gesso.to_device(backend, ts)
records=[]
@testset "Regime I primary Lava real-model lifecycle" begin
    @test gpu.rope == ts.rope
    @test eltype(gpu.embedding.storage)==Float32
    s=Gesso.Session(
        model,
        gpu;
        backend,
        context_length=64,
        page_size=4,
        eos_token_id=0,
        eps=cfg.rms_norm_eps,
        theta=cfg.rope_theta,
    )
    ws=s.ws
    for case in ref["cases"]
        ids=Int.(case["prompt_ids"])
        Gesso.Inference._session_reset!(s)
        logits=Gesso.prefill!(s, ids)
        sync()
        delta=maximum(
            abs.(Float64.(Array(logits)[:, end]) .- Float64.(case["last_logits"])),
        )
        @test delta <= 1e-2
        @test Gesso.generate(s, ids; max_new_tokens=8)==Int.(case["generated_ids"])
        @info "Primary Lava independent parity" prompt=case["prompt"] delta
        push!(records, (; prompt=case["prompt"], max_abs_delta=delta))
    end
    case=ref["cases"][2]
    ids=Int.(case["prompt_ids"])
    Gesso.generate(s, ids; max_new_tokens=8)
    clean_device()
    before=Base.gc_live_bytes()
    retained_before=owned_size(s)
    owned_before=owned_gpu_bytes(s)
    device_before=Lava.gpu_live_bytes()
    for repetition in 1:6
        @test Gesso.generate(s, ids; max_new_tokens=8)==Int.(case["generated_ids"])
        @test s.ws===ws
        pages=2*length(model.blocks)*cld(s.seqlen, s.page_size)
        @test Gesso.Inference.page_count(s.mgr)==pages
        @test Gesso.Inference.kv_bytes(s.mgr)==pages*s.page_size*s.n_kv_heads*s.d_head*sizeof(
            Float32,
        )
    end
    clean_device()
    after=Base.gc_live_bytes()
    retained_after=owned_size(s)
    owned_after=owned_gpu_bytes(s)
    device_after=Lava.gpu_live_bytes()
    @test owned_after==owned_before
    @test device_after-device_before < 64*1024^2
    @test after-before < 20*1024^2
    @test retained_after-retained_before < 1024^2
    Gesso.Inference._session_reset!(s)
    @test Gesso.Inference.page_count(s.mgr)==0
    small=Gesso.Session(
        model,
        gpu;
        backend,
        context_length=length(ids),
        page_size=4,
        eos_token_id=0,
        eps=cfg.rms_norm_eps,
        theta=cfg.rope_theta,
    )
    Gesso.prefill!(small, ids)
    caught=try
        Gesso.decode!(small)
        nothing
    catch err
        err
    end
    @test caught isa Gesso.GessoError && caught.code==Gesso.ERR_RESOURCE_LIMIT
    @test Gesso.generate(small, [0]; max_new_tokens=1)==Gesso.generate(
        s,
        [0];
        max_new_tokens=1,
    )
    # Reload owns fresh materialization, then releases it after GC.
    let (rm, rt, rc)=Gesso.load_llama(checkpoint)
        reloaded=Gesso.to_device(backend, rt)
        rs=Gesso.Session(
            rm,
            reloaded;
            backend,
            context_length=64,
            eos_token_id=0,
            eps=rc.rms_norm_eps,
            theta=rc.rope_theta,
        )
        @test Gesso.generate(rs, ids; max_new_tokens=8)==Int.(case["generated_ids"])
    end
    sync()
    GC.gc(true)
    write(
        ARGS[1],
        JSON.json((;
            schema="gesso-primary-device-lifecycle-v1",
            backend="lava",
            dtype="Float32",
            hardware="NVIDIA RTX 5060 via Vulkan",
            live_before=before,
            live_after=after,
            retained_before,
            retained_after,
            owned_before,
            owned_after,
            device_before,
            device_after,
            records,
        )),
    )
end
