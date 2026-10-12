# Qwen3.5 XML chat template: a hand-ported renderer (no Jinja).
#
# Gate 3 is byte-for-byte parity with llama-server's /apply-template. The
# expected strings in test/fixtures/qwen35_chat were captured from that
# oracle (CPU llama-server, Ornith-35B Q2_K_XL) and are committed. This file
# needs no server and no GGUF.

using Gesso.Inference: render_chat, chat_family, _chat_json_parse, _fmt_g6, _ct_trim

const CHAT_FIXDIR = joinpath(@__DIR__, "fixtures", "qwen35_chat")

# Message text of the error a call raises, or nothing when it does not raise.
function _error_text(f)
    try
        f()
        return nothing
    catch err
        return sprint(showerror, err)
    end
end

@testset "chat template gate 3: byte-for-byte vs llama-server /apply-template" begin
    files = sort(filter(f -> endswith(f, ".json"), readdir(CHAT_FIXDIR)))
    @test length(files) >= 15
    for f in files
        fx = _chat_json_parse(read(joinpath(CHAT_FIXDIR, f), String))
        inp = fx["input"]
        got = render_chat(inp["messages"];
                          tools = get(inp, "tools", nothing),
                          add_generation_prompt = inp["add_generation_prompt"],
                          enable_thinking = inp["enable_thinking"])
        @test got == fx["expected"]
    end
end

@testset "chat template: tojson float formatting (printf %g, precision 6)" begin
    @test _fmt_g6(1.0) == "1"
    @test _fmt_g6(2.5) == "2.5"
    @test _fmt_g6(100000.0) == "100000"
    @test _fmt_g6(1.0e6) == "1e+06"
    @test _fmt_g6(999999.5) == "1e+06"
    @test _fmt_g6(0.000123456789) == "0.000123457"
    @test _fmt_g6(1.0e20) == "1e+20"
    @test _fmt_g6(1.0e-7) == "1e-07"
    @test _fmt_g6(-2.5e-5) == "-2.5e-05"
    @test _fmt_g6(-1.5e-300) == "-1.5e-300"
    @test _fmt_g6(1 / 3) == "0.333333"
    @test _fmt_g6(0.1) == "0.1"
end

@testset "chat template: trim set is ASCII only" begin
    @test _ct_trim(" \t\n\v\f\r a \t\n\v\f\r ") == "a"
    @test _ct_trim(" a ") == " a "    # NBSP kept
    @test _ct_trim("\u0085a\u0085") == "\u0085a\u0085"    # NEL kept
    @test _ct_trim("   ") == ""
end

@testset "chat_family: Qwen3.5 XML markers" begin
    tpl = read(joinpath(CHAT_FIXDIR, "ornith_chat_template.jinja"), String)
    @test chat_family(tpl) === :qwen35_xml
    @test chat_family("<function=x> <tool_response> <think>") === :qwen35_xml
    # Each marker is required on its own.
    @test _error_text(() -> chat_family("<tool_response> <think>")) !== nothing
    @test _error_text(() -> chat_family("<function=x> <think>")) !== nothing
    @test _error_text(() -> chat_family("<function=x> <tool_response>")) !== nothing
    @test _error_text(() -> chat_family("{% for m in messages %}{{ m.content }}{% endfor %}")) !== nothing
end

@testset "chat template: input errors are explicit" begin
    user(c) = Dict{String, Any}("role" => "user", "content" => c)
    asst(; kw...) = Dict{String, Any}("role" => "assistant", "content" => "",
        (string(k) => v for (k, v) in kw)...)
    image = [Dict{String, Any}("type" => "image_url", "image_url" => Dict{String, Any}("url" => "x"))]
    video = [Dict{String, Any}("type" => "video", "video" => "x")]

    @test occursin("image", something(_error_text(() -> render_chat([user(image)])), ""))
    @test occursin("video", something(_error_text(() -> render_chat([user(video)])), ""))
    @test occursin("no messages", something(_error_text(() -> render_chat(Any[])), ""))
    # A mid-conversation assistant is rendered in full, so its arguments are parsed.
    # A trailing assistant is a prefill and drops its tool calls, so it is not used here.
    bad_call = [Dict{String, Any}("type" => "function",
        "function" => Dict{String, Any}("name" => "f", "arguments" => "not json"))]
    @test occursin("JSON", something(_error_text(() ->
        render_chat([user("q"), asst(tool_calls = bad_call), user("next")])), ""))
    @test _error_text(() -> render_chat([user(5)])) !== nothing
    @test _error_text(() -> render_chat([user("q"), asst(reasoning_content = 3)])) !== nothing
    @test _error_text(() -> render_chat([user([Dict{String, Any}("type" => "text")])])) !== nothing
    @test _error_text(() -> render_chat([user("q")]; enable_thinking = "false")) !== nothing
    @test _error_text(() -> render_chat([user("q")]; add_generation_prompt = "true")) !== nothing
end
