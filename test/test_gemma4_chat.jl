using Test
using JSON
using Gesso

const _G4_TK = Gesso.Inference
if !isdefined(_G4_TK, :render_gemma4_chat)
    Base.include(_G4_TK, joinpath(@__DIR__, "..", "src", "Inference", "gemma4_chat.jl"))
end
const render_gemma4_chat = _G4_TK.render_gemma4_chat
const chat_family = _G4_TK.chat_family

@testset "chat_family detection" begin
    gemma4_tpl = "<bos><|turn>system\n<turn|><|turn>user\n<turn|><|channel>thought\n<channel|>"
    @test chat_family(gemma4_tpl) === :gemma4

    # Regressions for existing templates
    lfm25_tpl = "CONTINUE_FINAL_MESSAGE_TAG ... render_tool_calls"
    @test chat_family(lfm25_tpl) === :lfm25

    lfm2_tpl = "<|tool_list_start|> ... <|tool_response_start|>"
    @test chat_family(lfm2_tpl) === :lfm2

    qwen_tpl = "<function=...><tool_response>...<think>..."
    @test chat_family(qwen_tpl) === :qwen35_xml

    @test_throws ErrorException chat_family("unknown template")
end

@testset "render_gemma4_chat: basic conversation and generation prompt" begin
    # 1. Simple user message, enable_thinking=false, add_generation_prompt=true
    msgs = [Dict("role" => "user", "content" => "Hello")]
    rendered = render_gemma4_chat(msgs; enable_thinking=false, add_generation_prompt=true)
    @test rendered == "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"

    # 2. Simple user message, enable_thinking=true, add_generation_prompt=true
    rendered_think = render_gemma4_chat(msgs; enable_thinking=true, add_generation_prompt=true)
    # enable_thinking=true injects initial system turn with <|think|>\n
    @test startswith(rendered_think, "<bos><|turn>system\n<|think|>\n<turn|>\n")
    @test occursin("<|turn>user\nHello<turn|>\n", rendered_think)
    @test endswith(rendered_think, "<|turn>model\n")
    @test !occursin("<|channel>thought\n<channel|>", rendered_think)

    # 3. add_generation_prompt=false
    rendered_nogp = render_gemma4_chat(msgs; add_generation_prompt=false)
    @test rendered_nogp == "<bos><|turn>user\nHello<turn|>\n"
end

@testset "render_gemma4_chat: system and developer prompts" begin
    # System role
    msgs = [
        Dict("role" => "system", "content" => "You are an expert mathematician."),
        Dict("role" => "user", "content" => "What is 2+2?"),
    ]
    rendered = render_gemma4_chat(msgs; enable_thinking=false, add_generation_prompt=true)
    @test startswith(rendered, "<bos><|turn>system\nYou are an expert mathematician.<turn|>\n")
    @test occursin("<|turn>user\nWhat is 2+2?<turn|>\n", rendered)

    # Developer role acts as system
    msgs_dev = [
        Dict("role" => "developer", "content" => "System instruction."),
        Dict("role" => "user", "content" => "Hi"),
    ]
    rendered_dev = render_gemma4_chat(msgs_dev; enable_thinking=false, add_generation_prompt=false)
    @test rendered_dev == "<bos><|turn>system\nSystem instruction.<turn|>\n<|turn>user\nHi<turn|>\n"

    # Multi-part content in system
    msgs_parts = [
        Dict("role" => "system", "content" => [
            Dict("type" => "text", "text" => "Part 1."),
            Dict("type" => "text", "text" => "Part 2."),
        ]),
        Dict("role" => "user", "content" => "Hi"),
    ]
    rendered_parts = render_gemma4_chat(msgs_parts; add_generation_prompt=false)
    @test occursin("Part 1. Part 2. <turn|>\n", rendered_parts)
end

@testset "render_gemma4_chat: thinking stripping and reasoning channel" begin
    # strip_thinking on prior model content
    msgs = [
        Dict("role" => "user", "content" => "Solve X."),
        Dict("role" => "assistant", "content" => "<|channel>thought\nInternal scratchpad\n<channel|>Final answer is 42."),
        Dict("role" => "user", "content" => "Great, thanks!"),
    ]
    rendered = render_gemma4_chat(msgs; add_generation_prompt=false)
    @test !occursin("Internal scratchpad", rendered)
    @test occursin("Final answer is 42.<turn|>\n", rendered)

    # reasoning_content emitted on turns after last user message
    msgs_reason = [
        Dict("role" => "user", "content" => "Question 1"),
        Dict("role" => "assistant", "reasoning_content" => "Early thought", "content" => "Answer 1"),
        Dict("role" => "user", "content" => "Question 2"),
        Dict("role" => "assistant", "reasoning" => "Late thought", "content" => "Answer 2"),
    ]
    # Without preserve_thinking, reasoning on the turn before Question 2 is suppressed
    rendered_r = render_gemma4_chat(msgs_reason; preserve_thinking=false, add_generation_prompt=false)
    @test !occursin("Early thought", rendered_r)
    @test occursin("<|channel>thought\nLate thought\n<channel|>", rendered_r)

    # preserve_thinking on assistant with tool calls before last user
    msgs_preserve = [
        Dict("role" => "user", "content" => "Call tool"),
        Dict(
            "role" => "assistant",
            "reasoning" => "Planning tool call",
            "tool_calls" => [
                Dict("id" => "c1", "function" => Dict("name" => "f", "arguments" => Dict()))
            ]
        ),
        Dict("role" => "tool", "tool_call_id" => "c1", "content" => "ok"),
        Dict("role" => "user", "content" => "Next turn"),
    ]
    r_no_pres = render_gemma4_chat(msgs_preserve; preserve_thinking=false, add_generation_prompt=false)
    @test !occursin("Planning tool call", r_no_pres)
    r_pres = render_gemma4_chat(msgs_preserve; preserve_thinking=true, add_generation_prompt=false)
    @test occursin("Planning tool call", r_pres)
end

@testset "render_gemma4_chat: tool declarations and formatting" begin
    tools = [
        Dict(
            "type" => "function",
            "function" => Dict(
                "name" => "get_temperature",
                "description" => "Look up temperature for a location",
                "parameters" => Dict(
                    "type" => "object",
                    "properties" => Dict(
                        "location" => Dict("type" => "string", "description" => "City name"),
                        "unit" => Dict("type" => "string", "enum" => ["C", "F"], "nullable" => true),
                    ),
                    "required" => ["location"],
                ),
            ),
        ),
    ]
    msgs = [Dict("role" => "user", "content" => "What's the weather?")]
    rendered = render_gemma4_chat(msgs; tools=tools, add_generation_prompt=false)
    @test startswith(rendered, "<bos><|turn>system\n<|tool>declaration:get_temperature{description:<|\"|>Look up temperature for a location<|\"|>")
    @test occursin("properties:{location:{description:<|\"|>City name<|\"|>,type:<|\"|>STRING<|\"|>}", rendered)
    @test occursin("unit:{enum:[<|\"|>C<|\"|>,<|\"|>F<|\"|>],nullable:true,type:<|\"|>STRING<|\"|>}}", rendered)
    @test occursin("required:[<|\"|>location<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n", rendered)
end

@testset "render_gemma4_chat: tool calls and OpenAI-style responses" begin
    # Tool call with Dict arguments, no tool response -> ends with <|tool_response>
    msgs_call = [
        Dict("role" => "user", "content" => "Search books"),
        Dict(
            "role" => "assistant",
            "tool_calls" => [
                Dict(
                    "id" => "tc_101",
                    "function" => Dict(
                        "name" => "search",
                        "arguments" => Dict("query" => "Julia lang", "limit" => 5),
                    ),
                ),
            ],
        ),
    ]
    rendered_call = render_gemma4_chat(msgs_call; add_generation_prompt=true)
    @test occursin("<|turn>model\n<|tool_call>call:search{limit:5,query:<|\"|>Julia lang<|\"|>}<tool_call|><|tool_response>", rendered_call)
    @test !endswith(rendered_call, "<turn|>\n")

    # Tool call with pre-serialized string arguments
    msgs_call_str = [
        Dict("role" => "user", "content" => "Search"),
        Dict(
            "role" => "assistant",
            "tool_calls" => [
                Dict(
                    "id" => "tc_102",
                    "function" => Dict(
                        "name" => "find",
                        "arguments" => "{\"id\": 42}",
                    ),
                ),
            ],
        ),
    ]
    rendered_str = render_gemma4_chat(msgs_call_str; add_generation_prompt=false)
    @test occursin("<|tool_call>call:find{\"id\": 42}<tool_call|>", rendered_str)

    # Tool call with consecutive role:tool follow message (OpenAI style)
    msgs_complete = [
        Dict("role" => "user", "content" => "Search books"),
        Dict(
            "role" => "assistant",
            "tool_calls" => [
                Dict(
                    "id" => "tc_101",
                    "function" => Dict(
                        "name" => "search",
                        "arguments" => Dict("query" => "Julia"),
                    ),
                ),
            ],
        ),
        Dict(
            "role" => "tool",
            "tool_call_id" => "tc_101",
            "content" => "Found 3 results",
        ),
    ]
    # enable_thinking=true with generation prompt produces trailing <|channel>thought\n
    rendered_complete = render_gemma4_chat(msgs_complete; enable_thinking=true, add_generation_prompt=true)
    @test occursin("<|tool_call>call:search{query:<|\"|>Julia<|\"|>}<tool_call|>", rendered_complete)
    @test occursin("<|tool_response>response:search{value:<|\"|>Found 3 results<|\"|>}<tool_response|>", rendered_complete)
    @test endswith(rendered_complete, "<tool_response|><|channel>thought\n")

    # enable_thinking=false with generation prompt stops at <tool_response|>
    rendered_no_think = render_gemma4_chat(msgs_complete; enable_thinking=false, add_generation_prompt=true)
    @test endswith(rendered_no_think, "<tool_response|>")

    # Native tool_responses on assistant message
    msgs_native = [
        Dict("role" => "user", "content" => "Ping"),
        Dict(
            "role" => "assistant",
            "tool_responses" => [
                Dict(
                    "name" => "ping",
                    "response" => Dict("status" => "pong", "code" => 200),
                ),
            ],
            "content" => "All systems operational.",
        ),
    ]
    rendered_nat = render_gemma4_chat(msgs_native; add_generation_prompt=false)
    @test occursin("<|tool_response>response:ping{code:200,status:<|\"|>pong<|\"|>}<tool_response|>", rendered_nat)
    @test occursin("All systems operational.<turn|>\n", rendered_nat)
end
