"""Tests for tool_handler module."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.agent.tool_handler import (
    parse_openai_tools,
    parse_claude_tools,
    build_tool_prompt,
    detect_tool_call,
    detect_all_tool_calls,
    fallback_verbal_tool_call,
    format_chat_messages,
    build_openai_tool_stream_chunks,
    build_openai_tool_response,
    ToolCall,
)


def test_parse_openai_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    parsed = parse_openai_tools(tools)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "get_weather"


def test_parse_claude_tools():
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    parsed = parse_claude_tools(tools)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "get_weather"


def test_build_tool_prompt():
    tools = [{"name": "get_weather", "description": "Get weather", "parameters": {}}]
    prompt = build_tool_prompt(tools)
    assert "get_weather" in prompt
    assert "<tool_call>" in prompt
    assert "严禁口头承诺" in prompt


def test_detect_tool_call_standard():
    text = 'some text\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>'
    result = detect_tool_call(text)
    assert result is not None
    assert result.name == "get_weather"
    assert result.arguments == {"city": "Beijing"}


def test_detect_tool_call_markdown_codeblock():
    text = '好的，我来调用天气工具：\n```tool_call\n{"name": "get_weather", "arguments": {"city": "Shanghai"}}\n```'
    clean, calls = detect_all_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Shanghai"}


def test_detect_tool_call_deepseek_dsml():
    # 模拟真实线上截图中 DeepSeek 输出带有 </ | | DSML ...> 标签的案例
    raw = """我来创建脚本并直接运行。

<tool_call> {"name": "Write", "arguments": {"file_path": "/Users/sealos/.zcode/workspace/draft-2a46b959-7bd9-451e-86c0-dff3d9f5235c/hello.js", "content": "#!/usr/bin/env node\\nconsole.log(1);"}} </ | | DSML | | parameter> </ | | DSML | | invoke> </ | | DSML | | calls>"""
    clean, calls = detect_all_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0].name == "Write"
    assert calls[0].arguments["file_path"].endswith("hello.js")
    assert clean == "我来创建脚本并直接运行。"


def test_detect_tool_call_none():
    assert detect_tool_call("no tool call here") is None


def test_fallback_verbal_tool_call():
    # 模拟用户在 ZCode 中提问，模型口头承诺“进入任务模式”，但漏掉了 <tool_call>
    tools = [
        {
            "name": "Agent",
            "description": "Launch a new agent to handle complex, multi-step tasks",
            "parameters": {
                "properties": {
                    "prompt": {"type": "string", "description": "The task prompt"}
                }
            }
        }
    ]
    messages = [
        {"role": "user", "content": "帮我用 nodejs 写一个 hello world 的 api 并运行起来"}
    ]
    model_reply = "我来帮你完成这个任务。写一个 Node.js 的 Hello World API 并把它跑起来，需要进入任务模式来实际编写代码并执行。"
    
    calls = fallback_verbal_tool_call(model_reply, tools, messages)
    assert len(calls) == 1
    assert calls[0].name == "Agent"
    assert calls[0].arguments == {"prompt": "帮我用 nodejs 写一个 hello world 的 api 并运行起来"}


def test_format_chat_messages_directive():
    messages = [
        {"role": "user", "content": "帮我写代码"}
    ]
    formatted = format_chat_messages(messages, max_budget=4000, has_tools=True)
    assert "[System Action Directive]:" in formatted
    assert "[Assistant]:" in formatted
    assert formatted.strip().endswith("[Assistant]:")


def test_build_openai_tool_stream_chunks_pure_delta():
    call = ToolCall(id="call_123", name="Agent", arguments={"prompt": "test"})
    chunks = build_openai_tool_stream_chunks("cmpl-123", "glm-5-3-flash", [call])
    
    # 验证生成的 chunk 包含 tool_calls 且 delta 不带 role
    import json
    parsed_chunks = [json.loads(c.replace("data: ", "").strip()) for c in chunks if c.startswith("data: {")]
    tc_chunk = next(c for c in parsed_chunks if "tool_calls" in c["choices"][0]["delta"])
    assert "role" not in tc_chunk["choices"][0]["delta"]
    assert tc_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Agent"


def test_build_openai_tool_response():
    call = ToolCall(id="call_123", name="get_weather", arguments={"city": "Beijing"})
    resp = build_openai_tool_response("cmpl-123", "kimi-k3", call)
    assert resp["object"] == "chat.completion.chunk"
    choices = resp["choices"]
    assert len(choices) == 1
    tc = choices[0]["delta"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert "Beijing" in tc["function"]["arguments"]
    assert choices[0]["finish_reason"] == "tool_calls"
