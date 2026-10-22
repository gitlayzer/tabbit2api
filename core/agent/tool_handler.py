"""
Tool Handler — function calling / tool use support for OpenAI and Claude APIs.

Implements:
- Soft tool calling: inject tool schema and convert model output into standard tool_calls
- Multi-turn conversation formatter preserving full context and tool results
- Standard streaming & non-streaming responses for Agent desktop clients (ZCode, WorkBuddy, Claude, Trae)
"""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tabbit2agent.tools")

# ── Tool definitions ──

@dataclass
class ToolCall:
    """Represents a single tool call."""
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallChain:
    """Tracks a tool call chain within a single room."""
    room_id: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


# In-memory tool call chains (room_id → chain)
_tool_chains: Dict[str, ToolCallChain] = {}


# ── Tool definition parser ──

def parse_openai_tools(tools: List[Any]) -> List[dict]:
    """Parse OpenAI tools array into a normalized format."""
    if not tools:
        return []
    result = []
    for t in tools:
        if isinstance(t, dict):
            if t.get("type") == "function":
                func = t.get("function", {})
                result.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
        else:
            t_type = getattr(t, "type", "")
            func = getattr(t, "function", None)
            if t_type == "function" and func:
                result.append({
                    "name": getattr(func, "name", ""),
                    "description": getattr(func, "description", ""),
                    "parameters": getattr(func, "parameters", {}),
                })
    return result


def parse_claude_tools(tools: List[dict]) -> List[dict]:
    """Parse Claude tools array into a normalized format."""
    if not tools:
        return []
    result = []
    for t in tools:
        result.append({
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        })
    return result


CORE_TOOL_NAMES = {
    "agent", "bash", "write", "edit", "read", "askuserquestion",
    "taskoutput", "taskstop", "enterplanmode", "exitplanmode", "skill", "webfetch", "todo"
}


def build_tool_prompt(tools: List[dict], max_chars: int = 2200) -> str:
    """Build an authoritative, ultra-compact tool prompt to inject into model context.
    Prioritizes core tools (Bash, Write, Read, Agent, etc.) and caps non-core tools
    to strictly prevent exceeding Tabbit's safe character ceiling.
    """
    if not tools:
        return ""

    core_tools = []
    other_tools = []
    for t in tools:
        t_name = t.get("name", "").lower()
        if any(c in t_name for c in CORE_TOOL_NAMES):
            core_tools.append(t)
        else:
            other_tools.append(t)

    tool_lines = []
    for t in core_tools:
        name = t.get("name", "")
        desc = (t.get("description", "") or "").split("\n")[0][:80]
        params = t.get("parameters", {})
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        param_sigs = [
            f"{p_name}: {p_info.get('type', 'string') if isinstance(p_info, dict) else 'string'}"
            for p_name, p_info in props.items()
        ]
        sig = f"{name}({', '.join(param_sigs[:4])})"
        tool_lines.append(f"- {sig}: {desc}")

    curr_len = sum(len(l) for l in tool_lines)
    for t in other_tools:
        if curr_len > max_chars - 350:
            remaining_count = len(other_tools) - (len(tool_lines) - len(core_tools))
            if remaining_count > 0:
                tool_lines.append(f"- ... (plus {remaining_count} additional specialized tools)")
            break
        name = t.get("name", "")
        desc = (t.get("description", "") or "").split("\n")[0][:50]
        line = f"- {name}: {desc}"
        tool_lines.append(line)
        curr_len += len(line)

    prompt = (
        "【系统环境可用工具 / AVAILABLE TOOLS】\n"
        "你当前处于 Agent 自动化开发环境中，必须通过调用以下本地工具完成文件读写、代码编写与命令执行：\n"
        + "\n".join(tool_lines)
        + "\n\n"
        "【执行规则（最高优先级）/ MANDATORY RULES】：\n"
        "1. 【严禁自然语言口头承诺】：绝对禁止仅在口头上回复“我来帮你写...”、“我这就开始...”、“好的，我马上在本地为你搭建...”、“现在开始执行”等任何无意义的客套话！严禁只说不调！\n"
        "2. 【直接以工具调用开头】：当用户要求写代码、创建修改文件、运行终端或启动任务时，你的回复必须【直接以 <tool_call> 开头】！严禁在 <tool_call> 前面输出任何中文句子！\n"
        "3. 【工具调用格式】：\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {<json_arguments>}}\n'
        "</tool_call>\n\n"
        "【正确示范 / CORRECT EXAMPLE】：\n"
        "User: 帮我写一个 nodejs 的 api 并运行起来\n"
        "Assistant:\n"
        "<tool_call>\n"
        '{"name": "write", "arguments": {"path": "app.js", "content": "const http = require(\'http\');\\nconst server = http.createServer((req, res) => res.end(\'Hello World\'));\\nserver.listen(3000, () => console.log(\'Server on 3000\'));"}}\n'
        "</tool_call>\n\n"
        "【错误示范（绝对禁止）/ FORBIDDEN】：\n"
        "好的，我来帮你创建一个简单的 Node.js Hello World API 并在本地运行起来，现在开始执行。<--- 严重违规！绝对禁止输出纯口头回复！"
    )
    return prompt


# ── Tool call detection ──

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
TOOL_TAG_PATTERN = re.compile(
    r"<tool>\s*([\s\S]*?)\s*</tool>", re.DOTALL | re.IGNORECASE
)
TOOL_CODEBLOCK_PATTERN = re.compile(
    r"```(?:tool_call|tool)\s*([\s\S]*?)\s*```", re.DOTALL | re.IGNORECASE
)


def detect_tool_call(text: str) -> Optional[ToolCall]:
    """Detect single tool call (backward compatibility)."""
    _, calls = detect_all_tool_calls(text)
    return calls[0] if calls else None


def _try_parse_tool_json(raw_str: str) -> Optional[ToolCall]:
    """Attempt to parse a JSON string into a ToolCall object."""
    cleaned = raw_str.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        fn_name = data.get("name") or data.get("tool") or data.get("function") or ""
        fn_args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
        if isinstance(fn_args, str):
            try:
                fn_args = json.loads(fn_args)
            except Exception:
                pass
        if fn_name:
            return ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=fn_name,
                arguments=fn_args if isinstance(fn_args, dict) else {"input": fn_args},
            )
    except Exception:
        pass
    return None


def extract_json_objects(text: str) -> List[Tuple[dict, int, int]]:
    """Extract all valid JSON objects from text using depth-tracking balance parser.
    Returns list of (parsed_dict, start_pos, end_pos).
    Properly ignores braces inside strings and handles escapes.
    """
    results = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            start = i
            depth = 0
            in_string = False
            escape = False
            j = i
            while j < n:
                char = text[j]
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = not in_string
                elif not in_string:
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            raw_candidate = text[start:j+1]
                            try:
                                obj = json.loads(raw_candidate)
                                if isinstance(obj, dict):
                                    results.append((obj, start, j+1))
                                    i = j
                                    break
                            except Exception:
                                pass
                j += 1
        i += 1
    return results


def detect_all_tool_calls(text: str) -> Tuple[str, List[ToolCall]]:
    """Detect all tool call blocks in text, supporting:
    - Standard <tool_call>...</tool_call>
    - Unclosed or DSML-closed <tool_call>... </ | | DSML ...>
    - <tool>...</tool>
    - ```tool_call ... ```
    - ```json {"name": "...", "arguments": ...} ```
    - Raw JSON with name and arguments anywhere in context
    Returns (clean_text, [ToolCall, ...]).
    """
    tool_calls = []
    clean_text = text

    # Identify the earliest marker that indicates tool calling begun
    markers = ["<tool_call", "<tool", "```tool_call", "```tool", "<｜tool", "<|tool", "</ | | DSML", "<｜DSML"]
    first_marker_pos = -1
    for m in markers:
        pos = text.lower().find(m.lower())
        if pos != -1:
            if first_marker_pos == -1 or pos < first_marker_pos:
                first_marker_pos = pos

    # Extract all top-level JSON objects via balanced braces
    json_objs = extract_json_objects(text)
    for obj, start, end in json_objs:
        fn_name = obj.get("name") or obj.get("tool") or obj.get("function") or ""
        fn_args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}
        if isinstance(fn_args, str):
            try:
                fn_args = json.loads(fn_args)
            except Exception:
                pass
        if fn_name and isinstance(fn_name, str):
            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=fn_name,
                arguments=fn_args if isinstance(fn_args, dict) else {"input": fn_args},
            ))

    if tool_calls:
        if first_marker_pos != -1:
            clean_text = text[:first_marker_pos].strip()
        else:
            # If no clear marker prefix, strip each recognized tool call JSON block
            clean_text = text
            for obj, start, end in reversed(json_objs):
                fn_name = obj.get("name") or obj.get("tool") or obj.get("function") or ""
                if fn_name:
                    clean_text = clean_text[:start] + clean_text[end:]
            clean_text = clean_text.strip()

        # Strip any residual markup (DSML tokens, tool tags, unclosed brackets)
        clean_text = re.sub(r"<[^>]*DSML[^>]*>", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"<\/?tool_call[\s\S]*?>", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"<\/?tool[\s\S]*?>", "", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"</\s*\|\s*\|\s*[^>]*>", "", clean_text, flags=re.IGNORECASE)
        clean_text = clean_text.strip()

    return clean_text, tool_calls


def fallback_verbal_tool_call(
    text: str,
    tools: List[dict],
    messages: List[Any],
) -> List[ToolCall]:
    """If the model verbally promises to enter task mode or fulfill a task
    without emitting tool call tags, automatically wrap it into an Agent tool call
    to prevent desktop Agent clients (ZCode/Cline/Cursor) from stalling.
    """
    if not tools or not text:
        return []

    # Check for verbal promise markers across variations
    task_promise_keywords = [
        "agent 模式", "agent模式", "agent 流程", "agent流程", "agent 执行流程", "agent执行流程",
        "进入 agent", "启动 agent", "我来用 agent", "先进入 agent",
        "进入任务模式", "任务模式", "直接进入任务", "切换到任务模式",
        "我来帮你完成", "我来帮你写", "我来帮你创建", "帮你把", "我来为你搭建", "为你搭建",
        "我马上在本地为你", "在本地为你搭建", "我来创建", "我来用", "先进入", "准备为你",
        "我来为你编写", "我来为你生成", "我来编写", "我来运行",
        "enter task mode", "entering task mode", "agent mode", "I will create", "I will implement",
    ]
    has_promise = any(kw in text.lower() for kw in task_promise_keywords)
    if not has_promise:
        return []

    # Find an Agent orchestration tool (e.g. ZCode's "Agent" or "Task")
    # SAFETY: ONLY target tools whose parameter explicitly accepts a task prompt.
    # NEVER fallback to 'bash', 'write' or 'read' with natural language!
    target_tool = None
    for t in tools:
        t_name = t.get("name", "").lower()
        if t_name in ("agent", "task", "task_mode", "launch_agent", "execute_agent"):
            target_tool = t
            break

    if not target_tool:
        return []

    # Determine parameter name for the task prompt
    params = target_tool.get("parameters", {})
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    arg_name = "prompt"
    for candidate in ["prompt", "task", "instruction", "input", "query"]:
        if candidate in props:
            arg_name = candidate
            break

    # Extract user's latest task intent from messages
    latest_user_text = ""
    for m in reversed(messages):
        r = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
        if r == "user":
            c = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if isinstance(c, list):
                parts = [item.get("text", "") for item in c if isinstance(item, dict) and item.get("type") == "text"]
                latest_user_text = "\n".join(parts).strip()
            else:
                latest_user_text = str(c or "").strip()
            if latest_user_text:
                break

    task_payload = latest_user_text or text
    tool_args = {arg_name: task_payload}
    if "description" in props and "description" not in tool_args:
        tool_args["description"] = f"Run: {task_payload[:60]}"

    logger.warning(
        f"[FALLBACK] Automatically converted verbal promise to ToolCall: tool={target_tool.get('name')} args={list(tool_args.keys())}"
    )
    return [
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=target_tool.get("name", "Agent"),
            arguments=tool_args,
        )
    ]


# ── Budget-Aware Multi-Turn Chat Formatter ──

def format_chat_messages(messages: List[Any], max_budget: int = 5500, has_tools: bool = False) -> str:
    """Format multi-turn chat messages within a safe token/char budget for Tabbit.
    Preserves:
    1. System instructions (core instructions, compact Head+Tail preservation)
    2. Latest User query (100% complete)
    3. Recent tool execution results (key content preserved)
    4. Recent conversation turns, backwards until budget is reached
    5. Action directive suffix if tools are available
    """
    if not messages:
        return ""

    parsed = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)
            tool_call_id = getattr(msg, "tool_call_id", None)

        if isinstance(content, list):
            text_items = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_items.append(item.get("text", ""))
                    elif item.get("type") == "tool_result":
                        text_items.append(str(item.get("content", "")))
                    else:
                        text_items.append(str(item.get("content", "")))
                else:
                    text_items.append(str(item))
            content_str = "\n".join(text_items)
        else:
            content_str = str(content or "")

        parsed.append({
            "role": role,
            "content": content_str.strip(),
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
        })

    # Separate system messages and dialog messages
    system_texts = []
    dialog_msgs = []
    for m in parsed:
        if m["role"] == "system":
            if m["content"]:
                system_texts.append(m["content"])
        else:
            dialog_msgs.append(m)

    # System instruction: smart Head+Tail preservation up to 2400 chars
    system_block = ""
    if system_texts:
        full_sys = "\n".join(system_texts)
        if len(full_sys) > 2400:
            full_sys = full_sys[:1600] + "\n...(system prompt middle omitted)...\n" + full_sys[-800:]
        system_block = f"[System Instruction]:\n{full_sys}"

    remaining_budget = max_budget - len(system_block) - 150

    # From newest to oldest, accumulate dialog messages
    collected_dialog = []
    used_len = 0

    for m in reversed(dialog_msgs):
        role = m["role"]
        c_str = m["content"]
        
        if role == "user":
            part = f"[User]:\n{c_str}"
        elif role == "assistant":
            tc_desc = ""
            if m["tool_calls"]:
                calls = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                    name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                    calls.append(name)
                tc_desc = f"[Assistant Invoked Tools]: {', '.join(calls)}\n"
            part = f"{tc_desc}[Assistant]:\n{c_str}".strip()
        elif role == "tool":
            if len(c_str) > 1500:
                c_str = c_str[:1500] + "\n...(truncated)"
            tag = f" ({m['tool_call_id']})" if m.get("tool_call_id") else ""
            part = f"[Tool Result{tag}]:\n{c_str}"
        else:
            part = f"[{role.capitalize()}]:\n{c_str}"

        if used_len + len(part) > remaining_budget and collected_dialog:
            break

        collected_dialog.insert(0, part)
        used_len += len(part) + 2

    final_parts = []
    if system_block:
        final_parts.append(system_block)
    final_parts.extend(collected_dialog)

    if has_tools:
        final_parts.append(
            "[System Action Directive]: If this request involves actions (writing code, files, running commands), you MUST start your reply directly with <tool_call>. DO NOT say '我来帮你' or '现在开始执行'. Output <tool_call> immediately."
        )
    final_parts.append("[Assistant]:")

    return "\n\n".join(final_parts)


# ── OpenAI Tool Call Responses ──

def build_openai_tool_response(
    completion_id: str,
    model: str,
    tool_call: ToolCall,
    index: int = 0,
) -> dict:
    """Build an OpenAI-compatible tool call response chunk."""
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": index,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False) if isinstance(tool_call.arguments, dict) else str(tool_call.arguments),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }


def build_openai_tool_stream_chunks(
    completion_id: str,
    model: str,
    tool_calls: List[ToolCall],
    content: str = "",
) -> List[str]:
    """Build a list of SSE chunks delivering tool_calls to OpenAI-compatible clients."""
    chunks = []

    # If there is preceding reasoning or commentary, output it first
    if content:
        c_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }],
        }
        chunks.append(f"data: {json.dumps(c_chunk, ensure_ascii=False)}\n\n")

    # Output tool_calls delta (standard delta without redundant role)
    serialized_tools = []
    for idx, tc in enumerate(tool_calls):
        args_str = json.dumps(tc.arguments, ensure_ascii=False) if isinstance(tc.arguments, dict) else str(tc.arguments)
        serialized_tools.append({
            "index": idx,
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": args_str,
            },
        })

    t_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": serialized_tools},
            "finish_reason": None,
        }],
    }
    chunks.append(f"data: {json.dumps(t_chunk, ensure_ascii=False)}\n\n")

    # Final finish chunk with finish_reason: "tool_calls"
    finish_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls",
        }],
    }
    chunks.append(f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return chunks


def build_openai_tool_completion(
    completion_id: str,
    model: str,
    tool_calls: List[ToolCall],
    content: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict:
    """Build a non-streaming OpenAI chat.completion response with tool_calls."""
    serialized_calls = []
    for tc in tool_calls:
        args_str = json.dumps(tc.arguments, ensure_ascii=False) if isinstance(tc.arguments, dict) else str(tc.arguments)
        serialized_calls.append({
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": args_str,
            },
        })

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": serialized_calls,
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


# ── Tool chain utilities ──

def get_tool_chain(room_id: str) -> Optional[ToolCallChain]:
    return _tool_chains.get(room_id)


def start_tool_chain(room_id: str) -> ToolCallChain:
    chain = ToolCallChain(room_id=room_id)
    _tool_chains[room_id] = chain
    return chain


def add_tool_call(room_id: str, call: ToolCall):
    chain = _tool_chains.get(room_id)
    if not chain:
        chain = start_tool_chain(room_id)
    chain.tool_calls.append(call)


def clear_tool_chain(room_id: str):
    _tool_chains.pop(room_id, None)
