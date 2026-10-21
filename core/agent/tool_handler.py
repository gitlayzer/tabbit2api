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


def _compact_parameters(params: Any) -> Any:
    if not isinstance(params, dict):
        return {}
    props = params.get("properties")
    if not isinstance(props, dict):
        return params
    compact_props = {}
    for p_name, p_val in props.items():
        if isinstance(p_val, dict):
            compact_props[p_name] = {
                "type": p_val.get("type", "string"),
                "description": (p_val.get("description", "") or "")[:150],
            }
            if "enum" in p_val:
                compact_props[p_name]["enum"] = p_val["enum"][:10]
        else:
            compact_props[p_name] = {"type": "string"}
    result = {
        "type": "object",
        "properties": compact_props,
    }
    if "required" in params and isinstance(params["required"], list):
        result["required"] = params["required"]
    return result


def build_tool_prompt(tools: List[dict]) -> str:
    """Build an authoritative tool-calling prompt to inject into the model context.
    Instructs the model to output strict <tool_call> XML format when invoking tools.
    """
    if not tools:
        return ""

    tool_descriptions = []
    for t in tools:
        name = t.get("name", "")
        desc = (t.get("description", "") or "")[:250]
        compact_params = _compact_parameters(t.get("parameters", {}))
        params_str = json.dumps(compact_params, ensure_ascii=False)
        tool_descriptions.append(
            f"### Tool: {name}\n- Description: {desc}\n- Parameters Schema: {params_str}"
        )

    prompt = (
        "【CRITICAL SYSTEM INSTRUCTION: TOOLS AVAILABLE】\n"
        "You have access to the following local environment tools to read/write files, execute terminal commands, or interact with the user's workspace:\n\n"
        + "\n\n".join(tool_descriptions)
        + "\n\n"
        "IMPORTANT RULES FOR TOOL INVOCATION:\n"
        "1. When you need to inspect files, run terminal commands, or execute tasks on the user's computer, you MUST output a tool call block.\n"
        "2. Do NOT pretend or fabricate that you have executed commands without calling the tool.\n"
        "3. To invoke a tool, you MUST output strictly formatted XML like this:\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {<arguments_json_object>}}\n'
        "</tool_call>\n"
        "4. You can provide brief explanations before the tool call. Once outputting <tool_call>, the system will execute it and give you the result.\n"
        "5. After receiving the tool result, continue the task."
    )
    return prompt


# ── Tool call detection ──

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)


def detect_tool_call(text: str) -> Optional[ToolCall]:
    """Detect single tool call (backward compatibility)."""
    _, calls = detect_all_tool_calls(text)
    return calls[0] if calls else None


def detect_all_tool_calls(text: str) -> Tuple[str, List[ToolCall]]:
    """Detect all <tool_call> blocks in text, return (clean_text, [ToolCall, ...]).
    Also supports fallback code block format like ```json {"tool": "...", "parameters": {...}} ```.
    """
    tool_calls = []
    clean_text = text
    matches = list(TOOL_CALL_PATTERN.finditer(text))

    if matches:
        clean_text = TOOL_CALL_PATTERN.sub("", text).strip()
        for m in matches:
            raw_json = m.group(1).strip()
            # Strip markdown code blocks if wrapped inside
            raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
            raw_json = re.sub(r"\s*```$", "", raw_json)
            try:
                data = json.loads(raw_json)
                fn_name = data.get("name") or data.get("tool") or data.get("function") or ""
                fn_args = data.get("arguments") or data.get("parameters") or data.get("args") or {}
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        pass
                if fn_name:
                    tool_calls.append(ToolCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=fn_name,
                        arguments=fn_args if isinstance(fn_args, dict) else {"input": fn_args},
                    ))
            except Exception as e:
                logger.warning(f"Error parsing tool call JSON: {e}, raw: {raw_json[:120]}")

    return clean_text, tool_calls


# ── Full Multi-Turn Chat Formatter (No Aggressive Truncation) ──

def format_chat_messages(messages: List[Any]) -> str:
    """Format full multi-turn chat messages into structured conversation text.
    Preserves all user, assistant, system instructions, and tool execution results
    without aggressive 500-char truncation, ensuring seamless Agent loop performance.
    """
    if not messages:
        return ""

    formatted_parts = []

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

        # Normalize content to string
        if isinstance(content, list):
            text_items = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_items.append(item.get("text", ""))
                    elif item.get("type") == "tool_result":
                        text_items.append(str(item.get("content", "")))
                else:
                    text_items.append(str(item))
            content_str = "\n".join(text_items)
        else:
            content_str = str(content or "")

        if role == "system":
            # Preserve system instructions
            if content_str.strip():
                formatted_parts.append(f"[System Instruction]:\n{content_str}")
        elif role == "user":
            formatted_parts.append(f"[User]:\n{content_str}")
        elif role == "assistant":
            # If previous assistant executed tool calls, format them for model context
            if tool_calls:
                call_descs = []
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                    name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
                    args = fn.get("arguments", "") if isinstance(fn, dict) else getattr(fn, "arguments", "")
                    call_descs.append(f"{name}({args})")
                formatted_parts.append(f"[Assistant Invoked Tools]: {', '.join(call_descs)}")
            if content_str.strip():
                formatted_parts.append(f"[Assistant]:\n{content_str}")
        elif role == "tool":
            call_id_tag = f" ({tool_call_id})" if tool_call_id else ""
            formatted_parts.append(f"[Tool Execution Result{call_id_tag}]:\n{content_str}")

    return "\n\n".join(formatted_parts)


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
                "delta": {"role": "assistant", "content": content},
                "finish_reason": None,
            }],
        }
        chunks.append(f"data: {json.dumps(c_chunk, ensure_ascii=False)}\n\n")

    # Output tool_calls delta
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
            "delta": {"role": "assistant", "tool_calls": serialized_tools},
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
