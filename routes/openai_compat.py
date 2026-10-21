import json
import time
import uuid
import logging
import asyncio
from typing import Any, Optional, List, Dict

from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import core.tabbit_client as tabbit_client
from core.tabbit_client import TabbitClient
from core.token_manager import TokenManager
from core.log_store import LogStore, LogEntry
from core.config import ConfigManager

# Agent modules (optional, enabled via config)
from core.agent.message_cleaner import build_agent_content
from core.agent.context_manager import SlidingWindow, ContextCompressor, estimate_tokens
from core.agent.agent_router import ModelSelector
from core.agent.tool_handler import (
    parse_openai_tools, build_tool_prompt, detect_tool_call,
    detect_all_tool_calls, format_chat_messages,
    build_openai_tool_stream_chunks, build_openai_tool_completion,
    build_openai_tool_response, add_tool_call, get_tool_chain, clear_tool_chain,
    ToolCall,
)

logger = logging.getLogger("tabbit2openai")

router = APIRouter()

_tm: TokenManager | None = None
_cfg: ConfigManager | None = None
_logs: LogStore | None = None
_fallback_clients: dict[str, TabbitClient] = {}

# Session cache: key = (bearer_token, model_id) -> {"session_id": str, "created_at": float, "use_v3": bool, "last_used": float, "request_count": int}
_session_cache: dict[str, dict] = {}
SESSION_TTL = 604800  # 7 days — effectively permanent, user manages via admin panel
SESSION_ENABLED = True  # configurable

# Fixed session bindings: key = (bearer_token, model_id) -> room_id (manually set, never expires)
_fixed_sessions: dict[str, str] = {}

# Per-session locks to prevent concurrent runs (avoids 409 Conflict)
_session_locks: dict[str, asyncio.Lock] = {}


def get_session_list() -> list:
    """Return all active sessions for admin display."""
    result = []
    now = time.time()
    for key, entry in _session_cache.items():
        parts = key.split(":", 1)
        bearer = parts[0] if len(parts) > 0 else ""
        model_id = parts[1] if len(parts) > 1 else ""
        result.append({
            "cache_key": key,
            "api_key_preview": (bearer[:12] + "...") if len(bearer) > 12 else bearer,
            "model_id": model_id,
            "session_id": entry.get("session_id", ""),
            "created_at": entry.get("created_at", 0),
            "last_used": entry.get("last_used", entry.get("created_at", 0)),
            "request_count": entry.get("request_count", 0),
            "ttl_remaining": max(0, int(SESSION_TTL - (now - entry.get("created_at", 0)))),
            "use_v3": entry.get("use_v3", False),
        })
    # Add fixed bindings
    for key, room_id in _fixed_sessions.items():
        parts = key.split(":", 1)
        bearer = parts[0] if len(parts) > 0 else ""
        model_id = parts[1] if len(parts) > 1 else ""
        result.append({
            "cache_key": key,
            "api_key_preview": (bearer[:12] + "...") if len(bearer) > 12 else bearer,
            "model_id": model_id,
            "session_id": room_id,
            "created_at": 0,
            "last_used": 0,
            "request_count": 0,
            "ttl_remaining": -1,  # -1 = 永久
            "use_v3": False,
            "fixed": True,
        })
    return sorted(result, key=lambda x: x["last_used"], reverse=True)


def delete_session(cache_key: str) -> bool:
    """Delete a session by cache_key. Returns True if found."""
    if cache_key in _session_cache:
        del _session_cache[cache_key]
        return True
    return False


def clear_all_sessions():
    _session_cache.clear()


def _session_key(bearer: str, model_id: str) -> str:
    return f"{bearer}:{model_id}"


def _get_cached_session(bearer: str, model_id: str) -> Optional[str]:
    """Return cached session_id if valid, else None.
    Priority: fixed binding > auto-cached session."""
    sk = _session_key(bearer, model_id)
    
    # 1) Fixed binding (manually set, never expires)
    if sk in _fixed_sessions:
        return _fixed_sessions[sk]
    
    # 2) Auto-cached session (TTL-based)
    if not SESSION_ENABLED:
        return None
    entry = _session_cache.get(sk)
    if entry and time.time() - entry["created_at"] < SESSION_TTL:
        entry["last_used"] = time.time()
        entry["request_count"] = entry.get("request_count", 0) + 1
        return entry["session_id"]
    return None


def set_fixed_session(bearer: str, model_id: str, room_id: str):
    """Manually bind a Tabbit room to an API Key + model."""
    _fixed_sessions[_session_key(bearer, model_id)] = room_id


def remove_fixed_session(bearer: str, model_id: str) -> bool:
    """Remove a fixed binding. Returns True if found."""
    sk = _session_key(bearer, model_id)
    if sk in _fixed_sessions:
        del _fixed_sessions[sk]
        return True
    return False


def get_fixed_sessions() -> dict:
    """Return all fixed bindings."""
    return dict(_fixed_sessions)


def get_session_list() -> list:
    """Return all active sessions for admin display."""
    result = []
    now = time.time()
    for key, entry in _session_cache.items():
        parts = key.split(":", 1)
        bearer = parts[0] if len(parts) > 0 else ""
        model_id = parts[1] if len(parts) > 1 else ""
        result.append({
            "cache_key": key,
            "api_key_preview": (bearer[:12] + "...") if len(bearer) > 12 else bearer,
            "model_id": model_id,
            "session_id": entry.get("session_id", ""),
            "created_at": entry.get("created_at", 0),
            "last_used": entry.get("last_used", entry.get("created_at", 0)),
            "request_count": entry.get("request_count", 0),
            "ttl_remaining": max(0, int(SESSION_TTL - (now - entry.get("created_at", 0)))),
            "use_v3": entry.get("use_v3", False),
            "fixed": False,
        })
    # Add fixed bindings
    for key, room_id in _fixed_sessions.items():
        parts = key.split(":", 1)
        bearer = parts[0] if len(parts) > 0 else ""
        model_id = parts[1] if len(parts) > 1 else ""
        result.append({
            "cache_key": key,
            "api_key_preview": (bearer[:12] + "...") if len(bearer) > 12 else bearer,
            "model_id": model_id,
            "session_id": room_id,
            "created_at": 0,
            "last_used": 0,
            "request_count": 0,
            "ttl_remaining": -1,
            "use_v3": False,
            "fixed": True,
        })
    return sorted(result, key=lambda x: x["last_used"], reverse=True)


def _cache_session(bearer: str, model_id: str, session_id: str, use_v3: bool):
    _session_cache[_session_key(bearer, model_id)] = {
        "session_id": session_id,
        "created_at": time.time(),
        "last_used": time.time(),
        "request_count": 0,
        "use_v3": use_v3,
    }


def _invalidate_session(bearer: str, model_id: str):
    _session_cache.pop(_session_key(bearer, model_id), None)


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """Get or create a per-session async lock to serialize runs."""
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


def init(token_manager: TokenManager, config: ConfigManager, log_store: LogStore):
    global _tm, _cfg, _logs
    _tm = token_manager
    _cfg = config
    _logs = log_store


import re

def _clean_text(text: str) -> str:
    if not text:
        return ""
    return str(text).strip()

def _normalize_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    result.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    result.append(f"[Image: {item.get('image_url', {}).get('url', '')}]")
                elif item.get("type") == "tool_result":
                    result.append(str(item.get("content", "")))
                else:
                    result.append(str(item.get("content", "")))
            else:
                result.append(str(item))
        return "\n".join(result).strip()
    return str(content).strip()


class ChatMessageContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class ChatMessage(BaseModel):
    role: str
    content: Optional[Any] = ""
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None


class ToolChoice(BaseModel):
    type: str = "function"
    function: Optional[dict] = None


class Tool(BaseModel):
    type: str = "function"
    function: dict


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[Any] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None


class SimpleChatRequest(BaseModel):
    role: str = "user"
    content: Any


MAX_CONTENT_LENGTH = 8000
MAX_MESSAGES = 6

def _build_content(messages: List[ChatMessage]) -> str:
    system_prompt = _cfg.get("proxy", "system_prompt") if _cfg else ""
    
    recent_messages = messages[-MAX_MESSAGES:]
    
    parts = []
    if system_prompt:
        parts.append(f"[System]: {system_prompt}")
    
    for m in recent_messages:
        label = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            m.role, m.role.capitalize()
        )
        parts.append(f"[{label}]: {_normalize_content(m.content)}")
    
    full_content = "\n\n".join(parts) + "\n\n[Assistant]:"
    
    if len(full_content) > MAX_CONTENT_LENGTH:
        logger.warning(f"Content too long ({len(full_content)} chars), truncating to {MAX_CONTENT_LENGTH}")
        # Keep system prompt + last N messages, drop middle ones
        keep_last = min(len(recent_messages), 3)  # keep at most last 3 messages
        kept_messages = recent_messages[-keep_last:]
        parts = []
        if system_prompt:
            parts.append(f"[System]: {system_prompt}")
        for m in kept_messages:
            label = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
                m.role, m.role.capitalize()
            )
            content_str = _normalize_content(m.content)
            # Truncate each message to avoid single message dominating
            max_per_msg = MAX_CONTENT_LENGTH // max(keep_last, 1)
            if len(content_str) > max_per_msg:
                content_str = content_str[:max_per_msg] + "..."
            parts.append(f"[{label}]: {content_str}")
        full_content = "\n\n".join(parts) + "\n\n[Assistant]:"
        if len(full_content) > MAX_CONTENT_LENGTH:
            full_content = full_content[:MAX_CONTENT_LENGTH]
    
    return full_content


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    token = authorization.strip()
    for prefix in ["Bearer ", "Bearer=", "bearer ", "bearer=", "Bearar ", "bearar="]:
        if token.startswith(prefix):
            return token[len(prefix):]
    return token


async def _get_client_and_token(authorization: str | None) -> tuple[TabbitClient, str, str]:
    if _tm.has_tokens:
        api_key = _cfg.get("proxy", "api_key")
        if api_key:
            bearer = _extract_bearer_token(authorization)
            if bearer != api_key:
                raise HTTPException(status_code=401, detail="invalid api key")
        token_info, client = await _tm.get_next()
        if token_info is None:
            raise HTTPException(status_code=503, detail="no available tokens (all cooling down)")
        return client, token_info.get("name", "unknown"), token_info["id"]

    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="missing token")
    if token not in _fallback_clients:
        _fallback_clients[token] = TabbitClient(
            token,
            _cfg.get("tabbit", "base_url"),
            _cfg.get("tabbit", "client_id"),
        )
    return _fallback_clients[token], "bearer", ""


def _get_model_access_type(model_id: str) -> str:
    """Return the access_type for a given model_id."""
    info = MODEL_INFO.get(model_id, {})
    return info.get("access_type", "free_metered")


def _is_premium_model(model_id: str) -> bool:
    return _get_model_access_type(model_id) == "premium_only"


async def _stream_handler(
    client, session_id, content, tabbit_model, req_model,
    completion_id, token_name, token_id, use_v3=False, has_tools=False
):
    start = time.time()
    error_msg = ""
    buffered_text = ""
    streamed_text = ""
    chunk_count = 0
    in_tool_call_mode = False
    try:
        # 1. 始终第一时间发送首包（带有 role: assistant），让客户端建立状态机
        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"

        async def _iterate_events():
            if use_v3:
                async for event in client.send_message_v3(session_id, content, tabbit_model):
                    yield event
            else:
                async for event in client.send_message(session_id, content, tabbit_model):
                    yield event

        async for event in _iterate_events():
            et, ed = event["event"], event["data"]
            if et == "error":
                error_msg = ed.get("message", str(ed))
                logger.warning(f"[STREAM] Tabbit error event: {error_msg}")
                err_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": f"⚠️ Tabbit API 提示: {error_msg}"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
                return

            if et == "message_chunk":
                chunk_count += 1
                # 2. 支持 reasoning_content（思考过程），实时推流给客户端
                if "reasoning_content" in ed and ed["reasoning_content"]:
                    r_text = ed["reasoning_content"]
                    r_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"reasoning_content": r_text}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(r_chunk, ensure_ascii=False)}\n\n"

                # 3. 处理普通正文 content
                if "content" in ed and ed["content"]:
                    chunk_piece = ed["content"]
                    buffered_text += chunk_piece

                    if not has_tools:
                        # 普通无 tools 对话，直接纯实时流式输出
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": chunk_piece}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    else:
                        # 有 tools 对话：检查是否出现了 <tool_call
                        if "<tool_call" in buffered_text:
                            in_tool_call_mode = True

                        if not in_tool_call_mode:
                            # 尚未出现工具调用标记，安全流式发送给客户端，保持打字机效果
                            if "<" in buffered_text:
                                safe_len = buffered_text.rfind("<")
                                to_send = buffered_text[len(streamed_text):safe_len]
                            else:
                                to_send = buffered_text[len(streamed_text):]

                            if to_send:
                                streamed_text += to_send
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "choices": [{"index": 0, "delta": {"content": to_send}, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        logger.info(f"[STREAM] Event loop finished: chunks={chunk_count}, buffered_len={len(buffered_text)}, has_tools={has_tools}")

        if has_tools:
            clean_text, tool_calls = detect_all_tool_calls(buffered_text)
            if tool_calls:
                # 提取出工具调用，输出标准的 tool_calls delta 和 finish_reason: "tool_calls"
                chunks = build_openai_tool_stream_chunks(
                    completion_id=completion_id,
                    model=req_model,
                    tool_calls=tool_calls,
                    content="" if streamed_text else clean_text,
                )
                for c in chunks:
                    yield c
                return
            else:
                # 纯文本对话（没有调用工具），补齐尚未发送的文本
                remaining = buffered_text[len(streamed_text):]
                if remaining:
                    chunk_text = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": remaining}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk_text, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        else:
            yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"

        yield "data: [DONE]\n\n"
        if token_id:
            _tm.report_success(token_id)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[STREAM] Exception in stream_handler: {e}", exc_info=True)
        if token_id:
            _tm.report_error(token_id)
        raise
    finally:
        duration = time.time() - start
        if _logs:
            _logs.add(LogEntry(
                model=req_model, token_name=token_name, stream=True,
                status="success" if not error_msg else "error",
                duration=duration, error=error_msg
            ))


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest | SimpleChatRequest, authorization: str = Header(None)
):
    try:
        client, token_name, token_id = await _get_client_and_token(authorization)
    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    
    try:
        if isinstance(req, SimpleChatRequest):
            model_id = "best"
            tabbit_model = "最佳"
            content = _normalize_content(req.content)
            has_tools = False
            is_agent_request = False
        else:
            model_id = (req.model or _cfg.get("openai", "default_model", default="best")).lower() if _cfg else (req.model or "best").lower()
            
            # Agent router: detect phase and potentially remap model
            agent_config = _cfg.get("agent", default={}) if _cfg else {}
            if agent_config.get("router", {}).get("enabled", False):
                selector = ModelSelector()
                model_id = selector.resolve(model_id)
            
            tabbit_model = tabbit_client.MODEL_MAP.get(model_id, model_id)
            
            has_tools = bool(req.tools)
            is_agent_request = has_tools or any(
                getattr(m, "role", "") in ("tool", "system") for m in req.messages
            ) or len(req.messages) > 2

            # Tool calling: build compact tool prompt if tools are present
            tool_prompt = ""
            if has_tools:
                tools_parsed = parse_openai_tools(req.tools)
                tool_prompt = build_tool_prompt(tools_parsed)

            # Control total length safely within Tabbit 7500 char threshold
            remaining_budget = max(3000, 7200 - len(tool_prompt))
            formatted_content = format_chat_messages(req.messages, max_budget=remaining_budget)

            if tool_prompt:
                content = tool_prompt + "\n\n" + formatted_content
            else:
                content = formatted_content
            
            logger.info(f"[DEBUG] Final content ({len(content)} chars): {content[:300]}")
    except Exception as e:
        logger.error(f"Error preparing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid request format")
    
    use_v3 = _is_premium_model(model_id)
    is_stream = getattr(req, "stream", False)
    
    # Session handling:
    # For agent requests (tools / multi-turn tool chains), use an isolated session to prevent
    # concurrency lock bottlenecks and Tabbit 409 Conflict.
    bearer = _extract_bearer_token(authorization) if authorization else ""
    cached_session = None
    if not is_agent_request:
        cached_session = _get_cached_session(bearer, model_id)

    if cached_session:
        session_id = cached_session
        logger.info(f"Reusing cached session {session_id} for model {model_id}")
    else:
        try:
            session_id = await client.create_chat_session()
            if not is_agent_request:
                _cache_session(bearer, model_id, session_id, use_v3)
            logger.info(f"Created new session {session_id} for model {model_id} (agent={is_agent_request})")
        except Exception as e:
            if token_id:
                _tm.report_error(token_id)
            if _logs:
                _logs.add(LogEntry(
                    model=getattr(req, "model", "unknown"), token_name=token_name,
                    stream=is_stream, status="error", error=str(e)
                ))
            raise HTTPException(status_code=502, detail=f"Failed to create chat session: {e}")
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    if is_stream:
        if is_agent_request:
            # Isolated session, stream directly without global lock contention
            return StreamingResponse(
                _stream_handler(
                    client, session_id, content, tabbit_model, getattr(req, "model", "unknown"),
                    completion_id, token_name, token_id, use_v3=use_v3, has_tools=has_tools
                ),
                media_type="text/event-stream"
            )
        else:
            lock = _get_session_lock(session_id)
            async def locked_stream():
                async with lock:
                    async for chunk in _stream_handler(
                        client, session_id, content, tabbit_model, getattr(req, "model", "unknown"),
                        completion_id, token_name, token_id, use_v3=use_v3, has_tools=has_tools
                    ):
                        yield chunk
            return StreamingResponse(locked_stream(), media_type="text/event-stream")
    
    start = time.time()
    full_text = ""
    error_msg = ""
    
    async def _do_chat(sid):
        nonlocal full_text, error_msg
        if use_v3:
            async for event in client.send_message_v3(sid, content, tabbit_model):
                et, ed = event["event"], event["data"]
                if et == "error":
                    error_msg = ed.get("message", str(ed))
                    logger.warning(f"[DEBUG] V3 error: {error_msg}")
                    break
                if et == "message_chunk" and "content" in ed:
                    full_text += ed["content"]
        else:
            async for event in client.send_message(sid, content, tabbit_model):
                et = event["event"]
                ed = event["data"]
                if et == "message_chunk":
                    full_text += ed.get("content", "")
                elif et == "error":
                    error_msg = ed.get("message", str(ed))
                    logger.warning(f"[DEBUG] V1 error: code={ed.get('code')} msg={error_msg}")
                elif et not in ("message_chunk",):
                    logger.info(f"[DEBUG] V1 event: {et} data_keys={list(ed.keys()) if isinstance(ed, dict) else str(ed)[:100]}")
        logger.info(f"[DEBUG] _do_chat result: full_text_len={len(full_text)} error_msg={error_msg[:200] if error_msg else 'None'}")
    
    try:
        if is_agent_request:
            await _do_chat(session_id)
        else:
            lock = _get_session_lock(session_id)
            async with lock:
                await _do_chat(session_id)
            
            # Auto-retry: if session failed (e.g. expired), create new one
            if error_msg and cached_session:
                logger.warning(f"Session {session_id} failed: {error_msg}, retrying with new session")
                _invalidate_session(bearer, model_id)
                new_sid = await client.create_chat_session()
                _cache_session(bearer, model_id, new_sid, use_v3)
                full_text = ""
                error_msg = ""
                lock2 = _get_session_lock(new_sid)
                async with lock2:
                    await _do_chat(new_sid)
        
        if token_id and not error_msg:
            _tm.report_success(token_id)
    except Exception as e:
        error_msg = str(e)
        if token_id:
            _tm.report_error(token_id)
        if _logs:
            _logs.add(LogEntry(
                model=getattr(req, "model", "unknown"), token_name=token_name,
                stream=False, status="error", error=error_msg
            ))
        raise HTTPException(status_code=502, detail=str(e))
    
    duration = time.time() - start
    if _logs:
        _logs.add(LogEntry(
            model=getattr(req, "model", "unknown"), token_name=token_name,
            stream=False, status="success" if not error_msg else "error", duration=duration, error=error_msg
        ))

    input_tokens = len(content) // 4
    output_tokens = len(full_text) // 4

    # Check for tool calls in non-streaming mode
    if has_tools or "<tool_call>" in full_text:
        clean_text, tool_calls = detect_all_tool_calls(full_text)
        if tool_calls:
            return build_openai_tool_completion(
                completion_id=completion_id,
                model=getattr(req, "model", "unknown"),
                tool_calls=tool_calls,
                content=clean_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": getattr(req, "model", "unknown"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": full_text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }
    }


MODEL_INFO = {
    "best": {
        "id": "best",
        "name": "最佳",
        "description": "默认模式，该模式下不消耗模型用量，使用量无上限",
        "max_tokens": 100000,
        "supports_streaming": True,
        "supports_vision": False,
        "access_type": "free_unlimited",
    },
    "kimi-k3": {
        "id": "kimi-k3",
        "name": "Kimi-K3",
        "description": "Kimi 迄今能力最强的旗舰模型，支持 1M token 上下文与视觉理解，适合软件工程、知识工作和深度推理",
        "max_tokens": 1000000,
        "supports_streaming": True,
        "supports_vision": False,
        "access_type": "premium_only",
    },
    "longcat-2-0": {
        "id": "longcat-2-0",
        "name": "LongCat-2.0",
        "description": "美团最新的自研旗舰文本模型，增强了 Agentic 能力，原生支持 1M 上下文",
        "max_tokens": 1000000,
        "supports_streaming": True,
        "supports_vision": False,
        "access_type": "free_metered",
    },
    "glm-5-2": {
        "id": "glm-5-2",
        "name": "GLM-5.2",
        "description": "智谱最新的文本模型，深度优化了长程任务能力",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": False,
        "access_type": "free_metered",
    },
    "qwen3-7-max": {
        "id": "qwen3-7-max",
        "name": "Qwen3.7-Max",
        "description": "阿里千问的旗舰级文本模型，大幅提升了在编程与通用型智能体上的表现，适合复杂任务",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": False,
        "access_type": "free_metered",
    },
    "kimi-k2-7-code": {
        "id": "kimi-k2-7-code",
        "name": "Kimi-K2.7-Code",
        "description": "月之暗面最新的旗舰多模态 Coding 模型，适合复杂编程任务",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "deepseek-v4-pro": {
        "id": "deepseek-v4-pro",
        "name": "DeepSeek-V4-Pro",
        "description": "DeepSeek 的全新旗舰模型，Pro版",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "deepseek-v4.1-flash": {
        "id": "deepseek-v4.1-flash",
        "name": "DeepSeek-V4.1-Flash",
        "description": "DeepSeek 全新架构系列最小模型，具备原生视觉理解、更快推理、更高吞吐与 1M 上下文",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "deepseek-v4-1-flash": {
        "id": "deepseek-v4-1-flash",
        "name": "DeepSeek-V4.1-Flash",
        "description": "DeepSeek 全新架构系列最小模型，具备原生视觉理解、更快推理、更高吞吐与 1M 上下文",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "deepseek-v4-flash": {
        "id": "deepseek-v4-flash",
        "name": "DeepSeek-V4-Flash",
        "description": "DeepSeek 的全新旗舰模型，Flash版",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "doubao-seed-2-1-pro": {
        "id": "doubao-seed-2-1-pro",
        "name": "Doubao-Seed-2.1-Pro",
        "description": "字节跳动最新的旗舰多模态模型，Pro 版，适合复杂任务",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "doubao-seed-2-1-turbo": {
        "id": "doubao-seed-2-1-turbo",
        "name": "Doubao-Seed-2.1-Turbo",
        "description": "字节跳动最新的旗舰多模态模型，Turbo 版，适合日常使用",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "minimax-m3": {
        "id": "minimax-m3",
        "name": "MiniMax-M3",
        "description": "MiniMax 最新的旗舰级原生多模态模型",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "glm-5-1": {
        "id": "glm-5-1",
        "name": "GLM-5.1",
        "description": "智谱的文本模型，深度优化了长程任务能力",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": False,
        "access_type": "free_metered",
    },
    "glm-5v-turbo": {
        "id": "glm-5v-turbo",
        "name": "GLM-5V-Turbo",
        "description": "智谱最新的多模态模型，深度优化了图像理解能力",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "kimi-k2-6": {
        "id": "kimi-k2-6",
        "name": "Kimi-K2.6",
        "description": "Moonshot 最新的旗舰级多模态模型，适合大部分任务",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "kimi-k2-5": {
        "id": "kimi-k2-5",
        "name": "Kimi-K2.5",
        "description": "Moonshot 的旗舰级多模态模型，适合大部分任务",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "minimax-m2-7": {
        "id": "minimax-m2-7",
        "name": "MiniMax-M2.7",
        "description": "MiniMax 最新的旗舰级文本模型",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": False,
        "access_type": "free_metered",
    },
    "doubao-seed-2-0-lite": {
        "id": "doubao-seed-2-0-lite",
        "name": "Doubao-Seed-2.0-lite",
        "description": "字节跳动豆包大模型系列的最新多模态模型，适合日常使用",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "qwen3-5-plus": {
        "id": "qwen3-5-plus",
        "name": "Qwen3.5-Plus",
        "description": "阿里千问首个原生多模态大模型，整合语言推理与视觉感知，适合大多数任务",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": True,
        "access_type": "free_metered",
    },
    "longcat-flash-chat": {
        "id": "longcat-flash-chat",
        "name": "LongCat-Flash-Chat",
        "description": "美团的上一代自研旗舰模型，推理速度快、推理效果优",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": False,
        "access_type": "free_metered",
    },
    "longcat-flash-thinking": {
        "id": "longcat-flash-thinking",
        "name": "LongCat-Flash-Thinking",
        "description": "美团的上一代自研旗舰思考模型，性能更强、泛化效果更优",
        "max_tokens": 128000,
        "supports_streaming": True,
        "supports_vision": False,
        "access_type": "free_metered",
    },
}


@router.get("/v1/models")
async def list_models():
    models = []
    for model_id, model_name in tabbit_client.MODEL_MAP.items():
        info = MODEL_INFO.get(model_id, {})
        models.append({
            "id": model_id,
            "object": "model",
            "created": 1714502400,
            "owned_by": "tabbit",
            "name": model_name,
            "description": info.get("description", f"Tabbit model: {model_name}"),
            "max_tokens": info.get("max_tokens", 128000),
            "supports_streaming": info.get("supports_streaming", True),
            "supports_vision": info.get("supports_vision", True),
            "access_type": info.get("access_type", "free_metered"),
        })
    return {
        "object": "list",
        "data": models
    }


@router.get("/models")
async def list_models_v0():
    return await list_models()


@router.post("/chat/completions")
async def chat_completions_v0(
    req: ChatCompletionRequest | SimpleChatRequest, authorization: str = Header(None)
):
    return await chat_completions(req, authorization)