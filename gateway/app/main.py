"""
FastAPI Gateway for vLLM with Conversation Memory and RAG.

This gateway provides:
- Session-based conversation management
- Conversation history stored in DynamoDB
- Document-based RAG (Retrieval-Augmented Generation) knowledge base
- OpenAI-compatible API endpoints
- Automatic token management and history truncation
"""
import asyncio
import json
import logging
import secrets
import time
import uuid
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_settings, Settings
from .session import get_session_manager, SessionManager
from .rag import (
    get_opensearch_rag,
    get_embedding_model,
    get_reranker_model,
    embed_query,
    rewrite_query,
    is_chitchat,
    build_rag_context,
    build_base_system_prompt,
    build_rag_system_prompt,
    QUERY_REWRITE_HISTORY_TURNS,
)
from .documents import (
    get_document_manager,
    process_document_background,
)
from .guard import canned_refusal, classify_user_intent
from .models import (
    SessionCreate,
    SessionResponse,
    SessionHistory,
    SessionDeleteResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelsListResponse,
    HealthResponse,
    ErrorResponse,
    Message,
    DocumentUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentDeleteResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Constant slack budget for the system-prompt template wrapper itself
# (persona + scope rules in build_base_system_prompt + "--- Document Context ---"
# markers in build_rag_system_prompt). Subtracted before allocating RAG body
# tokens so we don't accidentally over-budget RAG and exceed the model's
# context ceiling. Only matters if the prompt template grows substantially --
# bump it to compensate.
SYSTEM_PROMPT_OVERHEAD_TOKENS = 500

# Floor on tokens reserved for the model's reply. The per-request
# max_tokens (default 300) wins when larger; this just guarantees we
# never starve the completion below this many tokens regardless of
# history/RAG pressure. Pure floor -- no real "tune for prod" use case.
RESERVED_COMPLETION_TOKENS = 512

# Hard ceiling on the RAG body tokens. Defensive cap only -- the actual
# budgeter already shrinks RAG to `available // 2` so it can't starve
# history. This ceiling kicks in for unusually long context windows
# (e.g. 32K+ models) to keep the prompt cost from ballooning.
RAG_MAX_CONTEXT_TOKENS = 2500


# HTTP client for vLLM backend
http_client: Optional[httpx.AsyncClient] = None
# Separate client for OpenAI (input guard). No vLLM base_url.
openai_http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global http_client, openai_http_client
    
    settings = get_settings()
    if not settings.api_key or not settings.api_key.strip():
        raise RuntimeError(
            "VLLM_API_KEY is empty. Refusing to start with auth disabled. "
            "Set VLLM_API_KEY in the gateway environment or Kubernetes secret."
        )
    logger.info("Starting API Gateway...")
    logger.info(f"vLLM Backend URL: {settings.vllm_internal_url}")
    logger.info(f"DynamoDB Table: {settings.dynamodb_table}")
    logger.info(f"RAG Enabled: {settings.rag_enabled}")
    logger.info(
        f"Input guard: enabled={settings.input_guard_enabled} "
        f"key_configured={bool(settings.openai_api_key)}"
    )
    
    # Initialize HTTP client
    http_client = httpx.AsyncClient(
        base_url=settings.vllm_internal_url,
        timeout=httpx.Timeout(300.0, connect=10.0)  # 5 min timeout for LLM responses
    )
    openai_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(5.0, connect=2.0)
    )
    
    # Initialize session manager (validates DynamoDB connection)
    try:
        get_session_manager()
        logger.info("DynamoDB connection established")
    except Exception as e:
        logger.error(f"Failed to connect to DynamoDB: {e}")
    
    # Initialize RAG components if enabled
    if settings.rag_enabled and settings.opensearch_endpoint:
        try:
            # Initialize OpenSearch client and create index
            opensearch_rag = get_opensearch_rag()
            opensearch_rag.create_index_if_not_exists()
            logger.info("OpenSearch RAG initialized")
            
            # Pre-load embedding model
            logger.info("Pre-loading embedding model...")
            get_embedding_model()
            logger.info("Embedding model ready")
            
            # Pre-load reranker (cross-encoder)
            logger.info("Pre-loading reranker model...")
            get_reranker_model()
            logger.info("Reranker model ready")
            
            # Initialize document manager
            get_document_manager()
            logger.info("Document manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            logger.warning("RAG features will be unavailable")
    else:
        logger.info("RAG is disabled or OpenSearch endpoint not configured")
    
    yield
    
    # Cleanup
    if http_client:
        await http_client.aclose()
    if openai_http_client:
        await openai_http_client.aclose()
    logger.info("API Gateway shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="vLLM Gateway API",
    description="API Gateway for vLLM with conversation memory and RAG knowledge base",
    version="2.0.0",
    lifespan=lifespan
)


# =============================================================================
# Token Budget Helpers
# =============================================================================

def _merge_system_content(
    rag_system_prompt: Optional[str],
    extra_contents: list,
) -> Optional[str]:
    """
    Normalize the server-controlled system prompt for vLLM.

    Client-provided system messages are rejected at the session API boundary
    and ignored if they appear in stored history. ``extra_contents`` is kept
    for unit tests; production always passes an empty list.
    """
    extras = [c for c in extra_contents if c and c.strip()]
    rag = rag_system_prompt.strip() if rag_system_prompt else ""

    if not rag and not extras:
        return None
    if not extras:
        return rag
    if not rag:
        return "\n\n".join(extras)
    return rag + "\n\nAdditional instructions:\n" + "\n\n".join(extras)


def _assemble_messages_for_budget(
    history_messages,
    new_request_messages: list,
    rag_system_prompt: Optional[str],
    rag_tokens_used: int,
    request_max_tokens: Optional[int],
    settings: Settings,
) -> list:
    """
    Build the final list of ChatMessages we send to vLLM, trimming history
    to fit a single coordinated token budget.

    Emits at most one role="system" message at position 0, containing only
    the server-injected RAG/base prompt. Any system-role messages in stored
    history are dropped (defense in depth; clients cannot send system msgs).

    Budget formula:
        completion_reserve = max(request_max_tokens or 300, RESERVED_COMPLETION_TOKENS)
        new_user_tokens    = est(non-system new request messages)
        system_tokens      = est(merged system content) if any, else 0
        history_budget     = settings.model_context_tokens
                             - completion_reserve
                             - new_user_tokens
                             - system_tokens

    Then non-system history is trimmed oldest-first until it fits the
    budget (with settings.max_history_tokens as a hard upper bound).
    The new user turn and the merged system prompt are always included;
    only history can shrink.

    A single info log line per turn shows the resulting allocation so
    the budget can be verified from production logs.
    """
    cpt = settings.chars_per_token

    # 1. Keep only user/assistant turns from history and the new request.
    # System-role content is never merged from clients or history.
    dropped_history_system = 0
    non_system_history = []
    for m in history_messages:
        if m.role == "system":
            dropped_history_system += 1
        else:
            non_system_history.append(m)
    if dropped_history_system:
        logger.warning(
            f"Ignored {dropped_history_system} system message(s) in session history"
        )

    non_system_new = [
        m for m in new_request_messages if m.role != "system"
    ]

    # 2. Server-controlled system prompt only.
    merged_system = _merge_system_content(rag_system_prompt, [])

    # 3. Compute budget.
    completion_reserve = max(
        request_max_tokens if request_max_tokens is not None else 300,
        RESERVED_COMPLETION_TOKENS,
    )
    new_user_tokens = sum(len(m.content) // cpt for m in non_system_new)
    system_total_tokens = (len(merged_system) // cpt) if merged_system else 0

    history_budget = (
        settings.model_context_tokens
        - completion_reserve
        - new_user_tokens
        - system_total_tokens
    )
    # MAX_HISTORY_TOKENS is a hard upper bound independent of the per-turn budget.
    effective_history_budget = max(0, min(history_budget, settings.max_history_tokens))

    # 4. Trim non-system history (oldest first) until it fits the budget.
    trimmed = list(non_system_history)
    history_tokens = sum(len(m.content) // cpt for m in trimmed)
    while trimmed and history_tokens > effective_history_budget:
        removed = trimmed.pop(0)
        history_tokens -= len(removed.content) // cpt

    # 5. Assemble final list: at most one system message at index 0.
    full: list = []
    if merged_system:
        full.append(ChatMessage(role="system", content=merged_system))
    full.extend(
        ChatMessage(role=msg.role, content=msg.content) for msg in trimmed
    )
    full.extend(non_system_new)

    logger.info(
        f"Token budget: ceiling={settings.model_context_tokens} "
        f"completion={completion_reserve} new_user={new_user_tokens} "
        f"system={system_total_tokens} (rag_body={rag_tokens_used}) "
        f"history={history_tokens}/{effective_history_budget} "
        f"(kept {len(trimmed)}/{len(non_system_history)} msgs)"
    )

    return full


# =============================================================================
# SSE Streaming Helpers
# =============================================================================

# Headers we attach to every SSE response. `X-Accel-Buffering: no` defangs
# nginx / CloudFront layers that otherwise buffer text/event-stream responses
# until full and defeat the entire point of streaming.
_SSE_HEADERS: Dict[str, str] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

# Marker appended to the persisted assistant content when a stream is cut
# short. Belt-and-suspenders with metadata["interrupted"]=True: even a UI
# that doesn't honor metadata renders this inline so the user can see the
# response was truncated.
_INTERRUPTED_MARKER = "\n\n_[response interrupted]_"


def _canned_refusal_response(model: str, topic: str) -> Dict[str, Any]:
    """OpenAI-compatible JSON body for a scope refusal (no vLLM call)."""
    return {
        "id": f"chatcmpl-guard-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": canned_refusal(topic)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _persist_canned_refusal(
    session_manager: SessionManager,
    session_id: str,
    new_user_messages: List[ChatMessage],
    topic: str,
) -> None:
    for msg in new_user_messages:
        session_manager.add_message(session_id, msg.role, msg.content)
    session_manager.add_message(
        session_id,
        "assistant",
        canned_refusal(topic),
        metadata={"finish_reason": "stop", "input_guard": "refuse"},
    )
    session_manager.refresh_session_ttl(session_id)


async def _stream_canned_refusal(model: str, topic: str) -> AsyncGenerator[bytes, None]:
    """Minimal SSE so stream=true clients still render the canned refusal."""
    chunk_id = f"chatcmpl-guard-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    first = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": canned_refusal(topic)},
                "finish_reason": None,
            }
        ],
    }
    last = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(first)}\n\n".encode("utf-8")
    yield f"data: {json.dumps(last)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


async def _stream_vllm_completion(
    vllm_request: Dict[str, Any],
    headers: Dict[str, str],
    state: Dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    """
    Async generator: open an SSE stream to vLLM, yield each line back
    byte-for-byte as `data: ...\\n\\n` frames, and side-effect `state`
    with the accumulated assistant content for downstream persistence.

    Why a mutable `state` holder instead of a return value? An async
    generator's return value isn't reachable from the caller of the
    StreamingResponse it feeds, so we use a shared dict that the outer
    wrapper inspects in its `finally` block.

    `state` is mutated with:
        buffer:         {choice_index: accumulated_str}
        finish_reasons: {choice_index: "stop" | "length" | ...}
        last_usage:     final usage dict if vLLM sends one
        saw_done:       True iff the `[DONE]` sentinel was received
        error:          (status, body) or ("http_error", msg) on failure

    The chunks we yield are bytes already SSE-framed. We forward exactly
    what vLLM sends (no re-serialization) so future fields like
    `logprobs`, `tool_calls`, or `usage` pass through automatically.
    """
    try:
        async with http_client.stream(
            "POST",
            "/v1/chat/completions",
            json=vllm_request,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                logger.error(
                    f"vLLM stream returned {response.status_code}: {body[:500]!r}"
                )
                state["error"] = (response.status_code, body)
                err_payload = json.dumps({
                    "error": {
                        "status": response.status_code,
                        "message": body.decode("utf-8", errors="replace")[:500],
                    }
                })
                yield f"data: {err_payload}\n\n".encode("utf-8")
                return

            async for line in response.aiter_lines():
                # Skip blank lines; we re-add the SSE frame separator
                # ourselves below so each event is a single canonical
                # `data: ...\n\n` frame.
                if not line:
                    continue

                yield (line + "\n\n").encode("utf-8")

                if not line.startswith("data:"):
                    continue

                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    state["saw_done"] = True
                    continue

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    # Not JSON; we already forwarded it, just don't parse.
                    continue

                for choice in (chunk.get("choices") or []):
                    idx = choice.get("index", 0)
                    delta = choice.get("delta") or {}
                    delta_content = delta.get("content")
                    if delta_content:
                        state["buffer"][idx] = (
                            state["buffer"].get(idx, "") + delta_content
                        )
                    finish = choice.get("finish_reason")
                    if finish:
                        state["finish_reasons"][idx] = finish

                usage = chunk.get("usage")
                if usage:
                    state["last_usage"] = usage

    except httpx.HTTPError as e:
        logger.error(f"Stream HTTP error: {e}")
        state["error"] = ("http_error", str(e))
        err = json.dumps(
            {"error": {"message": "vLLM backend error", "detail": str(e)}}
        )
        yield f"data: {err}\n\n".encode("utf-8")


async def _stream_and_persist(
    vllm_request: Dict[str, Any],
    headers: Dict[str, str],
    session_manager: SessionManager,
    session_id: str,
    new_user_messages: List["ChatMessage"],
) -> AsyncGenerator[bytes, None]:
    """
    Outer SSE generator for the session-aware path.

    Forwards every chunk from vLLM to the client, accumulates assistant
    content in a buffer, and on stream completion (clean OR interrupted)
    persists the user message + assistant message to DynamoDB.

    Persistence rules (matching the agreed disconnect contract):
      - Only persist if we got at least one assistant token. A 0-byte
        failure leaves no dangling user turn (parity with the
        non-streaming all-or-nothing semantic).
      - If the stream was interrupted (client disconnect, vLLM error, or
        missing `[DONE]`), the persisted content gets `_INTERRUPTED_MARKER`
        appended AND `metadata["interrupted"]=True` so both smart and
        dumb client UIs render the truncation cue.
      - The `finally` clause runs on disconnect (FastAPI raises
        CancelledError into the generator), so persistence is robust.
    """
    state: Dict[str, Any] = {
        "buffer": {},
        "finish_reasons": {},
        "last_usage": None,
        "saw_done": False,
        "error": None,
    }
    cancelled = False
    try:
        async for chunk in _stream_vllm_completion(vllm_request, headers, state):
            yield chunk
    except asyncio.CancelledError:
        cancelled = True
        # Propagate so the runtime knows the request was cancelled, but
        # the finally below still runs first and persists what we have.
        raise
    finally:
        interrupted = (
            cancelled
            or state["error"] is not None
            or not state["saw_done"]
        )
        assistant_content = state["buffer"].get(0, "")

        if assistant_content:
            persisted_content = (
                assistant_content + _INTERRUPTED_MARKER
                if interrupted
                else assistant_content
            )

            try:
                for msg in new_user_messages:
                    session_manager.add_message(
                        session_id, msg.role, msg.content
                    )

                finish_reason = state["finish_reasons"].get(
                    0, "interrupted" if interrupted else "stop"
                )
                session_manager.add_message(
                    session_id,
                    "assistant",
                    persisted_content,
                    metadata={
                        "tokens": state["last_usage"] or {},
                        "finish_reason": finish_reason,
                        "interrupted": interrupted,
                        "streamed": True,
                    },
                )
                session_manager.refresh_session_ttl(session_id)
            except Exception as e:
                # Persistence failure is loud but non-fatal: the client
                # has already received the stream, we just lost the
                # write. Surface it for the operator.
                logger.error(
                    f"Failed to persist streamed turn for session "
                    f"{session_id}: {e}",
                    exc_info=True,
                )

        logger.info(
            f"Stream done: finish={state['finish_reasons'].get(0, 'n/a')} "
            f"interrupted={interrupted} assistant_chars={len(assistant_content)}"
        )


# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """Verify the API key from Authorization header."""
    settings = get_settings()

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    token = authorization[7:]
    if not secrets.compare_digest(token, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


# =============================================================================
# Session Endpoints
# =============================================================================

@app.post(
    "/v1/sessions",
    response_model=SessionResponse,
    tags=["Sessions"],
    summary="Create a new conversation session"
)
async def create_session(
    request: SessionCreate = SessionCreate(),
    _: bool = Depends(verify_api_key)
):
    """
    Create a new conversation session.
    
    Returns a session_id that should be used for subsequent chat requests
    to maintain conversation context.
    """
    session_manager = get_session_manager()
    
    try:
        result = session_manager.create_session(metadata=request.metadata)
        return SessionResponse(**result)
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.get(
    "/v1/sessions/{session_id}",
    response_model=SessionHistory,
    tags=["Sessions"],
    summary="Get session history"
)
async def get_session(
    session_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Retrieve the conversation history for a session.
    """
    session_manager = get_session_manager()
    
    # Check if session exists
    session_info = session_manager.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get messages
    messages = session_manager.get_session_messages(session_id)
    
    return SessionHistory(
        session_id=session_id,
        messages=messages,
        message_count=len(messages),
        created_at=session_info.get('created_at'),
        expires_at=session_info.get('expires_at')
    )


@app.delete(
    "/v1/sessions/{session_id}",
    response_model=SessionDeleteResponse,
    tags=["Sessions"],
    summary="Delete a session"
)
async def delete_session(
    session_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Delete a session and all its messages.
    """
    session_manager = get_session_manager()
    
    try:
        deleted = session_manager.delete_session(session_id)
        return SessionDeleteResponse(
            session_id=session_id,
            deleted=deleted,
            message="Session deleted successfully" if deleted else "Session not found"
        )
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


# =============================================================================
# Chat Completion Endpoints
# =============================================================================

@app.post(
    "/v1/sessions/{session_id}/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["Chat"],
    summary="Chat completion with session context"
)
async def session_chat_completion(
    session_id: str,
    request: ChatCompletionRequest,
    _: bool = Depends(verify_api_key)
):
    """
    Send a chat completion request with session context.
    
    The conversation history is automatically retrieved from the session
    and included in the request to vLLM. The new user message and 
    assistant response are stored in the session.
    """
    session_manager = get_session_manager()
    settings = get_settings()

    if request.messages:
        for msg in request.messages:
            if msg.role == "system":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "System messages are not allowed on the session chat "
                        "endpoint; persona and scope are server-controlled."
                    ),
                )

    # Verify session exists
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Step 1: Load history with the existing hard upper bound. We may
    # trim further once we know how many tokens RAG context consumed,
    # but MAX_HISTORY_TOKENS still caps the ceiling.
    history_messages = session_manager.get_session_history_with_token_limit(session_id)

    # Build provisional full message list (history + new request messages).
    # We use this to locate the latest user message for retrieval/rewriting;
    # the assembled `full_messages` we send to vLLM is rebuilt below after
    # the token budget is settled.
    provisional_messages = [
        ChatMessage(role=msg.role, content=msg.content)
        for msg in history_messages
    ]
    new_user_messages = []
    new_request_messages: List[ChatMessage] = []
    if request.messages:
        for msg in request.messages:
            provisional_messages.append(msg)
            new_request_messages.append(msg)
            if msg.role == "user":
                new_user_messages.append(msg)

    # Cheap gpt-4o-mini classifier: jailbreak / off-scope before RAG + GPU.
    # Fail-open on errors. Skip when disabled or no OpenAI key.
    latest_user_text = next(
        (m.content for m in reversed(new_user_messages) if m.content),
        None,
    )
    if latest_user_text and openai_http_client is not None:
        prior_for_guard = [
            {"role": m.role, "content": m.content}
            for m in history_messages
            if m.role in ("user", "assistant")
        ]
        allowed = await classify_user_intent(
            openai_http_client,
            latest_user_text,
            history_messages=prior_for_guard,
            settings=settings,
        )
        if not allowed:
            topic = settings.desired_rag_topic
            _persist_canned_refusal(
                session_manager, session_id, new_user_messages, topic
            )
            if request.stream:
                return StreamingResponse(
                    _stream_canned_refusal(request.model, topic),
                    media_type="text/event-stream",
                    headers=_SSE_HEADERS,
                )
            return _canned_refusal_response(request.model, topic)

    # We assemble `full_messages` once at the very end, after RAG context
    # is sized and history is trimmed against the per-turn budget.
    full_messages: List[ChatMessage] = provisional_messages

    # Always inject the base system prompt (persona + scope + refusal rules)
    # so off-topic queries -- which produce zero retrieval results and would
    # otherwise bypass the prompt entirely -- still get the canned refusal
    # behavior instead of the bare model's default. The RAG block below may
    # upgrade this to the full version (base + document context) if retrieval
    # finds relevant chunks. Consumed by the budget step further down.
    rag_system_prompt: Optional[str] = build_base_system_prompt(
        settings.desired_rag_topic
    )
    rag_tokens_used = 0
    rag_chunks_kept = 0

    # RAG: Search knowledge base and inject context if enabled
    if settings.rag_enabled and settings.opensearch_endpoint:
        try:
            # Locate the latest user message and the history that precedes it.
            # The history slice is what the rewriter uses to disambiguate
            # follow-ups; the latest message itself is what gets rewritten.
            user_query = None
            latest_user_idx = None
            for idx in range(len(provisional_messages) - 1, -1, -1):
                if provisional_messages[idx].role == "user":
                    user_query = provisional_messages[idx].content
                    latest_user_idx = idx
                    break

            # Pre-retrieval gate: skip the whole RAG pipeline (rewriter +
            # search + rerank) for obvious chit-chat like "hi" or "thanks".
            # The downstream score floor inside search() is the safety net
            # for the cases this misses. We pass has_history so that short
            # follow-ups like "more please" or "what about ice" still hit
            # the rewriter when there are prior turns to anchor against.
            has_prior_history = (
                latest_user_idx is not None and latest_user_idx > 0
            )
            gate_skip = (
                user_query is not None
                and is_chitchat(user_query, has_history=has_prior_history)
            )
            if gate_skip:
                logger.info(
                    f"RAG gate: skipping retrieval for chit-chat: {user_query!r}"
                )

            if user_query and not gate_skip:
                opensearch_rag = get_opensearch_rag()
                if opensearch_rag.is_available():
                    # Rewrite follow-ups into standalone search queries
                    # (e.g. "tell me more about that" -> the actual subject)
                    # whenever there's prior history to anchor against.
                    # Falls back to user_query on any internal failure.
                    search_query = user_query
                    if latest_user_idx is not None and latest_user_idx > 0:
                        prior_msgs = provisional_messages[:latest_user_idx]
                        prior_msgs = [
                            m for m in prior_msgs if m.role in ("user", "assistant")
                        ]
                        trimmed_history = [
                            {"role": m.role, "content": m.content}
                            for m in prior_msgs[-QUERY_REWRITE_HISTORY_TURNS:]
                        ]
                        if trimmed_history:
                            search_query = await rewrite_query(
                                history_messages=trimmed_history,
                                latest_message=user_query,
                                session_id=session_id,
                                openai_api_key=settings.openai_api_key or None,
                            )
                            if search_query != user_query:
                                logger.info(
                                    f"Rewrote query: {user_query!r} -> {search_query!r}"
                                )

                    query_embedding = embed_query(search_query)
                    # Optional metadata filters from the client; Pydantic
                    # already validated/restricted the field set.
                    rag_filters = (
                        request.rag_filters.model_dump(exclude_none=True)
                        if request.rag_filters
                        else None
                    )
                    search_results = opensearch_rag.search(
                        query_text=search_query,
                        query_embedding=query_embedding,
                        top_k=settings.rag_top_k,
                        filters=rag_filters,
                    )

                    if search_results:
                        # Size the RAG body to a portion of the per-turn
                        # budget. Hard cap by RAG_MAX_CONTEXT_TOKENS, soft
                        # cap at half the available content budget so RAG
                        # can never starve all history.
                        cpt = settings.chars_per_token
                        completion_budget = max(
                            request.max_tokens if request.max_tokens is not None else 300,
                            RESERVED_COMPLETION_TOKENS,
                        )
                        new_user_tokens = sum(
                            len(m.content) // cpt for m in new_request_messages
                        )
                        available = (
                            settings.model_context_tokens
                            - completion_budget
                            - SYSTEM_PROMPT_OVERHEAD_TOKENS
                            - new_user_tokens
                        )
                        rag_budget = max(
                            0,
                            min(RAG_MAX_CONTEXT_TOKENS, available // 2),
                        )
                        context, rag_tokens_used = build_rag_context(
                            search_results,
                            max_tokens=rag_budget,
                            chars_per_token=cpt,
                        )
                        if context:
                            rag_system_prompt = build_rag_system_prompt(
                                context, settings.desired_rag_topic
                            )
                            rag_chunks_kept = context.count("\n\n") + 1
                            logger.info(
                                f"RAG: Injected {rag_chunks_kept}/{len(search_results)} "
                                f"chunks ({rag_tokens_used} tokens) into prompt"
                            )
                        else:
                            # The whole budget was consumed before any chunk fit;
                            # drop RAG for this turn rather than send an empty
                            # context block.
                            logger.warning(
                                f"RAG: no chunk fits within {rag_budget}-token "
                                f"budget; skipping injection"
                            )
        except Exception as e:
            logger.error(f"RAG search failed (continuing without context): {e}")

    # Token budget step: now that RAG context is sized, trim history to fit
    # the remaining budget and rebuild the final message list.
    full_messages = _assemble_messages_for_budget(
        history_messages=history_messages,
        new_request_messages=new_request_messages,
        rag_system_prompt=rag_system_prompt,
        rag_tokens_used=rag_tokens_used,
        request_max_tokens=request.max_tokens,
        settings=settings,
    )

    # Prepare request for vLLM
    vllm_request = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in full_messages],
    }
    
    # Add optional parameters if provided (default max_tokens to 300 for concise responses)
    vllm_request["max_tokens"] = request.max_tokens if request.max_tokens is not None else 300
    if request.temperature is not None:
        vllm_request["temperature"] = request.temperature
    if request.top_p is not None:
        vllm_request["top_p"] = request.top_p
    if request.stop is not None:
        vllm_request["stop"] = request.stop
    if request.presence_penalty is not None:
        vllm_request["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        vllm_request["frequency_penalty"] = request.frequency_penalty

    headers = {}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    # Streaming path: hand off to the buffered SSE proxy. The generator
    # persists the user + assistant turn (with interrupted marker on
    # disconnect) in its `finally` block. We return a StreamingResponse
    # so FastAPI knows to keep the connection open.
    if request.stream:
        vllm_request["stream"] = True
        # vLLM streams `usage` only when explicitly requested.
        vllm_request.setdefault("stream_options", {"include_usage": True})
        return StreamingResponse(
            _stream_and_persist(
                vllm_request=vllm_request,
                headers=headers,
                session_manager=session_manager,
                session_id=session_id,
                new_user_messages=new_user_messages,
            ),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    # Non-streaming path (original behavior).
    try:
        response = await http_client.post(
            "/v1/chat/completions",
            json=vllm_request,
            headers=headers
        )
        response.raise_for_status()
        vllm_response = response.json()
        
    except httpx.HTTPStatusError as e:
        logger.error(f"vLLM request failed: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"vLLM backend error: {e.response.text}"
        )
    except httpx.RequestError as e:
        logger.error(f"vLLM connection error: {e}")
        raise HTTPException(status_code=503, detail="vLLM backend unavailable")
    
    # Store new user messages in session
    for msg in new_user_messages:
        session_manager.add_message(session_id, msg.role, msg.content)
    
    # Store assistant response in session
    if vllm_response.get("choices"):
        assistant_message = vllm_response["choices"][0]["message"]
        session_manager.add_message(
            session_id,
            assistant_message["role"],
            assistant_message["content"],
            metadata={"tokens": vllm_response.get("usage", {})}
        )
    
    # Refresh session TTL
    session_manager.refresh_session_ttl(session_id)
    
    return vllm_response


# =============================================================================
# Document Endpoints (RAG Knowledge Base)
# =============================================================================

ALLOWED_FILE_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "txt",
    "text/csv": "txt",
}
# Also match by file extension as fallback
ALLOWED_EXTENSIONS = {".pdf": "pdf", ".txt": "txt", ".md": "txt", ".csv": "txt"}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@app.post(
    "/v1/documents",
    response_model=DocumentUploadResponse,
    tags=["Documents"],
    summary="Upload a document to the knowledge base"
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, MD, or CSV file to upload"),
    _: bool = Depends(verify_api_key)
):
    """
    Upload a document to the global knowledge base.
    
    The file is processed in the background: text extraction, chunking,
    embedding, and indexing. Use GET /v1/documents/{document_id} to check
    processing status.
    
    Supported formats: PDF, TXT, MD, CSV
    Max file size: 50 MB
    """
    settings = get_settings()
    
    if not settings.rag_enabled or not settings.opensearch_endpoint:
        raise HTTPException(
            status_code=503,
            detail="RAG is not enabled. Configure OPENSEARCH_ENDPOINT to enable document uploads."
        )
    
    # Determine file type
    file_type = None
    if file.content_type and file.content_type in ALLOWED_FILE_TYPES:
        file_type = ALLOWED_FILE_TYPES[file.content_type]
    
    if not file_type and file.filename:
        import os
        ext = os.path.splitext(file.filename)[1].lower()
        file_type = ALLOWED_EXTENSIONS.get(ext)
    
    if not file_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: PDF, TXT, MD, CSV. Got content_type={file.content_type}"
        )
    
    # Read file into memory
    file_bytes = await file.read()
    file_size = len(file_bytes)
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB"
        )
    
    # Create document record
    document_id = str(uuid.uuid4())
    filename = file.filename or f"document.{file_type}"
    
    doc_manager = get_document_manager()
    doc_manager.create_document_record(document_id, filename, file_type, file_size)
    
    # Start background processing
    asyncio.create_task(
        process_document_background(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            file_bytes=file_bytes,
            doc_manager=doc_manager,
        )
    )
    
    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        file_size=file_size,
        status="processing",
        message="Document uploaded and processing started. Check status at GET /v1/documents/{document_id}"
    )


@app.get(
    "/v1/documents",
    response_model=DocumentListResponse,
    tags=["Documents"],
    summary="List all documents in the knowledge base"
)
async def list_documents(_: bool = Depends(verify_api_key)):
    """
    List all documents in the global knowledge base with their processing status.
    """
    settings = get_settings()
    
    if not settings.rag_enabled or not settings.opensearch_endpoint:
        raise HTTPException(
            status_code=503,
            detail="RAG is not enabled. Configure OPENSEARCH_ENDPOINT to enable document features."
        )
    
    doc_manager = get_document_manager()
    documents = doc_manager.list_documents()
    
    doc_list = [
        DocumentInfo(
            document_id=doc["document_id"],
            filename=doc["filename"],
            file_type=doc["file_type"],
            file_size=int(doc["file_size"]),
            status=doc["status"],
            chunk_count=int(doc.get("chunk_count", 0)),
            uploaded_at=doc.get("uploaded_at"),
            error=doc.get("error")
        )
        for doc in documents
    ]
    
    return DocumentListResponse(documents=doc_list, total_count=len(doc_list))


@app.get(
    "/v1/documents/{document_id}",
    response_model=DocumentInfo,
    tags=["Documents"],
    summary="Get document info and processing status"
)
async def get_document(
    document_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Get detailed information about a specific document, including processing status.
    """
    settings = get_settings()
    
    if not settings.rag_enabled or not settings.opensearch_endpoint:
        raise HTTPException(
            status_code=503,
            detail="RAG is not enabled."
        )
    
    doc_manager = get_document_manager()
    doc = doc_manager.get_document(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentInfo(
        document_id=doc["document_id"],
        filename=doc["filename"],
        file_type=doc["file_type"],
        file_size=int(doc["file_size"]),
        status=doc["status"],
        chunk_count=int(doc.get("chunk_count", 0)),
        uploaded_at=doc.get("uploaded_at"),
        error=doc.get("error")
    )


@app.delete(
    "/v1/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    tags=["Documents"],
    summary="Delete a document from the knowledge base"
)
async def delete_document(
    document_id: str,
    _: bool = Depends(verify_api_key)
):
    """
    Delete a document and all its indexed chunks from the knowledge base.
    """
    settings = get_settings()
    
    if not settings.rag_enabled or not settings.opensearch_endpoint:
        raise HTTPException(
            status_code=503,
            detail="RAG is not enabled."
        )
    
    doc_manager = get_document_manager()
    deleted = doc_manager.delete_document(document_id)
    
    return DocumentDeleteResponse(
        document_id=document_id,
        deleted=deleted,
        message="Document deleted successfully" if deleted else "Document not found"
    )


# =============================================================================
# Models Endpoint
# =============================================================================

@app.get(
    "/v1/models",
    response_model=ModelsListResponse,
    tags=["Models"],
    summary="List available models"
)
async def list_models(_: bool = Depends(verify_api_key)):
    """
    List available models from vLLM backend.
    """
    settings = get_settings()
    
    try:
        headers = {}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        
        response = await http_client.get("/v1/models", headers=headers)
        response.raise_for_status()
        return response.json()
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail="Failed to retrieve models from vLLM"
        )
    except httpx.RequestError as e:
        logger.error(f"vLLM connection error: {e}")
        raise HTTPException(status_code=503, detail="vLLM backend unavailable")


# =============================================================================
# Health Check Endpoints
# =============================================================================

def _gateway_is_ready(health: HealthResponse, settings: Settings) -> bool:
    """
    Return True when the gateway can accept session traffic.

    Used by GET /health (readiness): vLLM + DynamoDB are always required;
    OpenSearch is required only when RAG is enabled and configured.
    """
    if health.vllm_backend != "healthy" or health.dynamodb != "healthy":
        return False
    if settings.rag_enabled and settings.opensearch_endpoint:
        return health.opensearch == "healthy"
    return True


@app.get(
    "/live",
    tags=["Health"],
    summary="Liveness probe (process up, no dependency checks)",
)
async def liveness():
    """
    Cheap liveness check: confirms the HTTP server is responding.

    Kubernetes liveness and startup probes should use this endpoint so
    downstream outages (vLLM, DynamoDB, OpenSearch) do not restart
    otherwise-healthy gateway pods.
    """
    return {"status": "ok"}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Readiness check (gateway + dependencies)",
)
async def health_check():
    """
    Readiness check: probes vLLM, DynamoDB, and (when RAG is on) OpenSearch.

    Returns HTTP 200 when ready to serve traffic, HTTP 503 when a required
    dependency is down. Kubernetes readiness probes and operators should
    use this endpoint.
    """
    settings = get_settings()
    health = HealthResponse(status="healthy", gateway="healthy")

    # Check vLLM backend
    try:
        response = await http_client.get("/health")
        if response.status_code == 200:
            health.vllm_backend = "healthy"
        else:
            health.vllm_backend = "unhealthy"
            health.status = "degraded"
    except Exception:
        health.vllm_backend = "unhealthy"
        health.status = "degraded"

    # Check DynamoDB
    try:
        session_manager = get_session_manager()
        session_manager.table.table_status
        health.dynamodb = "healthy"
    except Exception:
        health.dynamodb = "unhealthy"
        health.status = "degraded"

    # Check OpenSearch (if RAG enabled)
    if settings.rag_enabled and settings.opensearch_endpoint:
        try:
            opensearch_rag = get_opensearch_rag()
            if opensearch_rag.is_available():
                health.opensearch = "healthy"
            else:
                health.opensearch = "unhealthy"
                health.status = "degraded"
        except Exception:
            health.opensearch = "unhealthy"
            health.status = "degraded"

    payload = health.model_dump()
    if not _gateway_is_ready(health, settings):
        return JSONResponse(status_code=503, content=payload)
    return health


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "vLLM Gateway API",
        "version": "2.0.0",
        "docs": "/docs",
        "live": "/live",
        "health": "/health",
    }


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )
