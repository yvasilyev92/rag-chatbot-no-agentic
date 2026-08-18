"""
Pydantic models for API request/response schemas.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Session Models
# ============================================================================

class SessionCreate(BaseModel):
    """Request model for creating a new session."""
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata to associate with the session"
    )


class SessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(..., description="Session creation timestamp")
    expires_at: datetime = Field(..., description="Session expiration timestamp")


class Message(BaseModel):
    """A single message in the conversation."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    created_at: Optional[datetime] = Field(default=None, description="Message timestamp")


class SessionHistory(BaseModel):
    """Response model for session history."""
    session_id: str
    messages: List[Message]
    message_count: int
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class SessionDeleteResponse(BaseModel):
    """Response model for session deletion."""
    session_id: str
    deleted: bool
    message: str


# ============================================================================
# Chat Completion Models (OpenAI-compatible)
# ============================================================================

class ChatMessage(BaseModel):
    """Message in chat completion request."""
    role: str = Field(..., description="Role: 'user' or 'assistant'. Client 'system' is rejected with 400.")
    content: str = Field(..., description="Message content")


class RagFilters(BaseModel):
    """
    Optional filters narrowing which document chunks RAG retrieval may return.

    Semantics:
      - AND across fields, OR within a field.
        e.g. {"category": "Equipment", "topic": ["Weapons", "Armor"]}
        means "Equipment AND (Weapons OR Armor)".
      - Strict filter (excludes non-matching chunks; does not just boost).
      - Empty / null means no filter.
    """
    category: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Document category (e.g. 'Equipment', 'Policies')."
    )
    topic: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Document topic (e.g. 'Weapons', 'Adventure Events')."
    )
    document_name: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Document name as set in markdown frontmatter."
    )

    # Reject unknown keys at the API boundary so clients can't sneak in
    # fields like 'chunk_id' that would bypass the intended filter surface.
    model_config = ConfigDict(extra="forbid")


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion (OpenAI-compatible)."""
    model: str = Field(..., description="Model name")
    messages: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Messages for this turn (optional if using session history)"
    )
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling")
    stream: Optional[bool] = Field(default=False, description="Stream responses")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    
    # Additional OpenAI-compatible fields
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    n: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

    # RAG retrieval filters (session chat only).
    rag_filters: Optional[RagFilters] = Field(
        default=None,
        description="Optional filters to narrow RAG retrieval to specific document subsets."
    )


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response model for chat completion (OpenAI-compatible)."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None


# ============================================================================
# Models Endpoint
# ============================================================================

class ModelInfo(BaseModel):
    """Information about an available model."""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelsListResponse(BaseModel):
    """Response model for listing available models."""
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Health Check
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gateway: str = "healthy"
    vllm_backend: Optional[str] = None
    dynamodb: Optional[str] = None
    opensearch: Optional[str] = None


# ============================================================================
# Document Models (RAG)
# ============================================================================

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Stored type: 'pdf' or 'txt' (md and csv are stored as txt)")
    file_size: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Processing status: 'processing', 'ready', 'failed'")
    message: str = Field(..., description="Status message")


class DocumentInfo(BaseModel):
    """Information about an uploaded document."""
    document_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int = 0
    uploaded_at: Optional[str] = None
    error: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[DocumentInfo]
    total_count: int


class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion."""
    document_id: str
    deleted: bool
    message: str


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
