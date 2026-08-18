"""
Configuration settings for the API Gateway.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # AWS Configuration
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    dynamodb_table: str = os.getenv("DYNAMODB_TABLE", "vllm-conversations")
    
    # vLLM Backend Configuration
    vllm_internal_url: str = os.getenv("VLLM_INTERNAL_URL", "http://vllm-service:80")
    
    # Session Configuration
    session_ttl_hours: int = int(os.getenv("SESSION_TTL_HOURS", "24"))
    max_history_tokens: int = int(os.getenv("MAX_HISTORY_TOKENS", "3000"))
    
    # API Configuration
    api_key: str = os.getenv("VLLM_API_KEY", "")

    # gpt-4o-mini for the input guard and query rewrite. Fail-open without a key.
    input_guard_enabled: bool = os.getenv("INPUT_GUARD_ENABLED", "true").lower() == "true"
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Domain name interpolated into the persona, scope, and canned refusal.
    # Swap this (and the docs/ corpus) to point the demo at any knowledge base.
    desired_rag_topic: str = os.getenv("DESIRED_RAG_TOPIC", "Desired RAG Topic")
    
    # Server Configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))
    
    # Token estimation (approximate characters per token for Llama models)
    chars_per_token: int = 4
    
    # RAG Configuration
    opensearch_endpoint: str = os.getenv("OPENSEARCH_ENDPOINT", "")
    opensearch_index: str = os.getenv("OPENSEARCH_INDEX", "vllm-documents")
    rag_enabled: bool = os.getenv("RAG_ENABLED", "true").lower() == "true"
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "2000"))       # ~500 tokens in chars
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))  # ~50 tokens in chars

    # Post-rerank score floor (sigmoid-normalized to [0, 1]). The single
    # dial reviewers actually turn after seeing bad results in prod logs.
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.35"))

    # Total token budget per request. MUST equal vLLM's --max-model-len
    # (set in kubernetes/configmap.yaml -> MAX_MODEL_LEN). Pairs with that
    # value across two separate config files, so it stays an env var:
    # swapping the underlying model (Llama 4K -> Qwen 32K -> ...) requires
    # changing both in lockstep, and we don't want a code change for it.
    # Default matches the shipped MAX_MODEL_LEN=4096 so misconfigured
    # gateway pods fail safe instead of silently truncating.
    model_context_tokens: int = int(os.getenv("MODEL_CONTEXT_TOKENS", "4096"))

    # ------------------------------------------------------------------
    # Knobs intentionally NOT exposed as env vars
    # ------------------------------------------------------------------
    # The RAG pipeline has only one master kill-switch (`RAG_ENABLED`
    # above). Per-stage toggles for hybrid retrieval, reranking, query
    # rewriting, retrieval gating, and caching used to exist but were
    # removed: each stage has internal fallbacks (hybrid -> kNN, rewrite
    # -> literal message, rerank -> unranked top-K, etc.), so a per-stage
    # disable flag added optionality without solving any actual incident
    # any faster than a code change + redeploy would.
    #
    # The following are module-level constants in the code that owns
    # them, since they have no real ops use case (defensive ceilings the
    # budgeter already auto-bounds, prompt-coupled values, image-baked
    # model paths, or pod-local memory bounds at sub-MB scale):
    #   - RESERVED_COMPLETION_TOKENS, RAG_MAX_CONTEXT_TOKENS     (main.py)
    #   - SYSTEM_PROMPT_OVERHEAD_TOKENS                          (main.py)
    #   - RERANK_MODEL_NAME                                      (rag.py)
    #   - HYBRID_CANDIDATE_POOL, RRF_K, RERANK_POOL              (rag.py)
    #   - QUERY_REWRITE_HISTORY_TURNS, _MAX_TOKENS, _TEMPERATURE,
    #     _TIMEOUT_SECONDS, _CACHE_SIZE, QUERY_REWRITE_MODEL     (rag.py)
    #   - EMBEDDING_CACHE_SIZE, SEARCH_CACHE_SIZE,
    #     SEARCH_CACHE_TTL_SECONDS                               (rag.py)
    #
    # DESIRED_RAG_TOPIC is the exception: it is prompt-coupled but is an
    # env var so the same image can be pointed at any domain without a
    # code change.
    # ------------------------------------------------------------------

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
