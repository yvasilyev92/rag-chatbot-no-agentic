# =============================================================================
# Dockerfile for vLLM Gateway Service
# =============================================================================
# Multi-stage build for smaller final image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Make sure scripts in .local are usable
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application code
COPY app/ ./app/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# NOTE: We deliberately don't bake runtime defaults (HOST, PORT, RAG_*, AWS_*)
# into the image as ENV. Every value already has a default in
# gateway/app/config.py, and production overrides come from the
# `gateway-config` ConfigMap (Kubernetes) or docker-compose env. Duplicating
# defaults here just creates a third source of truth that drifts.

# Pre-download the embedding model at build time so there's no delay at startup
# BAAI/bge-small-en-v1.5: 384 dimensions, ~130MB, optimized for ONNX
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Pre-download the cross-encoder reranker model at build time
# Xenova/ms-marco-MiniLM-L-6-v2: ~22M params, ~80MB, fast CPU reranking
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

# Expose port
EXPOSE 8080

# Liveness only: /live never checks downstream deps (see GET /health for readiness).
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/live').raise_for_status()" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
