# vLLM OpenAI-compatible API server
# Using v0.6.4 for CUDA driver compatibility with EKS GPU nodes
FROM vllm/vllm-openai:v0.6.4

# Set environment variables with defaults (can be overridden at runtime)
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
ENV MAX_MODEL_LEN=4096
ENV GPU_MEMORY_UTILIZATION=0.9
ENV HOST=0.0.0.0
ENV PORT=8000

# HuggingFace token (REQUIRED for Llama 3.1 - it's a gated model). Empty
# default is intentional: HF only needs a token for gated models, so a
# missing token only matters at model-pull time and we'd rather fail loudly
# there than silently. Set in kubernetes/secret.yaml or via -e at runtime.
ENV HF_TOKEN=""

# API Key for authentication. NO default on purpose: vLLM is launched with
# `--api-key "$VLLM_API_KEY"`, and starting with an empty key disables auth
# entirely. Force the runtime to provide it via Secret / -e VLLM_API_KEY=...,
# or `/start.sh` will fail with a clear "VLLM_API_KEY is empty" error.

# Create cache directory for model downloads
RUN mkdir -p /root/.cache/huggingface

# Create startup script to handle environment variable expansion.
# v0.6.4 uses the old command syntax: python -m vllm.entrypoints.openai.api_server
# Refuses to start with an empty VLLM_API_KEY so a misconfigured pod
# fails fast instead of booting with auth disabled.
RUN echo '#!/bin/bash\n\
set -e\n\
if [ -z "$VLLM_API_KEY" ]; then\n\
    echo "ERROR: VLLM_API_KEY is empty. Refusing to start with auth disabled." >&2\n\
    echo "       Set it via kubernetes/secret.yaml (vllm-secrets) or -e VLLM_API_KEY=... at docker run." >&2\n\
    exit 1\n\
fi\n\
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$VLLM_API_KEY" \
    --trust-remote-code' > /start.sh && chmod +x /start.sh

# Expose the API port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Clear the base image's ENTRYPOINT and start vLLM server
ENTRYPOINT []
CMD ["/start.sh"]
