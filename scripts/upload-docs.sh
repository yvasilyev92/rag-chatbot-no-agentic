#!/bin/bash
# =============================================================================
# Upload Documents to RAG Knowledge Base
# =============================================================================
# Uploads files from the docs/ folder to the gateway's /v1/documents endpoint.
#
# Files are uploaded SERIALLY: the script waits for each document's background
# embedding + indexing to finish (status: processing -> ready) before starting
# the next upload. This caps peak pod memory at one in-flight ingest's worth,
# instead of N parallel ingests stacking on top of each other (which can OOM
# the gateway pod under load).
#
# Usage:
#   ./scripts/upload-docs.sh                  # Upload all files in docs/
#   ./scripts/upload-docs.sh docs/skills.md   # Upload a specific file
#
# Env vars (optional):
#   POLL_INTERVAL_SECONDS  poll cadence while waiting (default 3)
#   POLL_MAX_SECONDS       per-file timeout before giving up (default 300)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Configuration
API_KEY="${VLLM_API_KEY:-}"
API_URL="${GATEWAY_URL:-}"

# If no gateway URL set, try to get it from kubectl
if [ -z "$API_URL" ]; then
    API_URL=$(kubectl get svc gateway-service -n vllm -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    if [ -n "$API_URL" ]; then
        API_URL="https://$API_URL"
    fi
fi

if [ -z "$API_URL" ]; then
    echo "ERROR: Could not determine Gateway URL."
    echo "Set GATEWAY_URL in .env or pass it as an environment variable."
    exit 1
fi

if [ -z "$API_KEY" ]; then
    echo "ERROR: VLLM_API_KEY is not set."
    exit 1
fi

echo "=============================================="
echo "Uploading Documents to RAG Knowledge Base"
echo "=============================================="
echo "Gateway: $API_URL"
echo ""

# Determine which files to upload
if [ -n "$1" ]; then
    # Upload specific file(s) passed as arguments
    FILES="$@"
else
    # Upload all files in docs/
    FILES=$(find docs/ -type f \( -name "*.md" -o -name "*.txt" -o -name "*.pdf" -o -name "*.csv" \) 2>/dev/null)
fi

if [ -z "$FILES" ]; then
    echo "No files found to upload."
    exit 0
fi

# POST returns "processing" immediately and the gateway finishes embedding +
# indexing in a background task. Polling the per-document GET endpoint is what
# turns this from "fire and forget" into a true serial pipeline.
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-3}"
POLL_MAX_SECONDS="${POLL_MAX_SECONDS:-300}"

UPLOADED=0
FAILED=0

for FILE in $FILES; do
    if [ ! -f "$FILE" ]; then
        echo "SKIP: $FILE (not found)"
        FAILED=$((FAILED + 1))
        continue
    fi

    FILENAME=$(basename "$FILE")
    echo "Uploading: $FILE ..."

    RESPONSE=$(curl -s -X POST "$API_URL/v1/documents" \
        -H "Authorization: Bearer $API_KEY" \
        -F "file=@$FILE")

    STATUS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "error")
    DOC_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('document_id',''))" 2>/dev/null || echo "")

    if [ "$STATUS" != "processing" ] || [ -z "$DOC_ID" ]; then
        echo "  FAILED to accept: $FILENAME"
        echo "  Response: $RESPONSE"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "  Accepted: $DOC_ID -- waiting for indexing..."

    waited=0
    final_status=""
    while [ $waited -lt $POLL_MAX_SECONDS ]; do
        sleep "$POLL_INTERVAL_SECONDS"
        waited=$((waited + POLL_INTERVAL_SECONDS))

        DOC_JSON=$(curl -s "$API_URL/v1/documents/$DOC_ID" \
            -H "Authorization: Bearer $API_KEY")
        DOC_STATUS=$(echo "$DOC_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "error")

        case "$DOC_STATUS" in
            ready)
                CHUNKS=$(echo "$DOC_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('chunk_count','?'))" 2>/dev/null || echo "?")
                echo "  OK: $FILENAME -> $CHUNKS chunks indexed (took ${waited}s)"
                UPLOADED=$((UPLOADED + 1))
                final_status="ready"
                break
                ;;
            failed)
                ERR=$(echo "$DOC_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))" 2>/dev/null || echo "unknown")
                echo "  FAILED: $FILENAME -> $ERR"
                FAILED=$((FAILED + 1))
                final_status="failed"
                break
                ;;
            processing)
                ;;
            *)
                echo "  ? unexpected status '$DOC_STATUS' for $FILENAME, will keep polling"
                ;;
        esac
    done

    if [ -z "$final_status" ]; then
        echo "  TIMEOUT: $FILENAME still processing after ${POLL_MAX_SECONDS}s -- moving on"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=============================================="
echo "Upload Complete!"
echo "=============================================="
echo "Indexed: $UPLOADED"
echo "Failed:  $FAILED"
echo ""
echo "Verify all documents are present:"
echo "  curl -H 'Authorization: Bearer YOUR_API_KEY' $API_URL/v1/documents"
echo "=============================================="
