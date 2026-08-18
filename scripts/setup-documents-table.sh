#!/bin/bash
# =============================================================================
# DynamoDB Table Setup for Document Metadata (RAG)
# =============================================================================
# Creates the DynamoDB table that stores document metadata for the RAG
# knowledge base (document_id, filename, status, chunk_count, etc.)
#
# This is separate from the vllm-conversations table used for chat sessions.
#
# Usage:
#   ./scripts/setup-documents-table.sh
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
AWS_REGION="${AWS_REGION:-us-east-1}"
TABLE_NAME="vllm-documents"

echo "=============================================="
echo "Setting up DynamoDB Table for Documents"
echo "=============================================="
echo "Table: ${TABLE_NAME}"
echo "Region: ${AWS_REGION}"
echo ""

# Check if table already exists
EXISTING=$(aws dynamodb describe-table --table-name "$TABLE_NAME" --region "$AWS_REGION" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Table '${TABLE_NAME}' already exists!"
    exit 0
fi

echo "Creating DynamoDB table..."

aws dynamodb create-table \
    --table-name "$TABLE_NAME" \
    --attribute-definitions \
        AttributeName=document_id,AttributeType=S \
    --key-schema \
        AttributeName=document_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "$AWS_REGION"

echo ""
echo "Waiting for table to become active..."
aws dynamodb wait table-exists --table-name "$TABLE_NAME" --region "$AWS_REGION"

echo ""
echo "=============================================="
echo "Documents Table Ready!"
echo "=============================================="
echo "Table: ${TABLE_NAME}"
echo "Primary Key: document_id (String)"
echo "Billing: On-demand (pay per request)"
echo "=============================================="
