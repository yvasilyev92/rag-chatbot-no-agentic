#!/bin/bash
# =============================================================================
# DynamoDB Table Setup Script
# =============================================================================
# This script creates the DynamoDB table for storing conversation sessions.
#
# Table Design:
#   - Partition Key: session_id (String) - Unique conversation identifier
#   - Sort Key: message_id (String) - Timestamp-based for message ordering
#   - TTL Attribute: expires_at - Automatic cleanup of expired sessions
#   - Billing: On-Demand (pay-per-request, auto-scales)
#
# Prerequisites:
#   - AWS CLI configured with appropriate permissions
#   - AWS_REGION environment variable set (or uses default us-east-1)
#
# Usage:
#   ./scripts/setup-dynamodb.sh
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

TABLE_NAME="${DYNAMODB_TABLE:-vllm-conversations}"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "=============================================="
echo "Creating DynamoDB Table: ${TABLE_NAME}"
echo "Region: ${AWS_REGION}"
echo "=============================================="

# Check if table already exists
if aws dynamodb describe-table --table-name "${TABLE_NAME}" --region "${AWS_REGION}" 2>/dev/null; then
    echo ""
    echo "Table '${TABLE_NAME}' already exists."
    echo ""
    
    # Check TTL status
    TTL_STATUS=$(aws dynamodb describe-time-to-live --table-name "${TABLE_NAME}" --region "${AWS_REGION}" --query 'TimeToLiveDescription.TimeToLiveStatus' --output text)
    echo "TTL Status: ${TTL_STATUS}"
    
    if [ "${TTL_STATUS}" != "ENABLED" ]; then
        echo "Enabling TTL on expires_at attribute..."
        aws dynamodb update-time-to-live \
            --table-name "${TABLE_NAME}" \
            --region "${AWS_REGION}" \
            --time-to-live-specification "Enabled=true,AttributeName=expires_at"
        echo "TTL enabled successfully."
    fi
    
    exit 0
fi

# Create the table
echo ""
echo "Creating table..."
aws dynamodb create-table \
    --table-name "${TABLE_NAME}" \
    --region "${AWS_REGION}" \
    --attribute-definitions \
        AttributeName=session_id,AttributeType=S \
        AttributeName=message_id,AttributeType=S \
    --key-schema \
        AttributeName=session_id,KeyType=HASH \
        AttributeName=message_id,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    --tags \
        Key=Project,Value=vllm-deployment \
        Key=Component,Value=conversation-memory

echo ""
echo "Waiting for table to become active..."
aws dynamodb wait table-exists --table-name "${TABLE_NAME}" --region "${AWS_REGION}"

# Enable TTL
echo ""
echo "Enabling TTL on 'expires_at' attribute..."
aws dynamodb update-time-to-live \
    --table-name "${TABLE_NAME}" \
    --region "${AWS_REGION}" \
    --time-to-live-specification "Enabled=true,AttributeName=expires_at"

echo ""
echo "=============================================="
echo "DynamoDB table created successfully!"
echo "=============================================="
echo ""
echo "Table Details:"
echo "  Name: ${TABLE_NAME}"
echo "  Region: ${AWS_REGION}"
echo "  Partition Key: session_id (String)"
echo "  Sort Key: message_id (String)"
echo "  TTL Attribute: expires_at"
echo "  Billing Mode: On-Demand (pay-per-request)"
echo ""
echo "Next steps:"
echo "  1. Run ./scripts/setup-iam-gateway.sh to create IAM role"
echo "  2. Build and deploy the gateway service"
echo ""
