#!/bin/bash
# =============================================================================
# OpenSearch Domain Setup for RAG Vector Search
# =============================================================================
# This script creates an AWS OpenSearch managed domain with kNN (vector search)
# support for the RAG knowledge base.
#
# What it creates:
#   - An OpenSearch domain (t3.small.search, single node)
#   - IAM resource policy (same-account principals; no fine-grained access control)
#   - kNN plugin enabled for vector similarity search
#
# Cost: ~$26/month for t3.small.search
#
# Prerequisites:
#   - AWS CLI configured with appropriate permissions
#   - .env file with AWS_REGION and AWS_ACCOUNT_ID
#
# Usage:
#   ./scripts/setup-opensearch.sh
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
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
if [ -z "$AWS_ACCOUNT_ID" ] || [ "$AWS_ACCOUNT_ID" = "PLACEHOLDER_AWS_ACCOUNT_ID" ]; then
    AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
fi
DOMAIN_NAME="${OPENSEARCH_DOMAIN_NAME:-vllm-rag}"

echo "=============================================="
echo "Setting up OpenSearch Domain for RAG"
echo "=============================================="
echo "Domain: ${DOMAIN_NAME}"
echo "Region: ${AWS_REGION}"
echo "Account: ${AWS_ACCOUNT_ID}"
echo "Instance: t3.small.search (single node)"
echo ""
echo "Estimated cost: ~\$26/month"
echo "Creation time: ~15-20 minutes"
echo ""

# Check if domain already exists
EXISTING=$(aws opensearch describe-domain --domain-name "$DOMAIN_NAME" --region "$AWS_REGION" 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    ENDPOINT=$(echo "$EXISTING" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('DomainStatus',{}).get('Endpoint',''))" 2>/dev/null || echo "")
    if [ -n "$ENDPOINT" ]; then
        echo "Domain already exists!"
        echo "Endpoint: https://${ENDPOINT}"
        echo ""
        echo "Set this in your .env (deploy.sh injects it into the gateway ConfigMap):"
        echo "  OPENSEARCH_ENDPOINT=https://${ENDPOINT}"
        exit 0
    else
        echo "Domain exists but may still be creating. Check AWS console."
        exit 0
    fi
fi

read -p "Create OpenSearch domain? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Creating OpenSearch domain..."

# Create the domain with:
# - t3.small.search instance (cheapest option with kNN support)
# - Single node (no multi-AZ for cost savings)
# - 20 GB EBS storage
# - NO fine-grained access control (IAM resource policy is sufficient)
# - Public access (secured by IAM resource policy)
# - OpenSearch 2.11 (supports kNN/vector search)
aws opensearch create-domain \
    --domain-name "$DOMAIN_NAME" \
    --region "$AWS_REGION" \
    --engine-version "OpenSearch_2.11" \
    --cluster-config \
        InstanceType=t3.small.search,InstanceCount=1,DedicatedMasterEnabled=false,ZoneAwarenessEnabled=false,WarmEnabled=false \
    --ebs-options \
        EBSEnabled=true,VolumeType=gp3,VolumeSize=20 \
    --node-to-node-encryption-options Enabled=true \
    --encryption-at-rest-options Enabled=true \
    --domain-endpoint-options EnforceHTTPS=true,TLSSecurityPolicy=Policy-Min-TLS-1-2-PFS-2023-10 \
    --access-policies "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":\"*\"},\"Action\":\"es:*\",\"Resource\":\"arn:aws:es:${AWS_REGION}:${AWS_ACCOUNT_ID}:domain/${DOMAIN_NAME}/*\",\"Condition\":{\"StringEquals\":{\"aws:PrincipalAccount\":\"${AWS_ACCOUNT_ID}\"}}}]}"

echo ""
echo "=============================================="
echo "OpenSearch Domain Creation Initiated!"
echo "=============================================="
echo ""
echo "The domain is being created. This takes ~15-20 minutes."
echo ""
echo "Monitor progress:"
echo "  aws opensearch describe-domain --domain-name $DOMAIN_NAME --region $AWS_REGION --query 'DomainStatus.Processing'"
echo ""
echo "When 'Processing' shows 'false', get the endpoint:"
echo "  aws opensearch describe-domain --domain-name $DOMAIN_NAME --region $AWS_REGION --query 'DomainStatus.Endpoint' --output text"
echo ""

# Wait for domain to be created
echo "Waiting for domain to become active (this will take ~15-20 minutes)..."
echo "(You can Ctrl+C and check manually later)"
echo ""

while true; do
    STATUS=$(aws opensearch describe-domain --domain-name "$DOMAIN_NAME" --region "$AWS_REGION" --query 'DomainStatus.Processing' --output text 2>/dev/null || echo "True")
    
    if [ "$STATUS" = "False" ] || [ "$STATUS" = "false" ]; then
        break
    fi
    
    echo "  Still creating... ($(date '+%H:%M:%S'))"
    sleep 30
done

# Get the endpoint
ENDPOINT=$(aws opensearch describe-domain --domain-name "$DOMAIN_NAME" --region "$AWS_REGION" --query 'DomainStatus.Endpoint' --output text)

echo ""
echo "=============================================="
echo "OpenSearch Domain Ready!"
echo "=============================================="
echo ""
echo "Domain: ${DOMAIN_NAME}"
echo "Endpoint: https://${ENDPOINT}"
echo ""
echo "IMPORTANT: Add this to your .env file (deploy.sh injects it; do not edit the committed ConfigMap):"
echo "  OPENSEARCH_ENDPOINT=https://${ENDPOINT}"
echo ""
echo "Next steps:"
echo "  1. Update IAM permissions: ./scripts/setup-iam-gateway.sh"
echo "  2. Rebuild gateway image: ./scripts/build-and-push.sh gateway"
echo "  3. Deploy: ./scripts/deploy.sh"
echo "=============================================="
