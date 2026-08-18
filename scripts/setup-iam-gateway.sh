#!/bin/bash
# =============================================================================
# IAM Role Setup for Gateway Access (IRSA)
# =============================================================================
# This script creates an IAM role that allows EKS gateway pods to access:
#   - DynamoDB (conversation storage + document metadata)
#   - OpenSearch (RAG vector search)
# using IAM Roles for Service Accounts (IRSA).
#
# Single source of truth for the gateway IAM policy is the inline
# POLICY_DOCUMENT heredoc below.
#
# Prerequisites:
#   - EKS cluster with OIDC provider enabled
#   - AWS CLI configured with IAM permissions
#   - eksctl installed
#
# Usage:
#   ./scripts/setup-iam-gateway.sh
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

CLUSTER_NAME="${EKS_CLUSTER_NAME:-vllm-cluster}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
if [ -z "$AWS_ACCOUNT_ID" ] || [ "$AWS_ACCOUNT_ID" = "PLACEHOLDER_AWS_ACCOUNT_ID" ]; then
    AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
fi
TABLE_NAME="${DYNAMODB_TABLE:-vllm-conversations}"
DOCUMENTS_TABLE_NAME="vllm-documents"
OPENSEARCH_DOMAIN="${OPENSEARCH_DOMAIN_NAME:-vllm-rag}"
NAMESPACE="vllm"
SERVICE_ACCOUNT_NAME="gateway-sa"
ROLE_NAME="vllm-gateway-dynamodb-role"
POLICY_NAME="vllm-gateway-dynamodb-policy"

echo "=============================================="
echo "Setting up IAM Role for Gateway Access"
echo "=============================================="
echo "Cluster: ${CLUSTER_NAME}"
echo "Region: ${AWS_REGION}"
echo "Account: ${AWS_ACCOUNT_ID}"
echo "DynamoDB Tables: ${TABLE_NAME}, ${DOCUMENTS_TABLE_NAME}"
echo "OpenSearch Domain: ${OPENSEARCH_DOMAIN}"
echo ""

# Step 1: Check if OIDC provider exists for the cluster
echo "Step 1: Checking OIDC provider..."
OIDC_ID=$(aws eks describe-cluster --name "${CLUSTER_NAME}" --region "${AWS_REGION}" --query "cluster.identity.oidc.issuer" --output text | cut -d '/' -f 5)

if [ -z "${OIDC_ID}" ]; then
    echo "Error: Could not get OIDC provider ID. Make sure the cluster exists and has OIDC enabled."
    exit 1
fi

echo "OIDC Provider ID: ${OIDC_ID}"

# Check if OIDC provider is associated
OIDC_PROVIDER=$(aws iam list-open-id-connect-providers --query "OpenIDConnectProviderList[?ends_with(Arn, '${OIDC_ID}')].Arn" --output text)

if [ -z "${OIDC_PROVIDER}" ]; then
    echo "OIDC provider not found. Creating..."
    eksctl utils associate-iam-oidc-provider --cluster "${CLUSTER_NAME}" --region "${AWS_REGION}" --approve
    echo "OIDC provider created."
else
    echo "OIDC provider already exists: ${OIDC_PROVIDER}"
fi

# Step 2: Create IAM Policy
echo ""
echo "Step 2: Creating IAM Policy..."

POLICY_DOCUMENT=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DynamoDBTableAccess",
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:UpdateItem",
                "dynamodb:DeleteItem",
                "dynamodb:Query",
                "dynamodb:Scan",
                "dynamodb:BatchGetItem",
                "dynamodb:BatchWriteItem"
            ],
            "Resource": [
                "arn:aws:dynamodb:${AWS_REGION}:${AWS_ACCOUNT_ID}:table/${TABLE_NAME}",
                "arn:aws:dynamodb:${AWS_REGION}:${AWS_ACCOUNT_ID}:table/${TABLE_NAME}/index/*",
                "arn:aws:dynamodb:${AWS_REGION}:${AWS_ACCOUNT_ID}:table/${DOCUMENTS_TABLE_NAME}",
                "arn:aws:dynamodb:${AWS_REGION}:${AWS_ACCOUNT_ID}:table/${DOCUMENTS_TABLE_NAME}/index/*"
            ]
        },
        {
            "Sid": "DynamoDBDescribeTable",
            "Effect": "Allow",
            "Action": [
                "dynamodb:DescribeTable",
                "dynamodb:DescribeTimeToLive"
            ],
            "Resource": [
                "arn:aws:dynamodb:${AWS_REGION}:${AWS_ACCOUNT_ID}:table/${TABLE_NAME}",
                "arn:aws:dynamodb:${AWS_REGION}:${AWS_ACCOUNT_ID}:table/${DOCUMENTS_TABLE_NAME}"
            ]
        },
        {
            "Sid": "OpenSearchHTTPAccess",
            "Effect": "Allow",
            "Action": [
                "es:ESHttpGet",
                "es:ESHttpHead",
                "es:ESHttpPost",
                "es:ESHttpPut",
                "es:ESHttpDelete"
            ],
            "Resource": "arn:aws:es:${AWS_REGION}:${AWS_ACCOUNT_ID}:domain/${OPENSEARCH_DOMAIN}/*"
        }
    ]
}
EOF
)

# Check if policy exists
EXISTING_POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='${POLICY_NAME}'].Arn" --output text)

if [ -n "${EXISTING_POLICY_ARN}" ]; then
    echo "Policy already exists: ${EXISTING_POLICY_ARN}"
    echo "Updating policy with new permissions (DynamoDB + OpenSearch)..."
    
    # Create a new version of the policy (and set as default)
    aws iam create-policy-version \
        --policy-arn "${EXISTING_POLICY_ARN}" \
        --policy-document "${POLICY_DOCUMENT}" \
        --set-as-default 2>/dev/null || {
        # If we hit the 5-version limit, delete the oldest non-default version first
        echo "Cleaning up old policy versions..."
        OLD_VERSION=$(aws iam list-policy-versions --policy-arn "${EXISTING_POLICY_ARN}" \
            --query "Versions[?IsDefaultVersion==\`false\`].VersionId | [0]" --output text)
        if [ -n "$OLD_VERSION" ] && [ "$OLD_VERSION" != "None" ]; then
            aws iam delete-policy-version --policy-arn "${EXISTING_POLICY_ARN}" --version-id "$OLD_VERSION"
            aws iam create-policy-version \
                --policy-arn "${EXISTING_POLICY_ARN}" \
                --policy-document "${POLICY_DOCUMENT}" \
                --set-as-default
        fi
    }
    
    POLICY_ARN="${EXISTING_POLICY_ARN}"
    echo "Policy updated: ${POLICY_ARN}"
else
    POLICY_ARN=$(aws iam create-policy \
        --policy-name "${POLICY_NAME}" \
        --policy-document "${POLICY_DOCUMENT}" \
        --description "Allows vLLM gateway pods to access DynamoDB and OpenSearch" \
        --query 'Policy.Arn' \
        --output text)
    echo "Policy created: ${POLICY_ARN}"
fi

# Step 3: Create IAM Role with trust policy for EKS
echo ""
echo "Step 3: Creating IAM Role with IRSA..."

# Use eksctl to create the service account and role (simplest approach)
eksctl create iamserviceaccount \
    --name "${SERVICE_ACCOUNT_NAME}" \
    --namespace "${NAMESPACE}" \
    --cluster "${CLUSTER_NAME}" \
    --region "${AWS_REGION}" \
    --attach-policy-arn "${POLICY_ARN}" \
    --approve \
    --override-existing-serviceaccounts

echo ""
echo "=============================================="
echo "IAM Setup Complete!"
echo "=============================================="
echo ""
echo "Created Resources:"
echo "  - IAM Policy: ${POLICY_NAME}"
echo "  - IAM Role: Created by eksctl"
echo "  - ServiceAccount: ${SERVICE_ACCOUNT_NAME} (namespace: ${NAMESPACE})"
echo ""
echo "The gateway deployment should use:"
echo "  serviceAccountName: ${SERVICE_ACCOUNT_NAME}"
echo ""
echo "Next steps:"
echo "  1. Build the gateway Docker image"
echo "  2. Deploy the gateway service"
echo ""
