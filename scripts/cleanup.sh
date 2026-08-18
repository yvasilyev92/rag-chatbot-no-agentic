#!/bin/bash
# ============================================
# Cleanup - Delete EKS Cluster and Resources
# ============================================
# This script deletes the EKS cluster and all associated resources
# to stop incurring AWS charges.
#
# WARNING: This is destructive and cannot be undone!

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
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-vllm-cluster}"
K8S_NAMESPACE="${K8S_NAMESPACE:-vllm}"
VLLM_REPO_NAME="${ECR_REPOSITORY_NAME:-vllm-server}"
GATEWAY_REPO_NAME="vllm-gateway"
DYNAMODB_TABLE="${DYNAMODB_TABLE:-vllm-conversations}"
DOCUMENTS_TABLE="vllm-documents"
OPENSEARCH_DOMAIN="${OPENSEARCH_DOMAIN_NAME:-vllm-rag}"
IAM_POLICY_NAME="vllm-gateway-dynamodb-policy"
LBC_POLICY_NAME="AWSLoadBalancerControllerIAMPolicy"

echo "============================================"
echo "EKS Cluster Cleanup"
echo "============================================"
echo "Cluster: $EKS_CLUSTER_NAME"
echo "Region: $AWS_REGION"
echo "Kubernetes namespace: $K8S_NAMESPACE"
echo "DynamoDB tables: $DYNAMODB_TABLE, $DOCUMENTS_TABLE"
echo "============================================"
echo ""
echo "WARNING: This will delete:"
echo "  - EKS cluster and all node groups"
echo "  - Associated VPC, subnets, and networking"
echo "  - All deployed pods and services"
echo "  - IAM service account roles (created by eksctl)"
echo ""
echo "Optional deletions (will ask separately):"
echo "  - DynamoDB tables: $DYNAMODB_TABLE, $DOCUMENTS_TABLE"
echo "  - OpenSearch domain: $OPENSEARCH_DOMAIN"
echo "  - ECR repositories"
echo "  - IAM policies: $IAM_POLICY_NAME, $LBC_POLICY_NAME"
echo ""
echo "This action CANNOT be undone!"
echo ""

read -p "Are you sure you want to delete the cluster? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting cleanup..."

# Delete in-cluster resources ONLY from the intended EKS cluster and namespace.
# Never run `kubectl delete -f kubernetes/` — that uses whatever kubeconfig
# context is active and can hit the wrong cluster. We also avoid applying
# committed PLACEHOLDER_* manifests; namespace deletion is scoped and safe.
echo "Configuring kubectl for cluster: $EKS_CLUSTER_NAME ..."
aws eks update-kubeconfig --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" >/dev/null

CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "")
if [[ "$CURRENT_CONTEXT" != *"$EKS_CLUSTER_NAME"* ]]; then
    echo "ERROR: kubectl context '$CURRENT_CONTEXT' does not match cluster '$EKS_CLUSTER_NAME'."
    echo "Refusing to delete Kubernetes resources to avoid hitting the wrong cluster."
    exit 1
fi

echo "Deleting namespace '$K8S_NAMESPACE' in cluster '$EKS_CLUSTER_NAME' ..."
kubectl delete namespace "$K8S_NAMESPACE" --ignore-not-found=true --timeout=180s || true

# Delete IAM service account (this also deletes the associated IAM role)
echo ""
echo "Deleting IAM service account..."
eksctl delete iamserviceaccount \
    --name gateway-sa \
    --namespace "$K8S_NAMESPACE" \
    --cluster "$EKS_CLUSTER_NAME" \
    --region "$AWS_REGION" \
    --wait 2>/dev/null || true

# Delete the EKS cluster
echo ""
echo "Deleting EKS cluster (this takes ~10 minutes)..."
eksctl delete cluster --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" --wait

echo ""
echo "============================================"
echo "Cluster Deleted Successfully!"
echo "============================================"
echo ""
echo "The following resources were deleted:"
echo "  - EKS cluster: $EKS_CLUSTER_NAME"
echo "  - All node groups"
echo "  - VPC and networking components"
echo "  - IAM service account and role"
echo ""

# Ask about DynamoDB tables
echo ""
read -p "Do you want to delete the DynamoDB tables ($DYNAMODB_TABLE, $DOCUMENTS_TABLE)? (yes/no): " delete_dynamo
if [ "$delete_dynamo" = "yes" ]; then
    echo "Deleting DynamoDB tables..."
    aws dynamodb delete-table --table-name "$DYNAMODB_TABLE" --region "$AWS_REGION" 2>/dev/null || echo "Table $DYNAMODB_TABLE not found or already deleted"
    aws dynamodb delete-table --table-name "$DOCUMENTS_TABLE" --region "$AWS_REGION" 2>/dev/null || echo "Table $DOCUMENTS_TABLE not found or already deleted"
    echo "DynamoDB tables deleted."
fi

# Ask about OpenSearch domain
echo ""
read -p "Do you want to delete the OpenSearch domain ($OPENSEARCH_DOMAIN)? (~\$26/month) (yes/no): " delete_opensearch
if [ "$delete_opensearch" = "yes" ]; then
    echo "Deleting OpenSearch domain (this takes a few minutes)..."
    aws opensearch delete-domain --domain-name "$OPENSEARCH_DOMAIN" --region "$AWS_REGION" 2>/dev/null || echo "Domain not found or already deleted"
    echo "OpenSearch domain deletion initiated."
fi

# Ask about ECR repositories
echo ""
read -p "Do you want to delete ECR repositories and images? (yes/no): " delete_ecr
if [ "$delete_ecr" = "yes" ]; then
    echo "Deleting ECR repositories..."
    aws ecr delete-repository --repository-name "$VLLM_REPO_NAME" --force --region "$AWS_REGION" 2>/dev/null || echo "Repository $VLLM_REPO_NAME not found"
    aws ecr delete-repository --repository-name "$GATEWAY_REPO_NAME" --force --region "$AWS_REGION" 2>/dev/null || echo "Repository $GATEWAY_REPO_NAME not found"
    echo "ECR repositories deleted."
fi

# Delete IAM policies
echo ""
read -p "Do you want to delete IAM policies ($IAM_POLICY_NAME, $LBC_POLICY_NAME)? (yes/no): " delete_policy
if [ "$delete_policy" = "yes" ]; then
    for pname in "$IAM_POLICY_NAME" "$LBC_POLICY_NAME"; do
        POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='$pname'].Arn" --output text)
        if [ -n "$POLICY_ARN" ] && [ "$POLICY_ARN" != "None" ]; then
            echo "Deleting IAM policy $pname..."
            aws iam delete-policy --policy-arn "$POLICY_ARN" 2>/dev/null || echo "Could not delete $pname (detach versions/roles first)"
        else
            echo "IAM policy $pname not found."
        fi
    done
fi

echo ""
echo "============================================"
echo "Cleanup Complete!"
echo "============================================"
echo ""
echo "All major AWS resources have been cleaned up."
echo "Check your AWS console to verify no resources remain."
echo "============================================"