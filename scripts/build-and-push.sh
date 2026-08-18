#!/bin/bash
# ============================================
# Build and Push Docker Images to ECR
# ============================================
# This script builds and pushes both:
#   1. vLLM server image (Dockerfile)
#   2. API Gateway image (gateway/Dockerfile)
#
# Usage:
#   ./scripts/build-and-push.sh          # Build both images
#   ./scripts/build-and-push.sh vllm     # Build only vLLM image
#   ./scripts/build-and-push.sh gateway  # Build only gateway image

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
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BUILD_TARGET="${1:-all}"  # all, vllm, or gateway

# Repository names
VLLM_REPO_NAME="${ECR_REPOSITORY_NAME:-vllm-server}"
GATEWAY_REPO_NAME="vllm-gateway"

# Validate required variables
if [ -z "$AWS_ACCOUNT_ID" ] || [ "$AWS_ACCOUNT_ID" = "PLACEHOLDER_AWS_ACCOUNT_ID" ]; then
    echo "ERROR: AWS_ACCOUNT_ID is not set"
    echo "Copy .env.example to .env and set AWS_ACCOUNT_ID"
    exit 1
fi

# ECR URI
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "============================================"
echo "Building and Pushing Docker Images"
echo "============================================"
echo "AWS Account ID: $AWS_ACCOUNT_ID"
echo "AWS Region: $AWS_REGION"
echo "Build Target: $BUILD_TARGET"
echo "Tag: $IMAGE_TAG"
echo "============================================"

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon is not running"
    exit 1
fi

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_URI"

# Function to build and push an image
build_and_push() {
    local name=$1
    local dockerfile=$2
    local context=$3
    local repo=$4
    
    local full_uri="$ECR_URI/$repo:$IMAGE_TAG"
    
    echo ""
    echo "--------------------------------------------"
    echo "Building $name..."
    echo "  Dockerfile: $dockerfile"
    echo "  Context: $context"
    echo "  Repository: $repo"
    echo "--------------------------------------------"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names "$repo" --region "$AWS_REGION" 2>/dev/null || \
        aws ecr create-repository --repository-name "$repo" --region "$AWS_REGION"
    
    # Build the image
    docker build --platform linux/amd64 -f "$dockerfile" -t "$repo:$IMAGE_TAG" "$context"
    
    # Tag and push
    docker tag "$repo:$IMAGE_TAG" "$full_uri"
    docker push "$full_uri"
    
    echo "$name image pushed: $full_uri"
}

# Build vLLM image
if [ "$BUILD_TARGET" = "all" ] || [ "$BUILD_TARGET" = "vllm" ]; then
    build_and_push "vLLM Server" "Dockerfile" "." "$VLLM_REPO_NAME"
fi

# Build Gateway image
if [ "$BUILD_TARGET" = "all" ] || [ "$BUILD_TARGET" = "gateway" ]; then
    build_and_push "API Gateway" "gateway/Dockerfile" "gateway" "$GATEWAY_REPO_NAME"
fi

echo ""
echo "============================================"
echo "Docker Image Build Complete!"
echo "============================================"
if [ "$BUILD_TARGET" = "all" ] || [ "$BUILD_TARGET" = "vllm" ]; then
    echo "vLLM Server:  $ECR_URI/$VLLM_REPO_NAME:$IMAGE_TAG"
fi
if [ "$BUILD_TARGET" = "all" ] || [ "$BUILD_TARGET" = "gateway" ]; then
    echo "API Gateway:  $ECR_URI/$GATEWAY_REPO_NAME:$IMAGE_TAG"
fi
echo ""
echo "Next steps:"
echo "1. Create EKS cluster: eksctl create cluster -f eks/cluster.yaml"
echo "2. Setup DynamoDB: ./scripts/setup-dynamodb.sh && ./scripts/setup-documents-table.sh"
echo "3. Setup OpenSearch: ./scripts/setup-opensearch.sh"
echo "4. Setup IAM for gateway: ./scripts/setup-iam-gateway.sh"
echo "5. Deploy to Kubernetes: ./scripts/deploy.sh"
echo "============================================"
