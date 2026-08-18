#!/bin/bash
# ============================================
# Deploy vLLM + Gateway to EKS
# ============================================
# This script deploys all Kubernetes resources to your EKS cluster.
#
# Architecture:
#   Client → Gateway (LoadBalancer) → vLLM (internal ClusterIP)
#                ↓                ↓
#            DynamoDB          OpenSearch (RAG vector search)
#
# What this script does:
#   1. Connects kubectl to your EKS cluster (updates kubeconfig)
#   2. Installs NVIDIA device plugin for GPU support
#   3. Installs AWS Load Balancer Controller if missing (NLB TLS)
#   4. Renders the manifests in kubernetes/ to a tmp dir, substituting your
#      ECR account/region, secrets, OpenSearch endpoint, and DESIRED_RAG_TOPIC
#      from .env (the committed manifests stay account-agnostic).
#   5. Applies Kubernetes manifests in order:
#      - namespace.yaml          (creates 'vllm' namespace)
#      - configmap.yaml          (vLLM model configuration)
#      - gateway-configmap.yaml  (gateway + RAG configuration)
#      - secret.yaml             (HF token + API keys from .env)
#      - vllm-hf-cache-pvc.yaml  (EBS volume for HuggingFace model cache)
#      - deployment.yaml         (vLLM pod with GPU)
#      - service.yaml            (ClusterIP - internal only)
#      - gateway-deployment.yaml (gateway pods, CPU)
#      - gateway-service.yaml    (LoadBalancer - external access)
#   6. Optionally enables the Gateway Horizontal Pod Autoscaler
#   7. Waits for pods to be ready
#   8. Retrieves the Gateway LoadBalancer URL
#
# Prerequisites:
#   - EKS cluster created: eksctl create cluster -f eks/cluster.yaml
#   - DynamoDB tables created: ./scripts/setup-dynamodb.sh + ./scripts/setup-documents-table.sh
#   - OpenSearch domain created: ./scripts/setup-opensearch.sh
#     (set OPENSEARCH_ENDPOINT in .env after setup-opensearch.sh)
#   - IAM role configured: ./scripts/setup-iam-gateway.sh
#   - Docker images pushed: ./scripts/build-and-push.sh
#   - Secrets in .env: HF_TOKEN, VLLM_API_KEY, ACM_CERT_ARN (required);
#     OPENAI_API_KEY (optional — input guard + query rewrite)
#     DESIRED_RAG_TOPIC (optional — persona / canned refusal)
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Load environment variables from .env (repo root)
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Escape a string for use as a sed replacement (delimiter is |)
escape_sed_repl() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//&/\\&}"
    s="${s//|/\\|}"
    printf '%s' "$s"
}

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-vllm-cluster}"
VLLM_REPO_NAME="${ECR_REPOSITORY_NAME:-vllm-server}"
GATEWAY_REPO_NAME="vllm-gateway"
IMAGE_TAG="${IMAGE_TAG:-latest}"
HF_TOKEN="${HF_TOKEN:-}"
VLLM_API_KEY="${VLLM_API_KEY:-}"
OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-}"
ACM_CERT_ARN="${ACM_CERT_ARN:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
DESIRED_RAG_TOPIC="${DESIRED_RAG_TOPIC:-Desired RAG Topic}"

# Validate required variables
if [ -z "$AWS_ACCOUNT_ID" ] || [ "$AWS_ACCOUNT_ID" = "PLACEHOLDER_AWS_ACCOUNT_ID" ]; then
    echo "ERROR: AWS_ACCOUNT_ID is not set"
    echo "Copy .env.example to .env and set AWS_ACCOUNT_ID"
    exit 1
fi

if [ -z "$VLLM_API_KEY" ] || [ "$VLLM_API_KEY" = "PLACEHOLDER_API_KEY" ] || [ "$VLLM_API_KEY" = "PLACEHOLDER_VLLM_API_KEY" ]; then
    echo "ERROR: VLLM_API_KEY is not set"
    echo "Copy .env.example to .env and set VLLM_API_KEY (e.g. openssl rand -hex 32)"
    exit 1
fi

if [ -z "$ACM_CERT_ARN" ] || [ "$ACM_CERT_ARN" = "PLACEHOLDER_ACM_CERT_ARN" ]; then
    echo "ERROR: ACM_CERT_ARN is not set"
    echo "Request a public ACM certificate for your API hostname (same region as EKS),"
    echo "then set ACM_CERT_ARN in .env (e.g. arn:aws:acm:us-east-1:123456789012:certificate/abc-...)"
    exit 1
fi

ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
VLLM_IMAGE_URI="$ECR_URI/$VLLM_REPO_NAME:$IMAGE_TAG"
GATEWAY_IMAGE_URI="$ECR_URI/$GATEWAY_REPO_NAME:$IMAGE_TAG"

echo "============================================"
echo "Deploying vLLM + Gateway to EKS"
echo "============================================"
echo "Cluster: $EKS_CLUSTER_NAME"
echo "Region: $AWS_REGION"
echo "vLLM Image: $VLLM_IMAGE_URI"
echo "Gateway Image: $GATEWAY_IMAGE_URI"
echo "============================================"

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "ERROR: kubectl is not installed"
    echo "Install it from: https://kubernetes.io/docs/tasks/tools/"
    exit 1
fi

# Update kubeconfig for EKS
echo "Updating kubeconfig for EKS cluster..."
aws eks update-kubeconfig --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION"

# Verify cluster connection
echo "Verifying cluster connection..."
kubectl cluster-info

# Install NVIDIA device plugin (if not already installed)
echo "Installing NVIDIA device plugin..."
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# NLB TLS annotations on gateway-service need the AWS Load Balancer Controller.
if kubectl get deployment aws-load-balancer-controller -n kube-system &>/dev/null; then
    echo "AWS Load Balancer Controller already installed."
else
    echo "AWS Load Balancer Controller not found; installing..."
    "$SCRIPT_DIR/setup-load-balancer-controller.sh"
fi

# Update deployment files with actual image URIs.
#
# We render to a tmp directory rather than rewriting the committed YAML files,
# so the repo always shows PLACEHOLDER_* (account-agnostic) and we never leave
# .bak files behind from sed's mac/BSD vs GNU portability quirks.
RENDERED_DIR="$(mktemp -d -t vllm-deploy-XXXXXX)"
trap 'rm -rf "$RENDERED_DIR"' EXIT
echo "Rendering manifests to $RENDERED_DIR ..."

# Copy every k8s manifest verbatim, then patch image URIs and region.
cp kubernetes/*.yaml "$RENDERED_DIR/"

sed_inplace() {
    # Portable in-place sed (no .bak siblings on macOS). $1=expr, $2=file
    local tmp
    tmp="$(mktemp)"
    sed "$1" "$2" >"$tmp" && mv "$tmp" "$2"
}

VLLM_IMAGE_URI_ESC="$(escape_sed_repl "$VLLM_IMAGE_URI")"
GATEWAY_IMAGE_URI_ESC="$(escape_sed_repl "$GATEWAY_IMAGE_URI")"
sed_inplace "s|PLACEHOLDER_AWS_ACCOUNT_ID.dkr.ecr.PLACEHOLDER_REGION.amazonaws.com/vllm-server:latest|${VLLM_IMAGE_URI_ESC}|g" "$RENDERED_DIR/deployment.yaml"
sed_inplace "s|PLACEHOLDER_AWS_ACCOUNT_ID.dkr.ecr.PLACEHOLDER_REGION.amazonaws.com/vllm-gateway:latest|${GATEWAY_IMAGE_URI_ESC}|g" "$RENDERED_DIR/gateway-deployment.yaml"
sed_inplace "s|AWS_REGION: \"us-east-1\"|AWS_REGION: \"$AWS_REGION\"|g" "$RENDERED_DIR/gateway-configmap.yaml"

HF_TOKEN_ESC="$(escape_sed_repl "$HF_TOKEN")"
VLLM_API_KEY_ESC="$(escape_sed_repl "$VLLM_API_KEY")"
OPENSEARCH_ENDPOINT_ESC="$(escape_sed_repl "$OPENSEARCH_ENDPOINT")"
ACM_CERT_ARN_ESC="$(escape_sed_repl "$ACM_CERT_ARN")"
sed_inplace "s|PLACEHOLDER_HF_TOKEN|${HF_TOKEN_ESC}|g" "$RENDERED_DIR/secret.yaml"
sed_inplace "s|PLACEHOLDER_VLLM_API_KEY|${VLLM_API_KEY_ESC}|g" "$RENDERED_DIR/secret.yaml"
sed_inplace "s|PLACEHOLDER_OPENSEARCH_ENDPOINT|${OPENSEARCH_ENDPOINT_ESC}|g" "$RENDERED_DIR/gateway-configmap.yaml"
sed_inplace "s|PLACEHOLDER_ACM_CERT_ARN|${ACM_CERT_ARN_ESC}|g" "$RENDERED_DIR/gateway-service.yaml"
OPENAI_API_KEY_ESC="$(escape_sed_repl "$OPENAI_API_KEY")"
sed_inplace "s|PLACEHOLDER_OPENAI_API_KEY|${OPENAI_API_KEY_ESC}|g" "$RENDERED_DIR/secret.yaml"
DESIRED_RAG_TOPIC_ESC="$(escape_sed_repl "$DESIRED_RAG_TOPIC")"
sed_inplace "s|PLACEHOLDER_DESIRED_RAG_TOPIC|${DESIRED_RAG_TOPIC_ESC}|g" "$RENDERED_DIR/gateway-configmap.yaml"

# Apply Kubernetes manifests in order
echo ""
echo "Applying Kubernetes manifests..."

echo "1. Creating namespace..."
kubectl apply -f "$RENDERED_DIR/namespace.yaml"

if ! kubectl get serviceaccount gateway-sa -n vllm &>/dev/null; then
    echo "ERROR: ServiceAccount gateway-sa not found in namespace vllm."
    echo "Run ./scripts/setup-iam-gateway.sh before deploy (IRSA for DynamoDB + OpenSearch)."
    exit 1
fi

echo "2. Creating ConfigMaps..."
kubectl apply -f "$RENDERED_DIR/configmap.yaml"
kubectl apply -f "$RENDERED_DIR/gateway-configmap.yaml"

echo "3. Creating Secret..."
kubectl apply -f "$RENDERED_DIR/secret.yaml"

echo "4. Creating HuggingFace cache PVC..."
if ! kubectl get csidriver ebs.csi.aws.com &>/dev/null; then
    echo "ERROR: EBS CSI driver not found (required for kubernetes/vllm-hf-cache-pvc.yaml)."
    echo "New clusters get it from eks/cluster.yaml. Existing clusters:"
    echo "  eksctl create addon --name aws-ebs-csi-driver --cluster $EKS_CLUSTER_NAME --force"
    exit 1
fi
kubectl apply -f "$RENDERED_DIR/vllm-hf-cache-pvc.yaml"

echo "5. Creating vLLM Deployment..."
kubectl apply -f "$RENDERED_DIR/deployment.yaml"

echo "6. Creating vLLM Service (internal)..."
kubectl apply -f "$RENDERED_DIR/service.yaml"

echo "7. Creating Gateway Deployment..."
kubectl apply -f "$RENDERED_DIR/gateway-deployment.yaml"

echo "8. Creating Gateway Service (external)..."
kubectl apply -f "$RENDERED_DIR/gateway-service.yaml"

# Optionally apply Gateway HPA
# (No vLLM HPA: GPU pods are slow to start and CPU-based scaling is meaningless on them.)
read -p "Do you want to enable the Gateway Horizontal Pod Autoscaler? (y/n): " enable_hpa
if [ "$enable_hpa" = "y" ]; then
    echo "9. Creating Gateway HPA..."
    kubectl apply -f "$RENDERED_DIR/gateway-hpa.yaml"
fi

echo ""
echo "============================================"
echo "Deployment Initiated!"
echo "============================================"
echo ""
echo "Waiting for vLLM to be ready (this may take 5-10 minutes for model download)..."
echo ""

# Wait for vLLM deployment to be ready (with timeout)
echo "Checking vLLM deployment status..."
kubectl rollout status deployment/vllm-server -n vllm --timeout=600s || {
    echo "vLLM deployment is still in progress. Check status with:"
    echo "  kubectl get pods -n vllm"
    echo "  kubectl logs -f deployment/vllm-server -n vllm"
}

# Wait for Gateway deployment
echo "Checking Gateway deployment status..."
kubectl rollout status deployment/gateway -n vllm --timeout=120s || {
    echo "Gateway deployment is still in progress."
}

# Get the Gateway LoadBalancer URL
echo ""
echo "Getting Gateway LoadBalancer URL..."
sleep 10  # Give AWS time to provision the LoadBalancer

LB_URL=$(kubectl get svc gateway-service -n vllm -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "pending")

echo ""
echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo ""
if [ "$LB_URL" = "pending" ] || [ -z "$LB_URL" ]; then
    echo "LoadBalancer is still being provisioned."
    echo "Run this command to get the URL when ready:"
    echo "  kubectl get svc gateway-service -n vllm"
else
    echo "API Gateway Endpoint: https://$LB_URL"
    echo ""
    echo "Test the API:"
    echo ""
    echo "1. List models:"
    echo "   curl -H 'Authorization: Bearer YOUR_API_KEY' https://$LB_URL/v1/models"
    echo ""
    echo "2. Create a session:"
    echo "   curl -X POST -H 'Authorization: Bearer YOUR_API_KEY' https://$LB_URL/v1/sessions"
    echo ""
    echo "3. Chat with session context:"
    echo "   curl -X POST https://$LB_URL/v1/sessions/SESSION_ID/chat/completions \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -H 'Authorization: Bearer YOUR_API_KEY' \\"
    echo "     -d '{\"model\": \"meta-llama/Llama-3.1-8B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
    echo ""
    echo "4. Get session history:"
    echo "   curl -H 'Authorization: Bearer YOUR_API_KEY' https://$LB_URL/v1/sessions/SESSION_ID"
    echo ""
    echo "5. Upload a document (RAG):"
    echo "   curl -X POST https://$LB_URL/v1/documents \\"
    echo "     -H 'Authorization: Bearer YOUR_API_KEY' \\"
    echo "     -F 'file=@/path/to/your/document.pdf'"
    echo ""
    echo "6. List documents:"
    echo "   curl -H 'Authorization: Bearer YOUR_API_KEY' https://$LB_URL/v1/documents"
fi
echo ""
echo "Useful commands:"
echo "  kubectl get pods -n vllm                     # Check all pod status"
echo "  kubectl logs -f deployment/vllm-server -n vllm   # vLLM logs"
echo "  kubectl logs -f deployment/gateway -n vllm       # Gateway logs"
echo "  kubectl describe pod -n vllm                 # Debug issues"
echo ""
echo "API Documentation: https://$LB_URL/docs"
echo "============================================"
