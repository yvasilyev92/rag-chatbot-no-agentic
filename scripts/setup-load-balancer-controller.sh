#!/bin/bash
# =============================================================================
# AWS Load Balancer Controller (NLB TLS on gateway-service)
# =============================================================================
# Installs the controller in kube-system from pinned upstream YAML (not Helm).
# gateway-service annotations (internet-facing NLB, ACM cert on 443) require
# this controller; the in-tree cloud provider will leave the LoadBalancer
# <pending>.
#
# Idempotent: safe to re-run. deploy.sh calls this if the Deployment is
# missing.
#
# Prerequisites:
#   - EKS cluster (eksctl create cluster -f eks/cluster.yaml)
#   - eksctl, kubectl, curl, python3, AWS CLI
#
# Usage:
#   ./scripts/setup-load-balancer-controller.sh
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

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
if [ -z "$AWS_ACCOUNT_ID" ] || [ "$AWS_ACCOUNT_ID" = "PLACEHOLDER_AWS_ACCOUNT_ID" ]; then
    AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
fi
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-vllm-cluster}"
SA_NAMESPACE="kube-system"
SA_NAME="aws-load-balancer-controller"
POLICY_NAME="AWSLoadBalancerControllerIAMPolicy"

# Pin to the version AWS documents for the YAML install path.
LBC_VERSION="v2.14.1"
CERT_MANAGER_VERSION="v1.13.5"
POLICY_URL="https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/${LBC_VERSION}/docs/install/iam_policy.json"
LBC_FULL_URL="https://github.com/kubernetes-sigs/aws-load-balancer-controller/releases/download/${LBC_VERSION}/v2_14_1_full.yaml"
LBC_INGCLASS_URL="https://github.com/kubernetes-sigs/aws-load-balancer-controller/releases/download/${LBC_VERSION}/v2_14_1_ingclass.yaml"
CERT_MANAGER_URL="https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml"

for cmd in eksctl kubectl curl python3 aws; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd is not installed"
        echo "AWS Load Balancer Controller YAML install needs eksctl, kubectl, curl, python3, and AWS CLI."
        exit 1
    fi
done

echo "=============================================="
echo "AWS Load Balancer Controller (${LBC_VERSION}, YAML)"
echo "=============================================="
echo "Cluster: ${EKS_CLUSTER_NAME}"
echo "Region: ${AWS_REGION}"
echo ""

aws eks update-kubeconfig --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION" >/dev/null

POLICY_ARN=$(aws iam list-policies --query "Policies[?PolicyName=='${POLICY_NAME}'].Arn" --output text)
if [ -z "$POLICY_ARN" ] || [ "$POLICY_ARN" = "None" ]; then
    echo "Creating IAM policy ${POLICY_NAME}..."
    tmp_policy="$(mktemp)"
    curl -fsSL "$POLICY_URL" -o "$tmp_policy"
    POLICY_ARN=$(aws iam create-policy \
        --policy-name "$POLICY_NAME" \
        --policy-document "file://${tmp_policy}" \
        --query 'Policy.Arn' \
        --output text)
    rm -f "$tmp_policy"
    echo "Policy created: ${POLICY_ARN}"
else
    echo "IAM policy already exists: ${POLICY_ARN}"
fi

echo "Ensuring IRSA ServiceAccount ${SA_NAME}..."
eksctl create iamserviceaccount \
    --cluster "$EKS_CLUSTER_NAME" \
    --region "$AWS_REGION" \
    --namespace "$SA_NAMESPACE" \
    --name "$SA_NAME" \
    --attach-policy-arn "$POLICY_ARN" \
    --approve \
    --override-existing-serviceaccounts

echo "Installing cert-manager ${CERT_MANAGER_VERSION} (webhook certs)..."
kubectl apply --validate=false -f "$CERT_MANAGER_URL"
kubectl wait --for=condition=Available deployment/cert-manager-webhook \
    -n cert-manager --timeout=180s

echo "Downloading controller manifests ${LBC_VERSION}..."
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
curl -fsSL "$LBC_FULL_URL" -o "$tmp_dir/lbc.yaml"
curl -fsSL "$LBC_INGCLASS_URL" -o "$tmp_dir/ingclass.yaml"

# eksctl already created the ServiceAccount with the IRSA annotation.
# The upstream full.yaml includes an un-annotated SA that would overwrite it.
python3 - "$tmp_dir/lbc.yaml" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
docs = path.read_text().split("\n---\n")
kept = []
for doc in docs:
    if "kind: ServiceAccount" in doc and "name: aws-load-balancer-controller" in doc:
        continue
    kept.append(doc)
path.write_text("\n---\n".join(kept))
PY

# Portable in-place replace (macOS / GNU sed).
python3 - "$tmp_dir/lbc.yaml" "$EKS_CLUSTER_NAME" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
cluster = sys.argv[2]
path.write_text(path.read_text().replace("your-cluster-name", cluster))
PY

echo "Applying controller + IngressClass..."
kubectl apply -f "$tmp_dir/lbc.yaml"
kubectl apply -f "$tmp_dir/ingclass.yaml"

echo "Waiting for controller to be ready..."
kubectl rollout status deployment/aws-load-balancer-controller -n kube-system --timeout=180s

echo ""
echo "=============================================="
echo "AWS Load Balancer Controller ready"
echo "=============================================="
