#!/bin/bash
# =============================================================================
# Create (or update a node group on) the EKS cluster
# =============================================================================
# Renders eks/cluster.yaml with AWS_REGION from .env, then calls eksctl.
# The committed file stays account-agnostic (PLACEHOLDER_AWS_REGION).
#
# Usage:
#   ./scripts/create-cluster.sh
#       eksctl create cluster (15-20 min)
#   ./scripts/create-cluster.sh nodegroup --include=gpu-nodes
#       eksctl create nodegroup -f <rendered> --include=gpu-nodes
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
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-vllm-cluster}"

if [ -z "$AWS_REGION" ] || [ "$AWS_REGION" = "PLACEHOLDER_AWS_REGION" ]; then
    echo "ERROR: AWS_REGION is not set"
    echo "Copy .env.example to .env and set AWS_REGION"
    exit 1
fi

if ! command -v eksctl &>/dev/null; then
    echo "ERROR: eksctl is not installed"
    echo "Install it from: https://eksctl.io"
    exit 1
fi

RENDERED="$(mktemp -t vllm-cluster-XXXXXX.yaml)"
trap 'rm -f "$RENDERED"' EXIT
cp eks/cluster.yaml "$RENDERED"

# Portable in-place replace (no sed -i).
python3 - "$RENDERED" "$AWS_REGION" "$EKS_CLUSTER_NAME" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
region = sys.argv[2]
cluster = sys.argv[3]
text = path.read_text()
text = text.replace("PLACEHOLDER_AWS_REGION", region)
text = text.replace("name: vllm-cluster", f"name: {cluster}", 1)
path.write_text(text)
PY

echo "=============================================="
echo "EKS cluster config"
echo "=============================================="
echo "Cluster: $EKS_CLUSTER_NAME"
echo "Region:  $AWS_REGION"
echo "=============================================="

if [ "${1:-}" = "nodegroup" ]; then
    shift
    echo "Running: eksctl create nodegroup -f <rendered> $*"
    eksctl create nodegroup -f "$RENDERED" "$@"
else
    echo "Creating cluster (15-20 minutes)..."
    eksctl create cluster -f "$RENDERED"
fi
