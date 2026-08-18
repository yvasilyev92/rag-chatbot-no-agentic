# Quick Start Commands

This guide provides the essential commands to deploy and manage your vLLM instance with conversation memory on AWS EKS.

**Default Model:** `meta-llama/Llama-3.1-8B-Instruct` on `g5.xlarge` (~$1/hr)

**What you'll deploy:**

- vLLM server on GPU for LLM inference
- API Gateway for conversation memory + RAG knowledge base
- DynamoDB tables for storing conversations and document metadata
- OpenSearch domain for RAG vector search (~$26/month)
- HTTPS NLB (TLS on 443) for external API access

**Important:** Llama 3.1 is a gated model. Before deploying, you must:

1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Accept the Llama 3.1 Community License Agreement
3. Get your HuggingFace token at https://huggingface.co/settings/tokens

## Prerequisites

Ensure you have installed:

- AWS CLI v2
- Docker (with Docker Desktop running)
- kubectl
- eksctl

---

## Full Deployment from Scratch

Follow these steps in order to deploy everything from a clean slate.

**Deployment Checklist:**

- [ ] Step 1: Configure environment variables
- [ ] Step 2: Configure AWS CLI
- [ ] Step 3: Create AWS resources (DynamoDB + OpenSearch)
- [ ] Step 4: Build and push Docker images
- [ ] Step 5: Create EKS cluster (~15-20 min)
- [ ] Step 6: Setup IAM for DynamoDB + OpenSearch
- [ ] Step 7: Paste OpenSearch endpoint into `.env`
- [ ] Step 8: Deploy to EKS (~5-10 min for model download)
- [ ] Step 9: Test the API + Upload documents

**Estimated total time:** 40-50 minutes (mostly waiting for AWS provisioning)

---

### Step 1: Configure Environment

```bash
# Copy and edit environment file
cp .env.example .env

# Edit .env with your values (see .env.example):
# - AWS_ACCOUNT_ID, AWS_REGION
# - HF_TOKEN, VLLM_API_KEY
# - ACM_CERT_ARN  (required — NLB TLS; same region as the cluster)
# - OPENAI_API_KEY  (optional — input guard + query rewrite; leave empty to skip)
# - DESIRED_RAG_TOPIC  (persona / refusal; deploy.sh injects into the ConfigMap)
nano .env
```

### Step 2: Configure AWS CLI

```bash
aws configure
```

### Step 3: Create AWS Resources

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Create DynamoDB tables
./scripts/setup-dynamodb.sh
./scripts/setup-documents-table.sh

# Create OpenSearch domain for RAG (~15-20 min wait, ~$26/month)
./scripts/setup-opensearch.sh
# When done, paste OPENSEARCH_ENDPOINT into .env
# (deploy.sh injects it into the gateway configmap — do not edit the YAML by hand)
```

> ECR repos for `vllm-server` and `vllm-gateway` are auto-created by
> `build-and-push.sh` on first push, so no separate setup step is required.

### Step 4: Build and Push Docker Images

```bash
# Builds both vLLM and Gateway images
# (automatically creates vllm-gateway ECR repo if needed)
./scripts/build-and-push.sh

# Re-deploying? Only build the gateway if vLLM image already exists:
# ./scripts/build-and-push.sh gateway
# kubectl scale deployment gateway -n vllm --replicas=0
# kubectl scale deployment gateway -n vllm --replicas=2
```

> **ARM Mac users:** `build-and-push.sh` builds with `--platform linux/amd64` for EKS compatibility.

### Step 5: Create EKS Cluster

```bash
# Optional: edit GPU instance type in eks/cluster.yaml
# Region and cluster name come from AWS_REGION / EKS_CLUSTER_NAME in .env
./scripts/create-cluster.sh
```

### Step 6: Setup IAM for DynamoDB + OpenSearch

```bash
./scripts/setup-iam-gateway.sh
```

### Step 7: OpenSearch Endpoint

Paste the domain URL from Step 3 into `.env` (HF_TOKEN, VLLM_API_KEY, and ACM_CERT_ARN should already be there from Step 1):

```bash
OPENSEARCH_ENDPOINT=https://search-vllm-rag-xxxxxxxx.us-east-1.es.amazonaws.com
```

`deploy.sh` injects `.env` into the Kubernetes manifests — do not edit `kubernetes/secret.yaml` or `gateway-configmap.yaml` by hand.

### Step 8: Deploy to EKS

```bash
# Reads .env, renders PLACEHOLDER_* ECR URIs, optionally applies gateway HPA
./scripts/deploy.sh
```

> Use `./scripts/deploy.sh` rather than `kubectl apply -f kubernetes/` directly.
> Committed manifests contain `PLACEHOLDER_*` image URIs that `deploy.sh` substitutes at deploy time.
>
> First deploy installs the [AWS Load Balancer Controller](https://kubernetes-sigs.github.io/aws-load-balancer-controller/) from pinned YAML via `scripts/setup-load-balancer-controller.sh`. Existing clusters also need the EBS CSI driver for the HuggingFace cache PVC:
> `eksctl create addon --name aws-ebs-csi-driver --cluster vllm-cluster --force`

### Step 9: Get API URL, Upload Docs, and Test

```bash
# Get the Gateway LoadBalancer URL
kubectl get svc gateway-service -n vllm

# TLS terminates at the NLB (ACM_CERT_ARN)
export API_URL="https://YOUR_LOADBALANCER_URL"
export API_KEY="your-api-key"  # same as VLLM_API_KEY in .env

# Health check (no auth required)
curl $API_URL/health

# Index the documents of your chosen topic (md / txt / pdf / csv in docs/)
# DESIRED_RAG_TOPIC in .env is injected by deploy.sh (persona / refusal).
./scripts/upload-docs.sh

# Or a single file:
# ./scripts/upload-docs.sh docs/heroes.md

# Test the API
curl -H "Authorization: Bearer $API_KEY" $API_URL/v1/models
```

---

## Using Conversation Memory

### Create a Session

```bash
curl -X POST "$API_URL/v1/sessions" \
  -H "Authorization: Bearer $API_KEY"

# Response: {"session_id": "abc123-...", "created_at": "...", "expires_at": "..."}
```

### Chat with Session

Send only the **new** user turn — history is loaded from DynamoDB. Default `max_tokens` is 300.

```bash
# First message
curl -X POST "$API_URL/v1/sessions/SESSION_ID/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What are relics?"}]
  }'

# Follow-up (query rewriter + history handle context)
curl -X POST "$API_URL/v1/sessions/SESSION_ID/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Which is best for a fire build?"}]
  }'

# Optional: SSE streaming
# Add "stream": true to the JSON body above

# Optional: narrow retrieval by metadata
# Add "rag_filters": {"category": "Equipment"} to the JSON body
```

### Get Session History

```bash
curl -H "Authorization: Bearer $API_KEY" "$API_URL/v1/sessions/SESSION_ID"
```

### Delete Session

```bash
curl -X DELETE -H "Authorization: Bearer $API_KEY" "$API_URL/v1/sessions/SESSION_ID"
```

---

## Using RAG (Document Knowledge Base)

### Bulk Upload Knowledge Base Docs (Recommended)

```bash
# Upload all md / txt / pdf / csv files in docs/
./scripts/upload-docs.sh

# Or a single file
./scripts/upload-docs.sh docs/heroes.md
```

### Upload via API

```bash
# Upload a PDF
curl -X POST "$API_URL/v1/documents" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@/path/to/your/document.pdf"

# Upload a text file
curl -X POST "$API_URL/v1/documents" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@/path/to/your/notes.txt"

# Response: {"document_id": "abc123-...", "status": "processing", ...}
```

### Check Document Status

```bash
# Check if processing is complete
curl -H "Authorization: Bearer $API_KEY" "$API_URL/v1/documents/DOCUMENT_ID"
# Wait for status: "ready"
```

### List All Documents

```bash
curl -H "Authorization: Bearer $API_KEY" "$API_URL/v1/documents"
```

### Ask Questions About Your Documents

```bash
# Chat as normal — full RAG pipeline runs automatically on session chat
curl -X POST "$API_URL/v1/sessions/SESSION_ID/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "What pets pair well with tank heroes?"}]
  }'
```

> After chunking or schema changes: delete the OpenSearch index, restart gateway, re-run `upload-docs.sh`.

### Delete a Document

```bash
curl -X DELETE -H "Authorization: Bearer $API_KEY" "$API_URL/v1/documents/DOCUMENT_ID"
```

---

## Switching Models

### Step 1: Edit ConfigMaps

```bash
nano kubernetes/configmap.yaml
nano kubernetes/gateway-configmap.yaml
```

Change `MODEL_NAME` and, if the new model has a different context window, update **both** in lockstep:

```yaml
# kubernetes/configmap.yaml
MAX_MODEL_LEN: "4096"

# kubernetes/gateway-configmap.yaml
MODEL_CONTEXT_TOKENS: "4096"
```

### Step 2: Redeploy

```bash
./scripts/deploy.sh
# Or restart vLLM only:
kubectl rollout restart deployment/vllm-server -n vllm
kubectl rollout status deployment/vllm-server -n vllm
```

### Popular Model Options

| Model                                | Size | GPU Memory Required        |
| ------------------------------------ | ---- | -------------------------- |
| `meta-llama/Llama-3.1-8B-Instruct`   | 8B   | ~16GB (requires HF token)  |
| `meta-llama/Llama-3.1-70B-Instruct`  | 70B  | ~140GB (requires HF token) |
| `Qwen/Qwen2.5-3B-Instruct`           | 3B   | ~6GB                       |
| `Qwen/Qwen2.5-7B-Instruct`           | 7B   | ~14GB                      |
| `Qwen/Qwen2.5-32B-Instruct`          | 32B  | ~65GB                      |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B   | ~14GB                      |

---

## Switching GPU Instance Types

### Step 1: Edit Cluster Configuration

```bash
nano eks/cluster.yaml
```

Change the `instanceType` under `managedNodeGroups`. The type must also appear in `kubernetes/deployment.yaml` nodeAffinity (vLLM will not schedule otherwise):

```yaml
managedNodeGroups:
  - name: gpu-nodes
    instanceType: g5.xlarge # Change to your desired instance type
```

### Step 2: Delete Old Node Group

```bash
eksctl delete nodegroup --cluster "$EKS_CLUSTER_NAME" --region "$AWS_REGION" --name gpu-nodes --wait
```

### Step 3: Create New Node Group

```bash
./scripts/create-cluster.sh nodegroup --include=gpu-nodes
```

### Step 4: Restart Deployment

```bash
kubectl rollout restart deployment/vllm-server -n vllm
```

### GPU Instance Type Reference

| Instance Type  | GPU     | GPU Memory | Best For         | Cost       |
| -------------- | ------- | ---------- | ---------------- | ---------- |
| `g5.xlarge`    | 1x A10G | 24GB       | Models up to 14B | ~$1.00/hr  |
| `g5.2xlarge`   | 1x A10G | 24GB       | More CPU/RAM     | ~$1.20/hr  |
| `p3.2xlarge`   | 1x V100 | 16GB       | Models up to 7B  | ~$3.00/hr  |
| `p4d.24xlarge` | 8x A100 | 80GB each  | Models up to 70B | ~$32.00/hr |

---

## Useful Commands

### Check Status

```bash
# View all pods
kubectl get pods -n vllm

# View vLLM logs
kubectl logs -f deployment/vllm-server -n vllm

# View Gateway logs
kubectl logs -f deployment/gateway -n vllm

# View Gateway URL
kubectl get svc gateway-service -n vllm
```

### Scaling

```bash
# Scale vLLM replicas (manual; no vLLM HPA — GPU pods are slow to start).
# HF cache PVC is ReadWriteOnce: keep vllm-server at 1 replica unless you add another PVC.
kubectl scale deployment vllm-server -n vllm --replicas=1

# Scale Gateway replicas (default is 2; avoid surge above 2 on small CPU nodes)
kubectl scale deployment gateway -n vllm --replicas=2

# Gateway HPA (2–3 replicas, matches cpu-nodes maxSize) — deploy.sh prompts, or:
kubectl apply -f kubernetes/gateway-hpa.yaml
```

### Local Testing

```bash
# Against EKS (port-forward). Service port is 443; the pod speaks HTTP, so
# use http://localhost:8080 (TLS exists only at the NLB, which this bypasses).
kubectl port-forward svc/gateway-service 8080:443 -n vllm

# Fully local (requires NVIDIA GPU + .env configured)
docker compose up
```

---

## Cleanup

```bash
# Interactive cleanup (recommended)
./scripts/cleanup.sh
# Say yes to cluster; no to DynamoDB/OpenSearch/ECR/IAM to preserve state

# Or manual cleanup (avoid kubectl delete -f kubernetes/ — manifests use PLACEHOLDER_* URIs):
eksctl delete cluster --name vllm-cluster
aws dynamodb delete-table --table-name vllm-conversations
aws dynamodb delete-table --table-name vllm-documents
aws opensearch delete-domain --domain-name vllm-rag
aws ecr delete-repository --repository-name vllm-server --force
aws ecr delete-repository --repository-name vllm-gateway --force
```
