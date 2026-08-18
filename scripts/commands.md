# Useful Commands Reference

Quick reference for managing your vLLM + Gateway deployment on EKS.

## Pod Management

```bash
# View all pods in vllm namespace
kubectl get pods -n vllm

# Watch pods in real-time (updates automatically)
kubectl get pods -n vllm -w

# View detailed pod information
kubectl describe pod -n vllm

# View vLLM logs (follow mode)
kubectl logs -f deployment/vllm-server -n vllm

# View Gateway logs (follow mode)
kubectl logs -f deployment/gateway -n vllm

# View last 100 lines of logs
kubectl logs --tail=100 deployment/vllm-server -n vllm
kubectl logs --tail=100 deployment/gateway -n vllm

# Execute shell inside vLLM pod (for debugging)
kubectl exec -it deployment/vllm-server -n vllm -- /bin/bash

# Execute shell inside Gateway pod
kubectl exec -it deployment/gateway -n vllm -- /bin/bash
```

## Service & Networking

```bash
# Get Gateway LoadBalancer URL (external access point)
kubectl get svc gateway-service -n vllm

# Get just the hostname
kubectl get svc gateway-service -n vllm -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# View internal vLLM service (ClusterIP)
kubectl get svc vllm-service -n vllm

# Port forward Gateway for local testing (NLB listens on 443)
kubectl port-forward svc/gateway-service 8080:443 -n vllm

# Port forward vLLM directly (bypass gateway, for debugging)
kubectl port-forward svc/vllm-service 8000:80 -n vllm
```

## Scaling

```bash
# Scale vLLM to 2 replicas (requires 2 GPU nodes AND a second HF-cache PVC;
# the shipped vllm-hf-cache PVC is ReadWriteOnce — keep 1 replica by default)
kubectl scale deployment vllm-server -n vllm --replicas=2

# Scale Gateway to 3 replicas
kubectl scale deployment gateway -n vllm --replicas=3

# Scale vLLM to 0 (stop pods but keep cluster)
kubectl scale deployment vllm-server -n vllm --replicas=0

# Scale back to 1
kubectl scale deployment vllm-server -n vllm --replicas=1
kubectl scale deployment gateway -n vllm --replicas=2
```

## Deployment Management

```bash
# Restart deployment (re-pulls image, restarts pods)
kubectl rollout restart deployment/vllm-server -n vllm

# Check rollout status
kubectl rollout status deployment/vllm-server -n vllm

# View deployment details
kubectl describe deployment vllm-server -n vllm
```

## Configuration

```bash
# View vLLM ConfigMap
kubectl get configmap vllm-config -n vllm -o yaml

# View Gateway ConfigMap
kubectl get configmap gateway-config -n vllm -o yaml

# View Secrets (base64 encoded)
kubectl get secret vllm-secrets -n vllm -o yaml

# Edit vLLM ConfigMap directly
kubectl edit configmap vllm-config -n vllm

# Edit Gateway ConfigMap
kubectl edit configmap gateway-config -n vllm

# Re-render placeholders from .env and apply. Do not `kubectl apply -f kubernetes/`
# directly — committed manifests contain PLACEHOLDER_* values.
./scripts/deploy.sh

# View ServiceAccount (for IAM role)
kubectl get serviceaccount gateway-sa -n vllm -o yaml
```

## Cluster & Nodes

```bash
# View all nodes
kubectl get nodes

# View GPU nodes
kubectl get nodes -l nvidia.com/gpu=true

# View node details (GPU info)
kubectl describe node <node-name>

# View resource usage
kubectl top nodes
kubectl top pods -n vllm
```

## Troubleshooting

```bash
# Check events (useful for debugging)
kubectl get events -n vllm --sort-by='.lastTimestamp'

# Check why pod is pending
kubectl describe pod -n vllm | grep -A 10 "Events:"

# Check container status
kubectl get pods -n vllm -o jsonpath='{.items[*].status.containerStatuses[*].state}'

# Force delete stuck pod
kubectl delete pod <pod-name> -n vllm --force --grace-period=0
```

## API Testing

```bash
# Set variables (NLB terminates TLS on 443)
API_URL=$(kubectl get svc gateway-service -n vllm -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
API_KEY="your-api-key"

# Health check (no auth)
curl https://$API_URL/health

# List models
curl -H "Authorization: Bearer $API_KEY" https://$API_URL/v1/models

# Create a session
curl -X POST -H "Authorization: Bearer $API_KEY" https://$API_URL/v1/sessions

# Chat with session context (replace SESSION_ID)
curl -X POST "https://$API_URL/v1/sessions/SESSION_ID/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Get session history
curl -H "Authorization: Bearer $API_KEY" "https://$API_URL/v1/sessions/SESSION_ID"

# Delete session
curl -X DELETE -H "Authorization: Bearer $API_KEY" "https://$API_URL/v1/sessions/SESSION_ID"
```

## DynamoDB Management

```bash
# View table info
aws dynamodb describe-table --table-name vllm-conversations

# Scan recent items (for debugging)
aws dynamodb scan --table-name vllm-conversations --limit 10

# Get specific session metadata
aws dynamodb get-item --table-name vllm-conversations \
  --key '{"session_id": {"S": "SESSION_ID"}, "message_id": {"S": "metadata"}}'

# Query all messages in a session
aws dynamodb query --table-name vllm-conversations \
  --key-condition-expression "session_id = :sid" \
  --expression-attribute-values '{":sid": {"S": "SESSION_ID"}}'

# Delete a specific session item
aws dynamodb delete-item --table-name vllm-conversations \
  --key '{"session_id": {"S": "SESSION_ID"}, "message_id": {"S": "metadata"}}'
```

## RAG / Document Management

```bash
# Set variables
API_URL=$(kubectl get svc gateway-service -n vllm -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
API_KEY="your-api-key"

# Upload a PDF document
curl -X POST "https://$API_URL/v1/documents" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@/path/to/document.pdf"

# Upload a text file
curl -X POST "https://$API_URL/v1/documents" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@/path/to/notes.txt"

# List all documents
curl -H "Authorization: Bearer $API_KEY" "https://$API_URL/v1/documents"

# Check document processing status
curl -H "Authorization: Bearer $API_KEY" "https://$API_URL/v1/documents/DOCUMENT_ID"

# Delete a document
curl -X DELETE -H "Authorization: Bearer $API_KEY" "https://$API_URL/v1/documents/DOCUMENT_ID"
```

## OpenSearch Management

```bash
# Check OpenSearch domain status
aws opensearch describe-domain --domain-name vllm-rag --query 'DomainStatus.{Endpoint:Endpoint,Processing:Processing}'

# Get domain endpoint
aws opensearch describe-domain --domain-name vllm-rag --query 'DomainStatus.Endpoint' --output text

# Delete OpenSearch domain (~$26/month savings)
aws opensearch delete-domain --domain-name vllm-rag
```

## Documents DynamoDB Table

```bash
# View documents table info
aws dynamodb describe-table --table-name vllm-documents

# Scan document records
aws dynamodb scan --table-name vllm-documents --limit 10

# Get specific document metadata
aws dynamodb get-item --table-name vllm-documents \
  --key '{"document_id": {"S": "DOCUMENT_ID"}}'

# Delete document record
aws dynamodb delete-item --table-name vllm-documents \
  --key '{"document_id": {"S": "DOCUMENT_ID"}}'
```

## Cleanup

```bash
# Delete the vllm namespace (keep the EKS cluster). Point kubeconfig at the
# intended cluster first — do not `kubectl delete -f kubernetes/` (PLACEHOLDER_*
# manifests and the wrong context are both easy mistakes).
#   aws eks update-kubeconfig --name vllm-cluster --region us-east-1
kubectl delete namespace vllm

# Delete entire cluster and resources (interactive)
./scripts/cleanup.sh

# Manual cleanup:
eksctl delete cluster --name vllm-cluster --region us-east-1
aws dynamodb delete-table --table-name vllm-conversations
aws ecr delete-repository --repository-name vllm-server --force
aws ecr delete-repository --repository-name vllm-gateway --force
```
