# M3-SLR Deployment Guide

This guide follows the 3 steps:
1. CI with GitHub Actions + GHCR
2. Manual deployment to Kubernetes
3. CD with ArgoCD (GitOps)

## 0) One-time repository setup

Create these GitHub secrets and permissions:
- Packages permission: write (already configured in workflow)
- If your repository is private, allow Kubernetes nodes to pull from GHCR using imagePullSecret.

Update placeholders before running:
- deploy/k8s/deployment.yaml: image (ghcr.io/doduc12306/m3-slr:latest)
- deploy/k8s/kustomization.yaml: image name
- deploy/argocd/application.yaml: repoURL (https://github.com/doduc12306/M3-SLR.git)

## 1) CI: Build and push Docker image to GHCR

Workflow file:
- .github/workflows/ci-ghcr.yml

Trigger:
- Push to main
- Push tag v*
- Manual workflow_dispatch

Output image format:
- ghcr.io/<owner>/<repo>:latest
- ghcr.io/<owner>/<repo>:sha-<commit>

## 2) Manual deploy to Kubernetes

From your jump server (202.191.100.9), confirm kubectl context points to your cluster.

Apply deployment manifests:

kubectl apply -k deploy/k8s

Check rollout:

kubectl -n m3-slr get pods
kubectl -n m3-slr rollout status deploy/m3-slr-infer
kubectl -n m3-slr get svc m3-slr-infer

Service exposure:
- NodePort 30080
- Access example: http://202.191.100.11:30080/health

Test inference with curl:

curl -X POST "http://202.191.100.11:30080/predict?top_k=5" \
  -F "file=@/path/to/video.mp4"

## 3) CD with ArgoCD

Install ArgoCD (run once on cluster):

kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

Create GitOps application:

kubectl apply -f deploy/argocd/application.yaml

Verify sync status:

kubectl -n argocd get applications

### Auto deploy new model image

Workflow file:
- .github/workflows/cd-update-k8s-tag.yml

How it works:
- You run workflow_dispatch with image_tag (example: sha-a1b2c3d)
- Workflow updates deploy/k8s/kustomization.yaml
- Workflow commits to main
- ArgoCD auto-syncs to cluster and rolls out new image

## Suggested production hardening

- Add imagePullSecrets for private GHCR repos.
- Add Ingress + TLS instead of NodePort.
- Add resource tuning after load testing.
- Add HPA for autoscaling.
- Add Prometheus metrics and centralized logs.
