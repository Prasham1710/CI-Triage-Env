#!/usr/bin/env bash
# Push this repo to both HuggingFace Spaces.
#
# Usage:
#   export HF_USERNAME="your_hf_username"
#   export HF_TOKEN="hf_xxxxxxxxxxxx"
#   bash push_to_hf.sh
#
# What it does:
#   1. Pushes main → ci-triage-env Space (uses root Dockerfile = env server)
#   2. Creates a temporary branch where Dockerfile.train becomes Dockerfile,
#      pushes that → ci-triage-training Space, then cleans up the branch.

set -euo pipefail

: "${HF_USERNAME:?Set HF_USERNAME before running}"
: "${HF_TOKEN:?Set HF_TOKEN before running}"

ENV_SPACE="ci-triage-env"
TRAIN_SPACE="ci-triage-training"

HF_ENV_REMOTE="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${ENV_SPACE}"
HF_TRAIN_REMOTE="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${TRAIN_SPACE}"

# ── 1. Env server Space ───────────────────────────────────────────────────────
echo ""
echo "==> Pushing env server to ${ENV_SPACE} …"
git remote remove hf-env 2>/dev/null || true
git remote add hf-env "$HF_ENV_REMOTE"
git push hf-env main:main --force
echo "    ✓ https://huggingface.co/spaces/${HF_USERNAME}/${ENV_SPACE}"

# ── 2. Training Space (Dockerfile.train → Dockerfile) ────────────────────────
echo ""
echo "==> Preparing training Space push (swapping Dockerfile) …"

TEMP_BRANCH="_hf_train_push_$(date +%s)"
git checkout -b "$TEMP_BRANCH"

# Overwrite root Dockerfile with training Dockerfile content
cp Dockerfile.train Dockerfile
git add Dockerfile
git commit -m "chore(spaces): use Dockerfile.train for training Space [skip ci]"

echo "==> Pushing training Space to ${TRAIN_SPACE} …"
git remote remove hf-train 2>/dev/null || true
git remote add hf-train "$HF_TRAIN_REMOTE"
git push hf-train "${TEMP_BRANCH}:main" --force
echo "    ✓ https://huggingface.co/spaces/${HF_USERNAME}/${TRAIN_SPACE}"

# ── Cleanup ───────────────────────────────────────────────────────────────────
git checkout main
git branch -D "$TEMP_BRANCH"
git remote remove hf-env  2>/dev/null || true
git remote remove hf-train 2>/dev/null || true

echo ""
echo "Both Spaces updated. Docker builds will start automatically."
echo "Watch build logs at:"
echo "  https://huggingface.co/spaces/${HF_USERNAME}/${ENV_SPACE}"
echo "  https://huggingface.co/spaces/${HF_USERNAME}/${TRAIN_SPACE}"
