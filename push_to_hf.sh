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

# Prepend HF Space YAML frontmatter to README.md then restore it afterward.
# HF requires this block so it knows the Space type and can render the UI correctly.
inject_readme() {
    local frontmatter="$1"
    local original
    original=$(cat README.md)
    printf '%s\n\n%s\n' "$frontmatter" "$original" > README.md
}

restore_readme() {
    git checkout -- README.md
}

# ── 1. Env server Space ───────────────────────────────────────────────────────
echo ""
echo "==> Pushing env server to ${ENV_SPACE} …"

ENV_FRONTMATTER="---
title: CI Triage Env
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---"

TEMP_BRANCH="_hf_env_push_$(date +%s)"
git checkout -b "$TEMP_BRANCH"
inject_readme "$ENV_FRONTMATTER"
git add README.md
git commit -m "chore(spaces): add HF Space config for env server [skip ci]"

git remote remove hf-env 2>/dev/null || true
git remote add hf-env "$HF_ENV_REMOTE"
git push hf-env "${TEMP_BRANCH}:main" --force
echo "    ✓ https://huggingface.co/spaces/${HF_USERNAME}/${ENV_SPACE}"

git checkout main
git branch -D "$TEMP_BRANCH"

# ── 2. Training Space (Dockerfile.train → Dockerfile + frontmatter) ──────────
echo ""
echo "==> Preparing training Space push …"

TRAIN_FRONTMATTER="---
title: CI Triage Training
emoji: 🏋️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---"

TEMP_BRANCH="_hf_train_push_$(date +%s)"
git checkout -b "$TEMP_BRANCH"

inject_readme "$TRAIN_FRONTMATTER"
cp Dockerfile.train Dockerfile          # training Space uses Dockerfile.train
git add README.md Dockerfile
git commit -m "chore(spaces): add HF Space config for training [skip ci]"

git remote remove hf-train 2>/dev/null || true
git remote add hf-train "$HF_TRAIN_REMOTE"
git push hf-train "${TEMP_BRANCH}:main" --force
echo "    ✓ https://huggingface.co/spaces/${HF_USERNAME}/${TRAIN_SPACE}"

# ── Cleanup ───────────────────────────────────────────────────────────────────
git checkout main
git branch -D "$TEMP_BRANCH"
git remote remove hf-env   2>/dev/null || true
git remote remove hf-train 2>/dev/null || true

echo ""
echo "Both Spaces updated. Docker builds will start automatically (~3 min env / ~15 min training)."
echo "Watch build logs:"
echo "  https://huggingface.co/spaces/${HF_USERNAME}/${ENV_SPACE}"
echo "  https://huggingface.co/spaces/${HF_USERNAME}/${TRAIN_SPACE}"
