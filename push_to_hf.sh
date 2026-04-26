#!/usr/bin/env bash
# Push this repo to both HuggingFace Spaces.
#
# Basic usage (same account for both):
#   export HF_USERNAME="Prasham1710"
#   export HF_TOKEN="hf_xxxxxxxxxxxx"
#   bash push_to_hf.sh
#
# Push training Space to a DIFFERENT account (e.g. priyanshimaheshwari):
#   export HF_USERNAME="Prasham1710"        # env server account (unchanged)
#   export HF_TOKEN="hf_prasham_token"      # env server token
#   export TRAIN_HF_USERNAME="priyanshimaheshwari"
#   export TRAIN_HF_TOKEN="hf_priyanshi_token"
#   export TRAIN_SPACE_NAME="ci-triage-env" # name of the Space in that account
#   bash push_to_hf.sh
#
# What it does:
#   1. Pushes main → ci-triage-env Space (uses root Dockerfile = env server)
#   2. Creates a temporary branch where Dockerfile.train becomes Dockerfile,
#      pushes that → training Space (possibly in a different account), then cleans up.

set -euo pipefail

: "${HF_USERNAME:?Set HF_USERNAME before running}"
: "${HF_TOKEN:?Set HF_TOKEN before running}"

# Training Space can live in a different account/space.
TRAIN_HF_USERNAME="${TRAIN_HF_USERNAME:-${HF_USERNAME}}"
TRAIN_HF_TOKEN="${TRAIN_HF_TOKEN:-${HF_TOKEN}}"
TRAIN_SPACE_NAME="${TRAIN_SPACE_NAME:-ci-triage-training}"

ENV_SPACE="ci-triage-env"

HF_ENV_REMOTE="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_USERNAME}/${ENV_SPACE}"
HF_TRAIN_REMOTE="https://${TRAIN_HF_USERNAME}:${TRAIN_HF_TOKEN}@huggingface.co/spaces/${TRAIN_HF_USERNAME}/${TRAIN_SPACE_NAME}"

# Prepend HF Space YAML frontmatter to README.md then restore it afterward.
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
echo "==> Pushing env server to ${HF_USERNAME}/${ENV_SPACE} …"

ENV_FRONTMATTER="---
title: CI Triage Env
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
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
echo "==> Pushing training Space to ${TRAIN_HF_USERNAME}/${TRAIN_SPACE_NAME} …"

TRAIN_FRONTMATTER="---
title: CI Triage Training
emoji: 🏋️
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
hardware: a10g-small
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
echo "    ✓ https://huggingface.co/spaces/${TRAIN_HF_USERNAME}/${TRAIN_SPACE_NAME}"

# ── Cleanup ───────────────────────────────────────────────────────────────────
git checkout main
git branch -D "$TEMP_BRANCH"
git remote remove hf-env   2>/dev/null || true
git remote remove hf-train 2>/dev/null || true

echo ""
echo "Both Spaces updated. Docker builds start automatically (~3 min env / ~15 min training)."
echo "Watch build logs:"
echo "  https://huggingface.co/spaces/${HF_USERNAME}/${ENV_SPACE}"
echo "  https://huggingface.co/spaces/${TRAIN_HF_USERNAME}/${TRAIN_SPACE_NAME}"
