"""Automated end-to-end training script for HF Spaces.

Runs: scenario download → SFT warmstart → GRPO fine-tuning → push to HF Hub.
All config comes from environment variables (set as Space secrets).

Optimised for A10G Large (46 GB VRAM, 12 vCPU).

Required env vars:
    HF_TOKEN         - HuggingFace write token
    HF_USERNAME      - your HF username
    WANDB_API_KEY    - Weights & Biases API key

Optional:
    HF_SCENARIOS_REPO   - default: {HF_USERNAME}/ci-triage-scenarios
    HF_SFT_DATASET_REPO - default: {HF_USERNAME}/ci-triage-sft
    HF_MODEL_REPO       - default: {HF_USERNAME}/ci-triage-agent
    GRPO_STEPS          - default: 100  (set lower to finish faster, higher for more training)
    SKIP_SFT            - set to "1" to skip SFT and jump straight to GRPO (if checkpoint exists)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── resolve config ────────────────────────────────────────────────────────────
HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
WANDB_KEY   = os.environ.get("WANDB_API_KEY", "")

SCENARIOS_REPO   = os.environ.get("HF_SCENARIOS_REPO",   f"{HF_USERNAME}/ci-triage-scenarios")
SFT_DATASET_REPO = os.environ.get("HF_SFT_DATASET_REPO", f"{HF_USERNAME}/ci-triage-sft")
MODEL_REPO       = os.environ.get("HF_MODEL_REPO",       f"{HF_USERNAME}/ci-triage-agent")
GRPO_STEPS       = int(os.environ.get("GRPO_STEPS", "100"))
SKIP_SFT         = os.environ.get("SKIP_SFT", "0") == "1"

DATA_ROOT  = Path("/data")
SCEN_DIR   = DATA_ROOT / "scenarios"
SFT_DS_DIR = DATA_ROOT / "sft_dataset"
SFT_CKPT   = DATA_ROOT / "checkpoints" / "sft"
GRPO_CKPT  = DATA_ROOT / "checkpoints" / "grpo"

# ── auth ──────────────────────────────────────────────────────────────────────
from huggingface_hub import login
login(token=HF_TOKEN)

if WANDB_KEY:
    import wandb
    wandb.login(key=WANDB_KEY)
    os.environ["WANDB_PROJECT"] = "ci-triage-env"
else:
    os.environ["WANDB_DISABLED"] = "true"

# ── Step 1: download scenario corpus ─────────────────────────────────────────
if not SCEN_DIR.exists() or not any(SCEN_DIR.rglob("*.json")):
    print(f"\n[1/4] Downloading scenarios from {SCENARIOS_REPO} …")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=SCENARIOS_REPO,
        repo_type="dataset",
        local_dir=str(SCEN_DIR),
        token=HF_TOKEN,
    )
else:
    n = sum(1 for _ in SCEN_DIR.rglob("*.json"))
    print(f"\n[1/4] Scenarios already present ({n} files) — skipping download.")

train_scen = list(SCEN_DIR.rglob("train/**/*.json")) or list(SCEN_DIR.rglob("*.json"))
print(f"      Train scenarios available: {len(train_scen)}")

# ── Step 2: download SFT dataset ─────────────────────────────────────────────
if not SFT_DS_DIR.exists():
    print(f"\n[2/4] Downloading SFT dataset from {SFT_DATASET_REPO} …")
    from datasets import load_dataset
    ds = load_dataset(SFT_DATASET_REPO, split="train", token=HF_TOKEN)
    SFT_DS_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(SFT_DS_DIR))
    print(f"      {len(ds)} SFT examples saved.")
else:
    from datasets import load_from_disk
    ds = load_from_disk(str(SFT_DS_DIR))
    print(f"\n[2/4] SFT dataset already present ({len(ds)} examples) — skipping download.")

# ── Step 3: SFT warmstart ─────────────────────────────────────────────────────
if SKIP_SFT and SFT_CKPT.exists():
    print(f"\n[3/4] SKIP_SFT=1 and checkpoint found at {SFT_CKPT} — skipping SFT.")
else:
    print(f"\n[3/4] SFT warmstart — {len(ds)} examples, A10G-optimised settings …")
    from ci_triage_env.training.sft import run_sft
    run_sft(
        dataset_path=str(SFT_DS_DIR),
        output_dir=str(SFT_CKPT),
        num_epochs=2,
        per_device_batch_size=4,      # 46 GB → fit 4 sequences comfortably
        gradient_accumulation_steps=4, # effective batch = 16
    )
    print(f"      SFT done → {SFT_CKPT}")

    # Push SFT checkpoint immediately so it's saved even if GRPO fails
    print("      Pushing SFT checkpoint to HF Hub …")
    from huggingface_hub import upload_folder
    upload_folder(
        folder_path=str(SFT_CKPT),
        repo_id=MODEL_REPO + "-sft",
        repo_type="model",
        token=HF_TOKEN,
        commit_message="SFT warmstart checkpoint",
    )

# ── Step 4: GRPO fine-tuning ──────────────────────────────────────────────────
print(f"\n[4/4] GRPO training — {GRPO_STEPS} steps, MockEnvClient in-process …")
print("      Monitoring: https://wandb.ai (search project ci-triage-env)")

from ci_triage_env.training.mock_env_client import MockEnvClient
from ci_triage_env.training.grpo import run_grpo

env_client = MockEnvClient(scenarios_dir=str(SCEN_DIR / "train"))
print(f"      Loaded {len(env_client.scenario_ids)} train scenarios into MockEnvClient")

# A10G Large optimised hyperparams.
# max_turns=4 + max_completion_length=256 keeps each rollout to ~15 sec so
# 100 steps × 4 rollouts ≈ 100 min total — fits the 2-3 hour budget.
run_grpo(
    sft_checkpoint_dir=str(SFT_CKPT),
    output_dir=str(GRPO_CKPT),
    total_steps=GRPO_STEPS,
    env_client=env_client,
    scenarios_train_path=str(SCEN_DIR / "train"),
    hyperparams={
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,   # effective batch = 4
        "num_generations": 4,
        "max_prompt_length": 2048,
        "max_completion_length": 256,
        "learning_rate": 5e-6,
        "kl_coef": 0.04,
        "temperature": 0.8,
        "top_p": 0.95,
        "logging_steps": 5,
        "save_steps": 50,
        "report_to": "wandb" if WANDB_KEY else "none",
    },
)
print(f"      GRPO done → {GRPO_CKPT}")

# ── Push final model ──────────────────────────────────────────────────────────
print(f"\n[done] Pushing final model to {MODEL_REPO} …")
from huggingface_hub import upload_folder
upload_folder(
    folder_path=str(GRPO_CKPT),
    repo_id=MODEL_REPO,
    repo_type="model",
    token=HF_TOKEN,
    commit_message=f"GRPO-trained adapter — {GRPO_STEPS} steps",
)
print(f"       Model at: https://huggingface.co/{MODEL_REPO}")
print("\nTraining complete.")
