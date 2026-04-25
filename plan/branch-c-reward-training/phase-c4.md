# Phase C4 — GRPO Training

**Owner:** Branch C.
**Prerequisite:** C3 merged (SFT dataset ready). Branch A through A3 merged (env runtime works).
**Estimated time:** 4–5 hours setup + ~30h wall-clock training (training runs unattended).
**Budget impact:** ~$35 (SFT smoke ~$1, GRPO smoke ~$3, full GRPO ~$30).

---

## Outcome

Trained Qwen3.5-4B + LoRA agent. By end of phase:

1. SFT warmstart trains Qwen3.5-4B + LoRA on the C3 dataset for 2–3 epochs.
2. GRPO training script wraps TRL's `GRPOTrainer` with a custom multi-turn rollout function.
3. 100-step smoke test runs end-to-end on real env + real scenarios.
4. Full 3000-step GRPO run produces a checkpointed adapter saved to HF Hub.
5. W&B logging captures reward curve, per-component breakdown, KL divergence, episode length.
6. Reward curve plot committed: clear improvement over baseline.
7. All C4 tests pass (mocked).

---

## Files to create

### `src/ci_triage_env/training/sft.py`

```python
import os
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

MODEL_NAME = "Qwen/Qwen3.5-4B"  # base, instruct variant if available — verify
MAX_SEQ_LEN = 8192

def load_model_for_sft(model_name: str = MODEL_NAME, max_seq_length: int = MAX_SEQ_LEN):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,   # auto
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer

def format_for_sft(traj: dict, tokenizer) -> dict:
    """Take a trajectory dict (with messages) and produce a single training example
    by formatting into the model's chat template."""
    text = tokenizer.apply_chat_template(traj["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

def run_sft(dataset_path: str, output_dir: str, num_epochs: int = 3,
            per_device_batch_size: int = 1, gradient_accumulation_steps: int = 4):
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = load_model_for_sft()
    raw = load_from_disk(dataset_path)
    train = raw.map(lambda t: format_for_sft(t, tokenizer))

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=100,
        report_to="wandb",
        max_seq_length=MAX_SEQ_LEN,
    )
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train, args=config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
```

### `src/ci_triage_env/training/rollout.py`

```python
class TrainingRollout:
    """Multi-turn rollout function for GRPO.
    Takes a model and a scenario_id; produces a full trajectory + reward."""

    def __init__(self, env_url: str, scenarios_train: list[str], system_prompt: str,
                 max_turns: int = 12):
        self.env = EnvClient(env_url)
        self.scenarios_train = scenarios_train
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self._quarantine_window: list[str] = []
        self.tools_listing = self._format_tools()

    def __call__(self, model, tokenizer, prompts: list[dict] | None = None) -> dict:
        """One rollout. Returns dict with prompt, response, reward.
        Used per-prompt within a GRPO group."""
        scenario_id = random.choice(self.scenarios_train)
        obs = self.env.reset(scenario_id=scenario_id)
        episode_id = obs.episode_id
        messages = [
            {"role": "system", "content": self.system_prompt.format(tools=self.tools_listing)},
            {"role": "user", "content": self._format_initial_obs(obs)},
        ]

        for turn in range(self.max_turns):
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=600, do_sample=True,
                    temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": response})

            action = parse_action(response)
            if action is None:
                # Format failure — terminate episode early
                break

            try:
                next_obs = self.env.step(episode_id, action)
            except Exception:
                break

            messages.append({"role": "user", "content": self._format_obs(next_obs)})
            if next_obs.is_terminal:
                # Counterfactual probe is deferred to v2 — env never populates
                # next_obs.probe_question in v1.
                break

        # Compute reward
        trace = self.env.get_trace(episode_id)
        scenario = load_scenario(trace.episode.scenario_id)
        reward = compute_reward(trace, scenario, quarantine_window=self._quarantine_window)

        # Update quarantine window
        if trace.episode.final_action:
            for sa in trace.episode.final_action.secondary_actions:
                self._quarantine_window.append(sa.name)
            self._quarantine_window = self._quarantine_window[-50:]

        return {
            "messages": messages,
            "reward": reward.total,
            "reward_breakdown": reward,
            "trajectory_length": len(messages),
        }
```

### `src/ci_triage_env/training/grpo.py`

```python
from trl import GRPOTrainer, GRPOConfig
from .sft import load_model_for_sft   # re-use loader
from .rollout import TrainingRollout

GRPO_HYPERPARAMS = {
    "learning_rate": 5e-6,
    "kl_coef": 0.04,
    "group_size": 8,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_prompt_length": 4096,
    "max_completion_length": 1024,
    "num_generations": 8,    # per group
    "temperature": 0.7,
    "top_p": 0.9,
    "logging_steps": 5,
    "save_steps": 200,
}

def run_grpo(sft_checkpoint_dir: str,
             output_dir: str,
             total_steps: int = 3000,
             env_url: str = "http://localhost:8000",
             scenarios_train_path: str = "data_artifacts/scenarios/train"):

    model, tokenizer = load_model_for_sft(model_name=sft_checkpoint_dir)

    train_scenarios = [p.stem for p in Path(scenarios_train_path).glob("*.json")]
    rollout = TrainingRollout(
        env_url=env_url,
        scenarios_train=train_scenarios,
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
    )

    config = GRPOConfig(
        output_dir=output_dir,
        max_steps=total_steps,
        **GRPO_HYPERPARAMS,
        report_to="wandb",
    )

    # GRPOTrainer takes a reward function. Our reward function wraps the rollout.
    def reward_fn(prompts, completions, **kwargs):
        # GRPOTrainer hands us prompts + completions; we already ran the env
        # via the rollout. So we use a stored mapping.
        # In practice: write a custom GRPOTrainer subclass that calls `rollout` directly.
        ...

    trainer = GRPOTrainerForToolUse(
        model=model, tokenizer=tokenizer,
        rollout_fn=rollout, args=config,
    )
    trainer.train()
    trainer.save_model(output_dir)
```

> **Critical engineering note**: standard `GRPOTrainer` expects single-turn prompts and uses the model's `generate` to produce completions, then scores them. Our task is multi-turn. We need a custom `GRPOTrainerForToolUse` subclass that:
> 1. For each prompt in the batch, runs `rollout` to get a (messages, reward) pair.
> 2. Concatenates all messages into a single sequence as the "completion" for loss purposes.
> 3. Computes group-relative advantage and loss as standard GRPO.
>
> This custom trainer is the most engineering-intensive piece in the project. Budget time accordingly. Reference TRL's source for `GRPOTrainer` and override `_generate_and_score`.

### `src/ci_triage_env/training/custom_trainer.py`

```python
class GRPOTrainerForToolUse(GRPOTrainer):
    """GRPO trainer that uses our multi-turn env-driven rollout instead of single-turn generate."""

    def __init__(self, *args, rollout_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_fn = rollout_fn

    def _generate_and_score(self, batch):
        """Override: generate via rollout, score via env-based reward."""
        rewards = []
        completions = []
        for _ in range(len(batch) * self.args.num_generations):
            result = self.rollout_fn(self.model, self.tokenizer)
            rewards.append(result["reward"])
            # Reconstruct the "completion" as concatenation of assistant turns
            assistant_text = "\n".join(
                m["content"] for m in result["messages"] if m["role"] == "assistant"
            )
            completions.append(assistant_text)
        return completions, rewards
```

### `notebooks/train_grpo.ipynb`

Top-to-bottom Colab-runnable notebook:

```python
# Cell 1: install
!pip install -q torch unsloth trl transformers accelerate wandb datasets pydantic httpx fastapi
!pip install -e .   # the ci_triage_env package

# Cell 2: env setup
import os
os.environ["WANDB_PROJECT"] = "ci-triage-env"
os.environ["HF_TOKEN"] = "..."

# Cell 3: pull scenarios from HF dataset
from huggingface_hub import snapshot_download
scen_dir = snapshot_download("USER/ci-triage-scenarios", repo_type="dataset")

# Cell 4: start env server in background
!python -m ci_triage_env.env.server &
import time; time.sleep(3)

# Cell 5: SFT warmstart
from ci_triage_env.training.sft import run_sft
run_sft(
    dataset_path="data_artifacts/sft_dataset/",
    output_dir="checkpoints/sft/",
    num_epochs=3,
)

# Cell 6: GRPO smoke test
from ci_triage_env.training.grpo import run_grpo
run_grpo(
    sft_checkpoint_dir="checkpoints/sft/",
    output_dir="checkpoints/grpo_smoke/",
    total_steps=100,
)

# Cell 7: full GRPO
run_grpo(
    sft_checkpoint_dir="checkpoints/sft/",
    output_dir="checkpoints/grpo_full/",
    total_steps=3000,
)

# Cell 8: push to HF
from huggingface_hub import upload_folder
upload_folder(
    folder_path="checkpoints/grpo_full/",
    repo_id="USER/ci-triage-trained-qwen3.5-4b",
    repo_type="model",
)
```

---

## Implementation notes

- **Custom GRPO trainer is the highest-risk part.** TRL's `GRPOTrainer` API may have changed by April 2026. Read the latest source. If multi-turn rollout integration breaks, fall back to a simpler approach: collect rollouts in batches, save (prompt, completion, reward) triples, fine-tune via standard GRPO on the saved data. This costs 1 epoch of pseudo-on-policy but is simpler.
- **Group size 8.** GRPO needs at least 4–8 generations per group to estimate advantage. Larger group = better estimate but more compute. 8 is a good balance for our budget.
- **KL coefficient.** Start at 0.04. If KL grows unboundedly, increase to 0.1. If KL stays near zero (model not exploring), drop to 0.01.
- **Multi-turn token accounting.** The "completion" for loss purposes is only the assistant turns, not the env's user turns. TRL's standard mask logic should handle this if we set up the training mask correctly (loss only on assistant tokens).
- **Checkpointing.** Save every 200 steps. Keep last 5 checkpoints; eval picks the best by val reward.
- **W&B project setup.** Create one project: `ci-triage-env`. Tag runs: `qwen3.5-4b-sft`, `qwen3.5-4b-grpo-smoke`, `qwen3.5-4b-grpo-full`, `ablation-no-cf`, etc.
- **Reward function determinism during training.** Our `compute_reward` is deterministic. The rollout's *generation* is stochastic (sampling). That's fine — different rollouts in the group should produce different rewards.

---

## Tests required (`tests/training/test_grpo.py`)

```python
def test_rollout_with_mock_env(monkeypatch):
    """TrainingRollout produces a (messages, reward) tuple with a mock env."""

def test_rollout_handles_format_failure():
    """Model emits malformed JSON → trajectory terminates, reward computed (low)."""

def test_quarantine_window_updates():
    """After running rollouts, _quarantine_window contains last 50 secondary actions."""

def test_sft_data_format(tmp_path):
    """format_for_sft produces a 'text' field that round-trips through tokenizer."""

# These need GPU — skip in CI, run manually
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_sft_smoke():
    """Run SFT for 5 steps on a 5-trajectory dataset; loss decreases."""

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_grpo_smoke():
    """Run GRPO for 5 steps; no NaN losses, reward distribution non-degenerate."""
```

---

## Smoke test (manual, requires GPU)

```bash
# Local Linux box with GPU OR HF compute Space
python -m ci_triage_env.env.server &
sleep 3

# SFT smoke (100 steps, ~10 min)
python -c "from ci_triage_env.training.sft import run_sft; \
  run_sft('data_artifacts/sft_dataset/', 'checkpoints/sft_smoke/', num_epochs=1)"

# GRPO smoke (100 steps, ~30 min)
python -c "from ci_triage_env.training.grpo import run_grpo; \
  run_grpo('checkpoints/sft_smoke/', 'checkpoints/grpo_smoke/', total_steps=100)"

# Check W&B for: reward going up, KL stable, format-gate-pass-rate going up
echo "Visit https://wandb.ai/<entity>/ci-triage-env"
```

Expected: reward curve climbs from baseline (~0) to a positive trend by step 50–100. Format-gate pass rate climbs from low to >80%.

---

## Full training run (onsite)

```bash
# Verify everything green
pytest -q
ruff check src/

# Push to HF compute (or run on local A100)
# Estimated wall time: ~30 hours
python -c "from ci_triage_env.training.grpo import run_grpo; \
  run_grpo('checkpoints/sft/', 'checkpoints/grpo_full/', total_steps=3000)"

# Monitor: wandb dashboard. If reward stalls for 500 steps, abort and debug.
```

---

## Hard-stop rules (from manual)

- Hit $60 cumulative without a "we have a bug" certainty → stop, debug.
- KL divergence spikes 10x baseline → stop, lower learning rate.
- Format-gate pass rate falls below 50% mid-training → stop, model is regressing on format. Reload SFT checkpoint and lower learning rate.
- Reward curve flat for 500 steps → stop, debug rollout/reward integration.

---

## Open questions

1. **Verify Qwen3.5-4B Unsloth + GRPO compatibility** before commit. Run a 5-step smoke on whatever GPU is available pre-onsite. If broken, fall back to Qwen3.5-2B.
2. **Custom trainer subclass complexity.** If TRL's `GRPOTrainer` doesn't support custom rollouts cleanly, the alternative is to implement a minimal GRPO loss + step ourselves. Add 4 hours of estimate if so.
3. **Should rollouts be batched within a step?** Yes — the env can be called sequentially within a single GRPO step's sampling phase. No parallelism benefit since the env is single-process.

---

## What's NOT in this phase

- Multi-baseline eval (C5)
- Ablation studies (C6)
