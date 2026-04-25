# Branch C — Reward & Training

**Branch name:** `branch-c/reward-training`
**Default owner:** Sahil (ML/RL lead).
**Prerequisite:** `phase-0-complete` tag on `main`.

---

## What this branch builds

The reward functions, the SFT warmstart pipeline, the GRPO training loop, the evaluation harness, and the ablation pipeline. After this branch merges and training completes:

```bash
# Re-run full training
jupyter nbconvert --execute notebooks/train_grpo.ipynb

# Evaluate the trained model against all baselines
python -m ci_triage_env.training.eval --baselines all --seeds 3

# Ablation matrix
python -m ci_triage_env.training.ablations
```

Outputs:
- A trained Qwen3.5-4B + LoRA agent saved to HF Hub.
- Reward curves, baseline comparison table, ablation plots in `data_artifacts/results/`.
- The "trained vs baseline" demo replays for the video.

---

## Phases

| Phase | Title | What lands | Depends on |
|---|---|---|---|
| C1 | Reward components | All 9 reward modules implementing `RewardComponent` interface; unit tests against mock trajectories | Phase 0 |
| C2 | Composite reward + format gate | `composite.py` wires components with weights; replay-from-trace verifier; integration tests | C1 |
| C3 | SFT trajectory generation | OpenAI API loop against env (mock or real); reward filtering; SFT dataset builder | C2 + Branch A merged (Gate-1) |
| C4 | GRPO training script | TRL + Unsloth + Qwen3.5-4B config; 100-step smoke test; checkpointing; W&B logging | C3 |
| C5 | Evaluation harness | Multi-baseline runner (5 baselines × 3 seeds); CSV + plot output | C4 |
| C6 | Curves + ablations | Reward layer ablation runner; matplotlib output committed to repo | C5 |

---

## What you consume

- All schemas from Phase 0. **Do not modify.**
- The OpenEnv HTTP server from Branch A. Until Gate-1, you use a `MockEnvClient` (you implement in C1) that simulates the env API by replaying mock trajectories from Phase 0.
- The scenario corpus from Branch B (HF dataset). Until Gate-1, you use mock scenarios.

## What you produce

- 9 reward component implementations + composite assembler.
- An SFT-ready dataset of ~180 high-quality trajectories.
- A trained model checkpoint on HF Hub.
- Reward curves, eval tables, ablation plots.

---

## Files this branch owns

```
src/ci_triage_env/rewards/
├── __init__.py
├── base.py                     # already from Phase 0
├── format_gate.py              # multiplicative gate
├── diagnosis.py                # asymmetric confusion-matrix
├── action_quality.py           # action × failure-type matrix
├── cost_efficiency.py          # negative weighted sum
├── investigation.py            # informative-tools + redundancy + ordering
├── time_penalty.py             # per-step penalty
├── anti_gaming.py              # quarantine guard + Brier + no-info-action
├── minimal_evidence.py         # min-evidence-set bonus
├── counterfactual_predict.py   # DORMANT in v1 — inert no-op preserved for v2
├── composite.py                # the assembler
└── weights.py                  # frozen weight constants

src/ci_triage_env/training/
├── __init__.py
├── env_client.py               # HTTP client that talks to Branch A's server
├── mock_env_client.py          # for use before Gate-1
├── rollout.py                  # multi-turn rollout function for GRPO
├── sft.py                      # SFT trainer wrapper
├── grpo.py                     # GRPO trainer wrapper
├── trajectory_gen.py           # OpenAI API loop for SFT data
├── eval.py                     # multi-baseline evaluation
├── ablations.py                # reward layer ablation runner
├── plotting.py                 # matplotlib helpers
└── baselines/
    ├── __init__.py
    ├── random_policy.py
    ├── heuristic_policy.py
    └── zero_shot.py            # wrapper around any HF model

notebooks/
├── train_grpo.ipynb            # the Colab-runnable training notebook
└── eval.ipynb                  # eval + plot generation

tests/rewards/
├── test_format_gate.py
├── test_diagnosis.py
├── test_action_quality.py
├── test_cost_efficiency.py
├── test_investigation.py
├── test_time_penalty.py
├── test_anti_gaming.py
├── test_minimal_evidence.py
├── test_counterfactual_predict.py
└── test_composite.py

tests/training/
├── test_env_client.py
├── test_mock_env_client.py
├── test_rollout.py
├── test_baselines.py
└── test_eval.py
```

You do **not** own:
- Anything under `src/ci_triage_env/env/` (Branch A)
- Anything under `src/ci_triage_env/data/` (Branch B)
- The schemas (Phase 0, frozen)

---

## External resources you will use

| Resource | Access | Budget |
|---|---|---|
| OpenAI API | `OPENAI_API_KEY` env var | $25 hard cap, $5 reserve. See Phase C3. |
| HF Hub | `HF_TOKEN` env var | For model + dataset download/upload, ~$0 cost |
| HF compute | Provided onsite | $90 across 3 accounts. See Phase C4. |
| W&B | optional, free tier | For training run tracking |

---

## Tooling convention (read once, applies to every phase doc)

This repo uses **uv** (see `INSTRUCTION-MANUAL.md` §0). Every command in the per-phase docs that says `python ...`, `pytest ...`, or `ruff ...` should be run as `uv run python ...`, `uv run pytest ...`, `uv run ruff ...`. New deps are added with `uv add <pkg>` (or `uv add --optional <extra> <pkg>`), never raw `pip install`. Commit `pyproject.toml` and `uv.lock` together.

**Colab / HF compute exception.** Colab notebooks under `notebooks/` install with `!pip install` because Colab doesn't ship with uv. That's the deliberate exception — see `INSTRUCTION-MANUAL.md` §0. Local development is uv-only.

## Test discipline

After every phase, run:

```bash
uv run pytest -q tests/rewards/
uv run pytest -q tests/training/
uv run pytest -q tests/schemas/   # sanity
uv run ruff check src/ci_triage_env/rewards/ src/ci_triage_env/training/
```

Specific tests required per phase: see each `phase-c<N>.md` file. Reward tests must verify:
- Replay-from-trace gives the same score (deterministic).
- Edge cases: format-gate-fail, empty trajectory, all-tools-called trajectory, abstain-correct, abstain-wrong.
- Counterfactual probe component is inert (default_weight=0) in v1 — tests verify it stays at zero contribution.
- Composite weighting sums correctly.

---

## Integration checkpoints

- **Gate-1 prerequisite:** C1, C2 merged. Composite reward works on mock trajectories.
- **Gate-2 prerequisite:** C3, C4 merged. SFT data generated, GRPO smoke-tested.
- **Pre-submission:** C5, C6 done. Eval table + ablation plots committed.

---

## Compute discipline (the budget you must respect)

This branch is the only branch consuming paid compute. Track every run in `plan/BUDGET-LOG.md`.

**Phase C3 — OpenAI API:**
- Use `gpt-5-mini` (or whichever current cheap reasoning model is live) for bulk trajectory generation.
- Generate ~600 trajectories total. Each ~5k input + ~1k output tokens.
- After each 100 trajectories, check budget log. Hard stop at $25 spent.
- Reserve $5 for spot-checks with full `gpt-5` on hardest scenarios.

**Phase C4 — HF compute:**
- SFT smoke test: 100 steps on Qwen3.5-4B + LoRA + a small fixture dataset. Budget: $1.
- GRPO smoke test: 100 steps on real env, group size 4 (smaller than final), batch 1. Budget: $3.
- Full GRPO: 3000 steps, group size 8, batch 2. Budget: $30.
- Eval: 5 baselines × 150 scenarios × 3 seeds. Budget: $5.
- Ablations: 5 ablation runs × 1000 steps. Budget: $15.
- Buffer: $30. Use only if needed.

**If GRPO blows budget at step 500:**
- Check loss / reward curve. If learning, finish 1000-step run and ship that.
- If not learning, debug rollout/reward integration before spending more.
- Do not hit $60 cumulative without a "we have a bug" certainty.

**If full GRPO finishes early (under $30):**
- Use saved budget for additional ablation seeds.

---

## Open questions you'll see in phase docs

Each `phase-c<N>.md` contains:
1. **Outcome** — definition of done
2. **Files to create / modify** — exact paths
3. **Implementation notes** — design decisions already made
4. **Tests required**
5. **Smoke test**
6. **Budget impact** — what this phase costs in $ (reward phases free; training phases priced)
7. **Open questions** — things to flag in chat

Open `phase-c1.md` and start there. C1 (rewards) and C2 (composite) can be done before Gate-1 using mock data.
