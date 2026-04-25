# CI-Triage-Env — README Plan

> This is the planning version of the README that ships in the submission repo. Sections marked **[FILL POST-TRAIN]** are placeholders for plots, numbers, and links produced after training completes. Everything else should hold from day one and is the source-of-truth for the team's pitch.

---

## TL;DR

CI-Triage-Env is an OpenEnv environment that trains a small LLM (Qwen3.5-4B + LoRA) to investigate ambiguous CI failures end-to-end: read logs, query commit history, check resource metrics, decide between rerunning, quarantining, filing a bug, or paging the owner. The reward signal is a five-component composable rubric verified entirely against simulator ground truth — no LLM-as-judge anywhere — featuring an asymmetric cost-grounded confusion matrix, a Brier-score calibration probe on a held-out *ambiguous* class, and an anti-gaming layer that defeats the obvious shortcuts.

After training, our 4B agent **[FILL POST-TRAIN: matches/beats] Qwen3.5-9B zero-shot** on diagnosis accuracy, action appropriateness, calibration on the held-out ambiguous class, and held-out generalization, while making **[FILL POST-TRAIN: X%] fewer expensive tool calls per episode**.

[Try the environment on Hugging Face Spaces](FILL_LINK) · [90-second demo video](FILL_LINK) · [Training notebook](FILL_LINK) · [Blog post](FILL_LINK)

---

## 1. The problem we're targeting

Every engineering org above 50 developers loses millions per year and sleep-quality at scale to one specific class of problem: **CI failures of ambiguous origin**. A test fails. The cause could be a real bug, a race-condition flake, a timing flake, an infra blip, a dependency drift, or a deliberately-mixed-cause case. Misclassification cost is asymmetric and often catastrophic:

- Quarantine a real bug → ships to production
- Rerun a real bug → hides it under "intermittent failure"
- File a bug for an infra issue → wastes engineering trust and on-call goodwill
- Page the wrong owner → burns paging-budget credibility

Existing tools (Trunk.io, Datadog Flaky Test Management, BuildPulse, GitHub's flaky-test detection) are **statistical rule-based detectors**. They flag patterns; they do not investigate. The actual triage — reading logs, correlating recent commits, checking resource metrics, deciding the right action under cost and risk constraints — is still done by humans, often at 3am, often by the most experienced engineer on the team.

This environment closes that gap. We train an LLM to do the multi-step investigation that the statistical tools refuse to do.

### Why this is an RL problem, not an SFT problem

CI triage cannot be cleanly supervised:

1. There is no public dataset of `(log → tool-call → tool-call → …  → diagnosis)` trajectories at scale. Senior-engineer triage knowledge is tacit.
2. The optimal action depends on what the agent has already discovered, which depends on which tools it chose to call earlier. This is sequential, not classification.
3. The interesting signal lives in the trajectory — investigation order, calibration of confidence, recovery from a wrong initial hypothesis. Trajectory cannot be supervised.
4. The reward shaping (informative-tools coverage, redundancy penalty, ordering bonus, minimal-evidence bonus) is dense and per-step — exactly the structure RL exploits. A classifier trained on `(failure_summary → label)` cannot learn investigation strategy because there's no investigation step in its data.

---

## 2. The environment

### 2.1 What the agent sees

At episode start the env emits an initial `Observation` containing:

- A failure summary: failing test name, suite, branch, last-passing commit
- A budget: max tool calls (default 12), wall-clock cost cap
- A list of available tools with their docstrings and cost weights

The agent then issues tool calls and receives observations until it submits a terminal action.

### 2.2 Tool surface

| Surface | Tool | Cost weight | Returns |
|---|---|---|---|
| Investigation | `read_logs(scope, lines)` | $0.001 / 1 unit time | Real-shaped log excerpt |
| Investigation | `inspect_test_code(test_id)` | $0.001 / 1 unit time | Test source code excerpt |
| Investigation | `run_diagnostic(name)` | $0.30 (rerun-equivalent) | Diagnostic output |
| Investigation | `cluster_metrics(window)` | $0.05 | CPU/mem/network deltas |
| Context | `query_flake_history(test_id)` | $0.01 | Past N runs of this test |
| Context | `recent_commits(window, paths)` | $0.01 | Commits touching paths |
| Context | `check_owner(path)` | $0.01 | Owner team & on-call |
| Action | `rerun_test()` | $0.30 + risk | Rerun outcome |
| Action | `quarantine_test()` | future-debt risk | Confirmation |
| Action | `file_bug(severity, summary)` | 30 min human time | Ticket ID |
| Action | `ping_owner(message)` | 5 min human time | Acknowledgement |
| Terminal | `submit_diagnosis(label, confidence)` | — | Episode end |

Exactly one terminal action is required per episode. `submit_diagnosis` accepts an `abstain` label for genuinely-ambiguous cases.

### 2.3 Failure taxonomy (7 classes)

Each is a *parametric scenario family* — a Python class that produces a fresh instance from a seed:

1. **Real bug** — a logic/data error introduced by a recent commit. Reproducible across reruns. Logs reference the buggy code path.
2. **Race-condition flake** — non-deterministic outcome from concurrent state. Reruns sometimes pass. Logs interleave thread/goroutine output.
3. **Timing flake** — assertion races a timeout. Manifests under CI load. Time-based logs.
4. **Infra-network** — DNS/TLS/connectivity blip during the run. Cluster-wide impact visible in metrics.
5. **Infra-resource** — OOM-kill, disk-full, CPU starvation. Visible in resource metrics; logs reference kernel/runtime symptoms.
6. **Dependency drift** — upstream package update changed semantics. Visible in lockfile diff and commit history.
7. **Ambiguous (abstain-correct)** — evidence is genuinely insufficient to determine cause. The correct terminal action is `submit_diagnosis(label="abstain", confidence_calibrated)`.

Per-family target: 30+ instances → ~200–300 total scenarios with a 70/15/15 train/val/held-out split.

### 2.4 Episode lifecycle

```
reset() → Observation0
loop:
  agent emits ToolCall or TerminalAction
  step(call) → Observation_t (or done)
  per-step rewards accumulate (cost, time, investigation shaping)
on terminal:
  compute final reward (diagnosis, action_quality, anti-gaming, …)
return final reward breakdown
```

---

## 3. The reward system (the centerpiece)

A composable rubric grounded in simulator ground-truth. Five components, format gate, no LLM judges.

```
R = R_format · (
    0.25 · R_diagnosis            # asymmetric confusion-matrix on final enum + abstain
  + 0.20 · R_action_quality       # action × ground-truth-failure-type cost matrix
  + 0.15 · R_cost_efficiency      # negative weighted sum of tool-call costs
  + 0.15 · R_investigation        # informative-tools coverage − redundancy − ordering
  + 0.10 · R_time                 # dense per-step penalty until terminal
  + 0.15 · R_anti_game            # quarantine-rate guard + Brier calibration on ambiguous
)
```

All weights tuned and frozen before final runs; ablations swap in zeros to isolate contributions.

### 3.1 R_format (multiplicative gate)

Strict JSON-schema validation on every model output. Bad-format rollouts get **R = 0**. This is what "hard to game" looks like in practice.

### 3.2 R_diagnosis (25%)

Asymmetric confusion-matrix loss on the final `submit_diagnosis(label)`. Weights grounded in published CI cost data:

- Quarantining a real bug = **−1.0** (Bell et al., DeFlaker, FSE 2018; data on shipped-bug incident cost)
- Rerunning a real bug = **−0.7** (hides bug, delays detection)
- Quarantining a flake correctly = **+1.0**
- Rerunning a flake = **+0.6** (acceptable if cheap)
- Filing a bug for an infra issue = **−0.5**
- Correct diagnosis on the ambiguous class = **+1.0** (only via `abstain`)

Cited weights documented in `rewards/diagnosis_weights.md`.

### 3.3 R_action_quality (20%)

A separate matrix scoring the agent's `terminal_action` choice given the ground-truth failure type. Distinct from diagnosis: the diagnosis enum is the agent's *belief*; the terminal action is the agent's *behavior*. They can disagree (correctly diagnosing a flake but wrongly filing a P0 bug should still be penalized).

### 3.4 R_cost_efficiency (15%)

Sum of tool-call costs in dollars-equivalent or minutes-equivalent, mapped to a smooth penalty. Exhaustive enumeration is punished; surgical investigation is rewarded.

### 3.5 R_investigation (15%)

The shaping reward for *trajectory quality*. Three sub-components:

- **Informative-tools coverage**: each scenario carries a hand-labeled set of tools whose outputs were necessary or sufficient to reach the correct diagnosis. Reward proportional to coverage.
- **Redundancy penalty**: repeated equivalent tool calls (same args, no new info) penalized.
- **Order-of-investigation bonus**: cheap tools before expensive ones, context tools before action tools (`recent_commits` before `ping_owner`).

This replaces a fragile entropy-reduction reward with a precomputed, deterministic label.

### 3.6 R_time (10%)

Dense per-step penalty: `−ε · num_steps`. Forces speed without sacrificing accuracy.

### 3.7 R_anti_game (15%)

Three sub-probes:

- **Quarantine-rate guard**: episode-level. If quarantine rate over the last 50 episodes exceeds a threshold (default 30%), all subsequent rewards scale down by `1 − excess_rate`. Defeats "quarantine everything."
- **Brier calibration on ambiguous**: on the `ambiguous` and held-out scenarios, `submit_diagnosis(confidence)` is required. Reward = `1 − BrierScore(confidence, correct)`. Overconfidence on hard cases is punished.
- **No-info-action guard**: terminal action without at least 2 informative tool calls = R scaled by 0.5. Defeats "guess from the failure summary."

### 3.8 R_minimal_evidence (folded into R_investigation)

Each scenario specifies a `minimal_evidence_set` — the smallest tool-output combination that uniquely determines the correct diagnosis. Bonus for reaching correct diagnosis using only elements from this set; penalty proportional to extra calls beyond it. Trains *strategic* investigation.

### 3.9 What's deferred

A counterfactual probe ("had you taken X instead of Y, what would have happened?") was explored as a possible additional component. We deferred it from v1 because:

- It splits the reward signal across two objectives (diagnose + predict counterfactuals), which can hurt main-task convergence under our 3000-step training budget on a 4B model.
- The env-side machinery (deterministic snapshot/replay, probe scheduling, outcome categorization) adds engineering risk on a 2-day onsite timeline.
- The same calibration angle is captured by §3.7's Brier probe on the held-out *ambiguous* class, which is far cheaper and proven to work.

The schema fields and reward-component scaffolding are kept as optional, so re-adding the probe in v2 is a purely additive change with no schema migration. See the *Future work* section.

---

## 4. Why this reward design is defensible against the "anyone could build this" critique

- Five components, every signal computed against simulator ground truth — replicating this requires building the simulator, not calling an API.
- The Brier-calibration probe on the held-out *ambiguous* family forces calibrated uncertainty, not pattern-matched confidence. Most teams will not design this.
- The minimal-sufficient-evidence label requires careful per-scenario annotation, not a generic heuristic. It rewards strategic investigation over exhaustive enumeration.
- The anti-gaming layer specifically defeats every shortcut a clever competitor would otherwise exploit ("quarantine everything," "always rerun," "guess from failure summary"). The quarantine-rate guard is episode-windowed, not per-rollout.
- The asymmetric confusion-matrix weights are grounded in real CI economics literature (DeFlaker FSE 2018, Google SRE book Ch. 31, CircleCI compute-cost public data) — auditable, not invented.
- Format gate is a multiplicative pre-condition, not a reward term. A model with bad output structure scores 0, full stop. This is a stronger anti-gaming signal than weighted format-validity.

---

## 5. Data sourcing

Real data is the bottleneck for any ML-shaped CI work. We use a **realism-by-seed** approach: real public failures provide log-shape and pattern templates; the generator parameterizes.

### 5.1 Public sources (all open / non-restricted)

| Source | Used for |
|---|---|
| **DeFlaker corpus** (Bell et al., FSE 2018) | Labels for flaky-vs-real-bug commits across 26 OSS projects |
| **iDFlakies dataset** (Lam et al., ICSE 2019) | Order-dependent flaky test instances |
| **FlakeFlagger** (Alshammari et al., ICSE 2021) | ~800 flaky tests with rich features |
| **LogHub** (Zhu et al., ISSRE 2019) | 19 system log datasets (HDFS, Hadoop, Spark) with anomaly labels |
| **GitHub Actions public logs** | Real CI failure logs from Kubernetes, React, TensorFlow, Rust, Go, etc. |
| **Chromium test history** | Sheriffed labels for flake/bug classification |

### 5.2 Pipeline

1. **Mine** ~300–500 failure logs from public GitHub Actions runs across major OSS repos.
2. **Cluster** offline by failure category using a one-shot LLM call (this is *data prep*, not the reward signal).
3. **Templatize** each cluster: extract structural pattern, parameterize varying slots (PIDs, paths, timestamps, error tokens) sampled from realistic distributions.
4. **Generate** parametric scenarios per family by inflating templates with seeded RNG, ensuring the failure pattern is preserved as ground truth.
5. **Label** correctness deterministically at generation time — the generator *is* the labeler.
6. **Annotate** each scenario with `informative_tools`, `minimal_evidence_set`, and `correct_terminal_action`.
7. **Validate** by spot-checking against the original real failures: do the synthetic logs preserve the diagnostic patterns?

### 5.3 Why we don't use parallel.ai or other generative-LLM data services

LLM-generated logs feel synthetic to humans and to small models trained on real distributions. We use real-failure seeds and templating instead. Generative LLMs are used (offline, one-shot) only for *clustering* mined real logs — not for generating new ones.

### 5.4 Minimum sufficient corpus

- **200** total scenarios for v1 (target: 300 if time permits)
- **Per family**: minimum 25, target 35
- **Train / val / held-out**: 70 / 15 / 15
- Held-out includes 100% of the `ambiguous` family for the calibration probe

---

## 6. Training pipeline

### 6.1 Base model

**Qwen3.5-4B-Instruct** with LoRA r=16 via Unsloth 4-bit. Fallback Qwen3.5-2B if 4B exceeds compute budget after first 50 GRPO steps.

### 6.2 SFT warmstart

The non-zero baseline required by the FAQ.

1. Run a controlled trajectory generation loop using a frontier model (Claude Opus / GPT-5) on the env. ~3000 trajectories.
2. Filter by reward score (top 30%) → ~1000 high-quality trajectories.
3. SFT for 2–3 epochs. Cost ~$1.

### 6.3 GRPO fine-tuning

TRL + Unsloth, group size 8, batch 2, LR 5e-6, KL coef 0.04 (tuned). 3000 optimization steps.

### 6.4 Evaluation

Multi-baseline matrix on the 30 held-out scenarios + 15 ambiguous scenarios:

| Baseline | Description |
|---|---|
| Random | Random tool calls + random terminal action |
| Heuristic | Hand-coded rule-based classifier |
| Qwen3.5-4B zero-shot | Same model, no training |
| Qwen3.5-9B zero-shot | Larger same-family, no training |
| **Trained (ours)** | Qwen3.5-4B + SFT + GRPO |

3 seeds per row.

Metrics: diagnosis accuracy (overall + per-class), action appropriateness, mean tool-call cost, time-to-resolution, Brier score on ambiguous.

### 6.5 Ablations

Ablate each reward layer (set its weight to 0, retrain a short run) and report the delta. Specifically required by judges' "ablation studies" effort surface.

---

## 7. Results [FILL POST-TRAIN]

### 7.1 Headline plot

**[FILL: training reward curve, baseline-vs-trained, x-axis = training steps, y-axis = mean episode reward]**

### 7.2 Baseline comparison table

[FILL: 5-row × 6-metric table]

### 7.3 Per-component reward breakdown

[FILL: stacked bar of R_diagnosis, R_action_quality, etc., before vs after training]

### 7.4 Calibration curve on ambiguous

[FILL: reliability diagram, baseline vs trained]

### 7.5 Side-by-side replay GIFs

[FILL: GIF of baseline quarantining a real bug → simulated outage; GIF of trained agent investigating + filing bug correctly]

### 7.6 Ablation table

[FILL: 6 rows, one per reward layer dropped, columns = key metrics]

---

## 8. Why it matters

CI-Triage is a $X-billion problem hiding in plain sight. Every dev org has it. Every dev has been burned by it. Today it is solved by the most expensive engineer on the team being woken at 3am to read logs. Statistical tools have failed at the investigative-reasoning layer because investigation is sequential and contextual, not pattern-matching.

This environment shows that an LLM trained with verifiable RL on a well-designed reward can do the investigation. Beyond the immediate use case, the methodology — multi-step tool-use with composable rubric rewards, dense per-step shaping, and calibrated abstain — generalizes to any ambiguous-diagnosis problem in software engineering: production incidents, security alert triage, code review for race conditions, etc.

We trained a 4B model. The same recipe scales to frontier scale. **What we built is a recipe, not a product.**

---

## 9. How to run

### 9.1 The environment

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and run
git clone https://github.com/<user>/ci-triage-env
cd ci-triage-env
uv sync --all-extras
uv run python -m ci_triage_env.env.server   # FastAPI server on port 8000
```

The HF Space mirror is auto-built from the same repo via GitHub Actions sync. The Space uses `requirements.txt` (auto-exported from `uv.lock`) for its container.

### 9.2 Re-run the training

[FILL: link to Colab notebook]

### 9.3 Inspect a trained agent on a single scenario

[FILL: link to interactive viewer / replay tool]

---

## 10. Engineering & reproducibility

- Standard Gym-style `reset / step / state` API
- OpenEnv `Environment` base class, client/server separation respected
- Valid `openenv.yaml` manifest
- No reserved tool names used for MCP tools
- All rewards deterministic given `(scenario_id, seed, action_history)`
- All scenarios JSON-serializable; corpus committed to repo (gitignored heavy artifacts referenced via HF dataset)

---

## 11. Limitations and honest disclosures

- **The env simulates CI; it is not a real CI system.** Realism is bounded by our scenario library. The pattern of *investigative reasoning* generalizes; specific log formats may not.
- **The trained 4B model is a methodology demo, not a deployable triage agent.** Production deployment would require larger model, real-system integration, human-in-the-loop, security/permissions hardening.
- **Frontier LLMs zero-shot are competitive on simple cases.** Our advantage is on the *trajectory-reasoning* and *calibration* dimensions, not raw classification accuracy.
- **The `ambiguous` class is the hardest to specify cleanly.** A scenario being "genuinely ambiguous" is itself a judgment call; we mitigate via panel-labeling and held-out validation.

---

## 12. Future work

The following extensions are explicitly scaffolded for but not enabled in v1. Schema fields, reward components, and weight constants are all in place; activating them requires only a code path to be re-enabled and a re-train.

- **Counterfactual probe.** Periodically asks the agent to predict the outcome of an alternate action at a snapshot point in the trajectory. Adds world-model learning. Deferred because (a) signal is sparse at our training scale, (b) env-side replay machinery adds engineering risk, (c) Brier-score calibration on the ambiguous family captures the same dimension at far lower cost.
- **Multi-app extension.** The current env simulates one CI surface. Extending to deploy / canary / rollback decisions multiplies the action space and would create a Multi-App RL Environment for Enterprise Workflows (a separate hackathon bonus track).
- **Live-system mode.** Replace the simulated tool outputs with calls to a real CI system (read_logs → actual log API; rerun_test → actual rerun) for production deployment. Requires safety/permissions hardening.

## 13. Team & acknowledgements

[FILL: team credits, mention of Scaler × Meta PyTorch OpenEnv hackathon, dataset citations]

---

## 14. Citations

[Adya et al. 2000, Bell et al. 2018, Lam et al. 2019, Alshammari et al. 2021, Zhu et al. 2019, Google SRE book Ch.31, OpenEnv documentation, TRL/Unsloth documentation, Qwen3.5 release notes]
