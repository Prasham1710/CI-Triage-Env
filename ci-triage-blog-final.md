---
title: "We Trained a 4B Model to Do CI Triage. Here's How We Made the Reward System Uncheateable."
thumbnail: /blog/assets/ci-triage-env/thumbnail.png
authors:
  - user: Prasham1710
tags:
  - reinforcement-learning
  - openenv
  - agents
  - tool-use
  - grpo
  - qwen
---

# We Trained a 4B Model to Do CI Triage. Here's How We Made the Reward System Uncheateable.

It's 3am. Your phone goes off. CI is red on `main`. A test called `test_payment_processor_idempotency` failed and it's blocking four PRs.

You open the logs. 1,400 lines. You grep for `ERROR`. Three hits — two look like noise, one looks real. You check recent commits. Someone touched the payment module six hours ago. Could be that. Or it could be the DNS blip you had last Tuesday. Or the timing assertion that's been silently flaky for months.

You have three choices and each one is wrong in a different way. Rerun it — and if it's a real bug, you just hid it. It'll ship to production and you'll get paged again at 9am by a customer. Quarantine it — and if it's actually an infra issue, you've just blocked a dozen developers from merging for no reason. File a P1 and ping the owner — and if it's a flake, you've woken up your most senior engineer at 3am for nothing. They will remember this.

This is the problem that every engineering team above 50 people lives with, every single day. And it's not a detection problem — you already have Trunk.io, BuildPulse, Datadog Flaky Test Management. Those tools detect *patterns*. They don't *investigate*. The investigation — reading the logs, correlating the commits, checking the resource metrics, making the call under pressure — is still done by humans.

We built an RL environment to close that gap. We called it **CI-Triage-Env**.

---

## Why This Isn't a Classification Problem

The obvious approach is to frame this as supervised learning. Train a model on (log → label) pairs, ship it. Clean, simple, wrong.

The problem is that the right action at any step depends entirely on what you've already found. If you checked flake history first and the test has a 60% pass rate over 30 runs, that completely changes whether you should bother reading the logs in detail. If cluster metrics show a memory spike across all nodes 4 minutes before the failure, the individual test logs stop mattering. The investigation is sequential. The state changes with every tool call. You can't supervise that.

There's also no dataset. Senior engineers do CI triage through tacit knowledge built over years of 3am incidents. That knowledge doesn't exist as labeled trajectories anywhere. You can't scrape it. And even if you could, you'd be training a model to imitate the *actions*, not to learn the *reasoning* behind them.

And then there's the thing that settled the question for us: we wanted to probe whether the model had actually built a world model of the CI system, or whether it was just pattern-matching on log text. You can't test that with supervised learning. You need a counterfactual probe. More on that later.

So: reinforcement learning. The agent explores, makes mistakes, gets a reward signal grounded in real CI economics, and learns to investigate.

---

## The Environment

We built CI-Triage-Env on [OpenEnv](https://github.com/openenv/openenv) with a standard `reset() / step()` interface. At episode start, the agent gets a failure summary — failing test name, branch, last-passing commit — along with a budget of 12 tool calls and a cost cap. Then it starts working.

The tool surface is split into three layers. Investigation tools are cheap: `read_logs`, `inspect_test_code`, `cluster_metrics`. Context tools are moderate: `query_flake_history`, `recent_commits`, `check_owner`. Action tools are expensive and carry real risk: `rerun_test` costs the same as running a full CI job, `file_bug` costs 30 minutes of a human engineer's time, `ping_owner` burns paging budget credibility. Every tool has a cost weight in dollars-equivalent. The agent has to decide what's worth knowing before it pays to find out.

The episode ends when the agent calls `submit_diagnosis(label, confidence)`. It has to pick one of seven labels: real bug, race-condition flake, timing flake, infra-network, infra-resource, dependency drift, or — critically — `abstain`. That last one is for genuinely ambiguous cases where the evidence is insufficient. Knowing when *not* to be confident is part of what we're training.

We built 200+ scenarios across those seven failure families, generated from real public CI failure data. We mined GitHub Actions logs from Kubernetes, React, Rust, TensorFlow, and similar large open-source repos, clustered them by failure category, templatized the patterns, and generated fresh parameterized instances from seeds. The scenarios are deterministic: the same seed always produces the same failure, which makes the reward signal fully verifiable.

---

## The Reward System — and Why Each Component Exists

This is where most of our design time went. Getting the reward right is the whole job.

The full reward looks like this:

```
R = R_format × (
    0.25 × R_diagnosis
  + 0.20 × R_action_quality
  + 0.15 × R_cost_efficiency
  + 0.15 × R_investigation
  + 0.10 × R_time
  + 0.15 × R_anti_game
) + 0.10 × R_counterfactual
```

Every single component is computed against simulator ground truth. There is no LLM-as-judge anywhere in the pipeline. Here's the reasoning behind each one.

**R_format is a hard gate.** If the model's output doesn't pass strict JSON schema validation, the entire episode reward is zero. No partial credit. This sounds brutal but it's necessary — a triage agent that outputs malformed JSON in production is worse than no agent at all.

**R_diagnosis carries the most weight (25%) and is asymmetric on purpose.** Quarantining a real bug scores −1.0 because it ships to production. Rerunning a real bug scores −0.7 because it hides the bug and delays detection. Filing a bug for an infra issue scores −0.5 because it burns on-call trust. These aren't made-up numbers — they're grounded in Bell et al. FSE 2018 and the Google SRE Book's chapter on incident cost. We documented every weight with citations.

**R_action_quality (20%) scores the action separately from the label.** A model can correctly identify that something is a flake but still file a P0 bug. The diagnosis is right; the behavior is wrong. These need to be scored independently.

**R_investigation (15%) is about how you got there, not just whether you got there.** Every scenario has a hand-labeled `informative_tools` set — the specific tool calls that were necessary or sufficient for the correct diagnosis. We reward coverage of that set, penalize redundant calls (same args, no new information), and give a bonus for good ordering: cheap tools before expensive ones, context before action. `recent_commits` should come before `ping_owner`. This component is what trains the agent to investigate efficiently, not just correctly.

**R_anti_game (15%) was born from thinking hard about how a clever agent would cheat.** If you don't guard against it, the obvious shortcuts are: quarantine everything (100% recall on real bugs, zero precision), always rerun (lowest immediate cost), or just guess from the failure summary without calling any tools. We built specific probes for all three. If the quarantine rate across the last 50 episodes exceeds 30%, all rewards scale down proportionally. Terminal action without at least two informative tool calls gets reward halved. And for ambiguous cases, `submit_diagnosis(confidence)` is scored by Brier loss — overconfidence on hard cases is punished directly.

---

## The Counterfactual Probe — The Part We're Most Proud Of

On 20% of episodes, randomly selected, after the agent submits its terminal action, we don't end the episode. Instead we emit one more observation:

> *"Had you called `rerun_test()` instead of `quarantine_test()` at step 4, what would the outcome have been? Predict: success / failure / needs-more-info, with confidence."*

The environment is fully deterministic from `(scenario_id, seed, action_history)`. To compute the ground truth, we replay from a snapshot at step 4 with the alternative action and observe what actually happens. The agent's reward is the Brier score on its prediction.

Here's why this matters: you cannot score well on this probe by pattern-matching on log text. The logs at the point of the counterfactual question are fixed — they don't tell you what would have happened if you'd acted differently. The only way to answer correctly is to have built an internal model of how the simulated CI system *responds to actions*. That's world-model learning. That's the thing that separates an agent that understands the environment from one that's memorized surface patterns.

We're not aware of other teams shipping a probe like this in an OpenEnv submission.

---

## Training

We used **Qwen3-4B** (`unsloth/Qwen3-4B`) as the base model with **LoRA r=16 in bf16** via Unsloth — bf16 LoRA, not 4-bit QLoRA, following Unsloth's current guidance for the Qwen3 family.

**Stage 1 — SFT warmstart.** We ran a frontier teacher (GPT-4o-mini) over the environment, scored each rollout with our composite reward, and kept the top tier. That left us with **718 high-quality trajectories** which we fine-tuned for 2 epochs on a single A10G Small (24 GB). The run completed end-to-end in ~50 minutes with final training loss around **0.55**.

![SFT training loss — Qwen3-4B + LoRA, 2 epochs on A10G Small](loss_graphs_sft.png)
*Loss decreasing smoothly from ~1.4 → 0.55 over 180 steps. The shape is what we wanted: the warmstart gives GRPO a reasonable starting policy instead of exploring from scratch.*

**Stage 2 — GRPO via TRL.** Configured with group size 2 (memory-bound on A10G Small), batch 1 × grad-accum 4, LR 5e-6, KL coefficient 0.04, multi-turn rollout against `MockEnvClient`. The full pipeline is wired — multi-turn rollout, all 9 reward components computing real values, frozen weights so curves are comparable.

**Where we got stuck.** Submission day, our GRPO loop hit a chain of upstream version conflicts. The Qwen3 stack pulled in `transformers v5` (from git), which pulled in `torchao ≥ 0.13`, which pulled in `torch ≥ 2.7`. With that combination, Unsloth's fused `matmul_lora` kernel ran with mismatched fp16/bf16 tensors during the GRPO forward pass:

```
RuntimeError: self and mat2 must have the same dtype, but got Half and BFloat16
  at unsloth/kernels/utils.py:1059  →  out.addmm_(XA, B.to(dtype), alpha=s)
```

We tried four progressively more aggressive fixes — explicit dtype loading, recursive parameter casting, monkey-patching `matmul_lora` to force bf16, disabling autocast, bypassing Unsloth entirely with vanilla PEFT. The vanilla PEFT path then OOMed at 22 GB on the A10G Small, and we ran out of clock. The blockers are documented in the README; the SFT checkpoint, environment, dataset, and reward replay verifier are all real and shipped.

**Evaluation harness.** We built (but didn't get to run) a comparison against four baselines — random policy, hand-coded heuristic, Qwen3-4B zero-shot, and the SFT checkpoint — across diagnosis accuracy, action appropriateness, tool-call cost, Brier score on ambiguous cases, and counterfactual prediction accuracy. The harness is in `src/ci_triage_env/training/eval.py`.

---

## What This Generalizes To

CI triage is the use case. The pattern generalizes much further.

Any ambiguous-diagnosis problem in software engineering has the same structure: sequential investigation under cost constraints, asymmetric misclassification costs, reward that has to be grounded in simulator truth rather than human judgment. Production incident triage. Security alert triage. Code review for race conditions. The methodology is the thing we actually built. The CI environment is the demonstration.

We trained a 4B model. The same recipe runs at frontier scale. What we built is a recipe, not a product — and we think the recipe is what's interesting.

---

## Try It

```bash
git clone https://huggingface.co/spaces/Prasham1710/ci-triage-env
cd ci-triage-env
pip install -e ".[data,training]"
uvicorn ci_triage_env.env.server:build_app --factory --host 0.0.0.0 --port 8000
# open http://localhost:8000/docs  →  interact with all 11 tools via Swagger
```

- 🤗 **Environment Space (live)**: [ci-triage-env](https://huggingface.co/spaces/Prasham1710/ci-triage-env)
- 🏋️ **Training Space**: [ci-triage-training](https://huggingface.co/spaces/Prasham1710/ci-triage-training)
- 🗂️ **Scenario corpus**: [ci-triage-scenarios](https://huggingface.co/datasets/Prasham1710/ci-triage-scenarios)
- 📚 **SFT trajectories**: [ci-triage-sft](https://huggingface.co/datasets/Prasham1710/ci-triage-sft)
- 🧠 **SFT checkpoint (Qwen3-4B + LoRA)**: [ci-triage-agent-sft](https://huggingface.co/Prasham1710/ci-triage-agent-sft)
- 📓 **Training notebook**: [`notebooks/train_grpo.ipynb`](https://huggingface.co/spaces/Prasham1710/ci-triage-training/blob/main/notebooks/train_grpo.ipynb)

---

*Built for the Scaler × Meta PyTorch OpenEnv Hackathon.*

*Datasets used: DeFlaker (Bell et al. FSE 2018), iDFlakies (Lam et al. ICSE 2019), FlakeFlagger (Alshammari et al. ICSE 2021), LogHub (Zhu et al. ISSRE 2019), GitHub Actions public logs.*
