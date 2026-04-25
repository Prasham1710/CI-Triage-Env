# Phase A4 — Counterfactual Probe (DEFERRED to v2)

> **Status:** Deferred. Do not implement in v1. This file is preserved as a re-add-back specification.
>
> **Decision date:** Cut after mentor review. Counterfactual probe was judged overkill for v1 given the 2-day onsite timeline, $90 compute budget, and the risk of splitting reward signal across two objectives on a 4B model with 3000 GRPO steps. The Brier-score calibration probe on the held-out *ambiguous* family (R_anti_game §3.7) covers the calibration angle at far lower engineering cost.

---

## What v1 ships instead

Nothing. Branch A goes A1 → A2 → A3 → A5. There is no A4 phase in v1.

The schema fields that the counterfactual probe would have used (`Observation.probe_question`, `EpisodeTrace.counterfactual_replay`) **stay in Phase 0 schemas as optional with default `None`**. The reward component file `src/ci_triage_env/rewards/counterfactual_predict.py` exists as an inert no-op (returns 0). The composite weight `COUNTERFACTUAL_WEIGHT = 0.0` in `weights.py`.

This means: if v2 ever happens, it's a purely additive change. No schema migration. No reward-component refactor. Just turn on the env-side machinery and flip a weight.

---

## How to re-enable in v2

If you have time onsite after v1 ships and training finishes early:

1. **Implement the env-side scaffolding** described in the original Phase A4 spec (preserved below): `src/ci_triage_env/env/counterfactual.py` with `EpisodeSnapshot`, `ProbeScheduler`, `CounterfactualProbe`. Wire `EpisodeManager.apply_action` to call `probe.maybe_capture` after each step, and the server's `/step` handler to inject the probe question after `is_terminal=True`.
2. **Set `COUNTERFACTUAL_WEIGHT = 0.10`** in `src/ci_triage_env/rewards/weights.py`. Bump `REWARD_VERSION` to `"1.1"`.
3. **Restore the probe handling** in `TrajectoryGenerator` (Phase C3) and `TrainingRollout` (Phase C4) — both currently strip the probe code path. The original handling is preserved in this file's appendix.
4. **Re-run GRPO** from the v1 SFT checkpoint for an additional 1000–1500 steps. Do not start from scratch — initialize from the v1 trained model so it doesn't lose primary-task capability.
5. **Re-eval** with the v1 baselines plus the new metric (`counterfactual_correct`).

Estimated v2 effort: ~6 hours (4h env, 1h training-side restoration, 1h re-eval). Cost: ~$8 compute.

---

## Original spec (preserved for v2 reference)

> The text below is the original Phase A4 specification. It is the source of truth for v2 if/when the probe is re-enabled. Read it only if you are working on v2.

### Outcome (in v2)

1. Episodes are deterministic given `(scenario_id, seed, action_history)`.
2. On 20% of episodes, at a randomly-selected step in the middle third, env fires a counterfactual probe.
3. Probe is emitted as a special "probe" observation after the terminal action.
4. Agent's prediction captured in `EpisodeTrace.counterfactual_replay`.
5. Env replays alternate trajectory from snapshot and computes ground-truth outcome.
6. Reward layer (`counterfactual_predict.py`) computes Brier score.

### Files (in v2)

- `src/ci_triage_env/env/counterfactual.py` — `EpisodeSnapshot`, `ProbeScheduler`, `CounterfactualProbe`.
- `src/ci_triage_env/env/episode.py` — extended with `probe: CounterfactualProbe | None = None` constructor arg and `apply_action` hook.
- `src/ci_triage_env/env/server.py` — `/step` handler routes probe-question / probe-response flow.

### Key implementation notes (preserved from original spec)

- Determinism is critical. Replay must reconstruct identical observations from `(seed, step_idx, tool_name)`-derived RNG.
- Outcome categorization rules are the trickiest part: map a tool's payload to one of {`reveals_real_bug`, `reveals_flake`, `reveals_infra`, `no_signal`} via hand-written rules over (tool, payload-pattern, scenario.ground_truth).
- Alternate action selection: pick a different tool than the one taken; never propose `submit_diagnosis` as alternate; never propose the same args.
- Probe-fire schedule deterministic per `episode_seed`, fires at random step in `[N/3, 2N/3]`.

### Tests required (in v2)

```python
def test_replay_determinism()
def test_probe_schedule_deterministic()
def test_probe_schedule_fires_at_expected_rate()
def test_alternate_action_is_different_from_taken()
def test_outcome_categorization_real_bug()
def test_outcome_categorization_flake()
def test_outcome_categorization_no_signal()
def test_full_episode_with_probe_fired()
def test_full_episode_without_probe()
def test_probe_response_required_after_probe_question()
def test_trace_includes_actual_and_predicted_outcomes()
```

---

## What's in v1 that survives

- `Observation.probe_question: ProbeQuestion | None = None` in Phase 0 schema.
- `EpisodeTrace.counterfactual_replay: CounterfactualReplay | None = None` in Phase 0 schema.
- `src/ci_triage_env/rewards/counterfactual_predict.py` — returns `ComponentScore(raw=0, weighted=0, weight=0)` when `trace.counterfactual_replay is None or not fired` (which is always in v1).
- `COUNTERFACTUAL_WEIGHT = 0.0` in `src/ci_triage_env/rewards/weights.py`.

This is the entire v2 hook surface. Everything else gets deleted/skipped in v1.
