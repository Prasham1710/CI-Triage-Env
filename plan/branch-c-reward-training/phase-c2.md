# Phase C2 — Composite Reward + Format Gate

**Owner:** Branch C.
**Prerequisite:** C1 merged.
**Estimated time:** 2–3 hours.
**Budget impact:** $0 (CPU only).

---

## Outcome

The composite reward function. By end of phase:

1. `composite.py` wires all 9 components with frozen weights.
2. Format gate is multiplicative — fail = total reward is 0.
3. `compute_reward(trace, scenario, **kwargs) -> RewardBreakdown` is the single entrypoint Branch A's trace writer and Branch C's training loop both call.
4. Replay verifier: given a trace JSON, reproduces the exact reward score that was computed during training.
5. Reward weights and ranges documented in `weights.md`.
6. Integration tests verify: ideal trajectory ~ +1.0, worst trajectory ~ -1.0, format-fail = 0.
7. **Gate-1 entry criterion met for Branch C.**

---

## Files to create

### `src/ci_triage_env/rewards/weights.py`

```python
"""Frozen reward weights. Changing these requires team approval."""

REWARD_WEIGHTS = {
    "diagnosis": 0.25,
    "action_quality": 0.20,
    "cost_efficiency": 0.15,
    "investigation": 0.15,
    "time": 0.10,
    "anti_gaming": 0.15,
    # minimal_evidence is folded into investigation, not added separately
}

# Counterfactual probe is deferred to v2 (see plan/branch-a-env-core/phase-a4.md).
# v1 keeps the term at zero weight so the composite code path exists but is dormant.
# To activate in v2: set this to 0.10 and bump REWARD_VERSION.
COUNTERFACTUAL_WEIGHT = 0.0

REWARD_VERSION = "1.0"
```

### `src/ci_triage_env/rewards/composite.py`

```python
from .format_gate import FormatGate
from .diagnosis import DiagnosisReward
from .action_quality import ActionQualityReward
from .cost_efficiency import CostEfficiencyReward
from .investigation import InvestigationReward
from .time_penalty import TimePenaltyReward
from .anti_gaming import AntiGamingReward
from .minimal_evidence import MinimalEvidenceReward
from .counterfactual_predict import CounterfactualPredictReward
from .weights import REWARD_WEIGHTS, COUNTERFACTUAL_WEIGHT, REWARD_VERSION
from ..schemas.reward import RewardBreakdown, ComponentScore, CounterfactualScore

class CompositeReward:
    """Single entrypoint for computing a full reward breakdown from a trace."""

    def __init__(self, weights: dict | None = None,
                 cf_weight: float | None = None,
                 quarantine_window: list[str] | None = None):
        self.weights = weights or REWARD_WEIGHTS
        self.cf_weight = cf_weight if cf_weight is not None else COUNTERFACTUAL_WEIGHT
        self.quarantine_window = quarantine_window or []

    def compute(self, trace: EpisodeTrace, scenario: Scenario) -> RewardBreakdown:
        # Format gate
        gate = FormatGate().score(trace, scenario)
        gate_passed = gate.raw > 0.5

        components: dict[str, ComponentScore] = {"format_gate": gate}

        # Compute each component (always, even if gate failed — for debug visibility)
        diag = DiagnosisReward().score(trace, scenario)
        diag = self._reweight(diag, self.weights["diagnosis"])
        components["diagnosis"] = diag

        action = ActionQualityReward().score(trace, scenario)
        action = self._reweight(action, self.weights["action_quality"])
        components["action_quality"] = action

        cost = CostEfficiencyReward().score(trace, scenario)
        cost = self._reweight(cost, self.weights["cost_efficiency"])
        components["cost_efficiency"] = cost

        investigation = InvestigationReward().score(trace, scenario)
        # Fold minimal_evidence into investigation
        min_ev = MinimalEvidenceReward().score(trace, scenario)
        investigation_combined = self._fold_minimal_evidence(investigation, min_ev)
        investigation_combined = self._reweight(investigation_combined, self.weights["investigation"])
        components["investigation"] = investigation_combined
        components["minimal_evidence"] = min_ev   # included in breakdown for visibility

        time_pen = TimePenaltyReward().score(trace, scenario)
        time_pen = self._reweight(time_pen, self.weights["time"])
        components["time"] = time_pen

        anti = AntiGamingReward(recent_episode_actions=self.quarantine_window).score(trace, scenario)
        anti = self._reweight(anti, self.weights["anti_gaming"])
        components["anti_gaming"] = anti

        # Sum the gated composite
        gated_sum = sum(c.weighted for k, c in components.items()
                        if k not in {"format_gate", "minimal_evidence"})

        if not gate_passed:
            total = 0.0
        else:
            total = gated_sum

        # Counterfactual: dormant in v1 (cf_weight=0.0). The code path is preserved
        # so v2 re-add is a single-line change in weights.py.
        cf_score = None
        if gate_passed and self.cf_weight > 0.0 and trace.counterfactual_replay and trace.counterfactual_replay.fired:
            cf = CounterfactualPredictReward().score(trace, scenario)
            cf_weighted = cf.raw * self.cf_weight
            total += cf_weighted
            cf_score = CounterfactualScore(
                fired=True,
                probe_step=trace.counterfactual_replay.probe_step,
                probe_action=trace.counterfactual_replay.alternate_action.tool_name if trace.counterfactual_replay.alternate_action else "",
                predicted_outcome=trace.counterfactual_replay.predicted_outcome or "",
                actual_outcome=trace.counterfactual_replay.actual_outcome or "",
                brier_score=cf.sub_scores.get("brier", 0.0) if hasattr(cf, "sub_scores") else 0.0,
            )

        return RewardBreakdown(
            schema_version=REWARD_VERSION,
            total=total,
            format_gate=gate_passed,
            components=components,
            counterfactual=cf_score,
        )

    @staticmethod
    def _reweight(comp: ComponentScore, new_weight: float) -> ComponentScore:
        return ComponentScore(
            raw=comp.raw,
            weighted=comp.raw * new_weight,
            weight=new_weight,
            sub_scores=comp.sub_scores,
        )

    @staticmethod
    def _fold_minimal_evidence(inv: ComponentScore, min_ev: ComponentScore) -> ComponentScore:
        # Investigation raw multiplied by (1 + 0.3 * min_ev.raw)
        # so a perfect minimal-evidence trajectory boosts investigation by 30%
        multiplier = 1.0 + 0.3 * max(min_ev.raw, 0.0)
        new_raw = inv.raw * multiplier
        new_raw = max(min(new_raw, 1.0), -1.0)
        return ComponentScore(
            raw=new_raw, weighted=new_raw, weight=inv.weight,
            sub_scores={**inv.sub_scores, "min_ev_multiplier": multiplier},
        )

# Convenience function used by Branch A and C
def compute_reward(trace: EpisodeTrace, scenario: Scenario, **kwargs) -> RewardBreakdown:
    return CompositeReward(**kwargs).compute(trace, scenario)
```

### `src/ci_triage_env/rewards/replay.py`

```python
def replay_reward_from_disk(trace_path: Path, scenario_path: Path) -> RewardBreakdown:
    """Recompute reward from trace JSON + scenario JSON. Used for verification."""
    trace = EpisodeTrace.model_validate_json(trace_path.read_text())
    scenario = Scenario.model_validate_json(scenario_path.read_text())
    return compute_reward(trace, scenario)

def assert_reward_reproducible(trace: EpisodeTrace, scenario: Scenario) -> None:
    """Compute twice; assert identical. Catches non-determinism early."""
    r1 = compute_reward(trace, scenario)
    r2 = compute_reward(trace, scenario)
    assert r1.total == r2.total, f"Reward not reproducible: {r1.total} != {r2.total}"
    for k in r1.components:
        assert r1.components[k].raw == r2.components[k].raw
```

### `src/ci_triage_env/rewards/weights.md`

A documentation file (markdown, not python) explaining provenance:

```markdown
# Reward Weights — Provenance and Justification

## Component weights (post format-gate)

| Component | Weight | Rationale |
|---|---|---|
| diagnosis | 0.25 | Largest single signal — getting the diagnosis right is the headline metric |
| action_quality | 0.20 | Second-largest; downstream behavior matters distinctly from belief |
| cost_efficiency | 0.15 | Forces surgical investigation; grounded in CircleCI/GitHub Actions compute pricing |
| investigation | 0.15 | Trajectory shaping; this is where RL beats classification |
| anti_gaming | 0.15 | Anti-exploit; prevents quarantine-everything and single-shot guessing |
| time | 0.10 | Mild speed pressure |
| counterfactual | 0.0 in v1 (deferred) | Scaffolded but dormant. v2 activates by setting weight to 0.10. |

## Diagnosis confusion-matrix weights

Asymmetry rationale:

- **Quarantine real bug = -1.0**: catastrophic; ships bug to production. From DeFlaker (Bell et al., FSE 2018), prod-shipped bug median cost is ~$100k vs. flake-investigation median cost ~$200.
- **Rerun real bug = -0.7**: hides bug under "intermittent failure" narrative; eventually surfaces but with delay cost.
- **Wrong infra-vs-bug ≈ -0.4 to -0.5**: paging wrong team is annoying but recoverable.
- **Mild flake-confusion (race vs timing) = +0.2**: both are flakes, both correctly trigger flake-handling pathway.
- **Default off-diagonal = -0.5**: penalty for any confidently-wrong answer.

References:
- Bell, J. et al. "DeFlaker: Automatically Detecting Flaky Tests." FSE 2018.
- Lam, W. et al. "iDFlakies: A Framework for Detecting and Partially Classifying Flaky Tests." ICSE 2019.
- Google SRE book, Ch. 31, "Communication and Collaboration in SRE": cost models for paging.
- CircleCI compute pricing: https://circleci.com/pricing/

## Anti-gaming thresholds

- Quarantine-rate guard threshold: 30% over 50-episode window. Above this, penalty proportional to excess. Calibrated to allow ~25% expected flake rate plus headroom.
- Brier calibration weight: 0.5 in raw [0, 1]; only fires on ambiguous-family scenarios.

## How to change weights

1. Propose change in team chat.
2. Run ablation: rerun GRPO with new weights for 1000 steps, compare reward curve and final eval scores.
3. If improvement, update `weights.py`, bump `REWARD_VERSION`, document in this file.
```

---

## Implementation notes

- **Format gate is computed but NOT in the weighted sum.** It's a multiplicative gate: fail → total = 0; pass → total = sum of weighted components.
- **Counterfactual is additive on top of the gated sum**, not weight-summed inside. Reason: not every episode fires the probe, so it can't be a fixed-fraction component without skewing distribution.
- **`minimal_evidence` is folded** into `investigation` via multiplier in `_fold_minimal_evidence`. It appears in `RewardBreakdown.components` for visibility but its `weighted` is not added to the sum (its weight is 0 in `weights.py`).
- **Quarantine window passed in via constructor.** Trainer maintains a sliding window of recent terminal actions and passes it to `CompositeReward(quarantine_window=...)` per call. For unit tests, pass `None` (no rate computed → no penalty).

---

## Tests required (`tests/rewards/test_composite.py`)

```python
def test_composite_returns_valid_reward_breakdown():
    """compute_reward returns RewardBreakdown that validates against schema."""

def test_format_gate_fail_zeros_total():
    """A trace with malformed action → total == 0, format_gate == False."""

def test_ideal_trajectory_high_score():
    """Mock 'good' trajectory on real_bug → total > 0.5."""

def test_worst_trajectory_low_score():
    """Quarantine on real_bug + redundant calls + format issues → total < -0.3."""

def test_replay_determinism():
    """Compute twice, get identical results."""

def test_replay_from_disk(tmp_path):
    """Write trace + scenario to disk, replay reward, equals in-memory result."""

def test_counterfactual_dormant_in_v1():
    """With COUNTERFACTUAL_WEIGHT=0, even a probe-fired trace contributes 0 to total."""

def test_counterfactual_zero_when_probe_not_fired():
    """Trace with no probe → cf_score is None, total unaffected."""

def test_weights_sum_to_one():
    """Sum of REWARD_WEIGHTS values == 1.0 (within float tolerance), excluding counterfactual."""

def test_minimal_evidence_boosts_investigation():
    """Trajectory using only min_set tools has higher investigation score than one using all tools."""

def test_quarantine_window_penalizes_after_threshold():
    """Recent_window with 80% quarantines → anti_gaming raw < 0."""

def test_reward_version_recorded():
    """RewardBreakdown.schema_version matches REWARD_VERSION."""

def test_compute_reward_handles_no_terminal_action():
    """Budget-exhausted trace (final_action=None) → total < 0, no exception."""
```

Plus integration test:

```python
def test_full_loop_real_bug_correct_diagnosis():
    """End-to-end: scenario(real_bug) + ideal trajectory → all components positive, total > 0.6."""

def test_full_loop_quarantine_real_bug_disaster():
    """End-to-end: scenario(real_bug) + trajectory ending in quarantine_test → action_quality < -1, total negative."""
```

---

## Smoke test (manual)

```python
from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.composite import compute_reward

for outcome in ["good", "bad", "abstain"]:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome=outcome)
    reward = compute_reward(trace, scenario)
    print(f"\n=== {outcome.upper()} ===")
    print(f"Total: {reward.total:+.3f}  Format gate: {'✓' if reward.format_gate else '✗'}")
    for name, score in reward.components.items():
        print(f"  {name:20s} raw={score.raw:+.3f}  weighted={score.weighted:+.3f}")
```

Expected: good trajectory total > bad trajectory total. Format gate True for all (mock trajectories are well-formed).

---

## Gate-1 readiness check

After C2 lands and CI green:

```bash
pytest -q tests/rewards/        # all green
pytest -q tests/schemas/        # sanity
python -c "from ci_triage_env.rewards.composite import compute_reward; print('OK')"
```

Branch C is now Gate-1 ready. C3 (SFT data gen) requires Branch A merged for the real env.

---

## Open questions

1. **Quarantine window size.** N=50 is a guess. Tune onsite based on observed training dynamics.
2. **Should `weights.py` be loaded from a YAML/JSON for ablation runs?** Recommend yes — Phase C6 ablations work by overriding individual weights. Add `compose_with_weights(weights_dict)` helper.

---

## What's NOT in this phase

- SFT trajectory generation (C3)
- Training loop (C4)
- Eval harness (C5)
