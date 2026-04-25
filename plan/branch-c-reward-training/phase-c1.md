# Phase C1 — Reward Components

**Owner:** Branch C.
**Prerequisite:** `phase-0-complete` on `main`. Can start before Gate-1 using mock data.
**Estimated time:** 5–6 hours. **The differentiator phase.**
**Budget impact:** $0 (CPU only).

---

## Outcome

All 9 reward components implemented as `RewardComponent` subclasses. By end of phase:

1. Each component is a separate file under `src/ci_triage_env/rewards/`.
2. Each implements `score(trace: EpisodeTrace, scenario: Scenario) -> ComponentScore`.
3. Each is deterministic, fully tested, and replay-safe (give it a trace, get back the same score every time).
4. All edge cases covered: format failure, empty trajectory, full-tool-call trajectory, abstain-correct, abstain-wrong, probe-fired, probe-not-fired.
5. All C1 unit tests pass.

This is where most of the submission's intellectual contribution lives. Take time on edge cases.

---

## Files to create

### `src/ci_triage_env/rewards/format_gate.py`

```python
class FormatGate(RewardComponent):
    """Validates trajectory follows MCP tool-call schema and final action schema.
    Returns 0 (gate fails) or 1 (passes). Used as multiplicative gate."""

    name = "format_gate"
    default_weight = 1.0   # multiplicative

    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore:
        # 1. Every ToolCall in history validates against its tool's args_schema
        # 2. The terminal action (if present) is a valid TerminalAction
        # 3. If probe fired, the probe response is well-formed
        for record in trace.episode.history:
            if isinstance(record.action, ToolCall):
                tool_def = TOOL_DEF_BY_NAME.get(record.action.tool_name)
                if tool_def is None:
                    return self._fail("unknown tool name")
                try:
                    jsonschema.validate(record.action.args, tool_def.args_schema)
                except jsonschema.ValidationError as e:
                    return self._fail(f"args validation failed: {e}")
            elif isinstance(record.action, TerminalAction):
                if record.action.diagnosis not in DiagnosisLabel:
                    return self._fail("invalid diagnosis label")
                if not (0 <= record.action.confidence <= 1):
                    return self._fail("confidence out of range")
        # Probe response check
        if trace.counterfactual_replay and not trace.counterfactual_replay.predicted_outcome:
            return self._fail("probe fired but no prediction")
        return ComponentScore(raw=1.0, weighted=1.0, weight=1.0, sub_scores={"valid": 1.0})

    def _fail(self, reason: str) -> ComponentScore:
        return ComponentScore(raw=0.0, weighted=0.0, weight=1.0, sub_scores={"reason_code": 0.0})
```

### `src/ci_triage_env/rewards/diagnosis.py`

```python
DIAGNOSIS_REWARD_MATRIX: dict[tuple[str, str], float] = {
    # (predicted, true) -> reward
    # diagonal = 1.0; off-diagonal asymmetric
    ("real_bug", "real_bug"): 1.0,
    ("race_flake", "race_flake"): 1.0,
    ("timing_flake", "timing_flake"): 1.0,
    ("infra_network", "infra_network"): 1.0,
    ("infra_resource", "infra_resource"): 1.0,
    ("dependency_drift", "dependency_drift"): 1.0,
    ("ambiguous", "ambiguous"): 1.0,

    # Worst: predicting flake when it's a real bug (ships to prod)
    ("race_flake", "real_bug"): -1.0,
    ("timing_flake", "real_bug"): -1.0,
    ("ambiguous", "real_bug"): -0.7,

    # Bad: predicting infra when it's a real bug (file with wrong team)
    ("infra_network", "real_bug"): -0.5,
    ("infra_resource", "real_bug"): -0.5,
    ("dependency_drift", "real_bug"): -0.4,

    # Bad: predicting bug when it's a flake (false-alarm noise)
    ("real_bug", "race_flake"): -0.3,
    ("real_bug", "timing_flake"): -0.3,

    # Bad: predicting bug when it's infra (wastes engineering time)
    ("real_bug", "infra_network"): -0.4,
    ("real_bug", "infra_resource"): -0.4,
    ("real_bug", "dependency_drift"): -0.2,   # dep drift kinda is a bug

    # Mild: confusing flake types
    ("race_flake", "timing_flake"): 0.2,
    ("timing_flake", "race_flake"): 0.2,
    ("infra_network", "infra_resource"): 0.1,
    ("infra_resource", "infra_network"): 0.1,

    # Confidently abstaining when there IS a clear cause
    ("ambiguous", "race_flake"): 0.0,
    ("ambiguous", "timing_flake"): 0.0,
    ("ambiguous", "infra_network"): 0.0,
    ("ambiguous", "infra_resource"): 0.0,
    ("ambiguous", "dependency_drift"): 0.0,
    # ambiguous on real_bug already covered above (-0.7) — most expensive

    # Default off-diagonal: -0.5
}

def lookup_reward(predicted: str, true: str) -> float:
    return DIAGNOSIS_REWARD_MATRIX.get((predicted, true), -0.5)

class DiagnosisReward(RewardComponent):
    name = "diagnosis"
    default_weight = 0.25

    def score(self, trace, scenario) -> ComponentScore:
        if trace.episode.final_action is None:
            # Budget exhausted without diagnosis = -1
            return ComponentScore(raw=-1.0, weighted=-1.0 * self.default_weight,
                                  weight=self.default_weight, sub_scores={"no_diagnosis": -1.0})
        predicted = trace.episode.final_action.diagnosis
        true = scenario.ground_truth.label
        raw = lookup_reward(predicted, true)
        return ComponentScore(
            raw=raw, weighted=raw * self.default_weight,
            weight=self.default_weight,
            sub_scores={
                "matrix_lookup": raw,
                "predicted": predicted,
                "true": true,
            },
        )
```

> **Document weights provenance.** Create `src/ci_triage_env/rewards/weights.md` citing the sources for each weight (DeFlaker, Google SRE book, etc.). README references this file.

### `src/ci_triage_env/rewards/action_quality.py`

```python
ACTION_REWARD_MATRIX: dict[tuple[str, str], float] = {
    # (action_name, ground_truth_family) -> reward
    ("file_bug", "real_bug"): 1.0,
    ("file_bug", "dependency_drift"): 0.7,    # right department, slightly different
    ("file_bug", "race_flake"): -0.5,         # wrong action for flake
    ("file_bug", "timing_flake"): -0.3,
    ("file_bug", "infra_network"): -0.5,
    ("file_bug", "infra_resource"): -0.5,
    ("file_bug", "ambiguous"): -0.2,          # premature

    ("quarantine_test", "race_flake"): 1.0,
    ("quarantine_test", "timing_flake"): 0.8,
    ("quarantine_test", "real_bug"): -1.5,    # CATASTROPHIC: ships bug to prod
    ("quarantine_test", "infra_network"): -0.3,
    ("quarantine_test", "infra_resource"): -0.3,
    ("quarantine_test", "dependency_drift"): -0.5,
    ("quarantine_test", "ambiguous"): -0.3,

    ("rerun_test", "race_flake"): 0.6,        # reasonable for flake
    ("rerun_test", "timing_flake"): 0.6,
    ("rerun_test", "infra_network"): 0.8,     # right action for transient infra
    ("rerun_test", "infra_resource"): 0.5,
    ("rerun_test", "real_bug"): -0.6,         # hides bug
    ("rerun_test", "dependency_drift"): -0.3,
    ("rerun_test", "ambiguous"): 0.2,         # safe

    ("ping_owner", "infra_resource"): 0.7,    # capacity team
    ("ping_owner", "infra_network"): 0.5,
    ("ping_owner", "real_bug"): 0.4,          # informing test owner
    ("ping_owner", "dependency_drift"): 0.6,
    ("ping_owner", "race_flake"): 0.0,
    ("ping_owner", "timing_flake"): 0.0,
    ("ping_owner", "ambiguous"): 0.3,
}

class ActionQualityReward(RewardComponent):
    name = "action_quality"
    default_weight = 0.20

    def score(self, trace, scenario) -> ComponentScore:
        if trace.episode.final_action is None:
            return ComponentScore(raw=-0.5, weighted=-0.5 * self.default_weight,
                                  weight=self.default_weight, sub_scores={"no_action": -0.5})
        true = scenario.ground_truth.label
        secondary = trace.episode.final_action.secondary_actions
        if not secondary:
            # Diagnosing without taking action — neutral
            return ComponentScore(raw=0.0, weighted=0.0, weight=self.default_weight,
                                  sub_scores={"no_secondary": 0.0})
        # Sum rewards across all secondary actions
        sub_scores = {}
        total = 0.0
        for sa in secondary:
            r = ACTION_REWARD_MATRIX.get((sa.name, true), 0.0)
            sub_scores[sa.name] = r
            total += r
        # Cap total to avoid stacking exploits
        capped = max(min(total, 1.5), -2.0)
        return ComponentScore(raw=capped, weighted=capped * self.default_weight,
                              weight=self.default_weight, sub_scores=sub_scores)
```

### `src/ci_triage_env/rewards/cost_efficiency.py`

```python
class CostEfficiencyReward(RewardComponent):
    name = "cost_efficiency"
    default_weight = 0.15

    BUDGET_REFERENCE = 5.0   # full budget; reaching 0% spend = max reward

    def score(self, trace, scenario) -> ComponentScore:
        total_spent = sum(rec.cost_charged for rec in trace.episode.history)
        # Map spend to [-1, 1]:
        # 0 cost = 1.0, full budget = -1.0, no over-budget penalty (budget already enforced)
        ratio = total_spent / self.BUDGET_REFERENCE
        raw = 1.0 - 2 * min(ratio, 1.0)
        return ComponentScore(raw=raw, weighted=raw * self.default_weight,
                              weight=self.default_weight,
                              sub_scores={"total_cost": total_spent, "ratio": ratio})
```

### `src/ci_triage_env/rewards/investigation.py`

The shaping reward — the most nuanced.

```python
class InvestigationReward(RewardComponent):
    name = "investigation"
    default_weight = 0.15

    def score(self, trace, scenario) -> ComponentScore:
        called_tools = [rec.action.tool_name for rec in trace.episode.history
                        if isinstance(rec.action, ToolCall)]

        # Coverage: fraction of informative_tools that were called
        informative = set(scenario.informative_tools)
        called_informative = sum(1 for t in called_tools if t in informative)
        coverage = called_informative / max(len(informative), 1)

        # Redundancy: tool calls that were repeats (same tool, same args)
        seen_calls = set()
        redundancy_count = 0
        for rec in trace.episode.history:
            if isinstance(rec.action, ToolCall):
                key = (rec.action.tool_name, json.dumps(rec.action.args, sort_keys=True))
                if key in seen_calls:
                    redundancy_count += 1
                seen_calls.add(key)
        redundancy_penalty = -0.1 * redundancy_count

        # Order: cheap-before-expensive bonus
        ordered_well = self._compute_ordering_score(called_tools)

        raw = 0.6 * coverage + 0.2 * ordered_well + redundancy_penalty
        raw = max(min(raw, 1.0), -1.0)
        return ComponentScore(
            raw=raw, weighted=raw * self.default_weight, weight=self.default_weight,
            sub_scores={"coverage": coverage, "ordering": ordered_well,
                        "redundancy_penalty": redundancy_penalty},
        )

    def _compute_ordering_score(self, tools: list[str]) -> float:
        # Ideal pattern: investigation tools (read_logs, query_*) before action tools (rerun_test, file_bug)
        cheap = {"read_logs", "query_flake_history", "recent_commits", "check_owner",
                 "inspect_test_code", "cluster_metrics"}
        expensive = {"rerun_test", "run_diagnostic", "file_bug", "ping_owner", "quarantine_test"}
        violations = 0
        seen_expensive = False
        for t in tools:
            if t in expensive:
                seen_expensive = True
            elif t in cheap and seen_expensive:
                violations += 1   # cheap call after expensive = bad ordering
        return max(1.0 - 0.2 * violations, 0.0)
```

### `src/ci_triage_env/rewards/time_penalty.py`

```python
class TimePenaltyReward(RewardComponent):
    name = "time"
    default_weight = 0.10

    PER_STEP_PENALTY = 0.02
    REFERENCE_STEPS = 6   # ideal episode length

    def score(self, trace, scenario) -> ComponentScore:
        steps = len([r for r in trace.episode.history if isinstance(r.action, ToolCall)])
        # 0 penalty at <= REFERENCE_STEPS, then linear
        excess = max(0, steps - self.REFERENCE_STEPS)
        raw = -self.PER_STEP_PENALTY * excess
        raw = max(raw, -1.0)
        return ComponentScore(raw=raw, weighted=raw * self.default_weight,
                              weight=self.default_weight,
                              sub_scores={"steps": steps, "excess": excess})
```

### `src/ci_triage_env/rewards/anti_gaming.py`

```python
class AntiGamingReward(RewardComponent):
    name = "anti_gaming"
    default_weight = 0.15

    def __init__(self, recent_episode_actions: list[str] | None = None):
        # In production, this is supplied by the trainer's rolling-window state
        self.recent_actions = recent_episode_actions or []

    def score(self, trace, scenario) -> ComponentScore:
        sub = {}

        # 1. No-info-action guard
        n_tool_calls = sum(1 for r in trace.episode.history if isinstance(r.action, ToolCall))
        if trace.episode.final_action and n_tool_calls < 2:
            no_info_penalty = -0.5
        else:
            no_info_penalty = 0.0
        sub["no_info_penalty"] = no_info_penalty

        # 2. Quarantine-rate guard (uses external state)
        quarantine_rate = self._compute_quarantine_rate()
        if quarantine_rate > 0.30:
            quarantine_penalty = -(quarantine_rate - 0.30) * 2.0
        else:
            quarantine_penalty = 0.0
        sub["quarantine_rate"] = quarantine_rate
        sub["quarantine_penalty"] = quarantine_penalty

        # 3. Brier calibration probe (only on ambiguous scenarios)
        brier_bonus = 0.0
        if scenario.ground_truth.is_ambiguous:
            target = scenario.ground_truth.confidence_target
            if trace.episode.final_action:
                pred_conf = trace.episode.final_action.confidence
                brier = (pred_conf - target) ** 2
                brier_bonus = 0.5 * (1.0 - brier)
            else:
                brier_bonus = -0.5
        sub["brier_bonus"] = brier_bonus

        raw = no_info_penalty + quarantine_penalty + brier_bonus
        raw = max(min(raw, 1.0), -1.5)
        return ComponentScore(raw=raw, weighted=raw * self.default_weight,
                              weight=self.default_weight, sub_scores=sub)

    def _compute_quarantine_rate(self) -> float:
        if not self.recent_actions:
            return 0.0
        return sum(1 for a in self.recent_actions if a == "quarantine_test") / len(self.recent_actions)
```

> **Note on quarantine-rate state.** The composite reward (C2) injects the rolling-window state from the trainer's run history. For unit tests, pass an empty list (no rate → no penalty).

### `src/ci_triage_env/rewards/minimal_evidence.py`

```python
class MinimalEvidenceReward(RewardComponent):
    name = "minimal_evidence"
    default_weight = 0.0   # folded into investigation in the composite

    def score(self, trace, scenario) -> ComponentScore:
        called = set()
        for rec in trace.episode.history:
            if isinstance(rec.action, ToolCall):
                called.add(rec.action.tool_name)
        min_set = set(scenario.minimal_evidence_set)
        if not min_set:
            return ComponentScore(raw=0.0, weighted=0.0, weight=0.0, sub_scores={})

        # Did the agent reach the diagnosis using only minimal_evidence tools?
        if trace.episode.final_action and trace.episode.final_action.diagnosis == scenario.ground_truth.label:
            extra = called - min_set
            min_used = called & min_set
            if min_used == min_set:
                # All minimal evidence used; bonus inversely proportional to extras
                bonus = 1.0 - 0.1 * len(extra)
            else:
                bonus = 0.3   # right answer but missing key evidence (lucky guess)
        else:
            bonus = 0.0
        bonus = max(min(bonus, 1.0), -0.5)
        return ComponentScore(raw=bonus, weighted=bonus * self.default_weight,
                              weight=self.default_weight,
                              sub_scores={"min_set_used": list(called & min_set),
                                          "extras": list(called - min_set)})
```

### `src/ci_triage_env/rewards/counterfactual_predict.py`

> **DORMANT in v1.** Counterfactual probe is deferred to v2 (see `plan/branch-a-env-core/phase-a4.md`). This component file exists so v2 re-add is a purely additive change — the env never fires probes in v1, so `trace.counterfactual_replay` is always `None`, and this component always returns zero. Keep the implementation; just know it never gets non-trivial input in v1.

```python
class CounterfactualPredictReward(RewardComponent):
    name = "counterfactual"
    default_weight = 0.0   # v1: dormant. v2: set to 0.10 in weights.py

    def score(self, trace, scenario) -> ComponentScore:
        cf = trace.counterfactual_replay
        if cf is None or not cf.fired:
            return ComponentScore(raw=0.0, weighted=0.0, weight=0.0,
                                  sub_scores={"fired": False})

        # Dead code in v1; reachable only when v2 enables the probe.
        predicted = cf.predicted_outcome
        actual = cf.actual_outcome
        if predicted == actual:
            raw = 1.0
        else:
            raw = -0.5

        return ComponentScore(raw=raw, weighted=raw * self.default_weight,
                              weight=self.default_weight,
                              sub_scores={"predicted": predicted, "actual": actual})
```

---

## Implementation notes

- **Weights are shared in `weights.py` (Phase C2 final).** In C1, each component declares a `default_weight` but the actual weights used at runtime live in `composite.py`'s WEIGHTS dict.
- **All raw scores are clamped to a defined range.** Some are [-1, 1], some [-1.5, 1.0], etc. Document the range in each component's docstring. The composite re-checks bounds.
- **Edge cases.** Every component must handle: `final_action is None` (budget-exhausted episode), empty history, all-tool-calls-but-no-terminal. Test all of these.
- **`MinimalEvidenceReward`'s default_weight is 0.0.** It's NOT in the additive composite. Its score is folded into `InvestigationReward` via a multiplier — the composite takes `min(investigation, 1.0 + minimal_evidence_bonus)`. Spec this clearly in C2.

---

## Tests required (`tests/rewards/`)

One test file per component. Each must verify:

```python
def test_<component>_correct_case_returns_high_score():
    """Trajectory matching the ideal returns score near 1.0."""

def test_<component>_wrong_case_returns_low_score():
    """Trajectory making the worst choice returns score near -1.0 (or component-specific min)."""

def test_<component>_handles_no_terminal_action():
    """Budget-exhausted trajectory (final_action=None) is handled."""

def test_<component>_deterministic():
    """Same trace, same scenario → same score."""

def test_<component>_score_is_in_documented_range():
    """For 100 random traces, raw score is within [min, max] of component's range."""

def test_<component>_subscores_are_meaningful():
    """sub_scores dict has the documented keys."""
```

Specific tests:

```python
# diagnosis.py
def test_diagonal_matches_return_one():
    """All (X, X) pairs in matrix return 1.0."""
def test_quarantine_real_bug_is_worst():
    """quarantine_test on real_bug action_quality is the most negative entry."""

# anti_gaming.py
def test_brier_calibration_perfect_match_bonus():
    """confidence_target=0.4, predicted=0.4 → bonus near 0.5."""
def test_quarantine_rate_above_threshold_penalizes():
    """recent_actions = ["quarantine_test"] * 50 → quarantine_penalty < 0."""

# minimal_evidence.py
def test_using_only_min_set_max_bonus():
    """Trajectory uses ONLY minimal_evidence tools, gets max bonus."""

# counterfactual_predict.py (dormant in v1; tests verify it stays inert)
def test_no_probe_returns_zero():
    """trace.counterfactual_replay = None → score 0, weight 0."""
def test_v1_default_weight_is_zero():
    """default_weight is 0.0 in v1 — proves component is dormant."""
```

---

## Smoke test (manual)

```python
from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.diagnosis import DiagnosisReward
from ci_triage_env.rewards.action_quality import ActionQualityReward
from ci_triage_env.rewards.cost_efficiency import CostEfficiencyReward
from ci_triage_env.rewards.investigation import InvestigationReward
from ci_triage_env.rewards.time_penalty import TimePenaltyReward
from ci_triage_env.rewards.anti_gaming import AntiGamingReward
from ci_triage_env.rewards.minimal_evidence import MinimalEvidenceReward
from ci_triage_env.rewards.counterfactual_predict import CounterfactualPredictReward
from ci_triage_env.rewards.format_gate import FormatGate

scenario = make_mock_scenario("real_bug")
trace_good = make_mock_trajectory(scenario, outcome="good")
trace_bad = make_mock_trajectory(scenario, outcome="bad")

components = [FormatGate(), DiagnosisReward(), ActionQualityReward(),
              CostEfficiencyReward(), InvestigationReward(), TimePenaltyReward(),
              AntiGamingReward(), MinimalEvidenceReward(),
              CounterfactualPredictReward()]
print("=== Good trajectory ===")
for c in components:
    s = c.score(trace_good, scenario)
    print(f"{c.name}: raw={s.raw:.3f} weighted={s.weighted:.3f}")

print("\n=== Bad trajectory ===")
for c in components:
    s = c.score(trace_bad, scenario)
    print(f"{c.name}: raw={s.raw:.3f} weighted={s.weighted:.3f}")
```

Expected: good trajectory has higher score on most components than bad.

---

## Open questions

1. **Confidence-matrix weights validation.** Once weights are set, run a sanity check: simulate 1000 random trajectory/scenario pairs and verify the *correct* trajectory always scores higher than the wrong one. If not, weights need rebalancing.
2. **Should `quarantine_rate` use a wall-clock window or last-N-episodes window?** Last-N is simpler. Recommend N=50.

---

## What's NOT in this phase

- The composite assembler (C2)
- Trajectory generation or training (C3+)
