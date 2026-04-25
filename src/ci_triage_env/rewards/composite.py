"""CompositeReward — wires all 9 components with frozen weights.

Format gate is multiplicative: if it fails, total reward is 0 regardless of other
component scores (but all components are still evaluated for debug visibility).

MinimalEvidenceReward is folded into InvestigationReward via a multiplier and is
NOT added as a separate term in the weighted sum.

CounterfactualPredictReward is dormant in v1 (COUNTERFACTUAL_WEIGHT = 0.0).
"""

from __future__ import annotations

from ci_triage_env.rewards.action_quality import ActionQualityReward
from ci_triage_env.rewards.anti_gaming import AntiGamingReward
from ci_triage_env.rewards.cost_efficiency import CostEfficiencyReward
from ci_triage_env.rewards.counterfactual_predict import CounterfactualPredictReward
from ci_triage_env.rewards.diagnosis import DiagnosisReward
from ci_triage_env.rewards.format_gate import FormatGate
from ci_triage_env.rewards.investigation import InvestigationReward
from ci_triage_env.rewards.minimal_evidence import MinimalEvidenceReward
from ci_triage_env.rewards.time_penalty import TimePenaltyReward
from ci_triage_env.rewards.weights import COUNTERFACTUAL_WEIGHT, REWARD_VERSION, REWARD_WEIGHTS
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import ComponentScore, RewardBreakdown
from ci_triage_env.schemas.scenario import Scenario

# Keys excluded from the additive weighted sum
_EXCLUDED_FROM_SUM = frozenset({"format_gate", "minimal_evidence"})


class CompositeReward:
    """Single entrypoint for computing a full reward breakdown from a trace.

    Args:
        weights: Override the frozen REWARD_WEIGHTS for ablation runs.
        cf_weight: Override COUNTERFACTUAL_WEIGHT (v1 default: 0.0).
        quarantine_window: Recent secondary-action names for anti-gaming quarantine-rate
            computation. Trainer injects its sliding window here; tests pass None.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        cf_weight: float | None = None,
        quarantine_window: list[str] | None = None,
    ) -> None:
        self.weights = weights or dict(REWARD_WEIGHTS)
        self.cf_weight = cf_weight if cf_weight is not None else COUNTERFACTUAL_WEIGHT
        self.quarantine_window: list[str] = quarantine_window or []

    def compute(self, trace: EpisodeTrace, scenario: Scenario) -> RewardBreakdown:
        components: dict[str, ComponentScore] = {}

        # Format gate — evaluated first; result determines multiplicative factor
        gate = FormatGate().score(trace, scenario)
        gate_passed = gate.raw > 0.5
        components["format_gate"] = gate

        # Diagnosis
        diag = DiagnosisReward().score(trace, scenario)
        components["diagnosis"] = self._reweight(diag, self.weights["diagnosis"])

        # Action quality
        action = ActionQualityReward().score(trace, scenario)
        components["action_quality"] = self._reweight(action, self.weights["action_quality"])

        # Cost efficiency
        cost = CostEfficiencyReward().score(trace, scenario)
        components["cost_efficiency"] = self._reweight(cost, self.weights["cost_efficiency"])

        # Investigation + minimal_evidence fold
        inv = InvestigationReward().score(trace, scenario)
        min_ev = MinimalEvidenceReward().score(trace, scenario)
        inv_combined = self._fold_minimal_evidence(inv, min_ev)
        components["investigation"] = self._reweight(inv_combined, self.weights["investigation"])
        components["minimal_evidence"] = min_ev  # visibility only; weight=0, not summed

        # Time penalty
        time_pen = TimePenaltyReward().score(trace, scenario)
        components["time"] = self._reweight(time_pen, self.weights["time"])

        # Anti-gaming
        anti = AntiGamingReward(recent_episode_actions=self.quarantine_window).score(trace, scenario)
        components["anti_gaming"] = self._reweight(anti, self.weights["anti_gaming"])

        # Weighted sum (excluding gate and minimal_evidence)
        gated_sum = sum(
            c.weighted for k, c in components.items() if k not in _EXCLUDED_FROM_SUM
        )

        total = gated_sum if gate_passed else 0.0

        # Counterfactual — dormant in v1 (cf_weight=0.0)
        # In v2: set cf_weight=0.10; counterfactual_replay will carry probe data.
        if gate_passed and self.cf_weight > 0.0 and trace.counterfactual_replay:
            cf = CounterfactualPredictReward().score(trace, scenario)
            total += cf.raw * self.cf_weight

        return RewardBreakdown(
            schema_version=REWARD_VERSION,
            total=total,
            format_gate=gate_passed,
            components=components,
            counterfactual=None,  # v1: probe never fires
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
        """Boost investigation score by up to 30% when minimal evidence is used optimally."""
        multiplier = 1.0 + 0.3 * max(min_ev.raw, 0.0)
        new_raw = max(min(inv.raw * multiplier, 1.0), -1.0)
        return ComponentScore(
            raw=new_raw,
            weighted=new_raw,
            weight=inv.weight,
            sub_scores={**inv.sub_scores, "min_ev_multiplier": multiplier},
        )


def compute_reward(
    trace: EpisodeTrace,
    scenario: Scenario,
    **kwargs,
) -> RewardBreakdown:
    """Convenience entrypoint used by Branch A's trace writer and Branch C's training loop."""
    return CompositeReward(**kwargs).compute(trace, scenario)
