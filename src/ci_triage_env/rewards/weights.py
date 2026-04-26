"""Frozen reward weights. Changing these requires team approval + ablation run."""

REWARD_WEIGHTS: dict[str, float] = {
    "diagnosis": 0.25,
    "action_quality": 0.20,
    "cost_efficiency": 0.15,
    "investigation": 0.15,
    "time": 0.10,
    "anti_gaming": 0.15,
    # minimal_evidence is folded into investigation, not added separately
}

# Counterfactual probe deferred to v2 (see plan/branch-a-env-core/phase-a4.md).
# Set to 0.10 and bump REWARD_VERSION to activate in v2.
COUNTERFACTUAL_WEIGHT: float = 0.0

REWARD_VERSION: str = "1.0"
