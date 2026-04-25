from ci_triage_env.schemas.reward import ComponentScore, RewardBreakdown


def _gated_total(rb: RewardBreakdown) -> float:
    return rb.total if rb.format_gate else 0.0


def test_reward_breakdown_round_trip():
    rb = RewardBreakdown(
        schema_version="1.0",
        total=0.75,
        format_gate=True,
        components={
            "diagnosis": ComponentScore(raw=0.8, weighted=0.6, weight=0.75, sub_scores={"a": 0.5}),
        },
        counterfactual=None,
    )
    restored = RewardBreakdown.model_validate_json(rb.model_dump_json())
    assert restored == rb


def test_format_gate_false_implies_zero_effective_total():
    rb = RewardBreakdown(total=0.9, format_gate=False)
    assert _gated_total(rb) == 0.0


def test_format_gate_true_passes_total_through():
    rb = RewardBreakdown(total=0.4, format_gate=True)
    assert _gated_total(rb) == 0.4


def test_counterfactual_dormant_by_default():
    rb = RewardBreakdown(total=0.0, format_gate=True)
    assert rb.counterfactual is None
