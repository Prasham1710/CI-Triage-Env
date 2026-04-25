"""Tests for CompositeReward and replay verifier (Phase C2)."""

from __future__ import annotations

from pathlib import Path

from ci_triage_env.mock import make_mock_scenario, make_mock_trajectory
from ci_triage_env.rewards.composite import CompositeReward, compute_reward
from ci_triage_env.rewards.replay import assert_reward_reproducible, replay_reward_from_disk
from ci_triage_env.rewards.weights import REWARD_VERSION, REWARD_WEIGHTS
from ci_triage_env.schemas.action import SecondaryAction, TerminalAction, ToolCall
from ci_triage_env.schemas.episode import StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation
from ci_triage_env.schemas.reward import RewardBreakdown


def _dummy_obs(step: int = 0) -> Observation:
    return Observation(
        episode_id="test",
        step=step,
        failure_summary=None,
        tool_response=None,
        budget_remaining=BudgetState(tool_calls_remaining=10, cost_remaining=1.0),
        is_terminal=False,
        probe_question=None,
    )


def _valid_tool_record(step: int = 0) -> StepRecord:
    return StepRecord(
        step=step,
        action=ToolCall(tool_name="read_logs", args={"scope": "full"}),
        observation=_dummy_obs(step),
        cost_charged=0.001,
    )


def _patch_secondary(trace, secondary_actions):
    new_terminal = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=trace.episode.final_action.diagnosis,
        confidence=trace.episode.final_action.confidence,
        secondary_actions=secondary_actions,
    )
    return trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"final_action": new_terminal})}
    )


# ---------------------------------------------------------------------------
# Basic schema + structure
# ---------------------------------------------------------------------------


def test_composite_returns_valid_reward_breakdown() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    result = compute_reward(trace, scenario)
    assert isinstance(result, RewardBreakdown)
    RewardBreakdown.model_validate(result.model_dump())


def test_reward_version_recorded() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    result = compute_reward(trace, scenario)
    assert result.schema_version == REWARD_VERSION


def test_weights_sum_to_one() -> None:
    total = sum(REWARD_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Format gate
# ---------------------------------------------------------------------------


def test_format_gate_fail_zeros_total() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Inject unknown tool to fail format gate
    bad_record = StepRecord(
        step=99,
        action=ToolCall(tool_name="__invalid__", args={}),
        observation=_dummy_obs(99),
        cost_charged=0.0,
    )
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={"history": [bad_record]}
        )}
    )
    result = compute_reward(patched, scenario)
    assert result.total == 0.0
    assert result.format_gate is False


def test_format_gate_pass_records_true() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Single valid tool call only
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(update={"history": [_valid_tool_record()]})}
    )
    result = compute_reward(patched, scenario)
    assert result.format_gate is True


# ---------------------------------------------------------------------------
# Score directions
# ---------------------------------------------------------------------------


def test_ideal_trajectory_high_score() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={"history": [_valid_tool_record(0), _valid_tool_record(1)]}
        )}
    )
    result = compute_reward(patched, scenario)
    assert result.total > 0.0


def test_worst_trajectory_low_score() -> None:
    # Wrong diagnosis + quarantine on real_bug
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="bad")
    wrong_terminal = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=trace.episode.final_action.diagnosis,  # wrong family
        confidence=0.9,
        secondary_actions=[SecondaryAction(name="quarantine_test", args={})],
    )
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={
                "history": [_valid_tool_record(0), _valid_tool_record(1)],
                "final_action": wrong_terminal,
            }
        )}
    )
    result = compute_reward(patched, scenario)
    assert result.total < 0.0


def test_good_beats_bad_trajectory() -> None:
    scenario = make_mock_scenario("race_flake")
    good = make_mock_trajectory(scenario, outcome="good")
    bad = make_mock_trajectory(scenario, outcome="bad")
    good_patched = good.model_copy(
        update={"episode": good.episode.model_copy(
            update={"history": [_valid_tool_record(0), _valid_tool_record(1)]}
        )}
    )
    bad_patched = bad.model_copy(
        update={"episode": bad.episode.model_copy(
            update={"history": [_valid_tool_record(0), _valid_tool_record(1)]}
        )}
    )
    r_good = compute_reward(good_patched, scenario)
    r_bad = compute_reward(bad_patched, scenario)
    assert r_good.total > r_bad.total


# ---------------------------------------------------------------------------
# Determinism + replay
# ---------------------------------------------------------------------------


def test_replay_determinism() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    r1 = compute_reward(trace, scenario)
    r2 = compute_reward(trace, scenario)
    assert r1.total == r2.total
    for k in r1.components:
        assert r1.components[k].raw == r2.components[k].raw


def test_assert_reward_reproducible_passes() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    assert_reward_reproducible(trace, scenario)  # must not raise


def test_replay_from_disk(tmp_path: Path) -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    in_memory = compute_reward(trace, scenario)

    trace_path = tmp_path / "trace.json"
    scenario_path = tmp_path / "scenario.json"
    trace_path.write_text(trace.model_dump_json())
    scenario_path.write_text(scenario.model_dump_json())

    from_disk = replay_reward_from_disk(trace_path, scenario_path)
    assert from_disk.total == in_memory.total
    assert from_disk.format_gate == in_memory.format_gate


# ---------------------------------------------------------------------------
# Counterfactual (dormant in v1)
# ---------------------------------------------------------------------------


def test_counterfactual_dormant_in_v1() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    result = compute_reward(trace, scenario)
    assert result.counterfactual is None


def test_counterfactual_zero_when_probe_not_fired() -> None:
    scenario = make_mock_scenario("race_flake")
    trace = make_mock_trajectory(scenario, outcome="good")
    assert trace.counterfactual_replay is None
    result = compute_reward(trace, scenario)
    assert result.counterfactual is None
    # Verify total is unaffected (same as without cf weight override)
    result_no_cf = CompositeReward(cf_weight=0.0).compute(trace, scenario)
    assert result.total == result_no_cf.total


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_compute_reward_handles_no_terminal_action() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    no_terminal = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={"final_action": None, "history": [_valid_tool_record(0), _valid_tool_record(1)]}
        )}
    )
    result = compute_reward(no_terminal, scenario)
    assert result.total < 0.0  # no diagnosis → diagnosis penalty dominates


def test_minimal_evidence_boosts_investigation() -> None:
    from ci_triage_env.data.generators import GENERATOR_REGISTRY

    # Use a generator scenario with a real minimal_evidence_set
    scenario = GENERATOR_REGISTRY["real_bug"]().generate(seed=42)
    min_set = scenario.minimal_evidence_set
    assert min_set

    # Build a trace using ONLY the minimal evidence tools + correct diagnosis
    min_records = [
        StepRecord(
            step=i,
            action=ToolCall(tool_name=t, args={}),
            observation=_dummy_obs(i),
            cost_charged=0.001,
        )
        for i, t in enumerate(min_set)
    ]
    correct_terminal = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=scenario.ground_truth.label,
        confidence=1.0,
        secondary_actions=[],
    )
    trace_min = make_mock_trajectory(scenario, outcome="good").model_copy(
        update={"episode": make_mock_trajectory(scenario, outcome="good").episode.model_copy(
            update={"history": min_records, "final_action": correct_terminal}
        )}
    )

    # Build a trace using all tools (many extras)
    all_tool_names = ["read_logs", "query_flake_history", "recent_commits",
                      "rerun_test", "cluster_metrics", "check_owner"]
    all_records = [
        StepRecord(
            step=i,
            action=ToolCall(tool_name=t, args={}),
            observation=_dummy_obs(i),
            cost_charged=0.001,
        )
        for i, t in enumerate(all_tool_names)
    ]
    trace_all = make_mock_trajectory(scenario, outcome="good").model_copy(
        update={"episode": make_mock_trajectory(scenario, outcome="good").episode.model_copy(
            update={"history": all_records, "final_action": correct_terminal}
        )}
    )

    r_min = compute_reward(trace_min, scenario)
    r_all = compute_reward(trace_all, scenario)
    # Min-only trace gets boost; check investigation component specifically
    assert r_min.components["investigation"].raw >= r_all.components["investigation"].raw - 0.3


def test_quarantine_window_penalizes_after_threshold() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={"history": [_valid_tool_record(0), _valid_tool_record(1)]}
        )}
    )
    heavy_quarantine = ["quarantine_test"] * 45 + ["file_bug"] * 5  # 90% quarantine rate
    result = CompositeReward(quarantine_window=heavy_quarantine).compute(patched, scenario)
    assert result.components["anti_gaming"].raw < 0.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_full_loop_real_bug_correct_diagnosis() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={"history": [_valid_tool_record(0), _valid_tool_record(1)]}
        )}
    )
    result = compute_reward(patched, scenario)
    assert result.total > 0.0
    assert result.format_gate is True
    # Diagnosis component must be positive (correct answer)
    assert result.components["diagnosis"].raw == 1.0


def test_full_loop_quarantine_real_bug_disaster() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    # Correct family diagnosis but quarantine secondary action
    quarantine_terminal = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=scenario.ground_truth.label,
        confidence=0.9,
        secondary_actions=[SecondaryAction(name="quarantine_test", args={})],
    )
    patched = trace.model_copy(
        update={"episode": trace.episode.model_copy(
            update={
                "history": [_valid_tool_record(0), _valid_tool_record(1)],
                "final_action": quarantine_terminal,
            }
        )}
    )
    result = compute_reward(patched, scenario)
    assert result.components["action_quality"].raw < -1.0
    assert result.total < 0.5  # action quality penalty drags total down


def test_all_component_keys_present() -> None:
    scenario = make_mock_scenario("real_bug")
    trace = make_mock_trajectory(scenario, outcome="good")
    result = compute_reward(trace, scenario)
    expected_keys = {
        "format_gate", "diagnosis", "action_quality", "cost_efficiency",
        "investigation", "minimal_evidence", "time", "anti_gaming",
    }
    assert expected_keys <= set(result.components.keys())
