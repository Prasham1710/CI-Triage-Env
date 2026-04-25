from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.scenario import (
    FailureSummary,
    GroundTruth,
    Scenario,
    ScenarioMetadata,
    TerminalActionSpec,
    ToolOutput,
)

_FAMILY_TO_LABEL: dict[str, DiagnosisLabel] = {
    "real_bug": DiagnosisLabel.REAL_BUG,
    "race_flake": DiagnosisLabel.RACE_FLAKE,
    "timing_flake": DiagnosisLabel.TIMING_FLAKE,
    "infra_network": DiagnosisLabel.INFRA_NETWORK,
    "infra_resource": DiagnosisLabel.INFRA_RESOURCE,
    "dependency_drift": DiagnosisLabel.DEPENDENCY_DRIFT,
    "ambiguous": DiagnosisLabel.AMBIGUOUS,
}


def make_mock_scenario(family: str = "race_flake", seed: int = 42) -> Scenario:
    """Return a fully-populated Scenario with toy tool outputs.

    Used by Branch C unit tests until real scenarios from Branch B are merged.
    """
    if family not in _FAMILY_TO_LABEL:
        raise ValueError(f"unknown family: {family}")

    label = _FAMILY_TO_LABEL[family]
    is_ambiguous = label is DiagnosisLabel.AMBIGUOUS

    tool_outputs = {
        "read_logs": ToolOutput(
            tool_name="read_logs",
            payload={"lines": ["LOG LINE 1", "LOG LINE 2"], "truncated": False},
            cost_units=0.001,
        ),
        "query_flake_history": ToolOutput(
            tool_name="query_flake_history",
            payload={"failure_count": 7, "pass_count": 93, "recent_failures": []},
            cost_units=0.002,
        ),
        "recent_commits": ToolOutput(
            tool_name="recent_commits",
            payload={"commits": [{"sha": "abc123", "msg": "noop"}]},
            cost_units=0.002,
        ),
        "rerun_test": ToolOutput(
            tool_name="rerun_test",
            payload={"results": [{"passed": True}, {"passed": True}]},
            cost_units=0.01,
        ),
    }

    return Scenario(
        schema_version="1.0",
        scenario_id=f"{family}-v1-seed{seed}-mock",
        family=family,
        seed=seed,
        ground_truth=GroundTruth(
            label=label,
            rationale=f"mock rationale for {family}",
            is_ambiguous=is_ambiguous,
            confidence_target=0.5 if is_ambiguous else 1.0,
        ),
        failure_summary=FailureSummary(
            test_name="tests/unit/test_widget.py::test_concurrent_update",
            suite="unit",
            branch="main",
            last_passing_commit="deadbeef",
            initial_log_excerpt="AssertionError: expected 2, got 1",
            timestamp="2026-04-25T12:00:00Z",
        ),
        tool_outputs=tool_outputs,
        informative_tools=["read_logs", "query_flake_history", "rerun_test"],
        minimal_evidence_set=["query_flake_history", "rerun_test"],
        correct_terminal_action=TerminalActionSpec(
            primary="submit_diagnosis",
            args={"diagnosis": label.value, "confidence": 0.9 if not is_ambiguous else 0.5},
            acceptable_alternatives=[],
        ),
        metadata=ScenarioMetadata(
            generator_version="mock-0.1",
            generated_at="2026-04-25T12:00:00Z",
            source_log_hash=None,
            difficulty="medium",
        ),
    )
