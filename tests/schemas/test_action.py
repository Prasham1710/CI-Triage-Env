import pytest
from pydantic import ValidationError

from ci_triage_env.schemas.action import SecondaryAction, TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel


def test_tool_call_round_trip():
    call = ToolCall(tool_name="read_logs", args={"scope": "test", "lines": 50})
    restored = ToolCall.model_validate_json(call.model_dump_json())
    assert restored == call


def test_terminal_action_round_trip_with_secondaries():
    action = TerminalAction(
        action_type="submit_diagnosis",
        diagnosis=DiagnosisLabel.RACE_FLAKE,
        confidence=0.85,
        secondary_actions=[
            SecondaryAction(name="quarantine_test", args={"test_name": "t1", "reason": "flake"}),
            SecondaryAction(name="file_bug", args={"title": "x"}),
        ],
    )
    restored = TerminalAction.model_validate_json(action.model_dump_json())
    assert restored == action
    assert restored.diagnosis is DiagnosisLabel.RACE_FLAKE


def test_terminal_action_confidence_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        TerminalAction(
            action_type="submit_diagnosis",
            diagnosis=DiagnosisLabel.REAL_BUG,
            confidence=1.5,
        )


def test_secondary_action_rejects_unknown_name():
    with pytest.raises(ValidationError):
        SecondaryAction(name="not_a_real_action", args={})
