from typing import Literal

from pydantic import BaseModel, Field

from ci_triage_env.schemas.diagnosis import DiagnosisLabel


class ToolCall(BaseModel):
    tool_name: str
    args: dict


class SecondaryAction(BaseModel):
    name: Literal["rerun_test", "quarantine_test", "file_bug", "ping_owner"]
    args: dict


class TerminalAction(BaseModel):
    action_type: Literal["submit_diagnosis"] = "submit_diagnosis"
    diagnosis: DiagnosisLabel
    confidence: float = Field(ge=0.0, le=1.0)
    secondary_actions: list[SecondaryAction] = Field(default_factory=list)
