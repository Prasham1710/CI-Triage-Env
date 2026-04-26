"""OpenEnv-compliant wire types for the CI triage env.

These wrap our frozen domain Pydantic schemas into OpenEnv's Action / Observation /
State envelope so the env is consumable by the canonical ``EnvClient``,
``serialize_observation`` / ``deserialize_action`` helpers, and HF Spaces' OpenEnv
runner. The domain schemas in ``ci_triage_env.schemas`` are unchanged.
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import ConfigDict, Field

from ci_triage_env.schemas.action import SecondaryAction, TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.episode import EpisodeState
from ci_triage_env.schemas.observation import Observation as DomainObservation


class CITriageAction(Action):
    """Wire action: discriminated on ``kind``.

    - ``kind="tool_call"`` carries ``tool_call`` (a domain ``ToolCall``).
    - ``kind="submit_diagnosis"`` carries ``terminal`` (a domain ``TerminalAction``).

    A separate envelope (rather than directly subclassing OpenEnv's MCP
    ``CallToolAction``) keeps tool-call cost/budget bookkeeping inside our env
    rather than at the FastMCP layer, while still letting agents discover tools
    via the standard MCP ``tools/list`` route.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: Literal["tool_call", "submit_diagnosis"]
    tool_call: ToolCall | None = None
    terminal: TerminalAction | None = None

    @classmethod
    def from_tool_call(cls, tool_name: str, args: dict | None = None) -> "CITriageAction":
        return cls(kind="tool_call", tool_call=ToolCall(tool_name=tool_name, args=args or {}))

    @classmethod
    def from_terminal(
        cls,
        diagnosis: DiagnosisLabel,
        confidence: float,
        secondary_actions: list[SecondaryAction] | None = None,
    ) -> "CITriageAction":
        return cls(
            kind="submit_diagnosis",
            terminal=TerminalAction(
                diagnosis=diagnosis,
                confidence=confidence,
                secondary_actions=secondary_actions or [],
            ),
        )


class CITriageObservation(Observation):
    """Wire observation: OpenEnv envelope (``done``, ``reward``, ``metadata``) plus
    the full domain ``Observation`` as ``payload`` for clients that need scenario,
    budget, tool response, etc."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    payload: DomainObservation = Field(description="Domain-level observation")


class CITriageState(State):
    """Wire state: OpenEnv envelope plus the full domain ``EpisodeState``."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    payload: EpisodeState | None = Field(
        default=None,
        description="Full domain episode state; None before the first reset.",
    )
