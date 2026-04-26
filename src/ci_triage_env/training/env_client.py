"""HTTP client for the CI-Triage OpenEnv server.

Wraps actions into the CITriageAction wire format ({kind, tool_call|terminal})
and unwraps CITriageObservation responses ({done, reward, metadata, payload})
back into domain Observation objects so the training loop can stay protocol-agnostic.

Use MockEnvClient for offline training (no server required).
"""

from __future__ import annotations

import httpx

from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.observation import Observation


def _to_wire_action(action: ToolCall | TerminalAction | dict) -> dict:
    """Convert a domain action into the CITriageAction wire envelope."""
    if isinstance(action, ToolCall):
        return {
            "kind": "tool_call",
            "tool_call": {"tool_name": action.tool_name, "args": action.args},
        }
    if isinstance(action, TerminalAction):
        return {
            "kind": "submit_diagnosis",
            "terminal": action.model_dump(),
        }
    # Raw dict — infer kind from content
    if "kind" in action:
        return action
    if "tool_name" in action:
        return {"kind": "tool_call", "tool_call": action}
    if "action_type" in action or "diagnosis" in action:
        return {"kind": "submit_diagnosis", "terminal": action}
    return action


def _unwrap_obs(data: dict) -> dict:
    """Unwrap CITriageObservation envelope → domain Observation dict."""
    return data.get("payload", data)


class EnvClient:
    """HTTP client for the CI-Triage env server (OpenEnv wire protocol).

    Args:
        base_url: Server base URL. Defaults to http://localhost:8000.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def reset(
        self,
        scenario_id: str | None = None,
        seed_override: int | None = None,
    ) -> Observation:
        """Start a new episode. Returns the initial observation."""
        resp = self._client.post(
            "/reset",
            json={"scenario_id": scenario_id, "seed": seed_override},
        )
        resp.raise_for_status()
        return Observation.model_validate(_unwrap_obs(resp.json()))

    def step(
        self,
        episode_id: str,
        action: ToolCall | TerminalAction | dict,
    ) -> Observation:
        """Send one action; returns the next observation."""
        resp = self._client.post(
            "/step",
            json={"episode_id": episode_id, "action": _to_wire_action(action)},
        )
        resp.raise_for_status()
        return Observation.model_validate(_unwrap_obs(resp.json()))

    def get_state(self, episode_id: str) -> dict:
        """Return raw episode state dict."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        data = resp.json()
        return data.get("payload", data) or {}

    def get_trace(self, episode_id: str) -> EpisodeTrace:
        """Return the full EpisodeTrace after episode termination."""
        resp = self._client.get(f"/trace/{episode_id}")
        resp.raise_for_status()
        return EpisodeTrace.model_validate(resp.json())

    def list_tools(self) -> list[dict]:
        """Return the MCP tool listing from the server."""
        resp = self._client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("result", {}).get("tools", [])

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EnvClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()
