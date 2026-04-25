"""HTTP client for the CI-Triage OpenEnv server (Branch A).

Before Gate-1, use MockEnvClient instead — this module requires the server running.
"""

from __future__ import annotations

import httpx

from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.observation import Observation


class EnvClient:
    """HTTP client for the CI-Triage env server.

    Args:
        base_url: Server base URL. Defaults to http://localhost:8000.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def reset(
        self,
        scenario_id: str | None = None,
        seed_override: int | None = None,
    ) -> Observation:
        """Start a new episode. Returns the initial observation (contains failure_summary)."""
        resp = self._client.post(
            "/reset",
            json={"scenario_id": scenario_id, "seed_override": seed_override},
        )
        resp.raise_for_status()
        return Observation.model_validate(resp.json())

    def step(self, episode_id: str, action: ToolCall | TerminalAction | dict) -> Observation:
        """Send one action; returns the next observation."""
        if isinstance(action, ToolCall):
            payload = {"tool_name": action.tool_name, "args": action.args}
        elif isinstance(action, TerminalAction):
            payload = action.model_dump()
        else:
            payload = action
        resp = self._client.post("/step", json={"episode_id": episode_id, "action": payload})
        resp.raise_for_status()
        return Observation.model_validate(resp.json())

    def get_state(self, episode_id: str) -> dict:
        """Return raw episode state dict."""
        resp = self._client.get(f"/state/{episode_id}")
        resp.raise_for_status()
        return resp.json()

    def get_trace(self, episode_id: str) -> EpisodeTrace:
        """Return the full EpisodeTrace after episode termination."""
        resp = self._client.get(f"/trace/{episode_id}")
        resp.raise_for_status()
        return EpisodeTrace.model_validate(resp.json())

    def list_tools(self) -> list[dict]:
        """Return the MCP tool listing from the server."""
        resp = self._client.get("/mcp/tools")
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EnvClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()
