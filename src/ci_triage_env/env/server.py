"""OpenEnv-compliant CI triage environment.

Public surface (registered automatically by ``openenv.core.env_server.http_server.create_app``):

- ``POST /reset``  → ``ResetResponse`` with the initial domain Observation
- ``POST /step``   → ``StepResponse`` for ``CITriageAction``
- ``GET  /state``  → ``CITriageState`` (current EpisodeState)
- ``GET  /schema`` → action / observation / state JSON Schemas
- ``POST /mcp``    → JSON-RPC ``tools/list`` and ``tools/call``
- ``WS   /mcp``    → MCP over WebSocket
- ``WS   /ws``     → persistent OpenEnv session (canonical ``EnvClient`` path)
- ``GET  /docs``   → Swagger UI

Tools are registered with a ``FastMCP`` server so MCP clients can discover them
via ``tools/list``. Step-level tool calls go through ``CITriageAction``
(``kind="tool_call"``) so budget/cost bookkeeping stays inside the env.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from fastmcp import FastMCP
from openenv.core.env_server.http_server import create_app as openenv_create_app
from openenv.core.env_server.mcp_environment import MCPEnvironment

from ci_triage_env.env.episode import EpisodeManager, EpisodeTerminatedError
from ci_triage_env.env.scenario_loader import load_scenarios
from ci_triage_env.env.tools import ALL_TOOL_HANDLERS, ToolHandler
from ci_triage_env.env.trace import trace_dir, write_trace
from ci_triage_env.env.wire import CITriageAction, CITriageObservation, CITriageState
from ci_triage_env.schemas.observation import Observation as DomainObservation
from ci_triage_env.schemas.scenario import Scenario, ToolOutput
from ci_triage_env.schemas.tools import ALL_TOOLS

logger = logging.getLogger(__name__)


def _build_mcp_server(handlers: list[ToolHandler]) -> FastMCP:
    """Build a FastMCP server advertising the 11 tools.

    The function bodies are stubs in Phase A1: they validate args against the
    schema and return placeholder payloads. Real routing into scenario-specific
    tool outputs lands in A2. Cost / budget bookkeeping happens on the env's
    ``/step`` path via ``CITriageAction(kind="tool_call")``, not here, so that
    one canonical path owns episode state.
    """
    mcp = FastMCP(name="ci-triage")
    handler_map = {h.name: h for h in handlers}
    tool_defs = {t.name: t for t in ALL_TOOLS}

    for tool_def in ALL_TOOLS:
        handler = handler_map[tool_def.name]

        def _make_fn(name: str, h: ToolHandler) -> Any:
            def _fn(arguments: dict | None = None) -> dict:
                args = arguments or {}
                h.validate_args(args)
                placeholder = ToolOutput(
                    tool_name=name,
                    payload={"stub": True, "tool": name},
                    cost_units=h.cost_unit,
                )
                return {"payload": placeholder.payload, "cost_units": placeholder.cost_units}

            _fn.__name__ = name
            return _fn

        mcp.tool(
            name=tool_def.name,
            description=tool_def.description,
        )(_make_fn(tool_def.name, handler))

    # Sanity: ALL_TOOLS already excludes reserved names; the MCPEnvironment will
    # re-validate in __init__ but assert here too for fast feedback.
    reserved = {"reset", "step", "state", "close"}
    assert not (set(tool_defs) & reserved)
    return mcp


class CITriageEnv(MCPEnvironment):
    """CI triage env exposing 11 MCP tools and a structured terminal action."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        scenario_source: str | None = None,
        scenarios: dict[str, Scenario] | None = None,
    ) -> None:
        if scenarios is not None:
            self._scenarios: dict[str, Scenario] = dict(scenarios)
        else:
            source = scenario_source or os.environ.get("CI_TRIAGE_SCENARIO_SOURCE")
            self._scenarios = load_scenarios(source)
        if not self._scenarios:
            raise RuntimeError(
                "no scenarios found; populate data_artifacts/scenarios/*.json or set "
                "CI_TRIAGE_SCENARIO_SOURCE"
            )

        self._handlers: dict[str, ToolHandler] = {h.name: h for h in ALL_TOOL_HANDLERS}
        super().__init__(mcp_server=_build_mcp_server(ALL_TOOL_HANDLERS))

        self._episode: EpisodeManager | None = None

    @property
    def scenarios(self) -> dict[str, Scenario]:
        return self._scenarios

    def _seed_for(self, scenario: Scenario, episode_id: str, override: int | None) -> int:
        if override is not None:
            return override
        return hash((scenario.seed, episode_id)) & 0xFFFFFFFF

    def _wrap(self, domain_obs: DomainObservation) -> CITriageObservation:
        return CITriageObservation(
            done=domain_obs.is_terminal,
            reward=None,
            metadata={},
            payload=domain_obs,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        scenario_id: str | None = None,
        **kwargs: Any,
    ) -> CITriageObservation:
        if scenario_id is None:
            scenario_id = next(iter(self._scenarios))
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise KeyError(f"unknown scenario_id: {scenario_id}")
        if episode_id is None:
            episode_id = str(uuid.uuid4())
        self._episode = EpisodeManager(
            scenario=scenario,
            episode_id=episode_id,
            seed=self._seed_for(scenario, episode_id, seed),
        )
        return self._wrap(self._episode.initial_observation())

    def step(
        self,
        action: CITriageAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> CITriageObservation:
        # Override MCPEnvironment.step so /step always lands here for our typed
        # action. ListToolsAction / CallToolAction still flow through MCP via
        # the /mcp endpoint; clients hitting /step always send CITriageAction.
        return self._step_impl(action, timeout_s=timeout_s, **kwargs)

    def _step_impl(
        self,
        action: CITriageAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> CITriageObservation:
        if self._episode is None:
            raise RuntimeError("env has no active episode; call reset first")
        if self._episode.is_terminated:
            raise EpisodeTerminatedError("episode already terminated")

        if action.kind == "tool_call":
            if action.tool_call is None:
                raise ValueError("kind='tool_call' requires a tool_call payload")
            handler = self._handlers.get(action.tool_call.tool_name)
            if handler is None:
                raise KeyError(f"unknown tool: {action.tool_call.tool_name}")
            domain_obs = self._episode.apply_tool_call(action.tool_call, handler)
        elif action.kind == "submit_diagnosis":
            if action.terminal is None:
                raise ValueError("kind='submit_diagnosis' requires a terminal payload")
            domain_obs = self._episode.apply_terminal(action.terminal)
        else:
            raise ValueError(f"unknown action kind: {action.kind}")

        if domain_obs.is_terminal:
            try:
                write_trace(self._episode, trace_dir())
            except OSError:
                # Trace write is best-effort: never fail an episode because the
                # trace dir is read-only or full. The caller can still inspect
                # state via /state.
                pass

        return self._wrap(domain_obs)

    @property
    def state(self) -> CITriageState:
        if self._episode is None:
            return CITriageState(
                episode_id=None,
                step_count=0,
                payload=None,
            )
        domain_state = self._episode.to_state()
        return CITriageState(
            episode_id=domain_state.episode_id,
            step_count=domain_state.step,
            payload=domain_state,
        )

    def close(self) -> None:
        # Inherit MCPEnvironment.close behavior (clears mcp client/server). We
        # also drop the episode reference so a stale env can't be re-used.
        self._episode = None
        super().close()


# ---------------------------------------------------------------------------
# Module-level FastAPI app for `python -m ci_triage_env.env.server` / uvicorn.
# Use ``build_app()`` from tests to inject scenarios.
# ---------------------------------------------------------------------------

def build_app(env_factory=None):
    """Build the OpenEnv-canonical FastAPI app.

    Args:
        env_factory: Callable returning a fresh ``CITriageEnv`` per HTTP request
            (OpenEnv's stateless HTTP semantics) and per WebSocket session.
            Defaults to ``CITriageEnv`` (loads scenarios from
            ``CI_TRIAGE_SCENARIO_SOURCE`` or ``data_artifacts/scenarios``).
    """
    factory = env_factory or CITriageEnv
    return openenv_create_app(
        env=factory,
        action_cls=CITriageAction,
        observation_cls=CITriageObservation,
        env_name="ci-triage",
        max_concurrent_envs=4,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_app(), host="0.0.0.0", port=8000)
