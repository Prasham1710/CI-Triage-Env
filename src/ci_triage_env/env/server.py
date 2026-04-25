import logging
import os
import random
import threading
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ci_triage_env.env.episode import EpisodeManager
from ci_triage_env.env.scenario_loader import load_scenarios
from ci_triage_env.env.tools import ALL_TOOL_HANDLERS, ToolHandler
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.episode import EpisodeState
from ci_triage_env.schemas.observation import Observation
from ci_triage_env.schemas.scenario import Scenario
from ci_triage_env.schemas.tools import ALL_TOOLS, MCPToolDef

logger = logging.getLogger(__name__)


class CITriageEnv:
    """OpenEnv-style CI triage environment.

    Public surface: 11 MCP tools + reset/step/state lifecycle. The PyPI ``openenv``
    package does not actually expose ``MCPEnvironment`` (its name collides with an
    unrelated gym-style library); per phase-a1.md "If the path differs, update", we
    implement the MCP listing endpoint directly on FastAPI rather than inheriting.
    """

    def __init__(
        self,
        scenario_source: str | None = None,
        scenarios: dict[str, Scenario] | None = None,
    ):
        self._episodes: dict[str, EpisodeManager] = {}
        self._lock = threading.Lock()
        if scenarios is not None:
            self._scenarios = dict(scenarios)
        else:
            self._scenarios = load_scenarios(scenario_source)
        if not self._scenarios:
            raise RuntimeError(
                "no scenarios found; populate data_artifacts/scenarios/*.json or set "
                "CI_TRIAGE_SCENARIO_SOURCE"
            )
        self._tools: dict[str, ToolHandler] = {h.name: h for h in ALL_TOOL_HANDLERS}
        self._tool_defs: dict[str, MCPToolDef] = {t.name: t for t in ALL_TOOLS}

    @property
    def scenarios(self) -> dict[str, Scenario]:
        return self._scenarios

    @property
    def tool_defs(self) -> list[MCPToolDef]:
        return list(self._tool_defs.values())

    def _new_episode_id(self) -> str:
        return str(uuid.uuid4())

    def _seed_for(self, scenario: Scenario, episode_id: str, override: int | None) -> int:
        if override is not None:
            return override
        return hash((scenario.seed, episode_id)) & 0xFFFFFFFF

    def reset(
        self,
        scenario_id: str | None = None,
        seed_override: int | None = None,
    ) -> Observation:
        if scenario_id is None:
            scenario_id = random.choice(list(self._scenarios.keys()))
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise KeyError(scenario_id)
        episode_id = self._new_episode_id()
        seed = self._seed_for(scenario, episode_id, seed_override)
        manager = EpisodeManager(scenario=scenario, episode_id=episode_id, seed=seed)
        with self._lock:
            self._episodes[episode_id] = manager
        return manager.initial_observation()

    def step(self, episode_id: str, action: dict) -> Observation:
        with self._lock:
            manager = self._episodes.get(episode_id)
        if manager is None:
            raise KeyError(episode_id)
        if manager.is_terminated:
            raise RuntimeError("episode already terminated")

        if action.get("action_type") == "submit_diagnosis":
            terminal = TerminalAction.model_validate(action)
            return manager.apply_terminal(terminal)

        if "tool_name" in action:
            tool_call = ToolCall.model_validate(action)
            handler = self._tools.get(tool_call.tool_name)
            if handler is None:
                raise KeyError(f"unknown tool: {tool_call.tool_name}")
            output = handler.call(tool_call.args, manager.scenario, manager.history)
            return manager.apply_tool_call(tool_call, output)

        raise ValueError(
            "action must be a ToolCall (with tool_name) or a TerminalAction "
            "(with action_type='submit_diagnosis')"
        )

    def state(self, episode_id: str) -> EpisodeState:
        with self._lock:
            manager = self._episodes.get(episode_id)
        if manager is None:
            raise KeyError(episode_id)
        return manager.to_state()


class ResetRequest(BaseModel):
    scenario_id: str | None = None
    seed_override: int | None = None


class StepRequest(BaseModel):
    episode_id: str
    action: dict


def create_app(env: CITriageEnv) -> FastAPI:
    app = FastAPI(title="CI Triage Env")

    @app.post("/reset")
    def reset(req: ResetRequest) -> Observation:
        try:
            return env.reset(scenario_id=req.scenario_id, seed_override=req.seed_override)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown scenario_id: {exc.args[0]}") from exc

    @app.post("/step")
    def step(req: StepRequest) -> Observation:
        try:
            return env.step(req.episode_id, req.action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown id: {exc.args[0]}") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (ValueError, Exception) as exc:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/state/{episode_id}")
    def state(episode_id: str) -> EpisodeState:
        try:
            return env.state(episode_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown episode_id: {exc.args[0]}") from exc

    @app.get("/mcp/tools")
    def list_mcp_tools() -> list[MCPToolDef]:
        return env.tool_defs

    return app


def _bootstrap() -> FastAPI:
    source = os.environ.get("CI_TRIAGE_SCENARIO_SOURCE")
    env = CITriageEnv(scenario_source=source)
    return create_app(env)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(_bootstrap(), host="0.0.0.0", port=8000)
