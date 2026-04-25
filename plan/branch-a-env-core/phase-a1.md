# Phase A1 — Server Scaffold

**Owner:** Branch A.
**Prerequisite:** `phase-0-complete` on `main`, branch `branch-a/env-core` checked out.
**Estimated time:** 3–4 hours.

---

## Outcome (definition of done)

By end of this phase:

1. `python -m ci_triage_env.env.server` starts a FastAPI server on port 8000.
2. `/reset` endpoint accepts a scenario_id (or random selection), returns a valid `Observation`.
3. `/step` endpoint accepts a `ToolCall` or `TerminalAction`, returns next `Observation`.
4. `/state` endpoint returns current `EpisodeState`.
5. `/mcp` endpoint exposes the 11 tools as MCP tools per OpenEnv `MCPEnvironment` spec.
6. In-memory episode store keyed by UUID, with concurrent-access safety.
7. Deterministic seeding: every episode is seeded from `(scenario.seed, episode_id_hash)`.
8. Loads scenarios from `data_artifacts/scenarios/*.json` OR (if present) from an HF dataset configured via env var `CI_TRIAGE_SCENARIO_SOURCE`.
9. All A1 unit tests pass.

> Tools themselves are stubs in this phase — they exist as `ToolHandler` subclasses that return placeholder data. Real routing to scenario tool outputs lands in A2.

---

## Files to create

### `src/ci_triage_env/env/server.py`

```python
from openenv import MCPEnvironment   # check exact import after pip install
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import threading
from .episode import EpisodeManager
from .tools import ALL_TOOL_HANDLERS
from ..schemas.scenario import Scenario
from ..schemas.observation import Observation
from ..schemas.action import ToolCall, TerminalAction
from ..schemas.episode import EpisodeState

class CITriageEnv(MCPEnvironment):
    """OpenEnv-compliant CI triage environment.
    
    Public surface: 11 MCP tools + standard reset/step/state lifecycle.
    """
    def __init__(self, scenario_source: str | None = None):
        super().__init__()
        self._episodes: dict[str, EpisodeManager] = {}
        self._lock = threading.Lock()
        self._scenarios = self._load_scenarios(scenario_source)
        self._register_tools()

    def _load_scenarios(self, source: str | None) -> dict[str, Scenario]:
        # If source is None: load from data_artifacts/scenarios/*.json
        # If source starts with "hf://": load from HF dataset
        ...

    def _register_tools(self):
        for handler in ALL_TOOL_HANDLERS:
            self.register_tool(handler.name, handler.call, handler.cost_unit)

    # standard OpenEnv lifecycle
    def reset(self, scenario_id: str | None = None) -> Observation:
        ...

    def step(self, episode_id: str, action: dict) -> Observation:
        ...

    def state(self, episode_id: str) -> EpisodeState:
        ...

# FastAPI surface
app = FastAPI(title="CI Triage Env")
env = CITriageEnv()

class ResetRequest(BaseModel):
    scenario_id: str | None = None
    seed_override: int | None = None

class StepRequest(BaseModel):
    episode_id: str
    action: dict   # discriminated union: ToolCall or TerminalAction

@app.post("/reset")
def reset(req: ResetRequest) -> Observation:
    ...

@app.post("/step")
def step(req: StepRequest) -> Observation:
    ...

@app.get("/state/{episode_id}")
def state(episode_id: str) -> EpisodeState:
    ...

# /mcp endpoint provided by MCPEnvironment base class

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### `src/ci_triage_env/env/episode.py`

```python
class EpisodeManager:
    """Owns state for a single in-flight episode. Thread-safe for single-episode access."""

    def __init__(self, scenario: Scenario, episode_id: str, seed: int):
        self.episode_id = episode_id
        self.scenario = scenario
        self.seed = seed
        self.step_idx = 0
        self.history: list[StepRecord] = []
        self.budget = BudgetState(tool_calls_remaining=12, cost_remaining=5.0)
        self.is_terminated = False
        self.final_action: TerminalAction | None = None

    def initial_observation(self) -> Observation:
        return Observation(
            episode_id=self.episode_id,
            step=0,
            failure_summary=self.scenario.failure_summary,
            tool_response=None,
            budget_remaining=self.budget,
            is_terminal=False,
        )

    def apply_action(self, action: ToolCall | TerminalAction) -> Observation:
        # In A1: stub — just increments step, returns dummy observation
        # Real logic lands in A2 + A3
        ...

    def to_state(self) -> EpisodeState:
        ...
```

### `src/ci_triage_env/env/tools/__init__.py`

```python
from .investigation import (
    ReadLogsHandler, InspectTestCodeHandler,
    RunDiagnosticHandler, ClusterMetricsHandler,
)
from .context import (
    QueryFlakeHistoryHandler, RecentCommitsHandler, CheckOwnerHandler,
)
from .actions import (
    RerunTestHandler, QuarantineTestHandler, FileBugHandler, PingOwnerHandler,
)

ALL_TOOL_HANDLERS = [
    ReadLogsHandler(), InspectTestCodeHandler(), RunDiagnosticHandler(),
    ClusterMetricsHandler(), QueryFlakeHistoryHandler(), RecentCommitsHandler(),
    CheckOwnerHandler(), RerunTestHandler(), QuarantineTestHandler(),
    FileBugHandler(), PingOwnerHandler(),
]
```

In A1, each handler is a stub that returns a placeholder `ToolOutput`. Real implementation in A2.

```python
# src/ci_triage_env/env/tools/investigation.py
class ReadLogsHandler(ToolHandler):
    name = "read_logs"
    cost_unit = 0.001

    def validate_args(self, args: dict) -> None:
        # Validate against MCPToolDef.args_schema
        ...

    def call(self, args, scenario, history) -> ToolOutput:
        # A1 STUB: return placeholder
        return ToolOutput(tool_name=self.name, payload={"lines": ["[stub]"], "truncated": False}, cost_units=self.cost_unit)
```

### `src/ci_triage_env/env/scenario_loader.py`

Loads scenarios from disk or HF dataset. Caches in memory. Used by `CITriageEnv._load_scenarios`.

```python
def load_from_disk(path: Path) -> dict[str, Scenario]:
    """Load all *.json under path as Scenario objects."""

def load_from_hf(dataset_name: str) -> dict[str, Scenario]:
    """Load all rows of an HF dataset as Scenario objects."""

def load_scenarios(source: str | None) -> dict[str, Scenario]:
    """Dispatch by source prefix."""
```

---

## Implementation notes

- **MCP base class**: import `MCPEnvironment` from `openenv`. Verify the import path with `python -c "from openenv import MCPEnvironment"` after `pip install openenv`. If the path differs, update.
- **Action dispatch**: `/step`'s `action` field is a `dict`. Discriminate on key: `"tool_name"` → `ToolCall`, `"action_type": "submit_diagnosis"` → `TerminalAction`. Use a Pydantic discriminated union if API surface allows.
- **Concurrency**: episodes are independent. The `_episodes` dict needs a lock for create/delete; per-episode operations don't need locking because each episode is a single-flight RPC pattern.
- **Determinism**: store `seed` on the episode at reset time. Any tool that internally randomizes (truncation length, log line selection) must derive its random from `(seed, step_idx, tool_name)` — never from a global RNG.
- **Scenario source resolution order**: env var `CI_TRIAGE_SCENARIO_SOURCE` > default `data_artifacts/scenarios/`. If the directory doesn't exist or is empty, raise `RuntimeError` with a clear "no scenarios found" message at startup.

---

## Tests required (`tests/env/test_server.py`)

```python
def test_server_boots():
    """The FastAPI app instantiates without error."""

def test_reset_returns_valid_observation():
    """POST /reset with no body returns a parseable Observation with failure_summary populated."""

def test_reset_with_specific_scenario_id():
    """POST /reset with scenario_id returns an episode bound to that scenario."""

def test_reset_with_unknown_scenario_id_404():
    """Unknown scenario_id returns 404."""

def test_step_with_tool_call_returns_observation():
    """After reset, POST /step with a valid ToolCall returns an Observation with tool_response populated."""
    # In A1, the response is a stub — just verify shape, not content.

def test_step_with_terminal_action_marks_done():
    """Submitting a TerminalAction marks episode is_terminated=True, returns is_terminal=True."""

def test_step_after_terminal_returns_400():
    """Stepping after terminal action returns 400."""

def test_state_endpoint_returns_episode_state():
    """GET /state/{episode_id} returns EpisodeState matching internal store."""

def test_state_unknown_episode_404():
    """Unknown episode_id returns 404."""

def test_concurrent_resets_get_distinct_episode_ids():
    """Two simultaneous resets produce two distinct episode_ids."""

def test_mcp_endpoint_lists_all_11_tools():
    """The MCP listing endpoint exposes exactly 11 tools with correct names."""

def test_episode_seeding_deterministic():
    """Two resets of the same scenario_id with the same seed_override produce identical EpisodeStates after the same action sequence."""
```

Run with `pytest -q tests/env/test_server.py`.

---

## Smoke test (manual)

```bash
# 1. Drop a fixture scenario into the data dir
cp tests/fixtures/mock_scenario.json data_artifacts/scenarios/

# 2. Start server
python -m ci_triage_env.env.server &
sleep 2

# 3. Reset
curl -X POST localhost:8000/reset -H 'Content-Type: application/json' -d '{}' | jq .

# 4. List MCP tools
curl localhost:8000/mcp/tools | jq .

# 5. Stop
kill %1
```

Expected: a valid Observation JSON in step 3, a list of 11 tools in step 4.

---

## Open questions

1. **OpenEnv MCPEnvironment exact import path / API.** Verify by reading `openenv` package after pip install. If their lifecycle methods differ from gym-style `reset/step/state`, conform to theirs and update this doc.
2. **Should `/reset` accept a config payload (e.g. `max_tool_calls`) for eval-time difficulty tuning?** Defer to A3 — Phase A1 hardcodes default budget.
3. **Logging**: stdlib `logging` or `structlog`? Pick one and standardize. Recommend stdlib `logging` with JSON formatter for HF Spaces compatibility.

---

## What's NOT in this phase

- Real tool output routing (A2)
- Budget enforcement / termination logic beyond stub (A3)
- Counterfactual probe (A4 — deferred to v2)
- Visualizer (A5)
