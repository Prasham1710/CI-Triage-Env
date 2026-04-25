# Phase 0 — Common Foundation

**Owner:** Prasham (executes solo on `main` before anyone else branches).
**Estimated time:** 2–3 hours.
**Output:** A clean repo where `pytest -q` passes with mock fixtures, all schemas defined, all 3 branches can compile against shared types.

This phase blocks every other branch. Until it's tagged `phase-0-complete`, Priyanshi and Sahil cannot start their work productively.

---

## 0.1 Outcomes (definition of done)

By the end of this phase, on `main`:

1. Repo skeleton committed, **uv project initialized** (`pyproject.toml`, `uv.lock`, `.python-version` all committed), `uv sync --all-extras` works on a fresh clone.
2. All Pydantic schemas defined and round-trip serializable.
3. MCP tool definitions defined for all 11 tools (signatures only, no implementation).
4. One mock scenario JSON committed, validates against schema.
5. One mock trajectory JSON committed, validates against schema.
6. Abstract base classes for `ToolHandler`, `ScenarioFamilyGenerator`, `RewardComponent`.
7. `openenv.yaml` skeleton committed and validates against the OpenEnv manifest spec.
8. CI workflow uses `uv` (via `astral-sh/setup-uv@v3`), runs `uv run pytest -q` and `uv run ruff check src/` on every push, green on `main`.
9. README points to README-PLAN.md and the planning folder.
10. `phase-0-complete` git tag pushed.

---

## 0.2 Repository layout to create

```
ci-triage-env/
├── README.md                       # links to plan/README-PLAN.md
├── openenv.yaml                    # MCP-style manifest
├── pyproject.toml                  # uv project metadata + deps
├── uv.lock                         # committed; reproducibility across laptops
├── .python-version                 # "3.11" — uv reads this
├── .gitignore                      # includes .venv/, data_artifacts/
├── .ruff.toml
├── requirements.txt                # auto-exported via `uv export` for HF Spaces
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── sync-to-hf.yml          # placeholder, disabled
├── plan/                           # already imported (this folder)
├── src/
│   └── ci_triage_env/
│       ├── __init__.py             # exports __version__
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── scenario.py         # Scenario, ScenarioFamily, etc.
│       │   ├── observation.py      # Observation, ToolOutput
│       │   ├── action.py           # ToolCall, TerminalAction
│       │   ├── diagnosis.py        # DiagnosisLabel enum
│       │   ├── reward.py           # RewardBreakdown
│       │   ├── tools.py            # MCP tool definitions
│       │   └── episode.py          # EpisodeState, EpisodeTrace
│       ├── env/
│       │   ├── __init__.py
│       │   └── tools/
│       │       ├── __init__.py
│       │       └── base.py         # ToolHandler ABC
│       ├── data/
│       │   ├── __init__.py
│       │   └── generators/
│       │       ├── __init__.py
│       │       └── base.py         # ScenarioFamilyGenerator ABC
│       ├── rewards/
│       │   ├── __init__.py
│       │   └── base.py             # RewardComponent ABC
│       └── mock/
│           ├── __init__.py
│           ├── scenario.py         # mock scenario factory
│           └── trajectory.py       # mock trajectory factory
├── data_artifacts/
│   └── .gitkeep                    # gitignored otherwise
└── tests/
    ├── __init__.py
    ├── schemas/
    │   ├── __init__.py
    │   ├── test_scenario.py
    │   ├── test_action.py
    │   ├── test_diagnosis.py
    │   ├── test_reward.py
    │   ├── test_tools.py
    │   └── test_episode.py
    └── fixtures/
        ├── mock_scenario.json
        └── mock_trajectory.json
```

---

## 0.3 uv project setup

### Initialize the project (Prasham, in the fresh repo)

```bash
# Pin Python version uv will use
echo "3.11" > .python-version

# Create the uv project (writes the initial pyproject.toml — we'll overwrite below)
uv init --no-readme --no-workspace --package
```

### `pyproject.toml`

```toml
[project]
name = "ci_triage_env"
version = "0.1.0"
description = "OpenEnv RL environment for CI failure triage"
readme = "README.md"
requires-python = ">=3.11,<3.13"
dependencies = [
    "pydantic>=2.7,<3.0",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "httpx>=0.27",
    "pyyaml>=6.0",
    "openenv",                       # latest from PyPI as of build date
    "datasets>=2.18",
    "huggingface_hub>=0.23",
    "jsonschema>=4.21",
]

[project.optional-dependencies]
training = [
    "torch>=2.3",
    "transformers>=4.45",
    "trl>=0.11",
    "unsloth>=2026.4",               # supports Qwen3.5; verify at install time
    "accelerate>=0.30",
    "wandb>=0.17",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "pandas>=2.2",
]
data = [
    "openai>=1.40",
    "tenacity>=8.2",
    "tqdm>=4.66",
    "scikit-learn>=1.4",            # for archetype clustering
]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "pytest-mock>=3.12",
    "ruff>=0.5",
    "mypy>=1.10",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ci_triage_env"]

[tool.uv]
package = true                       # treat root as a package, install in editable mode

[tool.ruff]
line-length = 110
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
ignore = ["E501"]
```

### Generate the lockfile (committed to git)

```bash
uv lock
git add pyproject.toml uv.lock .python-version
```

### Verify the install works

```bash
uv sync --all-extras
uv run python -c "import ci_triage_env; print('OK')"
```

### Auto-export `requirements.txt` for HF Spaces compatibility

HF Spaces builds containers from `requirements.txt`, not `uv.lock`. Add a one-liner CI step or a pre-push hook:

```bash
uv export --no-dev --extra training --format requirements-txt > requirements.txt
git add requirements.txt
```

This file is regenerated, never hand-edited. Stale `requirements.txt` causes Spaces builds to drift from local — verify it's regenerated whenever `uv.lock` changes.

---

## 0.4 Schema specifications

### 0.4.1 `schemas/diagnosis.py`

```python
from enum import StrEnum

class DiagnosisLabel(StrEnum):
    REAL_BUG = "real_bug"
    RACE_FLAKE = "race_flake"
    TIMING_FLAKE = "timing_flake"
    INFRA_NETWORK = "infra_network"
    INFRA_RESOURCE = "infra_resource"
    DEPENDENCY_DRIFT = "dependency_drift"
    AMBIGUOUS = "ambiguous"     # only correct on the abstain-correct family
```

Add `is_flake()`, `is_infra()`, `is_real_root_cause()` helper methods.

### 0.4.2 `schemas/scenario.py`

```python
class Scenario(BaseModel):
    schema_version: Literal["1.0"]
    scenario_id: str            # stable hash, e.g. "race_flake-v1-seed42-abc123"
    family: str                 # one of the 7 families
    seed: int
    ground_truth: GroundTruth   # see below
    failure_summary: FailureSummary
    tool_outputs: dict[str, ToolOutput]   # pre-computed per-tool responses
    informative_tools: list[str]          # which tools are informative for this scenario
    minimal_evidence_set: list[str]       # smallest sufficient tool set
    correct_terminal_action: TerminalActionSpec   # what action(s) score high
    metadata: ScenarioMetadata

class GroundTruth(BaseModel):
    label: DiagnosisLabel
    rationale: str              # human-readable explanation, NOT shown to agent
    is_ambiguous: bool          # true only for the ambiguous family
    confidence_target: float    # the calibrated confidence for ambiguous cases

class FailureSummary(BaseModel):
    test_name: str
    suite: str
    branch: str
    last_passing_commit: str
    initial_log_excerpt: str    # first ~200 lines of failure output
    timestamp: str              # ISO 8601

class ToolOutput(BaseModel):
    tool_name: str
    payload: dict | str         # tool-specific structure
    cost_units: float           # cost charged when this tool is called

class TerminalActionSpec(BaseModel):
    primary: str                # e.g. "submit_diagnosis"
    args: dict
    acceptable_alternatives: list[dict]   # alternative actions also scoring well

class ScenarioMetadata(BaseModel):
    generator_version: str
    generated_at: str
    source_log_hash: str | None    # hash of the real log used as seed (or None for purely synthetic)
    difficulty: Literal["easy", "medium", "hard"]
```

### 0.4.3 `schemas/observation.py`

```python
class Observation(BaseModel):
    episode_id: str
    step: int
    failure_summary: FailureSummary | None = None   # only on first observation
    tool_response: ToolResponse | None = None       # populated after each tool call
    budget_remaining: BudgetState
    is_terminal: bool
    probe_question: ProbeQuestion | None = None     # v1: always None (deferred); v2: populated when probe fires

class ProbeQuestion(BaseModel):
    """v1: dormant — env never emits this. v2 path preserved as schema."""
    step: int
    taken_action: dict | None
    alternate_action: dict

class ToolResponse(BaseModel):
    tool_name: str
    args: dict
    output: dict | str
    cost_charged: float

class BudgetState(BaseModel):
    tool_calls_remaining: int
    cost_remaining: float
```

### 0.4.4 `schemas/action.py`

```python
class ToolCall(BaseModel):
    tool_name: str
    args: dict

class TerminalAction(BaseModel):
    action_type: Literal["submit_diagnosis"]
    diagnosis: DiagnosisLabel
    confidence: float                # in [0, 1]
    secondary_actions: list[SecondaryAction]   # filed bug, quarantine, rerun, etc.

class SecondaryAction(BaseModel):
    name: Literal["rerun_test", "quarantine_test", "file_bug", "ping_owner"]
    args: dict
```

> Note: secondary actions are taken *as part of* the terminal step, not as separate tool calls. This forces the agent to commit to a coherent response.

### 0.4.5 `schemas/reward.py`

```python
class RewardBreakdown(BaseModel):
    schema_version: Literal["1.0"]
    total: float
    format_gate: bool                     # if False, total = 0
    components: dict[str, ComponentScore] # keyed by component name
    counterfactual: CounterfactualScore | None = None  # v1: always None (deferred); v2: populated on probe fire

class ComponentScore(BaseModel):
    raw: float
    weighted: float
    weight: float
    sub_scores: dict[str, float]          # per-component internal breakdown

class CounterfactualScore(BaseModel):
    """v1: dormant — never populated. v2 path preserved as schema."""
    fired: bool
    probe_step: int
    probe_action: str
    predicted_outcome: str
    actual_outcome: str
    brier_score: float
```

> **v1/v2 split.** The counterfactual probe is deferred from v1 (see `plan/branch-a-env-core/phase-a4.md`). The schema fields stay in place as optional with default `None` so re-adding the probe in v2 requires no schema migration. Branch A's env never fires the probe in v1; Branch C's reward composite gates on weight=0.

### 0.4.6 `schemas/tools.py`

MCP-style tool definitions (signature only, no implementation in Phase 0). Each entry:

```python
class MCPToolDef(BaseModel):
    name: str                            # MCP tool name (no "reset", "step", "state", "close")
    description: str                     # docstring shown to agent
    args_schema: dict                    # JSON schema for arguments
    output_schema: dict                  # JSON schema for output
    cost_unit: float                     # base cost weight

ALL_TOOLS: list[MCPToolDef] = [
    MCPToolDef(name="read_logs", ...),
    MCPToolDef(name="inspect_test_code", ...),
    MCPToolDef(name="run_diagnostic", ...),
    MCPToolDef(name="cluster_metrics", ...),
    MCPToolDef(name="query_flake_history", ...),
    MCPToolDef(name="recent_commits", ...),
    MCPToolDef(name="check_owner", ...),
    MCPToolDef(name="rerun_test", ...),
    MCPToolDef(name="quarantine_test", ...),
    MCPToolDef(name="file_bug", ...),
    MCPToolDef(name="ping_owner", ...),
    # submit_diagnosis is NOT an MCP tool — it's a terminal action emitted as a structured response
]
```

Spec each one fully in this phase. Examples:

```python
MCPToolDef(
    name="read_logs",
    description="Read log lines from the failed CI run. Use scope to narrow.",
    args_schema={
        "type": "object",
        "properties": {
            "scope": {"type": "string", "enum": ["full", "test", "stderr", "kernel", "build"]},
            "lines": {"type": "integer", "minimum": 10, "maximum": 2000, "default": 200},
        },
        "required": ["scope"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "lines": {"type": "array", "items": {"type": "string"}},
            "truncated": {"type": "boolean"},
        },
    },
    cost_unit=0.001,
)
```

### 0.4.7 `schemas/episode.py`

```python
class EpisodeState(BaseModel):
    episode_id: str
    scenario_id: str
    seed: int                                # for determinism
    step: int
    history: list[StepRecord]
    budget: BudgetState
    is_terminated: bool
    final_action: TerminalAction | None

class StepRecord(BaseModel):
    step: int
    action: ToolCall | TerminalAction
    observation: Observation
    cost_charged: float

class EpisodeTrace(BaseModel):
    schema_version: Literal["1.0"]
    episode: EpisodeState
    reward_breakdown: RewardBreakdown
    counterfactual_replay: list[StepRecord] | None = None  # v1: always None (deferred); v2: populated on probe fire
```

`EpisodeState` must be JSON-serializable and re-loadable — useful for replay/visualizer in v1, and required if v2 re-enables the counterfactual probe.

---

## 0.5 Abstract base classes

### 0.5.1 `env/tools/base.py`

```python
class ToolHandler(ABC):
    name: str
    cost_unit: float

    @abstractmethod
    def call(self, args: dict, scenario: Scenario, history: list[StepRecord]) -> ToolOutput: ...

    @abstractmethod
    def validate_args(self, args: dict) -> None:
        """Raise ValueError if args don't match this tool's schema."""
```

### 0.5.2 `data/generators/base.py`

```python
class ScenarioFamilyGenerator(ABC):
    family_name: str
    label: DiagnosisLabel

    @abstractmethod
    def generate(self, seed: int, source_log_hash: str | None = None) -> Scenario: ...

    @abstractmethod
    def informative_tools(self) -> list[str]:
        """Tools that are informative for this family by construction."""

    @abstractmethod
    def minimal_evidence_set(self) -> list[str]:
        """Smallest tool set that uniquely determines correct diagnosis."""
```

### 0.5.3 `rewards/base.py`

```python
class RewardComponent(ABC):
    name: str
    default_weight: float

    @abstractmethod
    def score(self, trace: EpisodeTrace, scenario: Scenario) -> ComponentScore: ...
```

---

## 0.6 Mock fixtures

### 0.6.1 `mock/scenario.py`

A factory `make_mock_scenario(family: str = "race_flake") -> Scenario` returning a fully-populated `Scenario` whose tool outputs are toy strings ("LOG LINE 1", etc.). Used by Branch C in unit tests until real scenarios from Branch B are merged.

### 0.6.2 `mock/trajectory.py`

A factory `make_mock_trajectory(scenario: Scenario, outcome: Literal["good", "bad", "abstain"]) -> EpisodeTrace`. Generates a deterministic trajectory of 4 tool calls + terminal. Three variants for testing different reward paths.

---

## 0.7 OpenEnv manifest (`openenv.yaml`)

```yaml
schema_version: "1.0"
name: ci-triage-env
display_name: "CI Triage Agent Environment"
description: "Train an LLM to investigate ambiguous CI failures with verifiable rewards."
type: mcp
entrypoint: "python -m ci_triage_env.env.server"
api:
  port: 8000
  host: "0.0.0.0"
mcp:
  tools:
    - read_logs
    - inspect_test_code
    - run_diagnostic
    - cluster_metrics
    - query_flake_history
    - recent_commits
    - check_owner
    - rerun_test
    - quarantine_test
    - file_bug
    - ping_owner
metadata:
  version: "0.1.0"
  authors: ["Sahil", "Prasham", "Priyanshi"]
  hackathon: "scaler-meta-pytorch-openenv-2026"
```

> The above structure is illustrative — verify against latest OpenEnv manifest spec from their docs and adjust field names as needed.

---

## 0.8 CI workflow (`.github/workflows/ci.yml`)

```yaml
name: CI
on:
  push:
    branches: [main, "branch-*/**"]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"
      - name: Install Python
        run: uv python install 3.11
      - name: Sync deps
        run: uv sync --extra dev --extra data
        # NOTE: skip --extra training in CI — torch/unsloth pulls 3GB+. Training tests
        # gate on @pytest.mark.skipif(not torch.cuda.is_available()) and run locally / onsite.
      - name: Lint
        run: uv run ruff check src/ tests/
      - name: Test
        run: uv run pytest -q tests/

  manifest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv run --with pyyaml python -c "import yaml; yaml.safe_load(open('openenv.yaml'))"

  requirements-txt-fresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - name: Verify requirements.txt is regenerated
        run: |
          uv export --no-dev --extra training --format requirements-txt > /tmp/exported.txt
          diff -q requirements.txt /tmp/exported.txt || (
            echo "::error::requirements.txt is stale. Run: uv export --no-dev --extra training --format requirements-txt > requirements.txt"
            exit 1
          )
```

---

## 0.9 Tests required in Phase 0

| Test file | Tests |
|---|---|
| `tests/schemas/test_scenario.py` | Round-trip serialize/deserialize a Scenario; validate failure on missing fields; validate ground_truth labels are valid enum values |
| `tests/schemas/test_action.py` | TerminalAction with diagnosis + secondary actions round-trips; ToolCall serializes |
| `tests/schemas/test_diagnosis.py` | All 7 enum values present; helper methods correct |
| `tests/schemas/test_reward.py` | RewardBreakdown round-trips; format_gate=False forces total=0 (logical assertion in test) |
| `tests/schemas/test_tools.py` | All 11 tool defs present; no duplicate names; no use of reserved names ("reset", "step", "state", "close") |
| `tests/schemas/test_episode.py` | EpisodeState round-trips; EpisodeTrace round-trips |
| `tests/test_mock_fixtures.py` | `make_mock_scenario` returns valid Scenario; `make_mock_trajectory` returns valid EpisodeTrace for all 3 outcome variants |

All tests must pass with `uv run pytest -q`. CI green is the merge gate.

---

## 0.10 Final steps

```bash
# Make sure requirements.txt is up-to-date for HF Spaces
uv export --no-dev --extra training --format requirements-txt > requirements.txt

# Final sync to verify everything resolves
uv sync --all-extras
uv run pytest -q && uv run ruff check src/ tests/

git add -A
git commit -m "feat(phase-0): foundation — uv project, schemas, mocks, manifest, CI"
git push origin main
# Wait for CI to go green (test, manifest, requirements-txt-fresh — all 3 jobs)
git tag -a phase-0-complete -m "Phase 0: foundation complete"
git push --tags
```

Notify team in chat. Other branches can now start.

---

## 0.11 What this phase explicitly does NOT include

These belong to later phases — do not implement here:

- Tool implementation logic (Branch A, A2)
- Scenario family generators (Branch B, B4)
- Reward component implementations (Branch C, C1)
- FastAPI server endpoints (Branch A, A1)
- Any training code (Branch C)
- Any data mining or scraping (Branch B)

Phase 0 is *only* contracts + scaffolding + tests on contracts.
