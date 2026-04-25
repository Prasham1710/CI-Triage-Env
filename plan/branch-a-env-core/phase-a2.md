# Phase A2 — Tool Implementations

**Owner:** Branch A.
**Prerequisite:** A1 merged into `branch-a/env-core`.
**Estimated time:** 3–4 hours.

---

## Outcome

By end of this phase:

1. All 11 `ToolHandler` subclasses route to scenario `tool_outputs` correctly.
2. Each handler validates its `args` against the MCP schema; invalid args raise `ValueError`.
3. Each tool charges its `cost_unit` against the episode budget.
4. Random-but-deterministic behavior (e.g. log line truncation, partial output for `read_logs(scope="full")`) seeded from `(scenario.seed, step, tool_name)`.
5. All A2 unit tests pass; A1 tests still pass.

---

## Files to modify / create

### `src/ci_triage_env/env/tools/investigation.py`

Implement the 4 investigation tools. Each follows the pattern:

```python
class ReadLogsHandler(ToolHandler):
    name = "read_logs"
    cost_unit = 0.001  # per 'line unit', actual cost computed from args

    def validate_args(self, args: dict) -> None:
        scope = args.get("scope")
        if scope not in {"full", "test", "stderr", "kernel", "build"}:
            raise ValueError(f"invalid scope: {scope}")
        lines = args.get("lines", 200)
        if not (10 <= lines <= 2000):
            raise ValueError(f"lines out of range: {lines}")

    def call(self, args, scenario, history) -> ToolOutput:
        self.validate_args(args)
        scope = args["scope"]
        lines = args.get("lines", 200)
        # Look up pre-computed payload for this scope
        payload = scenario.tool_outputs.get(f"read_logs:{scope}")
        if payload is None:
            # Fallback: empty
            output = {"lines": [], "truncated": False}
        else:
            all_lines = payload.payload["lines"]
            truncated = lines < len(all_lines)
            output = {"lines": all_lines[:lines], "truncated": truncated}
        cost = self.cost_unit * lines / 100   # cost scales with lines requested
        return ToolOutput(tool_name=self.name, payload=output, cost_units=cost)
```

Same pattern for `inspect_test_code`, `run_diagnostic`, `cluster_metrics`. Each tool's `tool_outputs` key convention:

| Tool | Key format | Notes |
|---|---|---|
| `read_logs` | `read_logs:<scope>` | One key per scope ("full", "test", "stderr", "kernel", "build") |
| `inspect_test_code` | `inspect_test_code:<test_id>` | One key per test_id allowed (typically just the failing test) |
| `run_diagnostic` | `run_diagnostic:<name>` | Diagnostic name from a fixed enum: "cpu_profile", "memory_profile", "race_detect", "leak_check" |
| `cluster_metrics` | `cluster_metrics:<window>` | Window: "1m", "5m", "15m" |

If the scenario's `tool_outputs` doesn't have the expected key, return an empty/null payload — this can happen if the scenario was generated with a smaller tool set. Don't crash.

### `src/ci_triage_env/env/tools/context.py`

```python
class QueryFlakeHistoryHandler(ToolHandler):
    name = "query_flake_history"
    cost_unit = 0.01

class RecentCommitsHandler(ToolHandler):
    name = "recent_commits"
    cost_unit = 0.01
    # args: {"window": "24h"|"7d"|"30d", "paths": [str] | None}
    # output: {"commits": [{"sha": str, "msg": str, "author": str, "files": [str], "ts": str}]}

class CheckOwnerHandler(ToolHandler):
    name = "check_owner"
    cost_unit = 0.01
    # args: {"path": str}
    # output: {"team": str, "primary_oncall": str, "escalation": str}
```

Tool output keys: `query_flake_history:<test_id>`, `recent_commits:<window>:<paths_hash>`, `check_owner:<path>`.

### `src/ci_triage_env/env/tools/actions.py`

The 4 secondary-action tools. These have side effects in the episode (advancing simulated state), but in A2 they just return their pre-computed output.

```python
class RerunTestHandler(ToolHandler):
    name = "rerun_test"
    cost_unit = 0.30
    # args: {} or {"test_id": str}
    # output: {"passed": bool, "duration_s": float, "log_excerpt": [str]}

class QuarantineTestHandler(ToolHandler):
    name = "quarantine_test"
    cost_unit = 0.0   # cost is reputational (anti-game guard handles this)
    # args: {"test_id": str, "reason": str}
    # output: {"quarantined": bool, "ticket_id": str}

class FileBugHandler(ToolHandler):
    name = "file_bug"
    cost_unit = 0.5   # 30 min human time at $60/hr
    # args: {"severity": str, "title": str, "summary": str}
    # output: {"ticket_id": str, "estimated_priority": str}

class PingOwnerHandler(ToolHandler):
    name = "ping_owner"
    cost_unit = 0.083   # 5 min human time
    # args: {"owner": str, "message": str}
    # output: {"ack": bool, "estimated_response_min": int}
```

Output keys: `rerun_test`, `quarantine_test`, `file_bug`, `ping_owner` (no parameter suffix — these are deterministic per scenario).

> **Important:** these are *secondary* action tools — they're invocations during investigation, not the terminal `submit_diagnosis`. Only one terminal action ends the episode. An agent may invoke `rerun_test` mid-investigation as evidence-gathering before submitting a final diagnosis.

### `src/ci_triage_env/env/tools/utils.py`

Shared helpers:

```python
def deterministic_rng(seed: int, step: int, tool_name: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{step}:{tool_name}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))

def args_hash(args: dict) -> str:
    return hashlib.sha1(json.dumps(args, sort_keys=True).encode()).hexdigest()[:12]
```

---

## Implementation notes

- **Scenario coverage isn't always full.** Some scenarios won't have `inspect_test_code:test_X` because the scenario family doesn't need that tool to be informative. Returning an empty payload is correct — it represents "you can call this tool but it doesn't help."
- **Cost scaling for `read_logs`.** Cost should scale roughly with information requested. The formula `cost_unit * lines / 100` treats 100 lines as the unit; 200 lines = 2x cost.
- **Repeated tool calls with identical args.** Allowed (the redundancy penalty in the reward layer handles this), but the env should still return the same output. Use `args_hash` to detect repeats and just return the cached output.
- **`run_diagnostic` semantics.** Choose 4 fixed diagnostic names to keep the action space bounded: `cpu_profile`, `memory_profile`, `race_detect`, `leak_check`. The model can call any of these, the scenario provides outputs for some, the rest return "no signal" payload.

---

## Tests required (`tests/env/test_tools.py`)

For each of the 11 tools:

```python
def test_<tool>_valid_args_returns_output():
    """Calling with valid args returns a ToolOutput with correct cost charged."""

def test_<tool>_invalid_args_raises():
    """Calling with malformed args raises ValueError."""

def test_<tool>_missing_scenario_data_returns_empty():
    """If scenario.tool_outputs lacks the expected key, returns empty payload, no crash."""

def test_<tool>_repeated_call_returns_same_output():
    """Calling twice with same args returns identical output (deterministic)."""
```

Plus integration:

```python
def test_full_tool_loop_against_mock_scenario():
    """Reset, call all 11 tools sequentially, verify each returns its expected output type."""

def test_cost_charging_accumulates_correctly():
    """After 5 tool calls, episode budget reflects correct total deducted."""
```

---

## Smoke test (manual)

```bash
python -m ci_triage_env.env.server &
sleep 2

# Reset to mock scenario
RESP=$(curl -sX POST localhost:8000/reset -H 'Content-Type: application/json' -d '{}')
EID=$(echo $RESP | jq -r .episode_id)

# Call read_logs
curl -sX POST localhost:8000/step -H 'Content-Type: application/json' -d "{
  \"episode_id\": \"$EID\",
  \"action\": {\"tool_name\": \"read_logs\", \"args\": {\"scope\": \"full\", \"lines\": 100}}
}" | jq .

# Call recent_commits
curl -sX POST localhost:8000/step -H 'Content-Type: application/json' -d "{
  \"episode_id\": \"$EID\",
  \"action\": {\"tool_name\": \"recent_commits\", \"args\": {\"window\": \"24h\"}}
}" | jq .

kill %1
```

Expected: each tool call returns a real (non-stub) payload sourced from the mock scenario.

---

## Open questions

1. **Should `submit_diagnosis` be exposed as an MCP tool?** No (per the master plan) — it's a structured terminal action. Confirm A1 already excludes it from MCP registration.
2. **How does the agent learn the available tool names?** Through the `/mcp/tools` listing at episode start. Branch C's rollout function injects the tool list into the system prompt.

---

## What's NOT in this phase

- Budget enforcement / termination on budget exhaustion (A3)
- Terminal action handling (A3)
- Counterfactual side-effects on `rerun_test` (A4)
