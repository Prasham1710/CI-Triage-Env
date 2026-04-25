# Phase A3 — Episode Lifecycle

**Owner:** Branch A.
**Prerequisite:** A2 merged.
**Estimated time:** 2–3 hours.

---

## Outcome

By end of this phase:

1. Budget is enforced: when `tool_calls_remaining` hits 0 OR `cost_remaining < 0`, env forces termination on next step with a budget-exhaustion observation.
2. `TerminalAction` (`submit_diagnosis`) is parsed, validated, and stored as `final_action`. Episode marked `is_terminated=True`.
3. Observations are properly formatted: `failure_summary` only on step 0, `tool_response` on subsequent steps, never both.
4. Long log payloads truncated to a server-side cap (default 4000 chars per response) to keep training context manageable.
5. `EpisodeTrace` JSON is written to `data_artifacts/traces/<episode_id>.json` on terminal step (configurable via env var).
6. All A3 unit tests pass; A1+A2 still pass.
7. **Gate-1 entry criterion met for Branch A:** the env can run a complete episode end-to-end against a mock scenario.

---

## Files to modify / create

### Modify `src/ci_triage_env/env/episode.py`

```python
class EpisodeManager:
    DEFAULT_MAX_TOOL_CALLS = 12
    DEFAULT_COST_BUDGET = 5.0       # in dollar-equivalents
    OBSERVATION_PAYLOAD_CAP = 4000  # chars per tool response payload

    def __init__(self, scenario: Scenario, episode_id: str, seed: int,
                 max_tool_calls: int | None = None, cost_budget: float | None = None):
        ...
        self.budget = BudgetState(
            tool_calls_remaining=max_tool_calls or self.DEFAULT_MAX_TOOL_CALLS,
            cost_remaining=cost_budget or self.DEFAULT_COST_BUDGET,
        )

    def apply_action(self, action: ToolCall | TerminalAction) -> Observation:
        if self.is_terminated:
            raise EpisodeTerminatedError("Episode already terminated.")

        if isinstance(action, TerminalAction):
            return self._apply_terminal(action)
        else:
            return self._apply_tool_call(action)

    def _apply_tool_call(self, call: ToolCall) -> Observation:
        # Budget check FIRST — if exhausted, force a budget-exhaustion terminal observation
        if self.budget.tool_calls_remaining <= 0 or self.budget.cost_remaining < 0:
            return self._force_terminate_budget_exhausted()

        handler = TOOL_HANDLER_BY_NAME[call.tool_name]
        try:
            handler.validate_args(call.args)
        except ValueError as e:
            # Invalid args still cost a tool call (cheap)
            self.budget.tool_calls_remaining -= 1
            return self._make_observation(
                tool_response=ToolResponse(
                    tool_name=call.tool_name,
                    args=call.args,
                    output={"error": str(e)},
                    cost_charged=0.001,
                )
            )

        output = handler.call(call.args, self.scenario, self.history)
        cost = output.cost_units
        self.budget.tool_calls_remaining -= 1
        self.budget.cost_remaining -= cost

        # Truncate large payloads
        truncated_payload = self._truncate_payload(output.payload)

        self.step_idx += 1
        record = StepRecord(
            step=self.step_idx,
            action=call,
            observation=...,
            cost_charged=cost,
        )
        self.history.append(record)
        return self._make_observation(
            tool_response=ToolResponse(
                tool_name=output.tool_name,
                args=call.args,
                output=truncated_payload,
                cost_charged=cost,
            )
        )

    def _apply_terminal(self, action: TerminalAction) -> Observation:
        # Validate action
        self._validate_terminal(action)
        self.final_action = action
        self.is_terminated = True
        self.step_idx += 1
        record = StepRecord(
            step=self.step_idx,
            action=action,
            observation=...,
            cost_charged=0.0,
        )
        self.history.append(record)
        # Trigger trace write (handled by env wrapper, not here)
        return Observation(
            episode_id=self.episode_id,
            step=self.step_idx,
            failure_summary=None,
            tool_response=None,
            budget_remaining=self.budget,
            is_terminal=True,
        )

    def _force_terminate_budget_exhausted(self) -> Observation:
        """When budget runs out without an explicit terminal action, force termination
        with a special 'no terminal action submitted' marker.
        Reward layer will treat this as a failure case."""
        self.is_terminated = True
        self.final_action = None
        return Observation(
            episode_id=self.episode_id,
            step=self.step_idx,
            failure_summary=None,
            tool_response=None,
            budget_remaining=self.budget,
            is_terminal=True,
        )

    def _truncate_payload(self, payload: dict | str) -> dict | str:
        """Truncate string fields that exceed OBSERVATION_PAYLOAD_CAP, append marker."""
        ...

    def _validate_terminal(self, action: TerminalAction):
        """Diagnosis must be a valid enum value; confidence in [0, 1]; secondary actions valid."""
        ...
```

### Create `src/ci_triage_env/env/trace.py`

```python
def write_trace(episode: EpisodeManager, output_dir: Path) -> Path:
    """Serialize EpisodeTrace to JSON under output_dir/<episode_id>.json.
    Returns the path written."""
    trace = EpisodeTrace(
        schema_version="1.0",
        episode=episode.to_state(),
        reward_breakdown=...,  # placeholder — Branch C populates after rewards run
        counterfactual_replay=None,  # always None in v1 (probe deferred to v2)
    )
    path = output_dir / f"{episode.episode_id}.json"
    path.write_text(trace.model_dump_json(indent=2))
    return path

def trace_dir() -> Path:
    """Resolve trace output dir from env var CI_TRIAGE_TRACE_DIR or default."""
    return Path(os.environ.get("CI_TRIAGE_TRACE_DIR", "data_artifacts/traces"))
```

### Modify `src/ci_triage_env/env/server.py`

In the `/step` handler, after `apply_action` returns `is_terminal=True`:
```python
if obs.is_terminal:
    write_trace(episode_manager, trace_dir())
```

---

## Implementation notes

- **Budget exhaustion is a failure mode, not an error.** Env returns a terminal observation with a flag that downstream reward computation can detect. Don't raise.
- **Order of operations on budget**: deduct cost AFTER validating args succeed. If args invalid, only the tool-call counter decrements (cheap penalty).
- **Truncation policy**: for `read_logs`, prefer keeping head + tail rather than just head. For `recent_commits`, keep all commits but truncate each commit's message to 200 chars. For others, simple truncation suffices.
- **Why force-terminate on budget exhaustion?** It lets the reward layer assign a meaningful score (low). Allowing infinite tool calls breaks training (budget pressure is part of the reward signal).
- **Secondary actions inside `submit_diagnosis`.** `TerminalAction.secondary_actions` is a list — the agent can simultaneously file a bug AND quarantine. Episode treats all of them as committed. Reward layer scores each.

---

## Tests required (`tests/env/test_episode.py`)

```python
def test_budget_exhaustion_forces_terminal():
    """Calling 12 tools in a row triggers budget-exhausted terminal on call 13."""

def test_terminal_action_records_final():
    """Submitting a TerminalAction sets final_action, is_terminated=True."""

def test_terminal_action_with_invalid_diagnosis_400():
    """An unknown DiagnosisLabel is rejected with 400."""

def test_terminal_action_with_secondary_actions():
    """Diagnosis + 2 secondary actions all stored on EpisodeState."""

def test_step_after_terminal_400():
    """Action submission after termination raises EpisodeTerminatedError → 400."""

def test_invalid_tool_args_charges_cheap_penalty():
    """Calling read_logs with scope='nonexistent' charges 1 tool call but no cost."""

def test_first_observation_has_failure_summary():
    """First observation post-reset has failure_summary populated."""

def test_subsequent_observations_have_tool_response_only():
    """After step 1+, observations have tool_response set, failure_summary None."""

def test_long_payload_truncated():
    """A scenario with 100k chars in tool_outputs is truncated to OBSERVATION_PAYLOAD_CAP."""

def test_trace_written_on_termination():
    """A complete episode produces an EpisodeTrace JSON in data_artifacts/traces/."""

def test_trace_round_trips_via_schema():
    """Loading the written trace and parsing through EpisodeTrace gives the same values."""
```

Integration test `tests/env/test_integration.py`:

```python
def test_full_episode_with_mock_scenario():
    """End-to-end: reset → 5 tool calls → submit_diagnosis(real_bug) → trace written."""
```

---

## Smoke test (manual)

```bash
python -m ci_triage_env.env.server &
sleep 2

RESP=$(curl -sX POST localhost:8000/reset -H 'Content-Type: application/json' -d '{}')
EID=$(echo $RESP | jq -r .episode_id)

# Make a few tool calls
for i in 1 2 3; do
  curl -sX POST localhost:8000/step -H 'Content-Type: application/json' -d "{
    \"episode_id\": \"$EID\",
    \"action\": {\"tool_name\": \"read_logs\", \"args\": {\"scope\": \"full\", \"lines\": 100}}
  }" > /dev/null
done

# Submit terminal
curl -sX POST localhost:8000/step -H 'Content-Type: application/json' -d "{
  \"episode_id\": \"$EID\",
  \"action\": {
    \"action_type\": \"submit_diagnosis\",
    \"diagnosis\": \"real_bug\",
    \"confidence\": 0.85,
    \"secondary_actions\": [{\"name\": \"file_bug\", \"args\": {\"severity\": \"high\", \"title\": \"X\", \"summary\": \"Y\"}}]
  }
}" | jq .

# Verify trace
ls data_artifacts/traces/

kill %1
```

Expected: terminal observation `is_terminal: true`, trace JSON file in `data_artifacts/traces/`.

---

## Gate-1 readiness check

After A3 lands and CI is green, the team can verify Gate-1 readiness for Branch A:

```bash
pytest -q tests/env/  # all green
python -m ci_triage_env.env.server &
# manual smoke test above
# Branch C team can now write env_client.py against this real server
```

---

## Open questions

1. **Should `confidence` be required or default to 1.0 for non-ambiguous diagnoses?** Required. If the model doesn't emit confidence, format-gate fails.
2. **Where do scenario JSONs live in the repo?** Branch B writes to `data_artifacts/scenarios/`. This phase loads from there. Real corpus from HF dataset replaces this at runtime via env var.

---

## What's NOT in this phase

- Counterfactual probe (A4)
- Visualizer (A5)
- Reward computation (Branch C — the trace's `reward_breakdown` is null until C2 runs against the trace)
