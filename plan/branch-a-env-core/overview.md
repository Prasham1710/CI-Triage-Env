# Branch A — Environment Core

**Branch name:** `branch-a/env-core`
**Default owner:** Prasham (team lead — runs FastAPI/MCP server, episode lifecycle, replay visualizer).
**Prerequisite:** `phase-0-complete` tag on `main`.

---

## What this branch builds

The runnable OpenEnv environment. After this branch merges, anyone can:

```bash
python -m ci_triage_env.env.server
# in another terminal
curl -X POST localhost:8000/reset -d '{"scenario_id": "..."}'
# and the server will return a valid Observation
```

Plus: the agent can call any of the 11 MCP tools, the env tracks episode state, computes per-step costs, and writes EpisodeTrace JSON for the visualizer.

---

## Phases

| Phase | Title | What lands | Depends on |
|---|---|---|---|
| A1 | Server scaffold | FastAPI app with `/reset`, `/step`, `/state`, MCP tool registration; in-memory episode store; deterministic seeding | Phase 0 |
| A2 | Tool implementations | All 11 `ToolHandler` subclasses; route to scenario `tool_outputs` dict; cost charging | A1 |
| A3 | Episode lifecycle | Budget enforcement, terminal-action handling, observation formatting, log truncation policy, EpisodeTrace serialization | A2 |
| ~~A4~~ | ~~Counterfactual probe~~ | **Deferred to v2.** Schema fields (`probe_question`, `counterfactual_replay`) stay optional in Phase 0 so re-adding is purely additive. See `phase-a4.md` for the dormant scaffolding. | — |
| A5 | Replay visualizer | Static HTML/JS that loads an EpisodeTrace JSON and renders tool calls, observations, reward breakdown over time | A3 |

---

## What you consume

- `Scenario` JSON files. Until Branch B merges, use `mock.scenario.make_mock_scenario()` for development. After B5 merges, load real scenarios from HF dataset.
- All schemas from Phase 0. Do not modify schemas in this branch — if you need a change, propose in chat first.

## What you produce

- A runnable OpenEnv-compliant FastAPI server.
- An HTTP API that Branch C's training rollout function will drive.
- `EpisodeTrace` JSON output, consumed by Branch C's reward layer and the visualizer.

---

## Files this branch owns

```
src/ci_triage_env/env/
├── server.py                  # FastAPI app + MCP registration
├── episode.py                 # Episode lifecycle, budget, termination
├── trace.py                   # EpisodeTrace builder
└── tools/
    ├── __init__.py
    ├── base.py                # already from Phase 0; do not modify
    ├── investigation.py       # read_logs, inspect_test_code, run_diagnostic, cluster_metrics
    ├── context.py             # query_flake_history, recent_commits, check_owner
    └── actions.py             # rerun_test, quarantine_test, file_bug, ping_owner

# counterfactual.py is NOT in v1. v2 would add it here — see phase-a4.md.

src/ci_triage_env/visualizer/
├── __init__.py
├── server.py                  # Optional small Flask/FastAPI server for the viewer
├── static/
│   ├── viewer.html
│   ├── viewer.js
│   └── viewer.css

tests/env/
├── test_server.py
├── test_tools.py
├── test_episode.py
└── test_integration.py        # full episode end-to-end
```

You do **not** own:
- Anything under `src/ci_triage_env/data/` (Branch B)
- Anything under `src/ci_triage_env/rewards/` (Branch C)
- Anything under `src/ci_triage_env/training/` (Branch C)

---

## Tooling convention (read once, applies to every phase doc)

This repo uses **uv** (see `INSTRUCTION-MANUAL.md` §0). Every command in the per-phase docs that says `python ...`, `pytest ...`, or `ruff ...` should be run as `uv run python ...`, `uv run pytest ...`, `uv run ruff ...`. New deps are added with `uv add <pkg>` (or `uv add --optional <extra> <pkg>`), never raw `pip install`. Commit `pyproject.toml` and `uv.lock` together.

## Test discipline

After every phase, run:

```bash
uv run pytest -q tests/env/
uv run pytest -q tests/schemas/   # sanity — should still pass since you don't touch schemas
uv run ruff check src/ci_triage_env/env/
```

CI must be green before opening a PR. PR title format: `feat(branch-a): A<N> <title>`.

Specific tests required per phase: see each `phase-a<N>.md` file.

---

## Integration checkpoints

- **Gate-1 prerequisite:** A1, A2, A3 must be merged into `main` by Sunday afternoon.
- **Gate-2 prerequisite:** A5 (visualizer) merged by Sunday onsite EOD. Cut to v2 if time-constrained — the env still works; you just lose the GIF assets for the demo video.
- **A4 (counterfactual)**: deferred. Do not work on it during v1.

---

## Quick map: what each phase doc tells you

Each `phase-a<N>.md` contains:
1. **Outcome** — the precise definition of done for that phase
2. **Files to create / modify** — exact paths
3. **Implementation notes** — design decisions already made
4. **Tests required** — what to add to `tests/env/`
5. **Smoke test** — manual verification command
6. **Open questions** — explicit list of things to ask the team about, never silently decide

Open `phase-a1.md` and start there.
