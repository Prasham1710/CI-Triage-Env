# Phase A5 — Replay Visualizer

**Owner:** Branch A.
**Prerequisite:** A4 merged, Gate-2 reached or close.
**Estimated time:** 2–3 hours.
**Priority:** Optional. Cut if behind schedule.

---

## Outcome

A static HTML+JS page that loads an `EpisodeTrace` JSON and renders:

1. A horizontal timeline of tool calls and the terminal action.
2. A side panel showing the failure summary, ground-truth label (revealed only after terminal), and budget tracking.
3. A reward breakdown panel (per-component scores, populated from `EpisodeTrace.reward_breakdown` if available).
4. A counterfactual probe panel — **in v1 this always renders "no probe fired" because the probe is deferred (see phase-a4.md). The panel exists as scaffolding for v2.**
5. Side-by-side comparison view: load two traces (e.g., baseline vs. trained) and step through synchronized.

This is what produces the GIFs for the demo video.

---

## Files to create

### `src/ci_triage_env/visualizer/static/viewer.html`

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CI-Triage Episode Replay</title>
  <link rel="stylesheet" href="viewer.css">
</head>
<body>
  <div class="header">
    <h1>CI-Triage Episode Replay</h1>
    <input type="file" id="trace-loader" accept=".json">
    <input type="file" id="trace-loader-2" accept=".json">  <!-- optional second trace -->
  </div>
  <div class="container">
    <div class="left-panel">
      <h2>Failure</h2>
      <div id="failure-summary"></div>
      <h2>Budget</h2>
      <div id="budget"></div>
      <h2>Ground Truth</h2>
      <div id="ground-truth" class="hidden-until-terminal"></div>
    </div>
    <div class="main-panel">
      <h2>Trajectory</h2>
      <div id="timeline"></div>
      <div id="step-detail"></div>
    </div>
    <div class="right-panel">
      <h2>Reward Breakdown</h2>
      <div id="reward-breakdown"></div>
      <h2>Counterfactual Probe</h2>
      <div id="counterfactual"></div>
    </div>
  </div>
  <script src="viewer.js"></script>
</body>
</html>
```

### `src/ci_triage_env/visualizer/static/viewer.js`

Vanilla JS, no framework. Logic outline:

```javascript
let trace1 = null;
let trace2 = null;

function loadTrace(file, slot) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const trace = JSON.parse(e.target.result);
    if (slot === 1) trace1 = trace;
    else trace2 = trace;
    render();
  };
  reader.readAsText(file);
}

function render() {
  if (!trace1) return;
  renderFailureSummary(trace1.episode.scenario.failure_summary);
  renderBudget(trace1.episode);
  renderTimeline(trace1.episode.history);
  renderRewardBreakdown(trace1.reward_breakdown);
  renderCounterfactual(trace1.counterfactual_replay);
  if (trace2) renderComparison(trace1, trace2);
}

function renderTimeline(history) {
  // Each step = a card showing tool name, args, output excerpt, cost
  // Click to expand into step-detail panel
  ...
}

function renderRewardBreakdown(breakdown) {
  // Bar chart of per-component scores (use simple SVG, no charting lib)
  ...
}

function renderCounterfactual(cf) {
  if (!cf || !cf.fired) {
    document.getElementById("counterfactual").innerHTML = "<em>No probe fired</em>";
    return;
  }
  // Show: alternate action, predicted outcome, actual outcome, Brier score
  ...
}

document.getElementById("trace-loader").addEventListener("change", (e) => loadTrace(e.target.files[0], 1));
document.getElementById("trace-loader-2").addEventListener("change", (e) => loadTrace(e.target.files[0], 2));
```

### `src/ci_triage_env/visualizer/static/viewer.css`

Minimal styling. Three-column layout, monospace for logs, color coding for cost (green=cheap, red=expensive).

### `src/ci_triage_env/visualizer/server.py` (optional)

A tiny FastAPI sub-app that serves the static files and accepts trace uploads.

```python
from fastapi import FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from pathlib import Path

viz_app = FastAPI()
static_dir = Path(__file__).parent / "static"
viz_app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

@viz_app.post("/upload-trace")
async def upload_trace(file: UploadFile):
    """Optionally accept trace JSON via HTTP for non-local browsing."""
    ...
```

Mount under `/viz` in the main `server.py` if hosting on the same port.

---

## Implementation notes

- **No JS framework.** Keep it static HTML + vanilla JS so it works on HF Spaces without a build step. Total page weight should be < 50KB.
- **No external CDN dependencies.** Submission must run offline. Self-host any required assets.
- **GIF capture instructions in README.** Explain how to capture a replay as GIF using browser DevTools or external tools (this is what produces the video assets).
- **The viewer is read-only.** It loads JSON, renders. No interaction with the live server beyond optional upload endpoint.
- **Optional comparison mode.** Loading two traces stacks them. If you can't get this working in 2 hours, ship single-trace only.

---

## Tests required

Heavy automated testing for a static viewer is overkill. Required:

```python
# tests/env/test_visualizer.py
def test_static_files_exist():
    """viewer.html, viewer.js, viewer.css are present in the package."""

def test_trace_json_loads_in_node():
    """Subprocess: run `node` to parse a sample trace.json — verifies JS can load it."""
    # Optional; skip if node not available in CI

def test_viz_server_mounts():
    """Optional FastAPI server mounts and returns 200 on /."""
```

Manual visual smoke test required before merge:

1. Generate a probe-fire trace with the env.
2. Load in browser via `python -m http.server` from the static dir.
3. Verify all panels populate.
4. Take a screenshot. Commit to `docs/screenshots/viewer-screenshot.png`.

---

## Smoke test

```bash
# Generate a sample trace
python -m ci_triage_env.env.server &
sleep 2
# ... run a full episode (use the smoke test from A4)
kill %1

# Serve the static visualizer
python -m http.server -d src/ci_triage_env/visualizer/static 8001 &
sleep 1
echo "Open http://localhost:8001 and load a trace from data_artifacts/traces/"
```

---

## Open questions

1. **Comparison mode necessary for v1?** No — single-trace viewer is enough for the demo video. Add comparison only if time permits post-Gate-2.
2. **HF Space hosting.** When deployed to HF Spaces, the visualizer should be reachable at `<space-url>/viz/`. Verify the FastAPI mounting works on Spaces.

---

## What's NOT in this phase

- Live websocket-based stepping (too much engineering)
- Embedded LLM playground (out of scope)
- Interactive scenario authoring (out of scope)
