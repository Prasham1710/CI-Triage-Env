"""Phase A5 visualizer tests.

The viewer itself is read-only static HTML/JS — heavy automated testing of
the rendering is overkill (a screenshot is the manual smoke gate). These
tests guard the package contract:

- The static assets ship with the package.
- The viz sub-app mounts and serves ``viewer.html`` from ``/viz/`` plus an
  ``/upload-trace`` endpoint that accepts JSON.
- A real EpisodeTrace JSON parses as JSON in pure Python (proxy for "the JS
  ``JSON.parse`` would also accept it"); we don't depend on Node in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ci_triage_env.env.episode import EpisodeManager
from ci_triage_env.env.server import build_app
from ci_triage_env.env.tools import ALL_TOOL_HANDLERS
from ci_triage_env.env.trace import build_trace, write_trace
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.visualizer.server import STATIC_DIR, build_visualizer_app
from tests.env.conftest import make_a2_scenario

HANDLERS = {h.name: h for h in ALL_TOOL_HANDLERS}


# ---------------------------------------------------------------------------
# Static assets are present and self-contained
# ---------------------------------------------------------------------------

def test_static_files_exist():
    for fname in ("viewer.html", "viewer.js", "viewer.css"):
        path = STATIC_DIR / fname
        assert path.exists(), f"missing static asset: {path}"
        assert path.stat().st_size > 0


def test_viewer_html_references_local_assets_only():
    """Submission must run offline: no external CDNs."""
    html = (STATIC_DIR / "viewer.html").read_text()
    # Local relative refs OK; reject anything that points at an external host.
    for needle in ("http://", "https://"):
        assert needle not in html, f"viewer.html must not pull from {needle}"


def test_viewer_total_bundle_under_50kb():
    """Phase A5 budget: keep the page weight tight."""
    total = sum((STATIC_DIR / fn).stat().st_size for fn in ("viewer.html", "viewer.js", "viewer.css"))
    assert total < 50 * 1024, f"static bundle is {total} bytes (cap 50KB)"


# ---------------------------------------------------------------------------
# Standalone visualizer sub-app
# ---------------------------------------------------------------------------

@pytest.fixture
def viz_client() -> TestClient:
    return TestClient(build_visualizer_app())


def test_viz_serves_viewer_html(viz_client):
    resp = viz_client.get("/")
    assert resp.status_code == 200
    assert "<title>CI-Triage Episode Replay</title>" in resp.text


def test_viz_serves_javascript(viz_client):
    resp = viz_client.get("/viewer.js")
    assert resp.status_code == 200
    assert "renderTimeline" in resp.text


def test_viz_serves_stylesheet(viz_client):
    resp = viz_client.get("/viewer.css")
    assert resp.status_code == 200
    assert ".timeline" in resp.text


def test_viz_upload_trace_writes_to_trace_dir(viz_client, tmp_path, monkeypatch):
    monkeypatch.setenv("CI_TRIAGE_TRACE_DIR", str(tmp_path))
    payload = {"hello": "world"}
    files = {"file": ("episode-1.json", json.dumps(payload), "application/json")}
    resp = viz_client.post("/upload-trace", files=files)
    assert resp.status_code == 200, resp.text
    saved = Path(resp.json()["saved_to"])
    assert saved.exists()
    assert json.loads(saved.read_text()) == payload


def test_viz_upload_rejects_non_json(viz_client):
    files = {"file": ("trace.txt", b"not json", "text/plain")}
    resp = viz_client.post("/upload-trace", files=files)
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Mounted under /viz on the main env app
# ---------------------------------------------------------------------------

def test_viz_mounted_on_main_app(env_factory):
    app = build_app(env_factory=env_factory)
    client = TestClient(app)
    # /viz/ resolves to viewer.html via StaticFiles(html=True)
    resp = client.get("/viz/")
    assert resp.status_code == 200
    assert "<title>CI-Triage Episode Replay</title>" in resp.text


def test_viz_can_be_disabled():
    app = build_app(mount_visualizer=False)
    client = TestClient(app)
    resp = client.get("/viz/")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# A real EpisodeTrace JSON is parseable (sanity for the JS-side JSON.parse)
# ---------------------------------------------------------------------------

def test_real_trace_json_parses_as_object(tmp_path):
    scenario = make_a2_scenario()
    mgr = EpisodeManager(scenario=scenario, episode_id="ep-viz", seed=1)
    mgr.apply_tool_call(
        ToolCall(tool_name="read_logs", args={"scope": "test", "lines": 50}),
        HANDLERS["read_logs"],
    )
    mgr.apply_terminal(TerminalAction(diagnosis=DiagnosisLabel.RACE_FLAKE, confidence=0.7))
    written = write_trace(mgr, tmp_path)
    parsed = json.loads(written.read_text())

    # The viewer reads these paths — guard the contract here so a future schema
    # tweak doesn't silently break the UI.
    assert "episode" in parsed
    assert "history" in parsed["episode"]
    assert parsed["episode"]["is_terminated"] is True
    assert parsed["episode"]["final_action"]["diagnosis"] == "race_flake"
    assert "reward_breakdown" in parsed
    # Counterfactual stays None in v1 (probe deferred to v2).
    assert parsed["counterfactual_replay"] is None


def test_build_trace_is_json_serializable_round_trip():
    scenario = make_a2_scenario()
    mgr = EpisodeManager(scenario=scenario, episode_id="ep-viz-2", seed=2)
    mgr.apply_terminal(TerminalAction(diagnosis=DiagnosisLabel.AMBIGUOUS, confidence=0.5))
    trace = build_trace(mgr)
    blob = trace.model_dump_json()
    restored = json.loads(blob)
    assert restored["episode"]["episode_id"] == "ep-viz-2"
