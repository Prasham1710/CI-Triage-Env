"""FastAPI sub-app that serves the static replay visualizer.

Mounted under ``/viz`` on the main env server (see ``ci_triage_env.env.server``).
The visualizer is read-only static HTML/JS — no live env interaction beyond
the optional ``/viz/upload-trace`` endpoint that writes a trace JSON into
``CI_TRIAGE_TRACE_DIR`` for debugging.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ci_triage_env.env.trace import trace_dir

STATIC_DIR = Path(__file__).parent / "static"


def build_visualizer_app() -> FastAPI:
    app = FastAPI(title="CI-Triage Replay Viewer")

    @app.get("/", include_in_schema=False)
    @app.get("", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "viewer.html", media_type="text/html")

    @app.post("/upload-trace")
    async def upload_trace(file: UploadFile) -> dict:
        if not file.filename or not file.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="expected a .json file")
        # Store under the configured trace dir; never escape it.
        target_dir = trace_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = os.path.basename(file.filename)
        path = target_dir / safe_name
        content = await file.read()
        path.write_bytes(content)
        return {"saved_to": str(path), "bytes": len(content)}

    # StaticFiles must be mounted last so the API routes above take precedence.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
