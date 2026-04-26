#!/usr/bin/env bash
set -euo pipefail

SCENARIO_DIR="/app/data_artifacts/scenarios"

# ── If a HF dataset is configured AND the local dir is empty, pull scenarios ──
if [[ -n "${CI_TRIAGE_SCENARIO_SOURCE:-}" && "${CI_TRIAGE_SCENARIO_SOURCE}" == hf://* ]]; then
    echo "[entrypoint] Downloading scenarios from ${CI_TRIAGE_SCENARIO_SOURCE} …"
    python - <<'PYEOF'
import os, json
from pathlib import Path
from datasets import load_dataset

source = os.environ["CI_TRIAGE_SCENARIO_SOURCE"][len("hf://"):]
token  = os.environ.get("HF_TOKEN") or None
ds     = load_dataset(source, split="train", token=token)
out    = Path("/app/data_artifacts/scenarios/train")
out.mkdir(parents=True, exist_ok=True)

for row in ds:
    if isinstance(row, dict) and "scenario_json" in row:
        payload = row["scenario_json"]
    else:
        payload = json.dumps(dict(row))
    import json as _json
    meta = _json.loads(payload)
    sid  = meta.get("scenario_id", "unknown")
    (out / f"{sid}.json").write_text(payload)

print(f"[entrypoint] Wrote {len(ds)} scenarios to {out}")
PYEOF
fi

# Verify at least one scenario exists before starting the server.
SCENARIO_COUNT=$(find "${SCENARIO_DIR}" -name "*.json" 2>/dev/null | wc -l)
if [[ "${SCENARIO_COUNT}" -eq 0 ]]; then
    echo "[entrypoint] ERROR: no scenario JSON files found under ${SCENARIO_DIR}."
    echo "  Set CI_TRIAGE_SCENARIO_SOURCE=hf://<org>/<dataset> to pull from HuggingFace,"
    echo "  or mount a local directory at ${SCENARIO_DIR}."
    exit 1
fi

echo "[entrypoint] Found ${SCENARIO_COUNT} scenario file(s). Starting env server on :8000 …"
exec uvicorn "ci_triage_env.env.server:build_app" \
    --factory \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
