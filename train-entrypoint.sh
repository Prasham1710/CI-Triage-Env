#!/usr/bin/env bash
set -euo pipefail

if [[ "${START_MODE:-jupyter}" == "auto" ]]; then
    echo "[train-entrypoint] START_MODE=auto — running train.py"
    exec python /workspace/train.py
else
    echo "[train-entrypoint] START_MODE=jupyter — launching JupyterLab on :7860"
    exec jupyter lab \
        --ip=0.0.0.0 \
        --port=7860 \
        --no-browser \
        --allow-root \
        --NotebookApp.token="" \
        --NotebookApp.password="" \
        --notebook-dir=/workspace
fi
