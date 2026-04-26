# CI-Triage OpenEnv submission container.
#
# The hackathon evaluation harness will:
#   1. docker build -t ci-triage .
#   2. docker run -p 8000:8000 ci-triage
#   3. POST /reset, POST /step, GET /state, POST /mcp … against port 8000
#
# Scenarios are downloaded from HuggingFace Hub at container start unless
# CI_TRIAGE_SCENARIO_SOURCE points to a local path that already has files.

FROM python:3.11-slim

# System deps: git is needed by huggingface_hub; curl for health checks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer).
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[data]"

# Copy the rest of the project.
COPY data_artifacts/ data_artifacts/
COPY openenv.yaml ./

# Entrypoint handles optional scenario download then launches uvicorn.
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# OpenEnv evaluation probes port 8000 by default.
EXPOSE 8000

# CI_TRIAGE_SCENARIO_SOURCE: override to "hf://org/dataset" to pull from HF Hub,
# or leave empty to use bundled data_artifacts/scenarios/.
ENV CI_TRIAGE_SCENARIO_SOURCE=""

# HF_TOKEN is injected by the evaluation harness when the dataset is private.
ENV HF_TOKEN=""

ENTRYPOINT ["/docker-entrypoint.sh"]
