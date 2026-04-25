"""EpisodeTrace serialization.

The env writes one trace JSON per terminated episode under
``CI_TRIAGE_TRACE_DIR`` (default ``data_artifacts/traces/``). The
``reward_breakdown`` field is a placeholder until Branch C runs the reward
layer over the trace; the ``counterfactual_replay`` field is permanently
``None`` in v1 (the probe is deferred to v2 — see plan/branch-a-env-core/phase-a4.md).
"""

from __future__ import annotations

import os
from pathlib import Path

from ci_triage_env.env.episode import EpisodeManager
from ci_triage_env.schemas.episode import EpisodeTrace
from ci_triage_env.schemas.reward import RewardBreakdown

_DEFAULT_TRACE_DIR = Path("data_artifacts/traces")


def trace_dir() -> Path:
    """Resolve the trace output directory from ``CI_TRIAGE_TRACE_DIR`` or default."""
    return Path(os.environ.get("CI_TRIAGE_TRACE_DIR", str(_DEFAULT_TRACE_DIR)))


def _placeholder_reward() -> RewardBreakdown:
    """Reward placeholder; Branch C overwrites this when scoring the trace."""
    return RewardBreakdown(total=0.0, format_gate=False)


def build_trace(episode: EpisodeManager) -> EpisodeTrace:
    return EpisodeTrace(
        schema_version="1.0",
        episode=episode.to_state(),
        reward_breakdown=_placeholder_reward(),
        counterfactual_replay=None,
    )


def write_trace(episode: EpisodeManager, output_dir: Path | None = None) -> Path:
    """Serialize an episode's trace to ``<output_dir>/<episode_id>.json``."""
    target = output_dir or trace_dir()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{episode.episode_id}.json"
    path.write_text(build_trace(episode).model_dump_json(indent=2))
    return path
