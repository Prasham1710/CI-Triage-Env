"""Annotation enrichment utilities for Phase B5."""

from __future__ import annotations

from ci_triage_env.schemas.scenario import Scenario
from ci_triage_env.schemas.tools import ALL_TOOLS

_ALL_TOOL_NAMES: list[str] = [t.name for t in ALL_TOOLS]


def enrich_annotations(scenario: Scenario) -> Scenario:
    """Return a copy of *scenario* with ``informative_tools`` auto-populated.

    If the scenario already has a non-empty ``informative_tools`` list it is
    returned unchanged.  Otherwise we fall back to listing every tool for which
    at least one ``tool_outputs`` key exists — a conservative but correct
    default.
    """
    if scenario.informative_tools:
        return scenario

    covered: list[str] = []
    for tool_name in _ALL_TOOL_NAMES:
        for key in scenario.tool_outputs:
            if key == tool_name or key.startswith(tool_name + ":"):
                covered.append(tool_name)
                break

    return scenario.model_copy(update={"informative_tools": covered})
