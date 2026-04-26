"""MockEnvClient — in-memory env replacement for pre-Gate-1 testing.

Replays mock trajectories by serving tool outputs directly from Scenario.tool_outputs.
No network required; deterministic given the scenario seed.

When ``scenarios_dir`` is provided (or ``CI_TRIAGE_SCENARIO_SOURCE`` env var points to a
directory), real generated scenarios are loaded from disk instead of synthetic mocks.
This is the recommended mode for training on HuggingFace Spaces.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ci_triage_env.mock.scenario import make_mock_scenario
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.episode import EpisodeState, EpisodeTrace, StepRecord
from ci_triage_env.schemas.observation import BudgetState, Observation, ToolResponse
from ci_triage_env.schemas.reward import ComponentScore, RewardBreakdown
from ci_triage_env.schemas.scenario import Scenario

_INITIAL_BUDGET = BudgetState(tool_calls_remaining=12, cost_remaining=1.0)
_DEFAULT_FAMILIES = [
    "real_bug", "race_flake", "timing_flake",
    "infra_network", "infra_resource", "dependency_drift", "ambiguous",
]


def _load_scenarios_from_dir(path: Path) -> list[Scenario]:
    """Load all *.json scenario files recursively under `path`."""
    scenarios = []
    for fp in sorted(path.rglob("*.json")):
        try:
            scenarios.append(Scenario.model_validate_json(fp.read_text()))
        except Exception:
            pass
    return scenarios


def _load_scenarios_from_hf(dataset_name: str) -> list[Scenario]:
    """Load scenarios from a HuggingFace dataset."""
    import json
    from datasets import load_dataset
    scenarios = []
    ds = load_dataset(dataset_name, split="train")
    for row in ds:
        try:
            if isinstance(row, dict) and "scenario_json" in row:
                scenarios.append(Scenario.model_validate_json(row["scenario_json"]))
            else:
                scenarios.append(Scenario.model_validate(json.loads(json.dumps(dict(row)))))
        except Exception:
            pass
    return scenarios


@dataclass
class _MockEpisode:
    episode_id: str
    scenario: Scenario
    step: int = 0
    history: list[StepRecord] = field(default_factory=list)
    budget: BudgetState = field(default_factory=lambda: BudgetState(
        tool_calls_remaining=12, cost_remaining=1.0
    ))
    is_terminated: bool = False
    final_action: TerminalAction | None = None


class MockEnvClient:
    """In-memory env replacement. Used before Gate-1 or in unit tests.

    When ``scenarios_dir`` or ``scenarios_hf`` is given (or the
    ``CI_TRIAGE_SCENARIO_SOURCE`` env var is set), real generated scenarios are
    served.  Otherwise, scenarios are generated synthetically via
    ``make_mock_scenario`` (cycles through families deterministically).

    Args:
        seed: Initial counter for scenario cycling / seed derivation.
        scenarios_dir: Path to a directory of scenario JSON files (recursive).
            Overrides ``CI_TRIAGE_SCENARIO_SOURCE`` if both are set.
        scenarios_hf: HuggingFace dataset name (e.g. ``"org/ci-triage-scenarios"``).
            Only used if ``scenarios_dir`` is None and ``CI_TRIAGE_SCENARIO_SOURCE``
            starts with ``"hf://"``.
    """

    def __init__(
        self,
        seed: int = 0,
        scenarios_dir: str | Path | None = None,
        scenarios_hf: str | None = None,
    ) -> None:
        self._episodes: dict[str, _MockEpisode] = {}
        self._call_count = seed
        self._real_scenarios: list[Scenario] = []

        # Resolve scenario source
        source = scenarios_dir or os.environ.get("CI_TRIAGE_SCENARIO_SOURCE")
        if source is not None:
            src_str = str(source)
            if src_str.startswith("hf://"):
                self._real_scenarios = _load_scenarios_from_hf(src_str[len("hf://"):])
            else:
                self._real_scenarios = _load_scenarios_from_dir(Path(src_str))
        elif scenarios_hf:
            self._real_scenarios = _load_scenarios_from_hf(scenarios_hf)

    # ------------------------------------------------------------------ API

    @property
    def scenario_ids(self) -> list[str]:
        """Return scenario IDs available from real scenarios (empty when using synthetic)."""
        return [s.scenario_id for s in self._real_scenarios]

    def reset(
        self,
        scenario_id: str | None = None,
        seed_override: int | None = None,
    ) -> Observation:
        seed = seed_override if seed_override is not None else self._call_count

        if self._real_scenarios:
            # Serve real scenarios, cycling by call_count or looking up by ID
            if scenario_id is not None:
                scenario = next(
                    (s for s in self._real_scenarios if s.scenario_id == scenario_id), None
                )
                if scenario is None:
                    raise KeyError(f"unknown scenario_id: {scenario_id}")
            else:
                scenario = self._real_scenarios[self._call_count % len(self._real_scenarios)]
        else:
            family = _DEFAULT_FAMILIES[self._call_count % len(_DEFAULT_FAMILIES)]
            scenario = make_mock_scenario(family=family, seed=seed)

        self._call_count += 1

        ep_id = f"mock-{uuid.uuid4().hex[:8]}"
        ep = _MockEpisode(episode_id=ep_id, scenario=scenario)
        self._episodes[ep_id] = ep

        return Observation(
            episode_id=ep_id,
            step=0,
            failure_summary=scenario.failure_summary,
            tool_response=None,
            budget_remaining=ep.budget,
            is_terminal=False,
            probe_question=None,
        )

    def step(self, episode_id: str, action: ToolCall | TerminalAction | dict) -> Observation:
        ep = self._episodes[episode_id]
        if ep.is_terminated:
            raise RuntimeError(f"Episode {episode_id} is already terminated.")

        # Normalise dict action
        if isinstance(action, dict):
            if "action_type" in action:
                action = TerminalAction.model_validate(action)
            else:
                action = ToolCall.model_validate(action)

        ep.step += 1

        if isinstance(action, TerminalAction):
            ep.is_terminated = True
            ep.final_action = action
            obs = Observation(
                episode_id=episode_id,
                step=ep.step,
                failure_summary=None,
                tool_response=None,
                budget_remaining=ep.budget,
                is_terminal=True,
                probe_question=None,
            )
            ep.history.append(
                StepRecord(step=ep.step, action=action, observation=obs, cost_charged=0.0)
            )
            return obs

        # ToolCall — look up in scenario.tool_outputs
        tool_name = action.tool_name
        # Try exact key, then prefix match (e.g. "read_logs:full")
        output_record = ep.scenario.tool_outputs.get(tool_name)
        if output_record is None:
            for key, rec in ep.scenario.tool_outputs.items():
                if key.startswith(tool_name + ":") or key == tool_name:
                    output_record = rec
                    break

        cost = output_record.cost_units if output_record is not None else 0.001
        payload = output_record.payload if output_record is not None else {}

        ep.budget = BudgetState(
            tool_calls_remaining=max(0, ep.budget.tool_calls_remaining - 1),
            cost_remaining=max(0.0, ep.budget.cost_remaining - cost),
        )

        tool_resp = ToolResponse(
            tool_name=tool_name,
            args=action.args,
            output=payload,
            cost_charged=cost,
        )
        is_budget_exhausted = ep.budget.tool_calls_remaining == 0
        obs = Observation(
            episode_id=episode_id,
            step=ep.step,
            failure_summary=None,
            tool_response=tool_resp,
            budget_remaining=ep.budget,
            is_terminal=is_budget_exhausted,
            probe_question=None,
        )
        ep.history.append(
            StepRecord(step=ep.step, action=action, observation=obs, cost_charged=cost)
        )
        if is_budget_exhausted:
            ep.is_terminated = True
        return obs

    def get_trace(self, episode_id: str) -> EpisodeTrace:
        ep = self._episodes[episode_id]
        episode_state = EpisodeState(
            episode_id=ep.episode_id,
            scenario_id=ep.scenario.scenario_id,
            seed=ep.scenario.seed,
            step=ep.step,
            history=ep.history,
            budget=ep.budget,
            is_terminated=ep.is_terminated,
            final_action=ep.final_action,
        )
        reward = RewardBreakdown(
            schema_version="1.0",
            total=0.0,
            format_gate=True,
            components={
                "placeholder": ComponentScore(raw=0.0, weighted=0.0, weight=0.0, sub_scores={})
            },
            counterfactual=None,
        )
        return EpisodeTrace(
            schema_version="1.0",
            episode=episode_state,
            reward_breakdown=reward,
            counterfactual_replay=None,
        )

    def get_scenario(self, episode_id: str) -> Scenario:
        return self._episodes[episode_id].scenario

    def list_tools(self) -> list[dict]:
        from ci_triage_env.schemas.tools import ALL_TOOLS
        return [
            {
                "name": t.name,
                "description": t.description,
                "args_schema": t.args_schema,
                "output_schema": t.output_schema,
                "cost_unit": t.cost_unit,
            }
            for t in ALL_TOOLS
        ]
