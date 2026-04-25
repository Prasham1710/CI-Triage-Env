"""Multi-turn rollout function for GRPO training.

TrainingRollout drives one full episode using the model's generate,
scores it with compute_reward, and maintains the quarantine rolling window.
"""

from __future__ import annotations

import random
from pathlib import Path

from ci_triage_env.rewards.composite import compute_reward
from ci_triage_env.schemas.scenario import Scenario
from ci_triage_env.training.trajectory_gen import TrajectoryGenerator

_parse_action = TrajectoryGenerator._parse_action

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert SRE investigating a CI failure. Determine the root cause \
with minimum cost.

Tools available:
{tools}

Call a tool: {{"tool_name": "<name>", "args": {{...}}}}
Submit diagnosis: {{"action_type": "submit_diagnosis", "diagnosis": "<family>", \
"confidence": <0-1>, "secondary_actions": []}}

Valid families: real_bug, race_flake, timing_flake, infra_network, \
infra_resource, dependency_drift, ambiguous.

Use cheap tools first. Do NOT quarantine unless you are certain it is a flake.\
"""


class TrainingRollout:
    """Single-call rollout for GRPO: runs one env episode, returns reward.

    Args:
        env_client: Any client with reset/step/get_trace/get_scenario/list_tools interface.
        scenarios_train: List of scenario_ids to sample from. If empty, env picks randomly.
        max_turns: Maximum tool calls before forced termination.
    """

    def __init__(
        self,
        env_client,
        scenarios_train: list[str] | None = None,
        max_turns: int = 12,
    ) -> None:
        self.env = env_client
        self.scenarios_train: list[str] = scenarios_train or []
        self.max_turns = max_turns
        self._quarantine_window: list[str] = []
        self._tools_listing: str | None = None

    def __call__(self, model, tokenizer, prompts=None) -> dict:
        """Run one episode; return messages, reward, and breakdown.

        Args:
            model: HF-compatible model with .generate().
            tokenizer: Matching tokenizer with apply_chat_template().
            prompts: Ignored (kept for TRL trainer compatibility).

        Returns:
            dict with 'messages', 'reward', 'reward_breakdown', 'trajectory_length'.
        """
        import torch  # type: ignore[import]

        scenario_id = random.choice(self.scenarios_train) if self.scenarios_train else None
        obs = self.env.reset(scenario_id=scenario_id)
        episode_id = obs.episode_id

        tools_text = self._get_tools_listing()
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT_TEMPLATE.format(tools=tools_text)},
            {"role": "user", "content": self._format_initial_obs(obs)},
        ]

        terminated = False
        for _turn in range(self.max_turns):
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True,
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=600,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                out[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            messages.append({"role": "assistant", "content": response})

            action = _parse_action(response)
            if action is None:
                break

            try:
                next_obs = self.env.step(episode_id, action)
            except Exception:
                break

            messages.append({"role": "user", "content": self._format_obs(next_obs)})
            if next_obs.is_terminal:
                # Counterfactual probe deferred to v2; probe_question always None in v1.
                terminated = True
                break

        trace = self.env.get_trace(episode_id)
        scenario = self._load_scenario(trace, episode_id)
        reward = compute_reward(
            trace, scenario, quarantine_window=self._quarantine_window
        )

        # Update rolling quarantine window (last 50 secondary actions)
        if trace.episode.final_action:
            for sa in trace.episode.final_action.secondary_actions:
                self._quarantine_window.append(sa.name)
            self._quarantine_window = self._quarantine_window[-50:]

        return {
            "messages": messages,
            "reward": reward.total,
            "reward_breakdown": reward,
            "trajectory_length": len(messages),
            "terminated": terminated,
        }

    # ------------------------------------------------------------------ helpers

    def _get_tools_listing(self) -> str:
        if self._tools_listing is None:
            import json
            tools = self.env.list_tools()
            lines: list[str] = []
            for t in tools:
                lines.append(f"- {t['name']}: {t.get('description', '')}")
                lines.append(f"  args: {json.dumps(t.get('args_schema', {}))}")
            self._tools_listing = "\n".join(lines)
        return self._tools_listing

    def _format_initial_obs(self, obs) -> str:
        if obs.failure_summary is None:
            return "CI failure detected. Begin investigation."
        fs = obs.failure_summary
        return (
            f"CI FAILURE ALERT\n"
            f"Test: {fs.test_name}\n"
            f"Suite: {fs.suite}  Branch: {fs.branch}\n"
            f"Last passing commit: {fs.last_passing_commit}\n"
            f"Log excerpt:\n{fs.initial_log_excerpt}\n\n"
            "Investigate and submit your diagnosis."
        )

    def _format_obs(self, obs) -> str:
        import json
        if obs.is_terminal:
            return "Episode terminated."
        if obs.tool_response is not None:
            tr = obs.tool_response
            out = json.dumps(tr.output, indent=2) if isinstance(tr.output, dict) else str(tr.output)
            return (
                f"Tool: {tr.tool_name}\nCost: ${tr.cost_charged:.4f}\n"
                f"Output:\n{out}\n"
                f"Budget: {obs.budget_remaining.tool_calls_remaining} calls left"
            )
        return "Observation received."

    def _load_scenario(self, trace, episode_id: str) -> Scenario:
        if hasattr(self.env, "get_scenario"):
            return self.env.get_scenario(episode_id)
        scenario_path = (
            Path("data_artifacts/scenarios") / f"{trace.episode.scenario_id}.json"
        )
        if scenario_path.exists():
            return Scenario.model_validate_json(scenario_path.read_text())
        family = trace.episode.scenario_id.split("-")[0]
        from ci_triage_env.mock.scenario import make_mock_scenario
        return make_mock_scenario(family=family if family in {
            "real_bug", "race_flake", "timing_flake",
            "infra_network", "infra_resource", "dependency_drift", "ambiguous",
        } else "real_bug")
