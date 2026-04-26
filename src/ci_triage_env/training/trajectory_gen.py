"""SFT trajectory generator — runs LLM-in-the-loop episodes and reward-filters results.

Usage:
    python -m ci_triage_env.training.trajectory_gen --count 600 --model gpt-5-mini

Each trajectory: env reset → model picks tool calls turn by turn → model submits
terminal action → reward computed. Top 30% by total reward saved as SFT dataset.

Budget cap: $25 hard limit. Update plan/BUDGET-LOG.md every 100 trajectories.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC
from pathlib import Path

from ci_triage_env.rewards.composite import compute_reward
from ci_triage_env.schemas.action import TerminalAction, ToolCall
from ci_triage_env.schemas.observation import Observation
from ci_triage_env.schemas.scenario import Scenario

SYSTEM_PROMPT = """\
You are an expert SRE investigating a CI failure. Your goal is to determine the root \
cause and take the right action with minimum cost.

You have these tools available:
{tools_listing}

To call a tool, respond with JSON:
{{"tool_name": "<name>", "args": {{...}}}}

After investigating, submit your final diagnosis:
{{"action_type": "submit_diagnosis", "diagnosis": "<family>", "confidence": <0-1>, \
"secondary_actions": []}}

Valid families: real_bug, race_flake, timing_flake, infra_network, infra_resource, \
dependency_drift, ambiguous.

Investigate efficiently. Cheap diagnostic tools (read_logs, query_flake_history, \
recent_commits) before expensive ones (rerun_test, file_bug). \
Do NOT quarantine unless certain it is a flake — quarantining a real bug ships it to prod.\
"""

# Approximate OpenAI pricing (verify at runtime with --count 5 first)
_PRICE_IN_PER_M: float = 0.15   # gpt-4o-mini input price per 1M tokens
_PRICE_OUT_PER_M: float = 0.60  # gpt-4o-mini output price per 1M tokens


class TrajectoryGenerator:
    """Runs LLM-in-the-loop episodes against the CI-Triage env and collects trajectories.

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model: OpenAI chat model name.
        budget_usd: Hard spend cap in USD. Generation stops when exceeded.
        env_client: Env client instance (real EnvClient or MockEnvClient).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        budget_usd: float = 25.0,
        env_client=None,
    ) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", ""))
        self.model = model
        self.budget = budget_usd
        self.spent: float = 0.0
        self.env = env_client
        self._tools_listing: str | None = None

    # ------------------------------------------------------------------ public

    def generate_one(self, scenario_id: str | None = None) -> dict | None:
        """Run one full episode. Returns trajectory dict or None on failure/budget exhaustion."""
        if self.spent >= self.budget:
            return None

        obs = self.env.reset(scenario_id=scenario_id)
        episode_id = obs.episode_id

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                tools_listing=self._get_tools_listing()
            )},
            {"role": "user", "content": self._format_initial_observation(obs)},
        ]

        max_turns = 12
        terminated = False
        for _turn in range(max_turns):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=600,
                )
            except Exception as exc:
                print(f"OpenAI error on turn {_turn}: {exc} — abandoning trajectory")
                return None

            self.spent += self._estimate_cost(completion)
            if self.spent >= self.budget:
                print(f"Budget exhausted (${self.spent:.2f}); stopping generation.")
                return None

            response_text = completion.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": response_text})

            action = self._parse_action(response_text)
            if action is None:
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response was not valid JSON for a tool call or terminal action. "
                        "Reply with valid JSON only."
                    ),
                })
                continue

            try:
                next_obs = self.env.step(episode_id, action)
            except Exception as exc:
                print(f"Env step error: {exc} — abandoning trajectory")
                return None

            messages.append({"role": "user", "content": self._format_observation(next_obs)})

            if next_obs.is_terminal:
                terminated = True
                break

        if not terminated:
            return None  # budget-exhausted trajectory — don't include in SFT set

        trace = self.env.get_trace(episode_id)
        scenario = self._load_scenario(trace, episode_id)
        reward = compute_reward(trace, scenario)

        if reward.format_gate is False:
            return None  # malformed trajectory

        return {
            "episode_id": episode_id,
            "scenario_id": trace.episode.scenario_id,
            "messages": messages,
            "reward": reward.total,
            "reward_breakdown": reward.model_dump(),
            "format_gate_passed": reward.format_gate,
        }

    # ------------------------------------------------------------------ internal helpers

    def _get_tools_listing(self) -> str:
        if self._tools_listing is None:
            tools = self.env.list_tools()
            lines: list[str] = []
            for t in tools:
                lines.append(f"- {t['name']}: {t.get('description', '')}")
                lines.append(f"  args: {json.dumps(t.get('args_schema', {}))}")
            self._tools_listing = "\n".join(lines)
        return self._tools_listing

    def _format_initial_observation(self, obs: Observation) -> str:
        if obs.failure_summary is None:
            return "A CI failure has been detected. Begin investigation."
        fs = obs.failure_summary
        parts = [
            "CI FAILURE ALERT",
            f"Test: {fs.test_name}",
            f"Suite: {fs.suite}  Branch: {fs.branch}",
            f"Last passing commit: {fs.last_passing_commit}",
            f"Log excerpt:\n{fs.initial_log_excerpt}",
            "\nInvestigate the failure and submit a diagnosis.",
        ]
        return "\n".join(parts)

    def _format_observation(self, obs: Observation) -> str:
        if obs.is_terminal:
            return "Episode terminated."
        if obs.tool_response is not None:
            tr = obs.tool_response
            output_str = json.dumps(tr.output, indent=2) if isinstance(tr.output, dict) else str(tr.output)
            return (
                f"Tool: {tr.tool_name}\n"
                f"Cost: ${tr.cost_charged:.4f}\n"
                f"Output:\n{output_str}\n"
                f"Budget remaining: {obs.budget_remaining.tool_calls_remaining} calls, "
                f"${obs.budget_remaining.cost_remaining:.3f}"
            )
        return "Observation received."

    @staticmethod
    def _parse_action(text: str) -> ToolCall | TerminalAction | None:
        """Extract JSON from model response; return ToolCall, TerminalAction, or None."""
        if not text:
            return None

        # Try to extract JSON from a ```json code block first
        block_match = re.search(r"```(?:json)?\s*(\{.*?})\s*```", text, re.DOTALL)
        candidates = [block_match.group(1)] if block_match else []

        # Also try raw JSON anywhere in the text
        raw_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
        if raw_match:
            candidates.append(raw_match.group(0))

        # Try the whole text as JSON
        candidates.append(text.strip())

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if not isinstance(data, dict):
                continue

            if "action_type" in data:
                try:
                    return TerminalAction.model_validate(data)
                except Exception:
                    continue
            if "tool_name" in data:
                try:
                    return ToolCall.model_validate(data)
                except Exception:
                    continue

        return None

    def _load_scenario(self, trace, episode_id: str) -> Scenario:
        # If the env client supports get_scenario (MockEnvClient), use it directly
        if hasattr(self.env, "get_scenario"):
            return self.env.get_scenario(episode_id)
        # Otherwise try loading from disk (real env writes scenarios alongside traces)
        scenario_path = (
            Path("data_artifacts/scenarios") / f"{trace.episode.scenario_id}.json"
        )
        if scenario_path.exists():
            return Scenario.model_validate_json(scenario_path.read_text())
        # Last resort: generate a mock scenario matching the family embedded in the ID
        family = trace.episode.scenario_id.split("-")[0]
        from ci_triage_env.mock.scenario import make_mock_scenario
        return make_mock_scenario(family=family if family in {
            "real_bug", "race_flake", "timing_flake",
            "infra_network", "infra_resource", "dependency_drift", "ambiguous",
        } else "real_bug")

    @staticmethod
    def _estimate_cost(completion) -> float:
        """Estimate USD cost from a chat completion's usage object."""
        usage = completion.usage
        return (
            usage.prompt_tokens * _PRICE_IN_PER_M
            + usage.completion_tokens * _PRICE_OUT_PER_M
        ) / 1_000_000


def _update_budget_log(spent: float, n_trajectories: int) -> None:
    from datetime import datetime
    log_path = Path("plan/BUDGET-LOG.md")
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    entry = f"[{timestamp}] C3 trajectory_gen: {n_trajectories} trajectories, ${spent:.2f} cumulative\n"
    if log_path.exists():
        with log_path.open("a") as f:
            f.write(entry)
    else:
        log_path.write_text(f"# Budget Log\n\n{entry}")


def _filter_top_fraction(
    trajectories: list[dict], fraction: float
) -> list[dict]:
    """Sort by reward descending; keep top `fraction`."""
    trajectories = [t for t in trajectories if t.get("format_gate_passed")]
    trajectories.sort(key=lambda t: t["reward"], reverse=True)
    keep_n = max(1, int(len(trajectories) * fraction))
    return trajectories[:keep_n]


def _run_parallel(
    api_key: str,
    model: str,
    scenarios_dir: str | None,
    count: int,
    budget_usd: float,
    checkpoint_path: Path,
    max_workers: int,
) -> tuple[list[dict], float]:
    """Run trajectory generation with a thread pool, writing each result to a JSONL checkpoint.

    Each worker gets its own MockEnvClient + TrajectoryGenerator so there is no shared
    mutable state between threads except the budget counter and the checkpoint file.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ── load existing checkpoint so we can resume ────────────────────────────
    done: list[dict] = []
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    done.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: loaded {len(done)} trajectories from {checkpoint_path}")

    remaining = max(0, count - len(done))
    if remaining == 0:
        print("Checkpoint already complete — nothing to generate.")
        return done, 0.0

    # ── shared state ─────────────────────────────────────────────────────────
    total_spent = 0.0
    collected = list(done)
    lock = threading.Lock()

    thread_local = threading.local()

    def get_worker():
        """One (env, gen) pair per thread — no cross-thread state sharing."""
        if not hasattr(thread_local, "gen"):
            if scenarios_dir:
                from ci_triage_env.training.mock_env_client import MockEnvClient
                env = MockEnvClient(scenarios_dir=scenarios_dir)
            else:
                from ci_triage_env.training.mock_env_client import MockEnvClient
                env = MockEnvClient()
            thread_local.gen = TrajectoryGenerator(
                api_key=api_key,
                model=model,
                budget_usd=1e9,  # unlimited per worker; global budget enforced below
                env_client=env,
            )
            thread_local.prev_spent = 0.0
        return thread_local.gen

    def run_one(_idx: int) -> dict | None:
        nonlocal total_spent
        with lock:
            if total_spent >= budget_usd:
                return None
        gen = get_worker()
        traj = gen.generate_one()
        delta = gen.spent - thread_local.prev_spent
        thread_local.prev_spent = gen.spent
        with lock:
            total_spent += delta
            if traj is not None:
                collected.append(traj)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("a") as f:
                    f.write(json.dumps(traj) + "\n")
        return traj

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_one, i) for i in range(remaining)]
        for future in as_completed(futures):
            completed += 1
            future.result()  # surface exceptions
            if completed % max(1, max_workers * 2) == 0:
                with lock:
                    print(
                        f"  [{len(done) + completed}/{count}] "
                        f"collected={len(collected)}  spent=${total_spent:.2f}"
                    )

    return collected, total_spent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ci_triage_env.training.trajectory_gen")
    parser.add_argument("--count", type=int, default=600, help="Trajectories to attempt")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    parser.add_argument("--budget", type=float, default=25.0, help="USD spend cap")
    parser.add_argument("--env-url", default="http://localhost:8000", help="Env server URL (ignored when --scenarios-dir is set)")
    parser.add_argument("--output", default="data_artifacts/sft_dataset/", help="Output dir")
    parser.add_argument("--top-fraction", type=float, default=0.50, help="Keep top N%%")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel worker threads (default 10; increase to 20 if not rate-limited)")
    parser.add_argument("--checkpoint", default="data_artifacts/traj_checkpoint.jsonl",
                        help="JSONL file written after each trajectory. Restart from here if interrupted.")
    parser.add_argument(
        "--scenarios-dir", default=None,
        help="Path to a directory of scenario JSON files. Uses MockEnvClient in-process — no server needed."
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use synthetic MockEnvClient (no scenarios-dir, no server; for smoke-testing only)"
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("warning: OPENAI_API_KEY not set — generation will fail on first call")

    scenarios_dir: str | None = None
    if args.scenarios_dir:
        scenarios_dir = args.scenarios_dir
        # Print count for info only — actual clients created per-thread
        from ci_triage_env.training.mock_env_client import MockEnvClient as _MC
        _probe = _MC(scenarios_dir=scenarios_dir)
        print(f"Using {len(_probe.scenario_ids)} real scenarios from {scenarios_dir}")
    elif not args.mock:
        # Falls back to EnvClient in per-thread workers if no scenarios-dir — not supported
        # in parallel mode; just use mock instead.
        print("No --scenarios-dir given; using synthetic MockEnvClient.")

    checkpoint_path = Path(args.checkpoint)
    print(f"Checkpoint: {checkpoint_path}  (safe to Ctrl+C and resume)")
    print(f"Workers: {args.workers}  |  target: {args.count} attempts  |  budget: ${args.budget}")

    trajectories, total_spent = _run_parallel(
        api_key=api_key,
        model=args.model,
        scenarios_dir=scenarios_dir,
        count=args.count,
        budget_usd=args.budget,
        checkpoint_path=checkpoint_path,
        max_workers=args.workers,
    )

    sft_set = _filter_top_fraction(trajectories, args.top_fraction)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    from datasets import Dataset
    ds = Dataset.from_list(sft_set)
    ds.save_to_disk(str(out_dir))

    if sft_set:
        rewards = [t["reward"] for t in sft_set]
        mid = len(sft_set) // 2
        print(
            f"\nGenerated {len(trajectories)}, kept top {len(sft_set)}\n"
            f"Reward: min={min(rewards):.3f}  max={max(rewards):.3f}  "
            f"median={sft_set[mid]['reward']:.3f}\n"
            f"Total spent: ${total_spent:.2f}"
        )
    else:
        print("No valid trajectories collected.")

    _update_budget_log(total_spent, len(trajectories))


if __name__ == "__main__":
    main()
