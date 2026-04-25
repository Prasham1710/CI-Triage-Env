# Phase C3 — SFT Trajectory Generation

**Owner:** Branch C.
**Prerequisite:** C2 merged. Branch A through A3 merged into main (Gate-1) so that real env is callable. Branch B through B5 merged so that real scenarios are available.
**Estimated time:** 4–5 hours wall time (mostly waiting on API).
**Budget impact:** $25 hard cap on OpenAI API. $5 reserve.

---

## Outcome

A reward-filtered SFT dataset of high-quality trajectories ready for warmstart. By end of phase:

1. `python -m ci_triage_env.training.trajectory_gen --count 600 --model gpt-5-mini` runs the generation loop.
2. Each trajectory: env reset → model picks tool calls turn by turn → model submits terminal → reward computed.
3. Top 30% by total reward (~180 trajectories) saved as SFT dataset.
4. SFT dataset format: HF `datasets.Dataset` with one row per trajectory (input prompt, full chat completion, final reward).
5. Budget tracker (`plan/BUDGET-LOG.md`) updated after every 100 trajectories. Hard stop at $25.
6. All C3 unit tests pass (small mocked tests).

---

## Files to create

### `src/ci_triage_env/training/env_client.py`

```python
import httpx
import json
from ..schemas.observation import Observation
from ..schemas.action import ToolCall, TerminalAction
from ..schemas.episode import EpisodeTrace

class EnvClient:
    """HTTP client for the CI-Triage env server."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def reset(self, scenario_id: str | None = None, seed_override: int | None = None) -> Observation:
        resp = self.client.post("/reset", json={"scenario_id": scenario_id, "seed_override": seed_override})
        resp.raise_for_status()
        return Observation.model_validate(resp.json())

    def step(self, episode_id: str, action: ToolCall | TerminalAction | dict) -> Observation:
        if isinstance(action, ToolCall):
            payload = {"tool_name": action.tool_name, "args": action.args}
        elif isinstance(action, TerminalAction):
            payload = action.model_dump()
        else:
            payload = action
        resp = self.client.post("/step", json={"episode_id": episode_id, "action": payload})
        resp.raise_for_status()
        return Observation.model_validate(resp.json())

    def get_state(self, episode_id: str):
        resp = self.client.get(f"/state/{episode_id}")
        resp.raise_for_status()
        return resp.json()

    def get_trace(self, episode_id: str) -> EpisodeTrace:
        """Read the EpisodeTrace JSON written by the server on terminal."""
        # Either via dedicated endpoint or by reading from data_artifacts/traces/
        ...

    def list_tools(self) -> list[dict]:
        resp = self.client.get("/mcp/tools")
        resp.raise_for_status()
        return resp.json()
```

### `src/ci_triage_env/training/mock_env_client.py`

```python
class MockEnvClient:
    """In-memory env replacement for tests / pre-Gate-1 development.
    Replays mock trajectories from the mock fixtures."""

    def __init__(self):
        self._episodes: dict[str, MockEpisode] = {}

    def reset(self, scenario_id=None, seed_override=None) -> Observation:
        ...

    def step(self, episode_id, action) -> Observation:
        ...
```

### `src/ci_triage_env/training/trajectory_gen.py`

```python
import os
import json
import time
from pathlib import Path
from openai import OpenAI
from .env_client import EnvClient
from ..rewards.composite import compute_reward
from ..schemas.scenario import Scenario

SYSTEM_PROMPT = """You are an expert SRE investigating a CI failure. Your goal is to determine the root cause and take the right action with minimum cost.

You have these tools available:
{tools_listing}

After investigating, submit your final diagnosis using:
{{
  "action_type": "submit_diagnosis",
  "diagnosis": "<one of: real_bug, race_flake, timing_flake, infra_network, infra_resource, dependency_drift, ambiguous>",
  "confidence": <float in [0, 1]>,
  "secondary_actions": [<optional list of {{name, args}} for file_bug, quarantine_test, rerun_test, ping_owner>]
}}

Investigate efficiently. Cheap diagnostic tools first. Don't quarantine unless you're sure it's a flake — that ships bugs to production.
"""

class TrajectoryGenerator:
    def __init__(self, api_key: str, model: str = "gpt-5-mini",
                 budget_usd: float = 25.0, env_url: str = "http://localhost:8000"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.budget = budget_usd
        self.spent = 0.0
        self.env = EnvClient(env_url)
        self.tools_listing = self._format_tools()

    def _format_tools(self) -> str:
        tools = self.env.list_tools()
        lines = []
        for t in tools:
            lines.append(f"- {t['name']}: {t.get('description', '')}")
            lines.append(f"  args: {json.dumps(t.get('args_schema', {}))}")
        return "\n".join(lines)

    def generate_one(self, scenario_id: str | None = None) -> dict | None:
        """Run one full episode with the model, return trajectory dict or None on failure."""
        if self.spent >= self.budget:
            return None

        obs = self.env.reset(scenario_id=scenario_id)
        episode_id = obs.episode_id
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(tools_listing=self.tools_listing)},
            {"role": "user", "content": self._format_initial_observation(obs)},
        ]

        max_turns = 12
        for turn in range(max_turns):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=600,
                )
            except Exception as e:
                print(f"OpenAI error: {e}, abandoning trajectory")
                return None
            self.spent += self._estimate_cost(completion)

            response_text = completion.choices[0].message.content
            messages.append({"role": "assistant", "content": response_text})

            action = self._parse_action(response_text)
            if action is None:
                # Malformed — give the model one more chance
                messages.append({"role": "user", "content": "Your response was not a valid tool call or terminal action. Reply with valid JSON."})
                continue

            try:
                next_obs = self.env.step(episode_id, action)
            except Exception as e:
                print(f"Env step error: {e}, abandoning")
                return None

            messages.append({"role": "user", "content": self._format_observation(next_obs)})

            if next_obs.is_terminal:
                # NOTE: counterfactual probe handling is deferred to v2. The schema
                # field `next_obs.probe_question` exists but the env never populates
                # it in v1 (see plan/branch-a-env-core/phase-a4.md).
                break

        # Get the trace and compute reward
        trace = self.env.get_trace(episode_id)
        scenario = self._load_scenario(trace.episode.scenario_id)
        reward = compute_reward(trace, scenario)

        return {
            "episode_id": episode_id,
            "scenario_id": trace.episode.scenario_id,
            "messages": messages,   # full chat history
            "reward": reward.total,
            "reward_breakdown": reward.model_dump(),
            "format_gate_passed": reward.format_gate,
        }

    def _format_initial_observation(self, obs: Observation) -> str:
        # Render the failure_summary as a readable user message
        ...

    def _format_observation(self, obs: Observation) -> str:
        # Render tool_response or probe_question as readable user message
        ...

    def _parse_action(self, text: str) -> ToolCall | TerminalAction | None:
        """Parse JSON from model's response. Try whole-text JSON first, then code-block extraction."""
        ...

    def _parse_probe_response(self, text: str) -> dict:
        """Parse {predicted_outcome, confidence}."""
        ...

    def _load_scenario(self, scenario_id: str) -> Scenario:
        ...

    def _estimate_cost(self, completion) -> float:
        usage = completion.usage
        # gpt-5-mini approx — verify at runtime
        in_per_M = 1.0
        out_per_M = 4.0
        return (usage.prompt_tokens * in_per_M + usage.completion_tokens * out_per_M) / 1_000_000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--budget", type=float, default=25.0)
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--output", default="data_artifacts/sft_dataset/")
    parser.add_argument("--top-fraction", type=float, default=0.30)
    args = parser.parse_args()

    api_key = os.environ["OPENAI_API_KEY"]
    gen = TrajectoryGenerator(api_key, model=args.model, budget_usd=args.budget, env_url=args.env_url)

    trajectories = []
    for i in range(args.count):
        if gen.spent >= gen.budget:
            print(f"Budget exhausted after {i} trajectories.")
            break
        traj = gen.generate_one()
        if traj is None:
            continue
        trajectories.append(traj)
        if i % 50 == 0:
            print(f"[{i}/{args.count}] spent=${gen.spent:.2f} kept={len(trajectories)}")
            _update_budget_log(gen.spent)

    # Filter top-N by reward
    trajectories.sort(key=lambda t: t["reward"], reverse=True)
    keep_n = int(len(trajectories) * args.top_fraction)
    sft_set = trajectories[:keep_n]

    # Save as HF dataset
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    from datasets import Dataset
    ds = Dataset.from_list(sft_set)
    ds.save_to_disk(str(out_dir))

    # Summary
    print(f"\nGenerated {len(trajectories)}, kept top {keep_n}")
    print(f"Reward distribution of kept: min={sft_set[-1]['reward']:.2f}  "
          f"max={sft_set[0]['reward']:.2f}  median={sft_set[keep_n//2]['reward']:.2f}")
    print(f"Total spent: ${gen.spent:.2f}")
```

### Update `plan/BUDGET-LOG.md` after every 100 trajectories:

```
[2026-04-26 14:33] C3 trajectory_gen: 100 trajectories, gpt-5-mini, $4.12 cumulative
[2026-04-26 14:51] C3 trajectory_gen: 200 trajectories, gpt-5-mini, $8.18 cumulative
...
```

---

## Implementation notes

- **OpenAI model name.** `gpt-5-mini` is a placeholder — verify the current cheap-but-capable reasoning model on OpenAI's dashboard at run time. Likely options: `gpt-5-mini`, `gpt-5-nano`, `o4-mini`. Whatever's cheapest with reasoning. Update `--model` default after checking pricing.
- **Pricing verification.** Before running the full 600, run with `--count 5` and check the actual `usage` data against your assumed pricing. Adjust budget cap if pricing is higher than expected.
- **Parallelism.** Don't run trajectories in parallel. The env server is single-process and handles one episode at a time. Even if it could, OpenAI rate limits will throttle you. Sequential is fine — the bottleneck is API latency, ~5–10s per turn × 6 turns = 30–60s per trajectory.
- **Failure modes:**
  - Model emits malformed JSON → 1 retry then abandon trajectory.
  - Env step fails → abandon trajectory.
  - OpenAI API error → exponential backoff, then abandon if persistent.
- **Trajectory shape for SFT.** The output is the full chat-completion sequence. SFT in C4 trains on completing each assistant turn given preceding context. This means the model learns the *full multi-turn behavior*, not just the final action.
- **Why 30% filter?** Generation includes failed and mediocre trajectories. The top 30% are the ones we want the model to imitate. After filter, expect ~180 trajectories, ~5MB of SFT data.
- **Trace retrieval.** Env server writes traces to disk. Generator reads them after each episode terminates. Alternative: add a `/trace/{episode_id}` endpoint to the env. Recommend adding the endpoint in this phase as a small follow-up to A3 — coordinate with Branch A.

---

## Tests required (`tests/training/test_trajectory_gen.py`)

```python
def test_parse_action_valid_tool_call():
    text = '{"tool_name": "read_logs", "args": {"scope": "full"}}'
    action = TrajectoryGenerator._parse_action(None, text)
    assert isinstance(action, ToolCall)
    assert action.tool_name == "read_logs"

def test_parse_action_valid_terminal():
    text = '{"action_type": "submit_diagnosis", "diagnosis": "real_bug", "confidence": 0.9, "secondary_actions": []}'
    action = TrajectoryGenerator._parse_action(None, text)
    assert isinstance(action, TerminalAction)

def test_parse_action_with_code_block():
    text = '```json\n{"tool_name": "read_logs", "args": {"scope": "full"}}\n```'
    action = TrajectoryGenerator._parse_action(None, text)
    assert action is not None

def test_parse_action_malformed_returns_none():
    text = "this is not JSON"
    action = TrajectoryGenerator._parse_action(None, text)
    assert action is None

def test_estimate_cost_uses_token_counts():
    """Mock completion with known usage; cost matches formula."""

def test_budget_check_stops_generation(monkeypatch):
    """Mock OpenAI to charge $30 per call; gen.spent exceeds 25 after 1 call; second call returns None."""

def test_generate_one_full_loop_with_mock_env(monkeypatch):
    """Mock env client + OpenAI; verify trajectory dict shape."""

def test_top_fraction_filter():
    """Given trajectories with rewards [0.5, 0.8, -0.2, 0.9, 0.3], top 0.4 fraction gives [0.9, 0.8]."""
```

---

## Smoke test (manual)

```bash
# Start env server
python -m ci_triage_env.env.server &
sleep 2

# Run small generation (5 trajectories) to verify pricing and pipeline
export OPENAI_API_KEY=sk-...
python -m ci_triage_env.training.trajectory_gen --count 5 --budget 1.0

# Check output
ls data_artifacts/sft_dataset/
python -c "
from datasets import load_from_disk
ds = load_from_disk('data_artifacts/sft_dataset/')
print(ds)
print('First reward:', ds[0]['reward'])
"

# Check budget log
cat plan/BUDGET-LOG.md | tail
```

Expected: ~5 trajectories generated, ~1 kept after 30% filter, budget reflects actual cost.

---

## Hard-stop discipline

If `spent` exceeds `budget` at any point, the loop terminates. Restart with `--budget 30` only after consulting team in chat.

If pricing turns out to be 2x what we estimated, immediately reduce `--count` proportionally.

---

## Open questions

1. **Should we re-generate trajectories on the same scenario multiple times?** Yes — diversity helps. The default loop picks scenarios at random; some will repeat. Don't enforce uniqueness.
2. **Should we include trajectories that hit budget exhaustion in the SFT set?** No — those traces have `final_action=None` which we don't want the model to imitate. Filter them out before reward sorting.
3. **Should we cap trajectory length in the SFT data?** Yes — drop any trajectory > 20 turns; those are probably the model getting confused. Pre-filter before reward sorting.

---

## What's NOT in this phase

- The actual SFT training run (C4)
- GRPO (C4)
- Eval (C5)
