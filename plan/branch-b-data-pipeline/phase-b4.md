# Phase B4 — Scenario Family Generators

**Owner:** Branch B.
**Prerequisite:** B3 merged.
**Estimated time:** 5–6 hours. **This is the longest phase in Branch B.**

---

## Outcome

Seven `ScenarioFamilyGenerator` subclasses, one per family. Each takes a seed and produces a fully-populated `Scenario` JSON. By end of phase:

1. All 7 generators implemented in `src/ci_triage_env/data/generators/`.
2. Each uses archetypes from `data_artifacts/clustering/<family>/archetypes.json` as templates.
3. Each is deterministic given a seed.
4. Each produces a `Scenario` that validates against the schema (Phase 0).
5. Each correctly populates `informative_tools` and `minimal_evidence_set`.
6. All B4 unit tests pass (one per generator + integration).

---

## Files to create

### `src/ci_triage_env/data/generators/_helpers.py`

Shared utilities used by all generators:

```python
def fill_template(template: str, slot_distributions: dict[str, list[str]], rng: random.Random) -> str:
    """Replace {SLOT} placeholders with random values from the distribution."""
    out = template
    for slot, values in slot_distributions.items():
        placeholder = "{" + slot + "}"
        while placeholder in out:
            out = out.replace(placeholder, rng.choice(values), 1)
    return out

def make_failure_summary(family: str, rng: random.Random, *, test_name: str, log_excerpt: str) -> FailureSummary:
    """Build a realistic-looking FailureSummary."""
    branch = rng.choice(["main", "develop", f"feature/{rng.choice(FEATURES)}", f"fix/{rng.choice(FIX_KEYWORDS)}"])
    return FailureSummary(
        test_name=test_name,
        suite=infer_suite(test_name),
        branch=branch,
        last_passing_commit=fake_sha(rng),
        initial_log_excerpt=log_excerpt[:400],
        timestamp=fake_timestamp(rng),
    )

def fake_sha(rng) -> str: ...
def fake_timestamp(rng) -> str: ...
def fake_commit(rng, *, touches: list[str] | None = None) -> dict: ...
def fake_owner(rng, path: str) -> dict: ...
```

### `src/ci_triage_env/data/generators/real_bug.py`

```python
class RealBugGenerator(ScenarioFamilyGenerator):
    family_name = "real_bug"
    label = DiagnosisLabel.REAL_BUG

    def informative_tools(self) -> list[str]:
        return ["read_logs", "inspect_test_code", "recent_commits", "rerun_test"]

    def minimal_evidence_set(self) -> list[str]:
        return ["recent_commits", "inspect_test_code"]

    def generate(self, seed: int, source_log_hash: str | None = None) -> Scenario:
        rng = random.Random(seed)
        archetype = self._pick_archetype(rng)
        log_text = fill_template(archetype.log_template, archetype.slot_distributions, rng)
        test_name = self._pick_test_name(rng)

        # Tool outputs:
        tool_outputs = {
            f"read_logs:full": ToolOutput(
                tool_name="read_logs",
                payload={"lines": log_text.split("\n"), "truncated": False},
                cost_units=0.001,
            ),
            f"read_logs:test": ToolOutput(...),  # narrower view
            f"inspect_test_code:{test_name}": ToolOutput(
                tool_name="inspect_test_code",
                payload={"code": self._buggy_code_excerpt(rng), "language": "python"},
                cost_units=0.001,
            ),
            f"recent_commits:24h": ToolOutput(
                tool_name="recent_commits",
                payload={"commits": self._commits_with_breaking_change(rng, test_name)},
                cost_units=0.01,
            ),
            "rerun_test": ToolOutput(
                tool_name="rerun_test",
                payload={"passed": False, "duration_s": rng.uniform(10, 60), "log_excerpt": log_text.split("\n")[:5]},
                cost_units=0.30,
            ),
            # Other tools — populated with non-informative payloads
            f"query_flake_history:{test_name}": ToolOutput(
                tool_name="query_flake_history",
                payload={"runs": [{"passed": True}] * 10 + [{"passed": False}]},  # mostly passing, recent fail
                cost_units=0.01,
            ),
            "cluster_metrics:5m": ToolOutput(...),  # normal metrics
            f"check_owner:src/...": ToolOutput(...),
            ...
        }

        return Scenario(
            schema_version="1.0",
            scenario_id=f"real_bug-{seed}-{rng_hash(rng)}",
            family=self.family_name,
            seed=seed,
            ground_truth=GroundTruth(
                label=self.label,
                rationale=f"The recent commit by {fake_user(rng)} introduced bug visible in logs at line ...",
                is_ambiguous=False,
                confidence_target=1.0,
            ),
            failure_summary=make_failure_summary(self.family_name, rng, test_name=test_name, log_excerpt=log_text),
            tool_outputs=tool_outputs,
            informative_tools=self.informative_tools(),
            minimal_evidence_set=self.minimal_evidence_set(),
            correct_terminal_action=TerminalActionSpec(
                primary="submit_diagnosis",
                args={"diagnosis": "real_bug", "confidence": 1.0,
                      "secondary_actions": [{"name": "file_bug", ...}]},
                acceptable_alternatives=[],
            ),
            metadata=ScenarioMetadata(
                generator_version="1.0",
                generated_at=datetime.utcnow().isoformat(),
                source_log_hash=source_log_hash,
                difficulty=rng.choice(["easy", "medium", "hard"]),
            ),
        )

    def _pick_archetype(self, rng): ...
    def _pick_test_name(self, rng): ...
    def _buggy_code_excerpt(self, rng): ...
    def _commits_with_breaking_change(self, rng, test_name): ...
```

Repeat the same structure for each of the other 6 families. Per-family specifics:

### `race_flake.py`

- Logs interleave thread/goroutine output. Race-detector output if applicable.
- `query_flake_history`: pass/fail mix (e.g., 7/10 pass).
- `rerun_test`: passes ~50% of the time (use seed-derived RNG so it's deterministic per scenario).
- `informative_tools`: `["read_logs", "query_flake_history", "rerun_test"]`
- `minimal_evidence_set`: `["query_flake_history"]` (history alone shows the flake)
- `correct_terminal_action.primary`: `submit_diagnosis(race_flake)` + secondary `quarantine_test`

### `timing_flake.py`

- Logs reference timeouts, deadline-exceeded.
- `query_flake_history`: fail rate correlates with CI load.
- `rerun_test`: passes more often when scheduler is less busy (we just fix a probability per scenario).
- `informative_tools`: `["read_logs", "query_flake_history", "cluster_metrics"]`
- `minimal_evidence_set`: `["query_flake_history", "cluster_metrics"]`

### `infra_network.py`

- Logs reference DNS/TLS/connectivity errors.
- `cluster_metrics`: shows network-related anomalies.
- `query_flake_history`: this test passed historically (not test-specific issue).
- `informative_tools`: `["read_logs", "cluster_metrics"]`
- `minimal_evidence_set`: `["cluster_metrics"]`
- `correct_terminal_action.primary`: `submit_diagnosis(infra_network)` + secondary `rerun_test` (no ticket; infra blip)

### `infra_resource.py`

- Logs reference OOM/disk-full/CPU-throttle.
- `cluster_metrics`: clear resource pressure.
- `informative_tools`: `["read_logs", "cluster_metrics"]`
- `minimal_evidence_set`: `["cluster_metrics"]`
- `correct_terminal_action.primary`: `submit_diagnosis(infra_resource)` + maybe `ping_owner` (capacity team)

### `dependency_drift.py`

- Logs reference version conflicts, lockfile diff.
- `recent_commits`: shows a lockfile/dependency-update commit.
- `inspect_test_code`: code unchanged, but imports use the drifted dep.
- `informative_tools`: `["read_logs", "recent_commits"]`
- `minimal_evidence_set`: `["recent_commits"]`
- `correct_terminal_action.primary`: `submit_diagnosis(dependency_drift)` + secondary `file_bug` to dep team

### `ambiguous.py`

- Logs show multiple plausible signals (e.g., resource pressure AND recent commit changes).
- `query_flake_history`: insufficient history (recently added test).
- `recent_commits`: changes touching the test, but not obviously buggy.
- `cluster_metrics`: borderline pressure, not clear-cut.
- `informative_tools`: list all but mark none as definitive.
- `minimal_evidence_set`: empty (no subset uniquely determines).
- `correct_terminal_action.primary`: `submit_diagnosis(ambiguous, confidence_calibrated)` — `confidence` should be ~0.4, NOT 1.0. The Brier-score reward in C1 penalizes overconfidence.
- `ground_truth.is_ambiguous`: `True`.
- `ground_truth.confidence_target`: depends on scenario — pre-computed per instance.

### `src/ci_triage_env/data/generators/__init__.py`

```python
GENERATOR_REGISTRY = {
    "real_bug": RealBugGenerator,
    "race_flake": RaceFlakeGenerator,
    "timing_flake": TimingFlakeGenerator,
    "infra_network": InfraNetworkGenerator,
    "infra_resource": InfraResourceGenerator,
    "dependency_drift": DependencyDriftGenerator,
    "ambiguous": AmbiguousGenerator,
}
```

---

## Implementation notes

- **One archetype per scenario isn't enough.** Each scenario picks one archetype as the *primary* signal but layers in *distractor* outputs from other archetypes (e.g., a `real_bug` scenario also has plausible-but-wrong cluster_metrics output that doesn't actually match). Distractors prevent the model from pattern-matching on tool name alone.
- **All 11 tools must have an output entry per scenario.** Even if non-informative, each tool needs an output (otherwise A2's tool returns "no data" which leaks information).
- **Ground truth `rationale` is for human review only.** Never shown to the agent. Must be detailed enough that a human can verify the scenario is correct.
- **`correct_terminal_action.acceptable_alternatives`.** Some scenarios have multiple right answers (e.g., for an infra issue, both `rerun_test` and `ping_owner(capacity_team)` are valid secondary actions). List both with similar reward weights.
- **Difficulty levels.** `easy`: signal is in 1–2 tools; `medium`: signal requires 3+; `hard`: requires correct *order* of investigation. Tag accordingly so eval can stratify.
- **Ambiguous family is special.** It's the calibration-probe target. Generator must produce scenarios where:
  - Multiple plausible labels exist
  - The correct response is `submit_diagnosis(ambiguous, confidence < 0.5)`
  - `ground_truth.confidence_target` matches what a calibrated agent should output

---

## Tests required (`tests/data/test_generators.py`)

For each of the 7 generators:

```python
def test_<family>_generator_determinism():
    """Same seed → identical Scenario."""
    s1 = GENERATOR_REGISTRY["<family>"]().generate(seed=42)
    s2 = GENERATOR_REGISTRY["<family>"]().generate(seed=42)
    assert s1 == s2

def test_<family>_generator_validates():
    """Output passes Scenario schema validation."""

def test_<family>_ground_truth_label():
    """ground_truth.label matches family."""

def test_<family>_all_tools_have_outputs():
    """For each of 11 tools, scenario.tool_outputs has at least one key."""

def test_<family>_informative_tools_nonempty():
    """informative_tools has ≥ 2 entries."""

def test_<family>_minimal_evidence_set_subset():
    """minimal_evidence_set ⊆ informative_tools."""

def test_<family>_minimal_evidence_actually_minimal():
    """For 5 random seeds, reaching the correct diagnosis using only minimal_evidence_set tools is feasible (manual review)."""
```

Special tests for ambiguous:

```python
def test_ambiguous_confidence_target_below_one():
    """ground_truth.confidence_target < 1.0."""

def test_ambiguous_is_flagged():
    """ground_truth.is_ambiguous == True."""
```

Integration test:

```python
def test_all_generators_produce_distinct_scenarios():
    """Generate 10 scenarios per family with different seeds; all scenario_ids are unique."""

def test_generators_use_archetypes_when_available():
    """If clustering archetypes exist, generator picks from them; otherwise falls back to defaults."""
```

---

## Smoke test (manual)

```bash
python -c "
from ci_triage_env.data.generators import GENERATOR_REGISTRY
from ci_triage_env.schemas.scenario import Scenario
import json

for family, GenCls in GENERATOR_REGISTRY.items():
    s = GenCls().generate(seed=1)
    Scenario.model_validate(s.model_dump())  # validates
    print(f'{family}: {s.scenario_id}, difficulty={s.metadata.difficulty}')
    print(f'  informative_tools: {s.informative_tools}')
    print(f'  minimal_evidence: {s.minimal_evidence_set}')
"
```

Expected: 7 lines, each with a unique scenario_id and well-formed metadata.

---

## Realism check before merging

For each family, generate 3 scenarios with random seeds. Manually inspect:

1. Does the failure_summary read like a real CI failure?
2. Do the tool outputs (especially `read_logs:full`) look like a real engineer would see?
3. Does `ground_truth.rationale` make sense — would a human reach the same conclusion from the tool outputs?
4. Are `informative_tools` actually informative for this scenario?
5. Is `minimal_evidence_set` actually minimal?

If any check fails, fix the generator before merging. Do this in a personal review session before opening a PR.

---

## Open questions

1. **Distractor strength.** How aggressive should distractors be? Recommend: 1–2 plausible-but-wrong signals per scenario. Too many = unsolvable; too few = trivial pattern matching.
2. **Buggy code excerpts for `real_bug`.** Hand-author a small library of ~20 buggy snippets across languages, sample from them. More variety can be added in B5 if time permits.
3. **`difficulty` distribution.** Aim for 40% easy, 40% medium, 20% hard per family. Eval will stratify on this.

---

## What's NOT in this phase

- Bulk instantiation and HF dataset publishing (B5)
- Hand-authored ambiguous edge cases (B5)
