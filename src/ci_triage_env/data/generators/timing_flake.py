"""TimingFlakeGenerator — scenario family: timing_flake."""

from __future__ import annotations

import random

from ci_triage_env.data.clustering.archetypes import Archetype
from ci_triage_env.data.generators._helpers import (
    ArchetypedGenerator,
    _metric_samples,
    build_base_outputs,
    fake_short_sha,
    fake_timestamp,
    fill_template,
    make_failure_summary,
    pick_test_name,
    scenario_id_for,
)
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.scenario import (
    GroundTruth,
    Scenario,
    ScenarioMetadata,
    TerminalActionSpec,
    ToolOutput,
)


class TimingFlakeGenerator(ArchetypedGenerator):
    family_name = "timing_flake"
    label = DiagnosisLabel.TIMING_FLAKE

    def informative_tools(self) -> list[str]:
        return ["read_logs", "query_flake_history", "cluster_metrics"]

    def minimal_evidence_set(self) -> list[str]:
        return ["query_flake_history", "cluster_metrics"]

    def _default_archetypes(self) -> list[Archetype]:
        return [
            Archetype(
                archetype_id="timing_flake_001",
                family="timing_flake",
                pattern_summary="Test timeout: deadline exceeded after N seconds",
                log_template=(
                    "--- FAIL: {TEST_FUNC} ({DURATION}s)\n"
                    "    {TEST_MODULE}_test.go:{NUM}: context canceled: deadline exceeded\n"
                    "FAIL\t{TEST_MODULE}\t{DURATION}s\n"
                    "panic: test timed out after {TIMEOUT}s"
                ),
                slot_distributions={
                    "TEST_FUNC": ["TestAPIResponse", "TestDBQuery", "TestCacheLoad"],
                    "TEST_MODULE": ["github.com/org/repo/api", "github.com/org/repo/storage"],
                    "DURATION": ["30.001", "60.002", "120.000"],
                    "TIMEOUT": ["30", "60", "120"],
                    "NUM": ["47", "83", "124"],
                },
                informative_tools_hint=["read_logs:full", "query_flake_history", "cluster_metrics:queue_depth"],
                minimal_evidence_hint=["query_flake_history", "cluster_metrics:queue_depth"],
            ),
            Archetype(
                archetype_id="timing_flake_002",
                family="timing_flake",
                pattern_summary="Context canceled / timeout exceeded in async call",
                log_template=(
                    "FAILED {TEST_MODULE}::{TEST_FUNC} - TimeoutError\n"
                    "  TimeoutError: Operation timed out after {DURATION} seconds\n"
                    "  During handling of the above exception:\n"
                    "  asyncio.exceptions.TimeoutError\n"
                    "short test summary info\n"
                    "FAILED {TEST_MODULE}::{TEST_FUNC} — timeout exceeded"
                ),
                slot_distributions={
                    "TEST_FUNC": ["test_async_call", "test_rpc_response", "test_batch_process"],
                    "TEST_MODULE": ["tests/integration/test_rpc", "tests/integration/test_batch"],
                    "DURATION": ["5.0", "10.0", "30.0"],
                },
                informative_tools_hint=["read_logs:full", "query_flake_history", "cluster_metrics:queue_depth"],
                minimal_evidence_hint=["cluster_metrics:queue_depth"],
            ),
        ]

    def generate(self, seed: int, source_log_hash: str | None = None) -> Scenario:
        rng = random.Random(seed)
        archetype = self._pick_archetype(rng)
        log_text = fill_template(archetype.log_template, archetype.slot_distributions, rng)
        test_name = pick_test_name(rng)

        summary = make_failure_summary(
            self.family_name, rng, test_name=test_name, log_excerpt=log_text
        )
        branch = summary.branch

        rerun_passes = (seed % 2 == 0)  # passes ~50% of time
        outputs = build_base_outputs(
            test_name, branch, rng,
            log_lines=log_text.splitlines(),
            rerun_passes=rerun_passes,
        )

        # --- informative overrides ---
        # queue_depth elevated — CI under load → timeouts
        outputs["cluster_metrics:queue_depth"] = ToolOutput(
            tool_name="cluster_metrics",
            payload={"samples": _metric_samples(rng, "queue_depth", elevated=True, n=6)},
            cost_units=0.003,
        )

        # Flake history: intermittent, correlates with CI load spikes
        total_runs = 30
        failures = rng.randint(5, 12)
        outputs[f"query_flake_history:{test_name}"] = ToolOutput(
            tool_name="query_flake_history",
            payload={
                "failure_count": failures,
                "pass_count": total_runs - failures,
                "recent_failures": [
                    {"run_id": fake_short_sha(rng), "at": fake_timestamp(rng),
                     "note": "CI queue was full"}
                    for _ in range(min(3, failures))
                ],
            },
            cost_units=0.002,
        )

        difficulty = rng.choice(["easy", "medium", "hard"])
        rationale = (
            f"query_flake_history shows {failures}/{total_runs} intermittent failures. "
            f"cluster_metrics:queue_depth is elevated, indicating CI is under load. "
            f"Timeouts correlate with scheduler pressure — not a code bug. "
            f"recent_commits shows no test-touching changes."
        )

        return Scenario(
            schema_version="1.0",
            scenario_id=scenario_id_for(self.family_name, seed),
            family=self.family_name,
            seed=seed,
            ground_truth=GroundTruth(
                label=self.label,
                rationale=rationale,
                is_ambiguous=False,
                confidence_target=1.0,
            ),
            failure_summary=summary,
            tool_outputs=outputs,
            informative_tools=self.informative_tools(),
            minimal_evidence_set=self.minimal_evidence_set(),
            correct_terminal_action=TerminalActionSpec(
                primary="submit_diagnosis",
                args={
                    "diagnosis": self.label.value,
                    "confidence": 0.9,
                    "secondary_actions": [{"name": "rerun_test"}],
                },
                acceptable_alternatives=[
                    {"primary": "submit_diagnosis",
                     "args": {"diagnosis": "timing_flake", "confidence": 0.85}},
                ],
            ),
            metadata=ScenarioMetadata(
                generator_version="1.0",
                generated_at=fake_timestamp(rng),
                source_log_hash=source_log_hash,
                difficulty=difficulty,
            ),
        )
