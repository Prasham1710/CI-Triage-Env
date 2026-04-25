"""AmbiguousGenerator — scenario family: ambiguous.

These scenarios deliberately blend signals from multiple families so that no
single tool reading uniquely determines the label.  The correct response is
``submit_diagnosis(ambiguous, confidence ≈ 0.4)``, NOT a high-confidence single
label.  Branch C's Brier-score reward penalises overconfidence here.
"""

from __future__ import annotations

import random

from ci_triage_env.data.clustering.archetypes import Archetype
from ci_triage_env.data.generators._helpers import (
    ArchetypedGenerator,
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

# Confidence that a calibrated agent should express for ambiguous scenarios
_AMBIGUOUS_CONFIDENCE_TARGET = 0.4


class AmbiguousGenerator(ArchetypedGenerator):
    family_name = "ambiguous"
    label = DiagnosisLabel.AMBIGUOUS

    def informative_tools(self) -> list[str]:
        # All tools are listed but none is definitive alone
        return [
            "read_logs",
            "query_flake_history",
            "recent_commits",
            "cluster_metrics",
            "inspect_test_code",
            "run_diagnostic",
        ]

    def minimal_evidence_set(self) -> list[str]:
        # No subset uniquely determines the label — empty by design
        return []

    def _default_archetypes(self) -> list[Archetype]:
        return [
            Archetype(
                archetype_id="ambiguous_001",
                family="ambiguous",
                pattern_summary="Memory pressure coincides with recent commit change",
                log_template=(
                    "FAILED {TEST_MODULE}::{TEST_FUNC} — signal: killed\n"
                    "kernel: Out of memory: Killed process {NUM} (pytest)\n"
                    "  Also noted: recent commit changed {MODULE} (author: {AUTHOR})\n"
                    "  query_flake_history shows {FAIL_COUNT}/{TOTAL} failures (borderline)"
                ),
                slot_distributions={
                    "TEST_MODULE": ["tests/unit/test_core", "tests/integration/test_batch"],
                    "TEST_FUNC": ["test_process_large", "test_batch_compute"],
                    "NUM": ["1234", "5678", "9012"],
                    "MODULE": ["src/core/processor.py", "src/batch/runner.py"],
                    "AUTHOR": ["@alice", "@bob"],
                    "FAIL_COUNT": ["3", "4"],
                    "TOTAL": ["15", "20"],
                },
                informative_tools_hint=["read_logs:kernel", "cluster_metrics:queue_depth", "recent_commits", "query_flake_history"],
                minimal_evidence_hint=[],
            ),
            Archetype(
                archetype_id="ambiguous_002",
                family="ambiguous",
                pattern_summary="Timeout overlaps with flake history AND borderline cluster load",
                log_template=(
                    "context canceled: deadline exceeded after {DURATION}s\n"
                    "FAILED {TEST_MODULE}::{TEST_FUNC}\n"
                    "  — queue_depth: {QUEUE} (borderline elevated)\n"
                    "  — flake_rate: {FAIL_COUNT}/{TOTAL} runs failed (inconclusive)\n"
                    "  — recent change to {MODULE} by {AUTHOR} ({COMMIT_MSG})"
                ),
                slot_distributions={
                    "DURATION": ["30", "60"],
                    "TEST_MODULE": ["tests/integration/test_rpc", "tests/unit/test_worker"],
                    "TEST_FUNC": ["test_rpc_call", "test_worker_drain"],
                    "QUEUE": ["0.45", "0.52", "0.48"],
                    "FAIL_COUNT": ["2", "3"],
                    "TOTAL": ["12", "18"],
                    "MODULE": ["src/rpc/client.py", "src/worker/loop.py"],
                    "AUTHOR": ["@carol", "@dave"],
                    "COMMIT_MSG": ["refactor: simplify timeout handling", "fix: adjust backoff"],
                },
                informative_tools_hint=["read_logs:full", "query_flake_history", "cluster_metrics:queue_depth", "recent_commits"],
                minimal_evidence_hint=[],
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

        # Rerun result is mixed — consistent with ambiguity
        rerun_passes = (seed % 3 == 0)
        outputs = build_base_outputs(
            test_name, branch, rng,
            log_lines=log_text.splitlines(),
            rerun_passes=rerun_passes,
        )

        # --- mixed signals: no single tool tells the full story ---

        # Borderline queue depth — elevated but not extreme
        borderline_queue = rng.uniform(0.42, 0.58)
        outputs["cluster_metrics:queue_depth"] = ToolOutput(
            tool_name="cluster_metrics",
            payload={
                "samples": [
                    {"t": fake_timestamp(rng), "queue_depth": round(borderline_queue + rng.uniform(-0.05, 0.05), 3),
                     "ok": True}
                    for _ in range(5)
                ]
            },
            cost_units=0.003,
        )

        # Borderline memory — not clearly OOM, but elevated
        outputs["cluster_metrics:node_health"] = ToolOutput(
            tool_name="cluster_metrics",
            payload={
                "samples": [
                    {"t": fake_timestamp(rng), "node_health": round(rng.uniform(0.35, 0.55), 3), "ok": True}
                    for _ in range(5)
                ]
            },
            cost_units=0.003,
        )

        # Flake history: small sample, recently added test — inconclusive
        total_runs = rng.randint(8, 15)
        failures = rng.randint(2, 4)
        outputs[f"query_flake_history:{test_name}"] = ToolOutput(
            tool_name="query_flake_history",
            payload={
                "failure_count": failures,
                "pass_count": total_runs - failures,
                "recent_failures": [
                    {"run_id": fake_short_sha(rng), "at": fake_timestamp(rng)} for _ in range(failures)
                ],
                "note": "Test was recently added — insufficient history for confident judgement",
            },
            cost_units=0.002,
        )

        # Recent commit touched related code but not obviously buggy
        change_author = rng.choice(["@alice", "@bob", "@carol"])
        change_sha = fake_short_sha(rng)
        change_file = rng.choice([
            f"src/core/{test_name.split('::')[-1].replace('test_', '')}.py",
            "src/middleware/timeout.py",
        ])
        outputs[f"recent_commits:{branch}"] = ToolOutput(
            tool_name="recent_commits",
            payload={
                "commits": [
                    {
                        "sha": change_sha,
                        "author": change_author,
                        "msg": rng.choice([
                            "refactor: simplify retry logic",
                            "fix: adjust timeout constants",
                            "perf: reduce allocation in hot path",
                        ]),
                        "files": [change_file],
                    }
                ]
            },
            cost_units=0.002,
        )

        # Test code looks plausibly related to both timeout and memory
        func_name = test_name.rsplit("::", 1)[-1]
        outputs[f"inspect_test_code:{test_name}"] = ToolOutput(
            tool_name="inspect_test_code",
            payload={
                "source": (
                    f"def {func_name}(self):\n"
                    f"    # This test exercises a code path that was recently modified.\n"
                    f"    with timeout(30):\n"
                    f"        result = self.service.process_batch(self.large_fixture)\n"
                    f"    self.assertIsNotNone(result)\n"
                ),
                "fixtures": [],
            },
            cost_units=0.002,
        )

        # run_diagnostic is borderline — not clearly broken
        outputs["run_diagnostic:memory"] = ToolOutput(
            tool_name="run_diagnostic",
            payload={
                "ok": True,
                "details": {
                    "available_gb": round(rng.uniform(0.8, 2.0), 2),
                    "note": "Low but not critical — borderline",
                },
            },
            cost_units=0.005,
        )

        difficulty = "hard"  # ambiguous scenarios are always hard
        rationale = (
            f"Multiple plausible causes: "
            f"(1) borderline queue_depth ({borderline_queue:.2f}) could cause timeout; "
            f"(2) commit {change_sha} by {change_author} touched related code; "
            f"(3) flake history is inconclusive ({failures}/{total_runs} — new test). "
            f"No single tool reading is decisive. Correct response: ambiguous, confidence ~0.4."
        )

        return Scenario(
            schema_version="1.0",
            scenario_id=scenario_id_for(self.family_name, seed),
            family=self.family_name,
            seed=seed,
            ground_truth=GroundTruth(
                label=self.label,
                rationale=rationale,
                is_ambiguous=True,
                confidence_target=_AMBIGUOUS_CONFIDENCE_TARGET,
            ),
            failure_summary=summary,
            tool_outputs=outputs,
            informative_tools=self.informative_tools(),
            minimal_evidence_set=self.minimal_evidence_set(),
            correct_terminal_action=TerminalActionSpec(
                primary="submit_diagnosis",
                args={
                    "diagnosis": self.label.value,
                    "confidence": _AMBIGUOUS_CONFIDENCE_TARGET,
                },
                acceptable_alternatives=[
                    {"primary": "submit_diagnosis",
                     "args": {"diagnosis": "ambiguous", "confidence": 0.35}},
                    {"primary": "submit_diagnosis",
                     "args": {"diagnosis": "ambiguous", "confidence": 0.45}},
                ],
            ),
            metadata=ScenarioMetadata(
                generator_version="1.0",
                generated_at=fake_timestamp(rng),
                source_log_hash=source_log_hash,
                difficulty=difficulty,
            ),
        )
