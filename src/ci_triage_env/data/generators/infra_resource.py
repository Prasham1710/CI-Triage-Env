"""InfraResourceGenerator — scenario family: infra_resource."""

from __future__ import annotations

import random

from ci_triage_env.data.clustering.archetypes import Archetype
from ci_triage_env.data.generators._helpers import (
    ArchetypedGenerator,
    _metric_samples,
    build_base_outputs,
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


class InfraResourceGenerator(ArchetypedGenerator):
    family_name = "infra_resource"
    label = DiagnosisLabel.INFRA_RESOURCE

    def informative_tools(self) -> list[str]:
        return ["read_logs", "cluster_metrics", "run_diagnostic"]

    def minimal_evidence_set(self) -> list[str]:
        return ["cluster_metrics"]

    def _default_archetypes(self) -> list[Archetype]:
        return [
            Archetype(
                archetype_id="infra_resource_001",
                family="infra_resource",
                pattern_summary="OOM-killer terminated test process",
                log_template=(
                    "[{TIMESTAMP}] kernel: Out of memory: Killed process {NUM} ({PROCESS}) "
                    "total-vm:{NUM}kB, anon-rss:{NUM}kB\n"
                    "[{TIMESTAMP}] systemd[1]: {SERVICE}.service: Main process exited, "
                    "code=killed, status=9/KILL\n"
                    "FAILED {TEST_MODULE}::{TEST_FUNC} ({DURATION}s)\n"
                    "signal: killed"
                ),
                slot_distributions={
                    "TIMESTAMP": ["2024-03-15T08:30:00", "2024-06-01T14:22:00"],
                    "NUM": ["1234", "5678", "9001", "2048576", "4194304"],
                    "PROCESS": ["pytest", "go test", "cargo test", "npm test"],
                    "SERVICE": ["test-runner", "ci-job", "build-agent"],
                    "TEST_MODULE": ["tests/unit/test_model", "tests/integration/test_batch"],
                    "TEST_FUNC": ["test_large_dataset", "test_matrix_multiply"],
                    "DURATION": ["120.5", "60.0", "300.1"],
                },
                informative_tools_hint=["read_logs:kernel", "cluster_metrics:queue_depth", "run_diagnostic:memory"],
                minimal_evidence_hint=["cluster_metrics:queue_depth"],
            ),
            Archetype(
                archetype_id="infra_resource_002",
                family="infra_resource",
                pattern_summary="ENOSPC: no space left on device",
                log_template=(
                    "write /var/lib/docker/overlay2/{NUM}/work/dir: "
                    "no space left on device\n"
                    "OSError: [Errno {NUM}] No space left on device: '/tmp/pytest-{NUM}'\n"
                    "FAILED {TEST_MODULE}::{TEST_FUNC}"
                ),
                slot_distributions={
                    "NUM": ["28", "1234abcd", "5678ef01", "42"],
                    "TEST_MODULE": ["tests/unit/test_io", "tests/integration/test_storage"],
                    "TEST_FUNC": ["test_write_large_file", "test_export_dataset"],
                },
                informative_tools_hint=["read_logs:kernel", "cluster_metrics:disk_io", "run_diagnostic:disk"],
                minimal_evidence_hint=["run_diagnostic:disk"],
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

        outputs = build_base_outputs(
            test_name, branch, rng,
            log_lines=log_text.splitlines(),
            rerun_passes=False,  # resource pressure is persistent
        )

        # --- informative overrides ---
        # Pick resource type (OOM or disk)
        resource_type = "memory" if (seed % 2 == 0) else "disk"

        if resource_type == "memory":
            outputs["cluster_metrics:queue_depth"] = ToolOutput(
                tool_name="cluster_metrics",
                payload={"samples": _metric_samples(rng, "queue_depth", elevated=True, n=6)},
                cost_units=0.003,
            )
            outputs["run_diagnostic:memory"] = ToolOutput(
                tool_name="run_diagnostic",
                payload={
                    "ok": False,
                    "details": {
                        "available_gb": round(rng.uniform(0.0, 0.3), 2),
                        "total_gb": rng.choice([8, 16, 32]),
                        "oom_events": rng.randint(1, 5),
                    },
                },
                cost_units=0.005,
            )
            kernel_lines = [
                f"[{fake_timestamp(rng)}] kernel: Out of memory: Killed process {rng.randint(100, 9999)} (pytest)",
                f"[{fake_timestamp(rng)}] kernel: Memory cgroup out of memory: Kill process {rng.randint(100, 9999)}",
            ]
            outputs["read_logs:kernel"] = ToolOutput(
                tool_name="read_logs",
                payload={"lines": kernel_lines, "truncated": False},
                cost_units=0.001,
            )
        else:
            outputs["cluster_metrics:disk_io"] = ToolOutput(
                tool_name="cluster_metrics",
                payload={"samples": _metric_samples(rng, "disk_io", elevated=True, n=6)},
                cost_units=0.003,
            )
            outputs["run_diagnostic:disk"] = ToolOutput(
                tool_name="run_diagnostic",
                payload={
                    "ok": False,
                    "details": {
                        "free_gb": round(rng.uniform(0.0, 0.5), 2),
                        "total_gb": rng.choice([50, 100, 200]),
                        "usage_pct": round(rng.uniform(95, 100), 1),
                    },
                },
                cost_units=0.005,
            )

        outputs[f"query_flake_history:{test_name}"] = ToolOutput(
            tool_name="query_flake_history",
            payload={"failure_count": 0, "pass_count": 40, "recent_failures": []},
            cost_units=0.002,
        )

        difficulty = rng.choice(["easy", "medium", "hard"])
        rationale = (
            f"Logs show {'OOM-kill' if resource_type == 'memory' else 'ENOSPC'} errors. "
            f"run_diagnostic:{resource_type} confirms resource exhaustion on the CI node. "
            f"cluster_metrics shows elevated pressure. "
            f"test was 100% passing before — not a code bug."
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
                    "confidence": 0.95,
                    "secondary_actions": [
                        {"name": "ping_owner", "owner": "@infra-team",
                         "message": f"CI node {resource_type} exhaustion"},
                    ],
                },
                acceptable_alternatives=[
                    {"primary": "submit_diagnosis",
                     "args": {"diagnosis": "infra_resource", "confidence": 0.9}},
                ],
            ),
            metadata=ScenarioMetadata(
                generator_version="1.0",
                generated_at=fake_timestamp(rng),
                source_log_hash=source_log_hash,
                difficulty=difficulty,
            ),
        )
