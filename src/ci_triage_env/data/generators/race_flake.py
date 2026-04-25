"""RaceFlakeGenerator — scenario family: race_flake."""

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


class RaceFlakeGenerator(ArchetypedGenerator):
    family_name = "race_flake"
    label = DiagnosisLabel.RACE_FLAKE

    def informative_tools(self) -> list[str]:
        return ["read_logs", "query_flake_history", "rerun_test"]

    def minimal_evidence_set(self) -> list[str]:
        return ["query_flake_history"]

    def _default_archetypes(self) -> list[Archetype]:
        return [
            Archetype(
                archetype_id="race_flake_001",
                family="race_flake",
                pattern_summary="Go race detector: concurrent map writes",
                log_template=(
                    "fatal error: concurrent map writes\n"
                    "goroutine {NUM} [running]:\n"
                    "runtime.throw({NUM}x{NUM})\n"
                    "\t/usr/local/go/src/runtime/panic.go:{NUM}\n"
                    "goroutine {NUM} [running]:\n"
                    "main.updateMap({NUM}x{NUM})\n"
                    "\t/home/runner/work/src/main.go:{NUM}\n"
                    "FAIL\t{TEST_MODULE}\t{DURATION}s"
                ),
                slot_distributions={
                    "NUM": ["1", "2", "42", "99", "1024", "4096"],
                    "TEST_MODULE": ["github.com/org/repo/pkg/store", "github.com/org/repo/pkg/cache"],
                    "DURATION": ["0.012", "0.034", "0.089"],
                },
                informative_tools_hint=["read_logs:full", "query_flake_history"],
                minimal_evidence_hint=["query_flake_history"],
            ),
            Archetype(
                archetype_id="race_flake_002",
                family="race_flake",
                pattern_summary="Go race detector: DATA RACE on shared variable",
                log_template=(
                    "==================\n"
                    "WARNING: DATA RACE\n"
                    "Write at {NUM}x{NUM} by goroutine {NUM}:\n"
                    "  {TEST_MODULE}.(*Handler).Update()\n"
                    "      /home/runner/work/src/handler.go:{NUM} +{NUM}x{NUM}\n"
                    "Previous read at {NUM}x{NUM} by goroutine {NUM}:\n"
                    "  {TEST_MODULE}.(*Handler).Get()\n"
                    "      /home/runner/work/src/handler.go:{NUM} +{NUM}x{NUM}\n"
                    "==================\n"
                    "FAIL\t{TEST_MODULE}\t{DURATION}s"
                ),
                slot_distributions={
                    "NUM": ["1", "7", "42", "0xc0001234ab", "0xc00deadbeef"],
                    "TEST_MODULE": ["github.com/org/repo/pkg/handler", "github.com/org/repo/internal/worker"],
                    "DURATION": ["0.023", "0.061", "0.102"],
                },
                informative_tools_hint=["read_logs:full", "query_flake_history"],
                minimal_evidence_hint=["query_flake_history"],
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

        # Race flakes pass sometimes — seed-derived to keep determinism
        rerun_passes = (seed % 3 != 0)
        outputs = build_base_outputs(
            test_name, branch, rng,
            log_lines=log_text.splitlines(),
            rerun_passes=rerun_passes,
        )

        # --- informative overrides ---
        # Flake history: intermittent failures (~30% failure rate)
        total_runs = 20
        failures = max(2, total_runs // 3)
        recent_failures = [
            {"run_id": fake_short_sha(rng), "at": fake_timestamp(rng)}
            for _ in range(min(3, failures))
        ]
        outputs[f"query_flake_history:{test_name}"] = ToolOutput(
            tool_name="query_flake_history",
            payload={
                "failure_count": failures,
                "pass_count": total_runs - failures,
                "recent_failures": recent_failures,
            },
            cost_units=0.002,
        )

        # No breaking commit — no obvious change caused the race
        outputs[f"recent_commits:{branch}"] = ToolOutput(
            tool_name="recent_commits",
            payload={"commits": [
                {"sha": fake_short_sha(rng), "author": "@dave",
                 "msg": "chore: update test fixtures", "files": ["tests/fixtures/data.json"]},
            ]},
            cost_units=0.002,
        )

        # Kernel log shows race detector output
        outputs["read_logs:kernel"] = ToolOutput(
            tool_name="read_logs",
            payload={"lines": ["(no kernel messages — race from user-space goroutines)"], "truncated": False},
            cost_units=0.001,
        )

        difficulty = rng.choice(["easy", "medium", "hard"])
        rationale = (
            f"query_flake_history shows {failures}/{total_runs} runs failing — clear intermittent pattern. "
            f"The log contains DATA RACE / concurrent map writes output from the Go race detector, "
            f"confirming a goroutine-level data race. No breaking commit in recent_commits. "
            f"rerun_test {'passes' if rerun_passes else 'fails'} (seed-derived), consistent with non-deterministic flake."
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
                    "secondary_actions": [{"name": "quarantine_test", "reason": "race condition"}],
                },
                acceptable_alternatives=[
                    {"primary": "submit_diagnosis",
                     "args": {"diagnosis": "race_flake", "confidence": 0.85}},
                ],
            ),
            metadata=ScenarioMetadata(
                generator_version="1.0",
                generated_at=fake_timestamp(rng),
                source_log_hash=source_log_hash,
                difficulty=difficulty,
            ),
        )
