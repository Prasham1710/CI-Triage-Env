"""RealBugGenerator — scenario family: real_bug."""

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

_DEFAULT_LOG_TEMPLATE = (
    "FAILED {TEST_MODULE}::{TEST_FUNC} - AssertionError\n"
    "  assert result == expected\n"
    "  where result   = {ACTUAL}\n"
    "  and   expected = {EXPECTED}\n"
    "E AssertionError: assertion failed at line {LINENO}\n"
    "short test summary info\n"
    "FAILED {TEST_MODULE}::{TEST_FUNC}"
)

_DEFAULT_BUGGY_CODE = (
    "def {TEST_FUNC}(self):\n"
    "    result = self.service.compute({INPUT})\n"
    "    assert result == {EXPECTED}  # broke after {COMMIT_MSG}\n"
)


class RealBugGenerator(ArchetypedGenerator):
    family_name = "real_bug"
    label = DiagnosisLabel.REAL_BUG

    def informative_tools(self) -> list[str]:
        return ["read_logs", "inspect_test_code", "recent_commits", "rerun_test"]

    def minimal_evidence_set(self) -> list[str]:
        return ["recent_commits", "inspect_test_code"]

    def _default_archetypes(self) -> list[Archetype]:
        return [
            Archetype(
                archetype_id="real_bug_001",
                family="real_bug",
                pattern_summary="AssertionError after recent commit changed return value",
                log_template=_DEFAULT_LOG_TEMPLATE,
                slot_distributions={
                    "TEST_MODULE": ["tests/unit/test_core", "tests/unit/test_api"],
                    "TEST_FUNC": ["test_compute", "test_process", "test_validate"],
                    "ACTUAL": ["None", "0", "-1", "[]"],
                    "EXPECTED": ["42", "True", "{'ok': True}"],
                    "LINENO": ["42", "87", "115", "203"],
                },
                informative_tools_hint=["read_logs:full", "inspect_test_code", "recent_commits"],
                minimal_evidence_hint=["recent_commits", "inspect_test_code"],
            ),
            Archetype(
                archetype_id="real_bug_002",
                family="real_bug",
                pattern_summary="AttributeError / NullPointerException in core logic",
                log_template=(
                    "AttributeError: 'NoneType' object has no attribute '{ATTR}'\n"
                    "  File \"{TEST_MODULE}.py\", line {LINENO}, in {TEST_FUNC}\n"
                    "    return obj.{ATTR}\n"
                    "FAILED {TEST_MODULE}::{TEST_FUNC}"
                ),
                slot_distributions={
                    "ATTR": ["name", "id", "value", "data", "result"],
                    "TEST_MODULE": ["tests/unit/test_models", "tests/unit/test_service"],
                    "TEST_FUNC": ["test_create", "test_update", "test_fetch"],
                    "LINENO": ["33", "67", "91", "144"],
                },
                informative_tools_hint=["read_logs:full", "inspect_test_code", "recent_commits"],
                minimal_evidence_hint=["inspect_test_code"],
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
            rerun_passes=False,
        )

        # --- informative overrides ---
        breaking_author = rng.choice(["@alice", "@bob", "@carol"])
        breaking_sha = fake_short_sha(rng)
        breaking_commit = {
            "sha": breaking_sha,
            "author": breaking_author,
            "msg": rng.choice([
                f"fix: update {test_name.split('::')[-1].replace('test_', '')} logic",
                "refactor: change return contract of compute()",
                f"feat: extend {test_name.split('::')[-1].split('_')[1]} API",
            ]),
            "files": [
                f"src/{test_name.split('/')[1].replace('test_', '')}.py",
                test_name.rsplit("::", 1)[0],
            ],
        }
        outputs[f"recent_commits:{branch}"] = ToolOutput(
            tool_name="recent_commits",
            payload={"commits": [breaking_commit, {
                "sha": fake_short_sha(rng),
                "author": rng.choice(["@dave", "@eve"]),
                "msg": "chore: update lockfile",
                "files": ["pyproject.toml"],
            }]},
            cost_units=0.002,
        )

        buggy_code = self._pick_buggy_code(rng)
        outputs[f"inspect_test_code:{test_name}"] = ToolOutput(
            tool_name="inspect_test_code",
            payload={"source": buggy_code, "fixtures": []},
            cost_units=0.002,
        )

        # Rerun also fails — it's a real bug, not a flake
        outputs["rerun_test"] = ToolOutput(
            tool_name="rerun_test",
            payload={"results": [{"passed": False, "duration_s": round(rng.uniform(5, 30), 2),
                                  "log_excerpt": log_text.splitlines()[:3]}]},
            cost_units=0.01,
        )

        # Flake history is clean (test was stable before the bad commit)
        outputs[f"query_flake_history:{test_name}"] = ToolOutput(
            tool_name="query_flake_history",
            payload={"failure_count": 0, "pass_count": 50, "recent_failures": []},
            cost_units=0.002,
        )

        difficulty = rng.choice(["easy", "medium", "hard"])
        rationale = (
            f"The commit {breaking_sha} by {breaking_author} changed the return contract "
            f"of the production code exercised by {test_name}. "
            f"inspect_test_code shows the assertion that now fails; "
            f"recent_commits:{branch} shows the introducing commit. "
            f"query_flake_history shows no prior failures — not a flake. "
            f"rerun_test fails again — confirms deterministic breakage."
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
                    "confidence": 1.0,
                    "secondary_actions": [{"name": "file_bug", "owner": breaking_author}],
                },
                acceptable_alternatives=[],
            ),
            metadata=ScenarioMetadata(
                generator_version="1.0",
                generated_at=fake_timestamp(rng),
                source_log_hash=source_log_hash,
                difficulty=difficulty,
            ),
        )
