"""DependencyDriftGenerator — scenario family: dependency_drift."""

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


class DependencyDriftGenerator(ArchetypedGenerator):
    family_name = "dependency_drift"
    label = DiagnosisLabel.DEPENDENCY_DRIFT

    def informative_tools(self) -> list[str]:
        return ["read_logs", "recent_commits", "inspect_test_code"]

    def minimal_evidence_set(self) -> list[str]:
        return ["recent_commits"]

    def _default_archetypes(self) -> list[Archetype]:
        return [
            Archetype(
                archetype_id="dependency_drift_001",
                family="dependency_drift",
                pattern_summary="npm/pip version conflict after lockfile update",
                log_template=(
                    "npm ERR! peer dep missing: {PACKAGE}@^{VERSION}, required by {REQUIRER}\n"
                    "npm ERR! version conflict: {PACKAGE}@{VERSION} vs {PACKAGE}@{NEW_VERSION}\n"
                    "FAILED {TEST_MODULE}::{TEST_FUNC}\n"
                    "  ImportError: cannot import name '{SYMBOL}' from '{PACKAGE}'"
                ),
                slot_distributions={
                    "PACKAGE": ["react", "lodash", "requests", "pydantic", "fastapi"],
                    "VERSION": ["17.0.0", "4.17.1", "2.28.0", "1.10.0"],
                    "NEW_VERSION": ["18.0.0", "4.18.0", "2.31.0", "2.0.0"],
                    "REQUIRER": ["react-dom", "react-router", "starlette"],
                    "SYMBOL": ["BaseModel", "Field", "validator"],
                    "TEST_MODULE": ["tests/unit/test_models", "tests/integration/test_schema"],
                    "TEST_FUNC": ["test_schema_validation", "test_import_contract"],
                },
                informative_tools_hint=["read_logs:build", "recent_commits:main"],
                minimal_evidence_hint=["recent_commits:main"],
            ),
            Archetype(
                archetype_id="dependency_drift_002",
                family="dependency_drift",
                pattern_summary="Cargo.lock / go.sum conflict after dependency bump",
                log_template=(
                    "error[E{NUM}]: use of unstable library feature '{FEATURE}'\n"
                    "  --> src/{MODULE}.rs:{NUM}:{NUM}\n"
                    "   = note: the library is in the Cargo registry but requires nightly\n"
                    "error: aborting due to {NUM} previous errors\n"
                    "Cargo.lock conflict: expected version {VERSION}, found {NEW_VERSION}"
                ),
                slot_distributions={
                    "NUM": ["123", "456", "789", "1", "2", "3"],
                    "FEATURE": ["proc_macro", "async_fn_traits", "adt_const_params"],
                    "MODULE": ["lib", "main", "service", "handler"],
                    "VERSION": ["1.2.3", "0.9.1", "2.0.0-beta.1"],
                    "NEW_VERSION": ["1.3.0", "0.9.2", "2.0.0"],
                },
                informative_tools_hint=["read_logs:build", "recent_commits:main"],
                minimal_evidence_hint=["recent_commits:main"],
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
            rerun_passes=False,  # version conflict is deterministic
        )

        # --- informative overrides ---
        dep_name = rng.choice(["pydantic", "requests", "react", "lodash", "serde"])
        old_ver = rng.choice(["1.10.0", "2.28.0", "17.0.0", "4.17.1"])
        new_ver = rng.choice(["2.0.0", "2.31.0", "18.0.0", "4.18.0"])
        bump_author = rng.choice(["@dependabot", "@alice", "@dep-bot"])
        bump_sha = fake_short_sha(rng)

        outputs[f"recent_commits:{branch}"] = ToolOutput(
            tool_name="recent_commits",
            payload={
                "commits": [
                    {
                        "sha": bump_sha,
                        "author": bump_author,
                        "msg": f"chore: bump {dep_name} from {old_ver} to {new_ver}",
                        "files": ["package-lock.json", "pyproject.toml", "Cargo.lock"],
                    },
                    {
                        "sha": fake_short_sha(rng),
                        "author": "@carol",
                        "msg": "docs: add changelog entry",
                        "files": ["CHANGELOG.md"],
                    },
                ]
            },
            cost_units=0.002,
        )
        outputs["recent_commits:main"] = ToolOutput(
            tool_name="recent_commits",
            payload={"commits": [
                {"sha": bump_sha, "author": bump_author,
                 "msg": f"chore: bump {dep_name} {old_ver} → {new_ver}",
                 "files": ["package-lock.json"]},
            ]},
            cost_units=0.002,
        )

        # Test code looks fine — the dep changed under it
        outputs[f"inspect_test_code:{test_name}"] = ToolOutput(
            tool_name="inspect_test_code",
            payload={
                "source": (
                    f"from {dep_name} import BaseModel, Field\n\n"
                    f"def {test_name.rsplit('::', 1)[-1]}():\n"
                    f"    model = MyModel(name='test')\n"
                    f"    assert model.dict()  # .dict() removed in {dep_name} v{new_ver}\n"
                ),
                "fixtures": [],
            },
            cost_units=0.002,
        )

        # build log shows the import/version conflict
        outputs["read_logs:build"] = ToolOutput(
            tool_name="read_logs",
            payload={
                "lines": [
                    f"ImportError: cannot import name 'validator' from '{dep_name}'",
                    f"  -- NOTE: {dep_name} was upgraded from {old_ver} to {new_ver}",
                    f"  -- Breaking change: 'validator' removed in {new_ver}",
                ],
                "truncated": False,
            },
            cost_units=0.001,
        )

        # Test was passing before the bump
        outputs[f"query_flake_history:{test_name}"] = ToolOutput(
            tool_name="query_flake_history",
            payload={
                "failure_count": 0,
                "pass_count": 30,
                "recent_failures": [],
                "note": f"All passes before {bump_sha}; first failure after dependency bump",
            },
            cost_units=0.002,
        )

        difficulty = rng.choice(["easy", "medium", "hard"])
        rationale = (
            f"recent_commits:{branch} shows commit {bump_sha} by {bump_author} bumped "
            f"{dep_name} from {old_ver} to {new_ver}. "
            f"inspect_test_code shows the test uses an API removed in the new version. "
            f"read_logs:build shows an ImportError for the changed symbol. "
            f"query_flake_history shows 100% pass rate before the bump — dependency drift."
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
                    "secondary_actions": [
                        {"name": "file_bug", "owner": "@dep-team",
                         "title": f"Breaking change from {dep_name} {old_ver}→{new_ver}"},
                    ],
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
