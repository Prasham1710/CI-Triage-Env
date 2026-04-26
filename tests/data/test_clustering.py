"""Tests for Phase B3 — Failure Clustering (classifier + archetype extractor)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ci_triage_env.data.clustering import (
    FAMILIES,
    Archetype,
    ArchetypeExtractor,
    LLMClassifier,
    RuleBasedClassifier,
    classify_all,
)
from ci_triage_env.data.datasets._base import FailureRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(log_text: str, record_id: str = "test-001") -> FailureRecord:
    return FailureRecord(
        record_id=record_id,
        source_dataset="deflaker",
        project="test/project",
        log_text=log_text,
    )


# ---------------------------------------------------------------------------
# RuleBasedClassifier
# ---------------------------------------------------------------------------

class TestRuleBasedClassifier:
    def setup_method(self) -> None:
        self.clf = RuleBasedClassifier()

    def test_rule_based_oom(self) -> None:
        record = _record("kernel: Out of memory: Killed process 123 (pytest)")
        family, conf = self.clf.classify(record)
        assert family == "infra_resource"
        assert conf > 0

    def test_rule_based_enospc(self) -> None:
        record = _record("write /var/lib/docker: no space left on device")
        family, conf = self.clf.classify(record)
        assert family == "infra_resource"
        assert conf > 0

    def test_rule_based_emfile(self) -> None:
        record = _record("Error: EMFILE: too many open files, open '/tmp/test'")
        family, conf = self.clf.classify(record)
        assert family == "infra_resource"
        assert conf > 0

    def test_rule_based_connection_refused(self) -> None:
        record = _record("dial tcp: connection refused localhost:5432")
        family, conf = self.clf.classify(record)
        assert family == "infra_network"
        assert conf > 0

    def test_rule_based_dns(self) -> None:
        record = _record("getaddrinfo failed: No such host is known")
        family, conf = self.clf.classify(record)
        assert family == "infra_network"
        assert conf > 0

    def test_rule_based_race(self) -> None:
        record = _record("WARNING: DATA RACE detected in goroutine 42")
        family, conf = self.clf.classify(record)
        assert family == "race_flake"
        assert conf > 0

    def test_rule_based_concurrent_map(self) -> None:
        record = _record("fatal error: concurrent map writes")
        family, conf = self.clf.classify(record)
        assert family == "race_flake"
        assert conf > 0

    def test_rule_based_timeout(self) -> None:
        record = _record("context canceled: deadline exceeded after 30s")
        family, conf = self.clf.classify(record)
        assert family == "timing_flake"
        assert conf > 0

    def test_rule_based_test_timed_out(self) -> None:
        record = _record("Test timed out after 120 seconds")
        family, conf = self.clf.classify(record)
        assert family == "timing_flake"
        assert conf > 0

    def test_rule_based_dependency_version_conflict(self) -> None:
        record = _record("npm ERR! peer dep missing: react@^17.0.0, version conflict")
        family, conf = self.clf.classify(record)
        assert family == "dependency_drift"
        assert conf > 0

    def test_rule_based_cargo_lock(self) -> None:
        record = _record("error: failed to select a version for the requirement\nCargo.lock conflict")
        family, conf = self.clf.classify(record)
        assert family == "dependency_drift"
        assert conf > 0

    def test_rule_based_unknown_returns_unknown(self) -> None:
        record = _record("generic test failure with no specific signal")
        family, conf = self.clf.classify(record)
        assert family == "unknown"
        assert conf == 0.0

    def test_rule_based_empty_log(self) -> None:
        record = _record("")
        family, conf = self.clf.classify(record)
        assert family == "unknown"
        assert conf == 0.0

    def test_ambiguous_when_multiple_families_match(self) -> None:
        # Both OOM (infra_resource) and DATA RACE (race_flake) in same log
        record = _record(
            "Out of memory: Killed process 999\nWARNING: DATA RACE in goroutine 1"
        )
        family, _conf = self.clf.classify(record)
        assert family == "ambiguous"

    def test_confidence_in_unit_range(self) -> None:
        record = _record("kernel: Out of memory: Killed process 1 (pytest)")
        _family, conf = self.clf.classify(record)
        assert 0.0 <= conf <= 1.0

    def test_real_bug_assertion_error(self) -> None:
        record = _record("AssertionError: expected 42, got 0\nassert result == expected")
        family, conf = self.clf.classify(record)
        assert family == "real_bug"
        assert conf > 0


# ---------------------------------------------------------------------------
# Archetype round-trip and extraction
# ---------------------------------------------------------------------------

class TestArchetype:
    def test_archetype_round_trip(self) -> None:
        arch = Archetype(
            archetype_id="infra_resource_001",
            family="infra_resource",
            pattern_summary="OOM-killer terminated test process",
            log_template="[{TIMESTAMP}] Out of memory: Killed process {PID}",
            slot_distributions={"TIMESTAMP": ["2024-01-01T00:00:00"], "PID": ["1234"]},
            informative_tools_hint=["read_logs:kernel"],
            minimal_evidence_hint=["cluster_metrics:5m"],
        )
        data = arch.model_dump()
        restored = Archetype.model_validate(data)
        assert restored.archetype_id == arch.archetype_id
        assert restored.family == arch.family
        assert restored.log_template == arch.log_template

    def test_archetype_json_round_trip(self) -> None:
        arch = Archetype(
            archetype_id="race_flake_001",
            family="race_flake",
            pattern_summary="Concurrent map write panic",
            log_template="fatal error: concurrent map writes at {TIMESTAMP}",
            slot_distributions={"TIMESTAMP": ["2024-06-01T12:00:00"]},
            informative_tools_hint=["read_logs:full"],
            minimal_evidence_hint=["read_logs:full"],
        )
        restored = Archetype.model_validate_json(arch.model_dump_json())
        assert restored.family == "race_flake"


class TestArchetypeExtractor:
    def setup_method(self) -> None:
        self.extractor = ArchetypeExtractor()

    def _make_records(self, log_texts: list[str]) -> list[FailureRecord]:
        return [
            _record(text, record_id=f"rec-{i:03d}")
            for i, text in enumerate(log_texts)
        ]

    def test_extract_returns_archetypes(self) -> None:
        records = self._make_records([
            "kernel: Out of memory: Killed process 123 (pytest) 2024-01-01T10:00:00",
            "kernel: Out of memory: Killed process 456 (cargo) 2024-01-02T11:00:00",
            "Out of memory: cannot allocate 4096kB",
            "OOMKilled: container exceeded memory limit 2048MB",
            "killed by OS: out of memory at 2024-01-03T09:00:00",
        ])
        archetypes = self.extractor.extract(records, "infra_resource", n_archetypes=4)
        assert len(archetypes) >= 1
        for arch in archetypes:
            assert arch.family == "infra_resource"
            assert arch.archetype_id.startswith("infra_resource_")
            assert arch.log_template
            assert arch.informative_tools_hint
            assert arch.minimal_evidence_hint

    def test_extract_with_single_record(self) -> None:
        records = self._make_records(["context canceled: deadline exceeded"])
        archetypes = self.extractor.extract(records, "timing_flake", n_archetypes=4)
        assert len(archetypes) == 1
        assert archetypes[0].family == "timing_flake"

    def test_extract_empty_records_returns_empty(self) -> None:
        archetypes = self.extractor.extract([], "real_bug", n_archetypes=4)
        assert archetypes == []

    def test_extract_slots_present_in_template(self) -> None:
        records = self._make_records([
            "2024-03-15T08:30:00 OOMKilled process 99999 after 120.5s",
        ])
        archetypes = self.extractor.extract(records, "infra_resource", n_archetypes=1)
        assert len(archetypes) == 1
        template = archetypes[0].log_template
        # At least one slot should have been extracted
        assert "{" in template

    def test_slot_distributions_are_populated(self) -> None:
        records = self._make_records([
            "2024-01-01T10:00:00 Out of memory: Killed process 1234",
            "2024-02-01T11:00:00 Out of memory: Killed process 5678",
        ])
        archetypes = self.extractor.extract(records, "infra_resource", n_archetypes=1)
        assert len(archetypes) >= 1
        # At least one slot family should have distribution values
        all_vals = []
        for arch in archetypes:
            for vals in arch.slot_distributions.values():
                all_vals.extend(vals)
        assert len(all_vals) > 0

    def test_archetype_extraction_from_fixture(self) -> None:
        """Given 5 fixture FailureRecords, extract at least 1 archetype with slots."""
        records = self._make_records([
            "fatal error: concurrent map writes at goroutine 12",
            "WARNING: DATA RACE in goroutine 42 on address 0xc000deadbeef",
            "data race detected: read at 0x00c0001234ab by goroutine 7",
            "deadlock detected: all goroutines are asleep",
            "fatal error: concurrent map iteration and map write",
        ])
        archetypes = self.extractor.extract(records, "race_flake", n_archetypes=4)
        assert len(archetypes) >= 1
        # Check that at least one archetype has slot placeholders
        has_slots = any("{" in a.log_template for a in archetypes)
        assert has_slots

    def test_n_archetypes_cap(self) -> None:
        """Never return more archetypes than records."""
        records = self._make_records(["timeout exceeded", "test timed out after 60s"])
        archetypes = self.extractor.extract(records, "timing_flake", n_archetypes=10)
        assert len(archetypes) <= len(records)


# ---------------------------------------------------------------------------
# classify_all
# ---------------------------------------------------------------------------

class TestClassifyAll:
    def test_classify_all_no_records(self) -> None:
        by_family = classify_all([])
        assert set(by_family.keys()) == set(FAMILIES)
        assert all(len(v) == 0 for v in by_family.values())

    def test_classify_all_routes_correctly(self) -> None:
        records = [
            _record("Out of memory: Killed process 1", "oom-1"),
            _record("WARNING: DATA RACE in goroutine 99", "race-1"),
            _record("context canceled: deadline exceeded", "timeout-1"),
        ]
        by_family = classify_all(records)
        assert len(by_family["infra_resource"]) >= 1
        assert len(by_family["race_flake"]) >= 1
        assert len(by_family["timing_flake"]) >= 1

    def test_classify_all_unknown_falls_into_real_bug(self) -> None:
        records = [_record("some completely generic text with no signal", "unknown-1")]
        by_family = classify_all(records, openai_api_key=None)
        # unknowns with no LLM go to real_bug
        assert len(by_family["real_bug"]) == 1

    def test_classify_all_returns_all_families(self) -> None:
        by_family = classify_all([_record("OOM error")])
        assert set(by_family.keys()) == set(FAMILIES)

    def test_classify_all_writes_per_family_files(self, tmp_path: Path) -> None:
        """classify_all + extract puts archetype files under each family dir."""
        records = [
            _record("Out of memory: Killed process 1", "oom-1"),
            _record("connection refused to db:5432", "net-1"),
            _record("deadline exceeded in test", "timeout-1"),
        ]
        by_family = classify_all(records)
        extractor = ArchetypeExtractor()
        for family, recs in by_family.items():
            if not recs:
                continue
            archetypes = extractor.extract(recs, family, n_archetypes=2)
            family_dir = tmp_path / family
            family_dir.mkdir(parents=True, exist_ok=True)
            (family_dir / "archetypes.json").write_text(
                json.dumps([a.model_dump() for a in archetypes], indent=2)
            )

        written_families = [d.name for d in tmp_path.iterdir() if d.is_dir()]
        assert len(written_families) >= 1
        for d in tmp_path.iterdir():
            arch_file = d / "archetypes.json"
            assert arch_file.exists()
            data = json.loads(arch_file.read_text())
            assert isinstance(data, list)
            assert len(data) >= 1


# ---------------------------------------------------------------------------
# LLMClassifier (mocked)
# ---------------------------------------------------------------------------

class TestLLMClassifier:
    def _make_mock_client(self, label: str = "infra_resource") -> MagicMock:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = label
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 5
        return mock_response

    def test_llm_classifier_respects_budget(self) -> None:
        """LLMClassifier stops calling after budget exhausted."""
        records = [
            _record(f"generic unknown failure {i}", f"rec-{i}")
            for i in range(5)
        ]

        mock_response = self._make_mock_client("real_bug")

        with patch("ci_triage_env.data.clustering.classifier.LLMClassifier.__init__", return_value=None):
            clf = LLMClassifier.__new__(LLMClassifier)
            clf.budget = 0.0  # already over budget
            clf.spent = 0.0
            clf.model = "gpt-4o-mini"

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            clf.client = mock_client

            results = clf.classify_batch(records)

        # All results should be ("unknown", 0.0) since budget is exhausted
        assert all(family == "unknown" for family, _ in results)
        # Client should NOT have been called
        mock_client.chat.completions.create.assert_not_called()

    def test_llm_classifier_classifies_records(self) -> None:
        """LLMClassifier calls the API and returns valid labels."""
        records = [_record("some unknown log text", "unk-1")]

        mock_response = self._make_mock_client("real_bug")

        with patch("ci_triage_env.data.clustering.classifier.LLMClassifier.__init__", return_value=None):
            clf = LLMClassifier.__new__(LLMClassifier)
            clf.budget = 5.0
            clf.spent = 0.0
            clf.model = "gpt-4o-mini"

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            clf.client = mock_client

            results = clf.classify_batch(records)

        assert len(results) == 1
        family, conf = results[0]
        assert family == "real_bug"
        assert conf == 0.7

    def test_llm_classifier_invalid_label_falls_back_to_unknown(self) -> None:
        records = [_record("something", "x-1")]
        mock_response = self._make_mock_client("not_a_valid_label")

        with patch("ci_triage_env.data.clustering.classifier.LLMClassifier.__init__", return_value=None):
            clf = LLMClassifier.__new__(LLMClassifier)
            clf.budget = 5.0
            clf.spent = 0.0
            clf.model = "gpt-4o-mini"

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            clf.client = mock_client

            results = clf.classify_batch(records)

        family, _ = results[0]
        assert family == "unknown"

    def test_classify_all_uses_llm_for_unknowns(self) -> None:
        """classify_all routes unknown records through LLM when api_key provided."""
        records = [_record("no signal here at all", "unk-1")]

        with (
            patch("ci_triage_env.data.clustering.classifier.LLMClassifier") as MockLLMClass,
        ):
            mock_llm = MagicMock()
            mock_llm.classify_batch.return_value = [("timing_flake", 0.7)]
            mock_llm.spent = 0.0001
            MockLLMClass.return_value = mock_llm

            by_family = classify_all(records, openai_api_key="sk-fake-key")

        assert len(by_family["timing_flake"]) == 1
        MockLLMClass.assert_called_once_with("sk-fake-key")
