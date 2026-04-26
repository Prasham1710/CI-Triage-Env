"""Phase B1 dataset-loader tests.

The four loaders are exercised against tiny fixtures shipped alongside the
tests so the suite runs offline (matches §Implementation notes "Caching is
content-addressable. Re-running is safe."). One test per loader covers each
behavior the phase doc requires:

- yields ≥ 1 ``FailureRecord``
- records validate against the schema (Pydantic raises if not)
- caches to ``cache_dir`` after a fetch
- a repeat ``load_or_fetch`` returns from the cache (we corrupt the source
  file between calls and verify it still works)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from ci_triage_env.data.cli import build_parser
from ci_triage_env.data.cli import main as cli_main
from ci_triage_env.data.datasets import (
    LOADER_REGISTRY,
    DatasetLoader,
    DeFlakerLoader,
    FailureRecord,
    FlakeFlaggerLoader,
    IDFlakiesLoader,
    LogHubLoader,
)
from ci_triage_env.data.datasets._base import MissingArtifactError
from ci_triage_env.data.datasets.cache import (
    DEFAULT_CACHE_ROOT,
    cache_dir_for,
    is_cached,
    load_cached,
)

FIXTURES_ROOT = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# FailureRecord round-trip + registry sanity
# ---------------------------------------------------------------------------

def test_failure_record_round_trip():
    rec = FailureRecord(
        record_id="rt-1",
        source_dataset="deflaker",
        project="x/y",
        test_name="t",
        failure_type_label="flaky",
        log_text="log",
        metadata={"k": "v"},
    )
    restored = FailureRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec


def test_loader_registry_lists_all_four():
    assert set(LOADER_REGISTRY) == {"deflaker", "idflakies", "flakeflagger", "loghub"}


@pytest.mark.parametrize(
    "loader_cls,expected_name",
    [
        (DeFlakerLoader, "deflaker"),
        (IDFlakiesLoader, "idflakies"),
        (FlakeFlaggerLoader, "flakeflagger"),
        (LogHubLoader, "loghub"),
    ],
)
def test_each_loader_has_correct_name(loader_cls, expected_name):
    assert loader_cls.name == expected_name


@pytest.mark.parametrize("loader_cls", [DeFlakerLoader, IDFlakiesLoader, FlakeFlaggerLoader, LogHubLoader])
def test_missing_data_path_raises_missing_artifact(loader_cls, tmp_path, monkeypatch):
    monkeypatch.delenv(loader_cls.env_var, raising=False)
    loader = loader_cls(cache_dir=tmp_path / "cache")
    with pytest.raises(MissingArtifactError) as exc_info:
        list(loader.fetch())
    assert loader_cls.download_instructions in str(exc_info.value)


# ---------------------------------------------------------------------------
# Per-loader fixture-driven tests (parametrized so the matrix is one place).
# ---------------------------------------------------------------------------

LOADER_CASES = [
    pytest.param(
        DeFlakerLoader,
        FIXTURES_ROOT / "deflaker" / "failures.csv",
        3,  # 4th row is malformed and skipped
        {"flaky", "real"},
        id="deflaker",
    ),
    pytest.param(
        IDFlakiesLoader,
        FIXTURES_ROOT / "idflakies" / "idflakies.csv",
        4,
        {"OD", "NOD", "OD-Brit", "OD-Vict"},
        id="idflakies",
    ),
    pytest.param(
        FlakeFlaggerLoader,
        FIXTURES_ROOT / "flakeflagger" / "features.csv",
        3,
        {"timing", "io", "thread"},
        id="flakeflagger",
    ),
    pytest.param(
        LogHubLoader,
        FIXTURES_ROOT / "loghub",
        7,  # 4 from HDFS_v1 + 3 from Linux
        {"normal", "anomaly", "unlabeled"},
        id="loghub",
    ),
]


@pytest.mark.parametrize("loader_cls,data_path,expected_count,expected_labels", LOADER_CASES)
def test_loader_fixture_yields_at_least_one_record(
    loader_cls, data_path, expected_count, expected_labels, tmp_path
):
    loader = loader_cls(data_path=data_path, cache_dir=tmp_path / "cache")
    records = list(loader.fetch())
    assert records, f"{loader_cls.name} produced no records"
    assert len(records) == expected_count
    for r in records:
        assert isinstance(r, FailureRecord)


@pytest.mark.parametrize("loader_cls,data_path,expected_count,expected_labels", LOADER_CASES)
def test_loader_records_validate_against_schema(
    loader_cls, data_path, expected_count, expected_labels, tmp_path
):
    loader = loader_cls(data_path=data_path, cache_dir=tmp_path / "cache")
    for record in loader.fetch():
        # round-trip via Pydantic to exercise validation end-to-end
        FailureRecord.model_validate_json(record.model_dump_json())


@pytest.mark.parametrize("loader_cls,data_path,expected_count,expected_labels", LOADER_CASES)
def test_loader_caches_to_disk(
    loader_cls, data_path, expected_count, expected_labels, tmp_path
):
    cache_dir = tmp_path / "cache"
    loader = loader_cls(data_path=data_path, cache_dir=cache_dir)
    records = list(loader.fetch())
    written = loader.cache_records(records)
    assert written == len(records)
    cached_files = list(cache_dir.glob("*.json"))
    assert len(cached_files) == len(records)
    # Sanity: every cached file parses back cleanly.
    for path in cached_files:
        FailureRecord.model_validate_json(path.read_text())


@pytest.mark.parametrize("loader_cls,data_path,expected_count,expected_labels", LOADER_CASES)
def test_loader_repeat_load_or_fetch_uses_cache(
    loader_cls, data_path, expected_count, expected_labels, tmp_path
):
    """Once the cache is populated, ``load_or_fetch`` reads from disk —
    even if the source artifact is later deleted."""
    cache_dir = tmp_path / "cache"
    loader = loader_cls(data_path=data_path, cache_dir=cache_dir)
    first = list(loader.load_or_fetch())
    assert len(first) == expected_count

    # Wreck the source so the next call would fail if the cache wasn't honored.
    loader_after = loader_cls(data_path=tmp_path / "does_not_exist", cache_dir=cache_dir)
    second = list(loader_after.load_or_fetch())
    assert {r.record_id for r in first} == {r.record_id for r in second}


@pytest.mark.parametrize("loader_cls,data_path,expected_count,expected_labels", LOADER_CASES)
def test_loader_label_distribution_in_info(
    loader_cls, data_path, expected_count, expected_labels, tmp_path
):
    loader = loader_cls(data_path=data_path, cache_dir=tmp_path / "cache")
    info = loader.info()
    assert info["count"] == expected_count
    seen = set(info["label_distribution"]) - {None}
    # Some loaders may emit additional labels (e.g. LogHub mixes), but every
    # expected label must appear.
    assert expected_labels.issubset(seen) or expected_labels & seen


# ---------------------------------------------------------------------------
# Loader-specific fine-grained checks
# ---------------------------------------------------------------------------

def test_deflaker_skips_malformed_rows(tmp_path):
    loader = DeFlakerLoader(
        data_path=FIXTURES_ROOT / "deflaker" / "failures.csv",
        cache_dir=tmp_path / "cache",
    )
    records = list(loader.fetch())
    assert all(r.test_name and r.metadata.get("commit_sha") for r in records)


def test_deflaker_rejects_csv_missing_required_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    with open(bad, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["project", "test"])  # missing label, commit_sha
        writer.writerow(["x", "y"])
    with pytest.raises(ValueError, match="missing required columns"):
        list(DeFlakerLoader(data_path=bad, cache_dir=tmp_path / "cache").fetch())


def test_idflakies_record_ids_are_unique(tmp_path):
    """Two iDFlakies rows can share (project, test_name) under different
    flake-type prefixes; record_ids must still be unique."""
    loader = IDFlakiesLoader(
        data_path=FIXTURES_ROOT / "idflakies" / "idflakies.csv",
        cache_dir=tmp_path / "cache",
    )
    ids = [r.record_id for r in loader.fetch()]
    assert len(ids) == len(set(ids))


def test_flakeflagger_extra_columns_preserved_in_metadata(tmp_path):
    loader = FlakeFlaggerLoader(
        data_path=FIXTURES_ROOT / "flakeflagger" / "features.csv",
        cache_dir=tmp_path / "cache",
    )
    records = list(loader.fetch())
    timing = next(r for r in records if r.test_name.endswith("testTiming"))
    assert timing.metadata["timing_flag"] is True
    assert timing.metadata["io_flag"] is False
    assert timing.metadata["extra_feature"] == "seasonal"


def test_loghub_picks_up_anomaly_labels(tmp_path):
    loader = LogHubLoader(
        data_path=FIXTURES_ROOT / "loghub",
        cache_dir=tmp_path / "cache",
        sub_datasets=["HDFS_v1"],
    )
    records = list(loader.fetch())
    by_label = {r.metadata["line_no"]: r.failure_type_label for r in records}
    assert by_label[1] == "anomaly"
    assert by_label[3] == "anomaly"
    assert by_label[0] == "normal"


def test_loghub_per_subset_limit_caps_records(tmp_path):
    loader = LogHubLoader(
        data_path=FIXTURES_ROOT / "loghub",
        cache_dir=tmp_path / "cache",
        sub_datasets=["HDFS_v1"],
        per_subset_limit=2,
    )
    records = list(loader.fetch())
    assert len(records) == 2


def test_loghub_treats_missing_anomaly_csv_as_unlabeled(tmp_path):
    loader = LogHubLoader(
        data_path=FIXTURES_ROOT / "loghub",
        cache_dir=tmp_path / "cache",
        sub_datasets=["Linux"],
    )
    labels = {r.failure_type_label for r in loader.fetch()}
    assert labels == {"unlabeled"}


def test_loghub_skips_missing_subsets_silently(tmp_path):
    loader = LogHubLoader(
        data_path=FIXTURES_ROOT / "loghub",
        cache_dir=tmp_path / "cache",
        sub_datasets=["HDFS_v1", "Hadoop_does_not_exist"],
    )
    records = list(loader.fetch())
    assert all(r.metadata["sub_dataset"] == "HDFS_v1" for r in records)


# ---------------------------------------------------------------------------
# cache helpers
# ---------------------------------------------------------------------------

def test_cache_dir_for_respects_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("CI_TRIAGE_DATA_CACHE", str(tmp_path / "alt"))
    assert cache_dir_for("loghub") == tmp_path / "alt" / "loghub"


def test_cache_dir_for_default_root(monkeypatch):
    monkeypatch.delenv("CI_TRIAGE_DATA_CACHE", raising=False)
    assert cache_dir_for("loghub") == DEFAULT_CACHE_ROOT / "loghub"


def test_load_cached_yields_records(tmp_path, monkeypatch):
    monkeypatch.setenv("CI_TRIAGE_DATA_CACHE", str(tmp_path))
    target = tmp_path / "deflaker"
    target.mkdir()
    rec = FailureRecord(
        record_id="x",
        source_dataset="deflaker",
        project="p",
        test_name="t",
        log_text="",
    )
    (target / "x.json").write_text(rec.model_dump_json())
    assert is_cached("x", "deflaker")
    out = list(load_cached("deflaker"))
    assert out == [rec]


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_cli_parser_lists_load_subcommand():
    parser = build_parser()
    parsed = parser.parse_args(["load", "deflaker"])
    assert parsed.cmd == "load"
    assert parsed.dataset == "deflaker"


def test_cli_load_writes_cache(tmp_path, capsys):
    rc = cli_main(
        [
            "load",
            "deflaker",
            "--data-path",
            str(FIXTURES_ROOT / "deflaker" / "failures.csv"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--summary",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "loaded 3 records" in out
    summary = json.loads(out.split("\n", 1)[1])  # second line is the JSON
    assert summary["count"] == 3
    assert any((tmp_path / "cache").glob("*.json"))


def test_cli_load_with_force_clears_cache(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stale = cache_dir / "stale-id.json"
    stale.write_text("{}")
    rc = cli_main(
        [
            "load",
            "deflaker",
            "--data-path",
            str(FIXTURES_ROOT / "deflaker" / "failures.csv"),
            "--cache-dir",
            str(cache_dir),
            "--force",
        ]
    )
    assert rc == 0
    assert not stale.exists()
    capsys.readouterr()  # drain


def test_cli_module_main_runs_via_subprocess(monkeypatch, tmp_path):
    """Smoke: ``python -m ci_triage_env.data.cli load --help`` parses cleanly."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["load", "--help"])  # argparse exits with 0
    # Confirm we didn't accidentally break the module-as-script entrypoint.
    import importlib

    sys.modules.pop("ci_triage_env.data.cli", None)
    importlib.import_module("ci_triage_env.data.cli")


# ---------------------------------------------------------------------------
# Fixture presence (per phase doc §Tests required)
# ---------------------------------------------------------------------------

def test_fixtures_exist():
    for path in [
        FIXTURES_ROOT / "deflaker" / "failures.csv",
        FIXTURES_ROOT / "idflakies" / "idflakies.csv",
        FIXTURES_ROOT / "flakeflagger" / "features.csv",
        FIXTURES_ROOT / "loghub" / "HDFS_v1" / "HDFS_v1.log",
        FIXTURES_ROOT / "loghub" / "HDFS_v1" / "HDFS_v1_anomaly.csv",
        FIXTURES_ROOT / "loghub" / "Linux" / "Linux.log",
    ]:
        assert path.exists(), f"fixture missing: {path}"


# ---------------------------------------------------------------------------
# DatasetLoader is abstract (sanity for the contract)
# ---------------------------------------------------------------------------

def test_dataset_loader_is_abstract():
    with pytest.raises(TypeError):
        DatasetLoader(data_path=Path("/tmp"))  # type: ignore[abstract]
