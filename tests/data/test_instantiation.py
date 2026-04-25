"""Tests for Phase B5 — CorpusBuilder, annotations, and CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ci_triage_env.data.annotations import enrich_annotations
from ci_triage_env.data.cli import build_parser, cmd_generate
from ci_triage_env.data.generators import GENERATOR_REGISTRY
from ci_triage_env.data.instantiation import CorpusBuilder
from ci_triage_env.schemas.scenario import Scenario

# ---------------------------------------------------------------------------
# CorpusBuilder
# ---------------------------------------------------------------------------


def test_corpus_builder_total_count(tmp_path: Path) -> None:
    builder = CorpusBuilder(total=20)
    summary = builder.build(tmp_path)
    assert summary["total"] == 20


def test_corpus_builder_split_dirs_created(tmp_path: Path) -> None:
    builder = CorpusBuilder(total=20)
    builder.build(tmp_path)
    for split in ("train", "val", "held_out"):
        assert (tmp_path / split).is_dir()


def test_corpus_builder_no_ambiguous_in_train_val(tmp_path: Path) -> None:
    builder = CorpusBuilder(total=40)
    builder.build(tmp_path)
    for split in ("train", "val"):
        split_dir = tmp_path / split
        for path in split_dir.glob("*.json"):
            scenario = json.loads(path.read_text())
            assert scenario["family"] != "ambiguous", (
                f"ambiguous scenario found in {split}: {path.name}"
            )


def test_corpus_builder_all_ambiguous_in_held_out(tmp_path: Path) -> None:
    builder = CorpusBuilder(total=40)
    builder.build(tmp_path)
    held_out_dir = tmp_path / "held_out"
    ambiguous_count = sum(
        1
        for p in held_out_dir.glob("*.json")
        if json.loads(p.read_text())["family"] == "ambiguous"
    )
    # We expect some ambiguous scenarios in held_out (distribution has 20% ambiguous)
    assert ambiguous_count > 0


def test_corpus_builder_determinism(tmp_path: Path) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    CorpusBuilder(total=20, base_seed=42).build(out1)
    CorpusBuilder(total=20, base_seed=42).build(out2)

    for split in ("train", "val", "held_out"):
        ids1 = sorted(p.stem for p in (out1 / split).glob("*.json"))
        ids2 = sorted(p.stem for p in (out2 / split).glob("*.json"))
        assert ids1 == ids2, f"non-deterministic split in {split}"


def test_corpus_builder_different_seeds_give_different_splits(tmp_path: Path) -> None:
    out1 = tmp_path / "s1"
    out2 = tmp_path / "s2"
    CorpusBuilder(total=30, base_seed=1).build(out1)
    CorpusBuilder(total=30, base_seed=2).build(out2)

    train1 = sorted(p.stem for p in (out1 / "train").glob("*.json"))
    train2 = sorted(p.stem for p in (out2 / "train").glob("*.json"))
    # Different seeds → different scenario IDs (different seeds in generators)
    assert train1 != train2


def test_corpus_builder_json_validates_schema(tmp_path: Path) -> None:
    builder = CorpusBuilder(total=14)
    builder.build(tmp_path)
    for split in ("train", "val", "held_out"):
        for path in (tmp_path / split).glob("*.json"):
            data = json.loads(path.read_text())
            Scenario.model_validate(data)


def test_corpus_builder_summary_keys(tmp_path: Path) -> None:
    summary = CorpusBuilder(total=14).build(tmp_path)
    assert set(summary.keys()) >= {"total", "train", "val", "held_out", "by_family"}
    assert summary["train"] + summary["val"] + summary["held_out"] == summary["total"]


# ---------------------------------------------------------------------------
# Annotations — enrich_annotations
# ---------------------------------------------------------------------------


def test_enrich_annotations_passthrough_when_already_set() -> None:
    scenario = GENERATOR_REGISTRY["real_bug"]().generate(seed=1)
    # real_bug generator always sets informative_tools
    assert scenario.informative_tools
    enriched = enrich_annotations(scenario)
    assert enriched.informative_tools == scenario.informative_tools


def test_enrich_annotations_fills_empty_informative_tools() -> None:
    scenario = GENERATOR_REGISTRY["real_bug"]().generate(seed=2)
    # Clear informative_tools to simulate a scenario that needs enrichment
    bare = scenario.model_copy(update={"informative_tools": []})
    enriched = enrich_annotations(bare)
    assert len(enriched.informative_tools) > 0


def test_enrich_annotations_only_lists_covered_tools() -> None:
    from ci_triage_env.schemas.tools import ALL_TOOLS

    scenario = GENERATOR_REGISTRY["real_bug"]().generate(seed=3)
    bare = scenario.model_copy(update={"informative_tools": []})
    enriched = enrich_annotations(bare)
    all_tool_names = {t.name for t in ALL_TOOLS}
    for tool_name in enriched.informative_tools:
        assert tool_name in all_tool_names


# ---------------------------------------------------------------------------
# CLI — generate subcommand
# ---------------------------------------------------------------------------


def test_cli_generate_creates_files(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "generate",
        "--total", "21",
        "--split", "70/15/15",
        "--seed", "999",
        "--output-dir", str(tmp_path),
    ])
    rc = cmd_generate(args)
    assert rc == 0
    total_files = sum(1 for _ in tmp_path.rglob("*.json"))
    assert total_files > 0


def test_cli_generate_bad_split_returns_2(tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "generate",
        "--total", "14",
        "--split", "70/15",  # only two parts — invalid
        "--output-dir", str(tmp_path),
    ])
    rc = cmd_generate(args)
    assert rc == 2


def test_cli_generate_default_output_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["generate", "--total", "7"])
    rc = cmd_generate(args)
    assert rc == 0
    default_out = tmp_path / "data_artifacts" / "scenarios"
    assert default_out.is_dir()
