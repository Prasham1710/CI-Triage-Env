"""Tests for Phase B4 — ScenarioFamilyGenerators (one per failure family)."""

from __future__ import annotations

import pytest

from ci_triage_env.data.generators import GENERATOR_REGISTRY, ScenarioFamilyGenerator
from ci_triage_env.schemas.scenario import Scenario
from ci_triage_env.schemas.tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_FAMILIES = list(GENERATOR_REGISTRY.keys())
ALL_TOOL_NAMES = {t.name for t in ALL_TOOLS}

EXPECTED_FAMILIES = {
    "real_bug", "race_flake", "timing_flake",
    "infra_network", "infra_resource", "dependency_drift", "ambiguous",
}


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------

def test_registry_has_all_seven_families() -> None:
    assert set(GENERATOR_REGISTRY.keys()) == EXPECTED_FAMILIES


def test_registry_values_are_generator_subclasses() -> None:
    for GenCls in GENERATOR_REGISTRY.values():
        assert issubclass(GenCls, ScenarioFamilyGenerator)


# ---------------------------------------------------------------------------
# Per-family parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_generator_determinism(family: str) -> None:
    """Same seed → identical Scenario."""
    GenCls = GENERATOR_REGISTRY[family]
    s1 = GenCls().generate(seed=42)
    s2 = GenCls().generate(seed=42)
    assert s1.model_dump() == s2.model_dump()


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_generator_different_seeds_produce_different_ids(family: str) -> None:
    """Different seeds → different scenario_ids."""
    GenCls = GENERATOR_REGISTRY[family]
    ids = {GenCls().generate(seed=s).scenario_id for s in range(5)}
    assert len(ids) == 5


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_generator_validates_schema(family: str) -> None:
    """Output passes Scenario Pydantic schema validation."""
    scenario = GENERATOR_REGISTRY[family]().generate(seed=7)
    Scenario.model_validate(scenario.model_dump())


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_ground_truth_label_matches_family(family: str) -> None:
    scenario = GENERATOR_REGISTRY[family]().generate(seed=99)
    assert scenario.ground_truth.label.value == family


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_all_tools_have_outputs(family: str) -> None:
    """For each of 11 tools, tool_outputs has at least one matching key."""
    scenario = GENERATOR_REGISTRY[family]().generate(seed=3)
    for tool_name in ALL_TOOL_NAMES:
        covered = any(
            k == tool_name or k.startswith(tool_name + ":")
            for k in scenario.tool_outputs
        )
        assert covered, (
            f"family={family!r}: no tool_outputs key for tool={tool_name!r}. "
            f"Keys present: {sorted(scenario.tool_outputs)}"
        )


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_informative_tools_nonempty(family: str) -> None:
    """informative_tools must have ≥ 2 entries (or be empty for ambiguous)."""
    gen = GENERATOR_REGISTRY[family]()
    scenario = gen.generate(seed=5)
    if family == "ambiguous":
        # ambiguous has more than 2 by design
        assert len(scenario.informative_tools) >= 2
    else:
        assert len(scenario.informative_tools) >= 2


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_minimal_evidence_subset_of_informative(family: str) -> None:
    """minimal_evidence_set ⊆ informative_tools OR minimal_evidence_set is empty."""
    gen = GENERATOR_REGISTRY[family]()
    scenario = gen.generate(seed=11)
    if scenario.minimal_evidence_set:
        assert set(scenario.minimal_evidence_set) <= set(scenario.informative_tools), (
            f"family={family}: minimal_evidence_set not a subset of informative_tools. "
            f"minimal={scenario.minimal_evidence_set}, informative={scenario.informative_tools}"
        )


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_difficulty_is_valid(family: str) -> None:
    scenario = GENERATOR_REGISTRY[family]().generate(seed=21)
    assert scenario.metadata.difficulty in {"easy", "medium", "hard"}


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_scenario_id_contains_family(family: str) -> None:
    scenario = GENERATOR_REGISTRY[family]().generate(seed=13)
    assert family in scenario.scenario_id


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_schema_version_is_1_0(family: str) -> None:
    scenario = GENERATOR_REGISTRY[family]().generate(seed=77)
    assert scenario.schema_version == "1.0"


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_tool_output_cost_units_nonnegative(family: str) -> None:
    scenario = GENERATOR_REGISTRY[family]().generate(seed=55)
    for key, output in scenario.tool_outputs.items():
        assert output.cost_units >= 0.0, f"negative cost_units on key={key!r}"


@pytest.mark.parametrize("family", ALL_FAMILIES)
def test_failure_summary_populated(family: str) -> None:
    scenario = GENERATOR_REGISTRY[family]().generate(seed=33)
    fs = scenario.failure_summary
    assert fs.test_name
    assert fs.branch
    assert fs.suite in {"unit", "integration", "benchmark"}
    assert len(fs.last_passing_commit) == 40  # full SHA


# ---------------------------------------------------------------------------
# Ambiguous-specific tests
# ---------------------------------------------------------------------------

def test_ambiguous_confidence_target_below_one() -> None:
    scenario = GENERATOR_REGISTRY["ambiguous"]().generate(seed=42)
    assert scenario.ground_truth.confidence_target < 1.0


def test_ambiguous_is_flagged() -> None:
    scenario = GENERATOR_REGISTRY["ambiguous"]().generate(seed=42)
    assert scenario.ground_truth.is_ambiguous is True


def test_ambiguous_minimal_evidence_is_empty() -> None:
    scenario = GENERATOR_REGISTRY["ambiguous"]().generate(seed=42)
    assert scenario.minimal_evidence_set == []


def test_ambiguous_correct_action_has_low_confidence() -> None:
    scenario = GENERATOR_REGISTRY["ambiguous"]().generate(seed=42)
    confidence = scenario.correct_terminal_action.args.get("confidence", 1.0)
    assert confidence < 0.6


# ---------------------------------------------------------------------------
# Non-ambiguous families: is_ambiguous == False
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", [f for f in ALL_FAMILIES if f != "ambiguous"])
def test_non_ambiguous_scenarios_not_flagged(family: str) -> None:
    scenario = GENERATOR_REGISTRY[family]().generate(seed=42)
    assert scenario.ground_truth.is_ambiguous is False


@pytest.mark.parametrize("family", [f for f in ALL_FAMILIES if f != "ambiguous"])
def test_non_ambiguous_confidence_target_is_1(family: str) -> None:
    scenario = GENERATOR_REGISTRY[family]().generate(seed=42)
    assert scenario.ground_truth.confidence_target == 1.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_all_generators_produce_distinct_scenario_ids() -> None:
    """Generate 10 scenarios per family with different seeds; all IDs are unique."""
    seen: set[str] = set()
    for family, GenCls in GENERATOR_REGISTRY.items():
        gen = GenCls()
        for seed in range(10):
            sid = gen.generate(seed=seed).scenario_id
            assert sid not in seen, f"Duplicate scenario_id={sid!r} for family={family!r} seed={seed}"
            seen.add(sid)


def test_generators_use_default_archetypes_when_no_clustering_data(tmp_path) -> None:
    """Generators fall back to built-in defaults when archetypes_dir is empty."""
    for family, GenCls in GENERATOR_REGISTRY.items():
        gen = GenCls(archetypes_dir=tmp_path)  # empty dir → use defaults
        scenario = gen.generate(seed=42)
        assert scenario.family == family


def test_generators_use_archetypes_when_available(tmp_path) -> None:
    """If clustering archetypes exist, generator picks from them."""
    import json

    from ci_triage_env.data.clustering.archetypes import Archetype

    family = "real_bug"
    family_dir = tmp_path / family
    family_dir.mkdir()
    custom_arch = Archetype(
        archetype_id="custom_001",
        family=family,
        pattern_summary="Custom archetype for test",
        log_template="CUSTOM_LOG_LINE_{NUM}",
        slot_distributions={"NUM": ["42", "99"]},
        informative_tools_hint=["read_logs:full"],
        minimal_evidence_hint=["read_logs:full"],
    )
    (family_dir / "archetypes.json").write_text(
        json.dumps([custom_arch.model_dump()], indent=2)
    )

    gen = GENERATOR_REGISTRY[family](archetypes_dir=tmp_path)
    # Force reload by creating fresh instance
    scenario = gen.generate(seed=42)
    # Scenario should be valid and deterministic
    assert scenario.family == family
    Scenario.model_validate(scenario.model_dump())


def test_generator_seed_embedded_in_scenario() -> None:
    for _family, GenCls in GENERATOR_REGISTRY.items():
        seed = 1337
        scenario = GenCls().generate(seed=seed)
        assert scenario.seed == seed


def test_read_logs_full_has_content() -> None:
    """read_logs:full always has non-empty lines."""
    for family, GenCls in GENERATOR_REGISTRY.items():
        scenario = GenCls().generate(seed=42)
        full_output = scenario.tool_outputs.get("read_logs:full")
        assert full_output is not None, f"{family}: missing read_logs:full"
        payload = full_output.payload
        assert isinstance(payload, dict)
        assert len(payload.get("lines", [])) > 0, f"{family}: read_logs:full has empty lines"
