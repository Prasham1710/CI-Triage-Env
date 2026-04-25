import pytest
from pydantic import ValidationError

from ci_triage_env.mock.scenario import make_mock_scenario
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.scenario import Scenario


def test_round_trip_serialize_deserialize():
    scenario = make_mock_scenario("race_flake")
    payload = scenario.model_dump_json()
    restored = Scenario.model_validate_json(payload)
    assert restored == scenario


def test_round_trip_via_dict():
    scenario = make_mock_scenario("real_bug")
    restored = Scenario.model_validate(scenario.model_dump())
    assert restored == scenario


def test_validation_fails_on_missing_fields():
    with pytest.raises(ValidationError):
        Scenario.model_validate({"scenario_id": "x"})


def test_ground_truth_label_is_valid_enum():
    scenario = make_mock_scenario("infra_network")
    assert scenario.ground_truth.label is DiagnosisLabel.INFRA_NETWORK
    assert scenario.ground_truth.label in DiagnosisLabel


def test_ground_truth_rejects_unknown_label():
    scenario = make_mock_scenario("real_bug")
    payload = scenario.model_dump()
    payload["ground_truth"]["label"] = "not_a_real_label"
    with pytest.raises(ValidationError):
        Scenario.model_validate(payload)


def test_all_seven_families_construct():
    for family in [
        "real_bug",
        "race_flake",
        "timing_flake",
        "infra_network",
        "infra_resource",
        "dependency_drift",
        "ambiguous",
    ]:
        scenario = make_mock_scenario(family)
        assert scenario.family == family
