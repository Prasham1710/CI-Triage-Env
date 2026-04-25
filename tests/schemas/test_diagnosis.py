from ci_triage_env.schemas.diagnosis import DiagnosisLabel


def test_all_seven_labels_present():
    assert {label.value for label in DiagnosisLabel} == {
        "real_bug",
        "race_flake",
        "timing_flake",
        "infra_network",
        "infra_resource",
        "dependency_drift",
        "ambiguous",
    }


def test_is_flake_helper():
    assert DiagnosisLabel.RACE_FLAKE.is_flake()
    assert DiagnosisLabel.TIMING_FLAKE.is_flake()
    assert not DiagnosisLabel.REAL_BUG.is_flake()
    assert not DiagnosisLabel.AMBIGUOUS.is_flake()


def test_is_infra_helper():
    assert DiagnosisLabel.INFRA_NETWORK.is_infra()
    assert DiagnosisLabel.INFRA_RESOURCE.is_infra()
    assert not DiagnosisLabel.REAL_BUG.is_infra()
    assert not DiagnosisLabel.RACE_FLAKE.is_infra()


def test_is_real_root_cause_helper():
    assert DiagnosisLabel.REAL_BUG.is_real_root_cause()
    assert DiagnosisLabel.DEPENDENCY_DRIFT.is_real_root_cause()
    assert not DiagnosisLabel.RACE_FLAKE.is_real_root_cause()
    assert not DiagnosisLabel.AMBIGUOUS.is_real_root_cause()
