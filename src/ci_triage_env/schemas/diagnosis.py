from enum import StrEnum


class DiagnosisLabel(StrEnum):
    REAL_BUG = "real_bug"
    RACE_FLAKE = "race_flake"
    TIMING_FLAKE = "timing_flake"
    INFRA_NETWORK = "infra_network"
    INFRA_RESOURCE = "infra_resource"
    DEPENDENCY_DRIFT = "dependency_drift"
    AMBIGUOUS = "ambiguous"

    def is_flake(self) -> bool:
        return self in {DiagnosisLabel.RACE_FLAKE, DiagnosisLabel.TIMING_FLAKE}

    def is_infra(self) -> bool:
        return self in {DiagnosisLabel.INFRA_NETWORK, DiagnosisLabel.INFRA_RESOURCE}

    def is_real_root_cause(self) -> bool:
        return self in {DiagnosisLabel.REAL_BUG, DiagnosisLabel.DEPENDENCY_DRIFT}
