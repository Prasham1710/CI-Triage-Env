from ci_triage_env.data.generators.ambiguous import AmbiguousGenerator
from ci_triage_env.data.generators.base import ScenarioFamilyGenerator
from ci_triage_env.data.generators.dependency_drift import DependencyDriftGenerator
from ci_triage_env.data.generators.infra_network import InfraNetworkGenerator
from ci_triage_env.data.generators.infra_resource import InfraResourceGenerator
from ci_triage_env.data.generators.race_flake import RaceFlakeGenerator
from ci_triage_env.data.generators.real_bug import RealBugGenerator
from ci_triage_env.data.generators.timing_flake import TimingFlakeGenerator

GENERATOR_REGISTRY: dict[str, type[ScenarioFamilyGenerator]] = {
    "real_bug": RealBugGenerator,
    "race_flake": RaceFlakeGenerator,
    "timing_flake": TimingFlakeGenerator,
    "infra_network": InfraNetworkGenerator,
    "infra_resource": InfraResourceGenerator,
    "dependency_drift": DependencyDriftGenerator,
    "ambiguous": AmbiguousGenerator,
}

__all__ = [
    "GENERATOR_REGISTRY",
    "AmbiguousGenerator",
    "DependencyDriftGenerator",
    "InfraNetworkGenerator",
    "InfraResourceGenerator",
    "RaceFlakeGenerator",
    "RealBugGenerator",
    "ScenarioFamilyGenerator",
    "TimingFlakeGenerator",
]
