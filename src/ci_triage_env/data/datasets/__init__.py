from ci_triage_env.data.datasets._base import DatasetLoader, FailureRecord
from ci_triage_env.data.datasets.deflaker import DeFlakerLoader
from ci_triage_env.data.datasets.flakeflagger import FlakeFlaggerLoader
from ci_triage_env.data.datasets.idflakies import IDFlakiesLoader
from ci_triage_env.data.datasets.loghub import LogHubLoader

LOADER_REGISTRY: dict[str, type[DatasetLoader]] = {
    "deflaker": DeFlakerLoader,
    "idflakies": IDFlakiesLoader,
    "flakeflagger": FlakeFlaggerLoader,
    "loghub": LogHubLoader,
}

__all__ = [
    "LOADER_REGISTRY",
    "DatasetLoader",
    "DeFlakerLoader",
    "FailureRecord",
    "FlakeFlaggerLoader",
    "IDFlakiesLoader",
    "LogHubLoader",
]
