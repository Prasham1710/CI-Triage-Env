"""Corpus builder for Phase B5 — mass-generate Scenario JSONs.

Generates a reproducible corpus from the 7 family generators, applies a
deterministic train/val/held-out split, and writes one JSON file per scenario.

Key invariant: ALL ``ambiguous`` scenarios land in ``held_out`` regardless of
the split ratios.  Train and val have no ambiguous instances — the model trains
on the 6 unambiguous families, then its calibration is evaluated on the
held-out ambiguous set.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ci_triage_env.schemas.scenario import Scenario

DEFAULT_DISTRIBUTION: dict[str, float] = {
    "real_bug": 0.20,
    "race_flake": 0.15,
    "timing_flake": 0.10,
    "infra_network": 0.10,
    "infra_resource": 0.15,
    "dependency_drift": 0.10,
    "ambiguous": 0.20,  # over-represented: calibration probe set
}


def _scenario_split_key(scenario: Scenario, base_seed: int) -> float:
    """Return a float in [0,1) derived deterministically from the scenario_id."""
    digest = hashlib.sha256(f"{base_seed}:{scenario.scenario_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class CorpusBuilder:
    """Generate and split a scenario corpus across all 7 failure families."""

    def __init__(
        self,
        total: int = 200,
        distribution: dict[str, float] | None = None,
        split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
        base_seed: int = 100_000,
    ) -> None:
        self.total = total
        self.distribution = distribution or dict(DEFAULT_DISTRIBUTION)
        self.split_ratios = split_ratios
        self.base_seed = base_seed

    # ------------------------------------------------------------------ public

    def build(self, output_dir: Path) -> dict:
        """Generate the full corpus; write files; return summary dict."""
        from ci_triage_env.data.generators import GENERATOR_REGISTRY

        per_family = self._compute_per_family()
        all_scenarios: list[Scenario] = []

        family_order = list(per_family)
        cumulative = 0
        for family in family_order:
            count = per_family[family]
            generator = GENERATOR_REGISTRY[family]()
            for i in range(count):
                seed = self.base_seed + cumulative + i
                all_scenarios.append(generator.generate(seed=seed))
            cumulative += count

        train, val, held_out = self._split(all_scenarios)

        for split_name, split in [("train", train), ("val", val), ("held_out", held_out)]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for scenario in split:
                (split_dir / f"{scenario.scenario_id}.json").write_text(
                    scenario.model_dump_json(indent=2)
                )

        return {
            "total": len(all_scenarios),
            "train": len(train),
            "val": len(val),
            "held_out": len(held_out),
            "by_family": dict(per_family),
        }

    # ------------------------------------------------------------------ internal

    def _compute_per_family(self) -> dict[str, int]:
        """Allocate scenario count per family; guarantee at least 1 per family."""
        counts = {f: max(1, round(self.total * w)) for f, w in self.distribution.items()}
        # Clamp to self.total by trimming the largest family if needed
        total_allocated = sum(counts.values())
        if total_allocated > self.total:
            largest = max(counts, key=lambda f: counts[f])
            counts[largest] -= total_allocated - self.total
        return counts

    def _split(
        self, scenarios: list[Scenario]
    ) -> tuple[list[Scenario], list[Scenario], list[Scenario]]:
        """Deterministically assign scenarios to train / val / held-out.

        Ambiguous scenarios always land in held-out.  The rest are sorted by
        their hash-based split key so the assignment is seed-stable.
        """
        ambiguous = [s for s in scenarios if s.family == "ambiguous"]
        rest = [s for s in scenarios if s.family != "ambiguous"]

        # Sort rest by deterministic key so shuffle is reproducible
        rest_sorted = sorted(rest, key=lambda s: _scenario_split_key(s, self.base_seed))

        n = len(rest_sorted)
        n_train = int(n * self.split_ratios[0])
        n_val = int(n * self.split_ratios[1])

        train = rest_sorted[:n_train]
        val = rest_sorted[n_train : n_train + n_val]
        held_out_rest = rest_sorted[n_train + n_val :]

        held_out = ambiguous + held_out_rest
        return train, val, held_out
