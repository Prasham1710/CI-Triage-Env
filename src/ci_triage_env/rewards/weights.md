# Reward Weights — Provenance and Justification

## Component weights (post format-gate)

| Component | Weight | Rationale |
|---|---|---|
| diagnosis | 0.25 | Largest single signal — getting the diagnosis right is the headline metric |
| action_quality | 0.20 | Second-largest; downstream behavior matters distinctly from belief |
| cost_efficiency | 0.15 | Forces surgical investigation; grounded in CircleCI/GitHub Actions compute pricing |
| investigation | 0.15 | Trajectory shaping; this is where RL beats classification |
| anti_gaming | 0.15 | Anti-exploit; prevents quarantine-everything and single-shot guessing |
| time | 0.10 | Mild speed pressure; discourages exhaustive tool-call fishing |
| counterfactual | 0.0 in v1 (deferred) | Scaffolded but dormant. v2 activates by setting weight to 0.10. |

Weights sum to 1.0 (excluding counterfactual). MinimalEvidenceReward is folded
into InvestigationReward as a multiplier (up to +30% boost), not a separate term.

## Diagnosis confusion-matrix weights

Asymmetry rationale:

- **Predict flake on real bug = -1.0**: catastrophic; ships bug to production. From DeFlaker (Bell et al., FSE 2018), prod-shipped bug median cost is ~$100k vs. flake-investigation median cost ~$200.
- **Predict infra on real bug ≈ -0.5**: pages wrong team; recoverable but costly.
- **Wrong flake type (race vs timing) = +0.2**: both trigger the flake-handling pathway; confusion is mild.
- **Default off-diagonal = -0.5**: penalty for any confidently-wrong answer.

## Action quality matrix

- **quarantine_test on real_bug = -1.5**: most catastrophic action; suppresses a real failure, ships bug silently.
- **file_bug on real_bug = +1.0**: ideal response to a real defect.
- **rerun_test on infra_network = +0.8**: correct for transient connectivity failures.
- **quarantine_test on race_flake = +1.0**: correct for non-deterministic flakes.

## Anti-gaming thresholds

- **Quarantine-rate guard threshold**: 30% over a 50-episode rolling window. Above this, penalty proportional to excess. Calibrated to allow ~25% expected flake rate plus headroom.
- **No-info-action guard**: diagnosing with < 2 tool calls = -0.5. Prevents single-shot guessing.
- **Brier calibration weight**: 0.5 raw on ambiguous scenarios. Penalises mis-calibrated confidence on scenarios with no clear answer.

## References

- Bell, J. et al. "DeFlaker: Automatically Detecting Flaky Tests." FSE 2018.
- Lam, W. et al. "iDFlakies: A Framework for Detecting and Partially Classifying Flaky Tests." ICSE 2019.
- Google SRE Book, Ch. 31, "Communication and Collaboration in SRE": cost models for paging wrong teams.
- CircleCI compute pricing (used for cost_efficiency reference budget of $5/episode).

## How to change weights

1. Propose change in team chat with motivation and expected effect.
2. Run ablation: rerun GRPO with new weights for 1000 steps, compare reward curve and final eval scores against baseline.
3. If improvement confirmed, update `weights.py`, bump `REWARD_VERSION`, document change in this file.
