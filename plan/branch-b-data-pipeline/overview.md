# Branch B — Data Pipeline

**Branch name:** `branch-b/data-pipeline`
**Default owner:** Priyanshi.
**Prerequisite:** `phase-0-complete` tag on `main`.

---

## What this branch builds

The 200-scenario corpus that Branch A serves and Branch C trains on. The corpus is built from:

1. Public flaky-test datasets (DeFlaker, iDFlakies, FlakeFlagger, LogHub).
2. Real GitHub Actions logs scraped from popular OSS repos (Kubernetes, React, TensorFlow, Rust, etc.).
3. Parametric scenario generators that take real failures as templates and produce many variants.

After this branch merges, on `main`:

```bash
python -m ci_triage_env.data.cli generate --family race_flake --count 30 --seed 42
# writes 30 scenarios to data_artifacts/scenarios/

python -m ci_triage_env.data.cli publish-hf --dataset-name USER/ci-triage-scenarios
# publishes the corpus to HF dataset hub
```

The published HF dataset is what Branch A loads at runtime.

---

## Phases

| Phase | Title | What lands | Depends on |
|---|---|---|---|
| B1 | Public dataset ingest | Loaders for DeFlaker, iDFlakies, FlakeFlagger, LogHub. Output: normalized labeled-failure records | Phase 0 |
| B2 | GitHub Actions log mining | `gh CLI`-based scraper for failed runs across 8–10 OSS repos; rate-limited, cached, anonymized | Phase 0 (parallel with B1) |
| B3 | Failure clustering | Offline one-shot LLM call (or rule-based) to cluster mined logs into the 7 failure families with per-family archetypes | B1 + B2 |
| B4 | Scenario family generators | 7 parametric `ScenarioFamilyGenerator` subclasses, one per failure category | B3 |
| B5 | Scenario instantiation + HF publish | CLI to generate N scenarios per family with `informative_tools` and `minimal_evidence_set` annotations; writes JSONs; uploads to HF dataset | B4 |

---

## What you consume

- All schemas from Phase 0. Do not modify them; if you need changes, propose in chat first.
- `mock.scenario.make_mock_scenario()` for sanity-checking generators.

## What you produce

- ~200 (target 300) `Scenario` JSON files in `data_artifacts/scenarios/`.
- An HF dataset hosting the corpus, structured so Branch A can pull by `scenario_id`.
- Per-family informative-tools and minimal-evidence-set annotations baked into each scenario.
- `data_artifacts/mined_logs/` (gitignored) — raw mined logs for reproducibility.

---

## Files this branch owns

```
src/ci_triage_env/data/
├── __init__.py
├── cli.py                          # `python -m ci_triage_env.data.cli ...`
├── mining/
│   ├── __init__.py
│   ├── github_actions.py           # gh CLI scraper
│   ├── anonymizer.py               # strip project-specific identifiers
│   └── cache.py                    # disk cache for scrape results
├── datasets/
│   ├── __init__.py
│   ├── deflaker.py                 # DeFlaker loader
│   ├── idflakies.py                # iDFlakies loader
│   ├── flakeflagger.py             # FlakeFlagger loader
│   └── loghub.py                   # LogHub loader
├── clustering/
│   ├── __init__.py
│   ├── classifier.py               # rule-based + optional LLM cluster
│   └── archetypes.py               # archetype extractor
├── generators/
│   ├── __init__.py
│   ├── base.py                     # already from Phase 0
│   ├── real_bug.py
│   ├── race_flake.py
│   ├── timing_flake.py
│   ├── infra_network.py
│   ├── infra_resource.py
│   ├── dependency_drift.py
│   └── ambiguous.py
├── publish.py                      # HF dataset upload
└── annotations/
    ├── __init__.py
    └── informative_tools.py        # logic to derive informative_tools per family

tests/data/
├── test_dataset_loaders.py
├── test_mining.py
├── test_clustering.py
├── test_generators.py              # one test per family
├── test_annotations.py
└── fixtures/
    ├── sample_real_log.txt
    └── sample_dataset_record.json
```

You do **not** own:
- Anything under `src/ci_triage_env/env/` (Branch A)
- Anything under `src/ci_triage_env/rewards/` or `training/` (Branch C)

---

## External resources you will use

| Resource | Access | Notes |
|---|---|---|
| **gh CLI** | `gh` installed locally; auth with `gh auth login` | Used to fetch failed-run logs from public repos. Rate limit: 5000/hr authenticated |
| **DeFlaker dataset** | GitHub releases or paper supplementary | Bell et al., FSE 2018 |
| **iDFlakies** | https://github.com/idflakies | Lam et al., ICSE 2019 |
| **FlakeFlagger** | https://github.com/AlshammariA/FlakeFlagger | Alshammari et al., ICSE 2021 |
| **LogHub** | https://github.com/logpai/loghub | Zhu et al., ISSRE 2019 |
| **OpenAI API** | Shared $30 budget, via `OPENAI_API_KEY` env var | For B3 clustering only — see Phase B3 budget |

OpenAI usage in this branch: at most ~$5 for offline clustering. Bulk SFT generation budget is owned by Branch C.

---

## OSS repos to mine in B2

Target list (final list locked in Phase B2 doc):

1. `kubernetes/kubernetes` — Go, large flake history
2. `facebook/react` — JS, well-documented test infra
3. `tensorflow/tensorflow` — Python+C++, GPU resource issues
4. `rust-lang/rust` — Rust, build/timing issues
5. `golang/go` — Go, race-detector flakes
6. `apache/spark` — Scala+Python, distributed test flakes
7. `pytorch/pytorch` — Python+C++, GPU/CPU resource issues
8. `nodejs/node` — JS, native bindings flakes

Fetch ~30–50 failed runs per repo via `gh run list --status failure --limit 50` plus `gh run view <id> --log`.

---

## Tooling convention (read once, applies to every phase doc)

This repo uses **uv** (see `INSTRUCTION-MANUAL.md` §0). Every command in the per-phase docs that says `python ...`, `pytest ...`, or `ruff ...` should be run as `uv run python ...`, `uv run pytest ...`, `uv run ruff ...`. New deps are added with `uv add <pkg>` (or `uv add --optional <extra> <pkg>`), never raw `pip install`. Commit `pyproject.toml` and `uv.lock` together.

## Test discipline

After every phase, run:

```bash
uv run pytest -q tests/data/
uv run pytest -q tests/schemas/   # sanity
uv run ruff check src/ci_triage_env/data/
```

Specific tests required per phase: see each `phase-b<N>.md` file. Generator tests must verify:
- Output validates against `Scenario` schema
- Determinism: same seed → same output
- Ground-truth label matches family
- `informative_tools` is non-empty
- `minimal_evidence_set` is a non-empty subset of available tools

---

## Integration checkpoints

- **Gate-1 prerequisite:** B1, B2, B3, B4, B5 ALL must be merged into `main` by Sunday afternoon. The full corpus must be on HF dataset before Branch C's eval phase.
- This is your hard deadline. If you slip, fall back to a smaller corpus (100 scenarios across the 7 families) — Branch A and C cannot proceed without scenarios.

---

## Realism checks (the quality bar)

Before B5 publishes, **manually inspect 5 random scenarios per family**. Each should:

1. Read like a real CI failure to a human engineer (not "AI-generated logs").
2. Have ground truth label that matches the failure pattern in the logs.
3. Have informative_tools that are actually informative (would a human use these?).
4. Have minimal_evidence_set that's actually minimal (could a human reach the diagnosis with strictly less?).

If any scenario fails these checks, fix the generator. Quality > quantity.

---

## Open questions you'll see in phase docs

Each `phase-b<N>.md` contains:
1. **Outcome** — definition of done
2. **Files to create / modify** — exact paths
3. **Implementation notes** — design decisions already made
4. **Tests required** — what to add to `tests/data/`
5. **Smoke test** — manual verification command
6. **Open questions** — things to flag in chat, never silently decide

Open `phase-b1.md` and `phase-b2.md` (these can run in parallel — start whichever you prefer).
