# CI-Triage-Env — Master Instruction Manual

Single source of truth for the build. Read this top-to-bottom before starting. Whenever the build state is ambiguous, this document wins; when this document conflicts with a phase doc, *update both* before proceeding.

---

## 0. Tooling — non-negotiable

We use **`uv`** as the Python package manager and runner everywhere. No `pip install`, no `conda`, no `poetry`, no system Python invocation outside `uv run`. Reasons: hermetic environments, identical dep resolution across all 3 laptops, single lockfile (`uv.lock`) committed to the repo, fast.

### Install uv (each laptop, once)

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# OR if you have brew:
brew install uv
# OR via pip into a system Python (least preferred):
pipx install uv

uv --version    # confirm ≥ 0.5.0
```

### Daily commands

| Action | Command |
|---|---|
| One-time setup after `git clone` | `uv sync --all-extras` |
| Run tests | `uv run pytest -q` |
| Run lint | `uv run ruff check src/ tests/` |
| Run the env server | `uv run python -m ci_triage_env.env.server` |
| Run a CLI subcommand | `uv run python -m ci_triage_env.data.cli <args>` |
| Add a new dependency | `uv add <pkg>` (commits to `pyproject.toml` + `uv.lock`) |
| Add a dev-only dep | `uv add --dev <pkg>` |
| Add to a specific extra | `uv add --optional training <pkg>` |
| Upgrade locked deps | `uv lock --upgrade` |
| Run a one-off Python | `uv run python -c "..."` |

### Rules

- **Never `pip install` into the venv directly.** It bypasses the lockfile and breaks reproducibility on the other laptops. If you need a package, `uv add <pkg>`, commit the updated `pyproject.toml` + `uv.lock`, push, others `git pull && uv sync`.
- **`uv.lock` is committed to git.** Treat it like a source file — review changes in PRs.
- **`.venv/` is gitignored.** Never commit it.
- **Colab notebooks** (training notebooks under `notebooks/`) install via `!pip install` because Colab doesn't ship with uv. That's a deliberate exception — the notebook environment is short-lived and ephemeral. Local development is uv-only.
- **HF Spaces** uses its own env; the OpenEnv server's `requirements.txt` (auto-generated from uv via `uv export`) is committed for Spaces compatibility. See Phase 0 for the export step.

### When uv breaks

- `uv sync` fails on dep conflict → read the error, then `uv lock --upgrade-package <pkg>` for the conflicting package, or `uv add` with version pin.
- Someone else's `uv.lock` won't apply on your machine → you may have a different platform marker; run `uv sync --refresh` to re-resolve.
- You hit a Python version mismatch → `pyproject.toml` requires Python 3.11. Install with `uv python install 3.11`.

---

## 1. Roles and laptops

| Person | Role | Branch | Folder of plan files |
|---|---|---|---|
| Prasham Jain | Team lead, repo owner | `branch-a/env-core` | `plan/branch-a-env-core/` |
| Priyanshi | Data pipeline | `branch-b/data-pipeline` | `plan/branch-b-data-pipeline/` |
| Sahil | ML / RL training | `branch-c/reward-training` | `plan/branch-c-reward-training/` |

> If team prefers different ownership: swap below. Per-phase docs are role-agnostic — anyone can pick up any folder, the docs assume nothing about who you are.

---

## 2. Critical contracts (DO NOT BREAK)

These must be locked at end of Phase 0. Changing them after split forces costly re-sync across branches.

1. **Scenario JSON schema** — defined in `src/ci_triage_env/schemas/scenario.py`. Branch B writes; Branch A reads; Branch C reads.
2. **MCP tool definitions** — name, args schema, return schema for all 11 tools, defined in `src/ci_triage_env/schemas/tools.py`.
3. **OpenEnv server HTTP contract** — endpoints `/reset`, `/step`, `/state`, `/mcp` (MCP server endpoint).
4. **Reward breakdown schema** — `src/ci_triage_env/schemas/reward.py`. Branch C writes; Branch A reads (for replay viz).
5. **Episode trace JSON schema** — what gets written to disk per episode for replay/visualizer.

Any change to these *after* Phase 0 must be: announced in team chat, PR'd to `main`, all branches rebase before continuing.

---

## 3. Repository setup

### 3.1 Prasham (one-time, day 0)

```bash
# On Prasham's laptop only
gh repo create ci-triage-env --public --description "OpenEnv RL env for CI failure triage"
cd ~ && git clone https://github.com/<prasham-user>/ci-triage-env
cd ci-triage-env

# Bootstrap with the planning artifacts
cp -r ~/hackathon-research/plan plan/
git add plan/
git commit -m "chore: import planning artifacts"
git push origin main
```

Then Prasham executes Phase 0 entirely on `main` (foundation must be on `main` before anyone branches). See `plan/shared/foundation-phase-0.md`. Phase 0 sets up the uv project (`pyproject.toml`, `uv.lock`, `.python-version`) and runs `uv sync --all-extras` for the first time.

After Phase 0 lands and CI passes:

```bash
git tag phase-0-complete
git push --tags
```

### 3.2 Priyanshi & Sahil (one-time)

```bash
git clone https://github.com/<prasham-user>/ci-triage-env
cd ci-triage-env

# Verify Phase 0 is on main
git log --oneline | head -5  # should see phase-0 commits

# Set up the venv (this is the only time you "install" anything by hand)
uv sync --all-extras
# uv reads pyproject.toml + uv.lock, creates .venv/, installs everything

# Sanity check
uv run pytest -q       # all green
uv run python -c "import ci_triage_env; print(ci_triage_env.__version__)"

# Create your branch from main
git checkout -b branch-b/data-pipeline   # for Priyanshi
git checkout -b branch-c/reward-training # for Sahil
git push -u origin <your-branch>
```

Prasham creates his own branch:

```bash
git checkout -b branch-a/env-core
git push -u origin branch-a/env-core
```

### 3.3 After every `git pull`

```bash
uv sync --all-extras   # pick up any dependency changes
```

If `uv.lock` changed, `uv sync` re-resolves and updates `.venv/`. Skip this and you risk import errors from missing deps another teammate added.

---

## 4. Branch protection and merge rules

After Phase 0, configure on GitHub (Prasham):

- `main` requires PR + 1 approval + green CI
- `main` requires linear history (no merge commits — use squash-merge or rebase-merge)
- All branches must pass: `pytest -q tests/` (full suite) and `ruff check src/`

CI runs on every push and every PR. Defined in `.github/workflows/ci.yml` (lands in Phase 0).

---

## 5. Build timeline

Times are *target* — slip is recoverable as long as merge gates hit roughly on schedule.

### Day 0 (today / pre-onsite, Saturday daytime)
- **Hour 0–2** (Prasham): Phase 0 — foundation on `main`. All teammates wait.
- **Hour 2** (Prasham): tag `phase-0-complete`, push. Notify team.
- **Hour 2–4** (everyone): clone, branch, read your branch's `overview.md` + `phase-1.md`.
- **Hour 4–end-of-day** (everyone): execute first phase of each branch in parallel.

### Day 0 evening / Day 1 morning (Saturday night → Sunday morning)
- Each branch progresses through phases 2, 3.
- Smoke-test integration: Branch C's mock-env runs against schemas only, no real merge yet.
- **Gate-1 target: Sunday afternoon.** Branch A through A3 (episode lifecycle works), Branch B through B5 (200 scenarios serialized), Branch C through C2 (composite reward works on mock trajectories).

### Day 1 afternoon — Gate-1
- Branch A and B PR into `main`. Branch C rebases onto new `main`.
- Verify: from `main`, you can `python -m ci_triage_env.env.server`, `curl /reset` returns a real scenario, model can call tools, env returns observations.
- Tag: `gate-1-integration-passed`.

### Day 2 onsite (Saturday-Sunday at venue, compute credits arrive)
- Branch A finishes A5 (visualizer). A4 (counterfactual probe) is **deferred to v2** — see `plan/branch-a-env-core/phase-a4.md` for the dormant scaffolding.
- Branch C runs C3 (SFT trajectory generation), C4 (GRPO smoke test on mock + real scenarios).
- **Gate-2 target: Sunday end-of-day.** All branches merged. SFT training begins.

### Day 3 onsite (Monday, training day)
- Full GRPO run on HF compute.
- Branch C executes C5 (multi-baseline eval) and C6 (ablations).
- README.md gets `[FILL POST-TRAIN]` sections populated.
- Demo video recorded.

### Day 3 evening — submission
- Final commit to `main`.
- Push to HF Space, verify env loads.
- Submit URL.

---

## 6. Per-phase test discipline

**Every phase concludes by running `pytest -q tests/<phase-folder>` and `pytest -q tests/integration` and seeing all tests green. No phase is "done" until tests pass.**

Each phase doc specifies:
- The unit tests that must be added in that phase (file paths + behaviors covered).
- The smoke test the phase's deliverable must pass.
- The integration test (if any) that the phase enables others to write.

CI enforces: `pytest -q` must pass on every push to `branch-*/*`.

If you can't make tests pass and you're blocked: post in team chat with the failing test output, **do not merge**.

---

## 7. Merge protocol

### Before opening a PR

1. Rebase onto latest `main`: `git fetch origin && git rebase origin/main`
2. Resolve conflicts in *your* code, never in shared schemas without team sign-off.
3. Run full local test: `pytest -q && ruff check src/`
4. Push: `git push --force-with-lease`

### Opening a PR

```
Title: [Branch X] Phase X<N>: <short title>
Body:
  - Summary of what landed
  - Tests added (links to test files)
  - Schemas touched (yes/no — if yes, get explicit team approval)
  - Closes / advances which planning doc
```

### Reviewing a PR

- One other team member must review and approve.
- Reviewer checks: tests cover the deliverable, no schema breakage, ruff clean, no `[FILL]` markers left in committed code.

### Merging

- **Squash-merge only.** Linear history.
- Author handles the merge after approval.
- If CI flakes: rerun once. If still flaking, investigate, do not bypass.

### Hot-fix exception

- If a schema bug is blocking another branch: open a `hotfix/<issue>` branch off `main`, fast-track review, merge, all branches rebase.

---

## 8. Communication protocol

- Team chat (Discord/Slack/etc): primary sync channel.
- When something blocks another branch: **announce in chat first**, then start fixing.
- When a schema needs to change: **propose in chat, get explicit OK from all 3, then PR**.
- Daily 15-minute standup at start of each day: status, blockers, today's target.

---

## 9. Compute and API budget tracker

| Resource | Budget | Used by | Notes |
|---|---|---|---|
| HF compute | $90 (3 × $30) | Branch C, training | Triggered onsite |
| OpenAI API | $30 | Branch C, Phase C3 | `gpt-5-mini` bulk + `gpt-5` spot |
| Anthropic API | $0 | none | Reserved fallback |

Budget tracker: maintained in `plan/BUDGET-LOG.md`. Update after every paid run.

Phase C3 hard-stop: **stop trajectory generation at $25 spent**, $5 reserve for re-runs.

GRPO hard-stop: **abort run at $60 spent**, regardless of training progress.

---

## 10. What to do if everything is on fire

| Symptom | Action |
|---|---|
| Phase 0 not done by hour 3 | Prasham asks for help, splits Phase 0 into pieces, Priyanshi and Sahil each take a piece |
| Gate-1 missed by EOD Sunday | Reduce scenario corpus to 100, cut A5 visualizer to v2, ship a headless replay-from-trace JSON dump. |
| Gate-2 missed by Sunday EOD onsite | Reduce GRPO step count to 1000, skip ablations C6, ship the headline curve only |
| GRPO blowing budget | Switch to Qwen3.5-2B, cut group size from 8 → 4, cut steps from 3000 → 1500 |
| OpenAI budget exhausted before SFT done | Use whatever trajectories generated, even if 50; SFT is for format conditioning, not capability |
| Scenario corpus < 100 by Gate-1 | Hand-author additional scenarios using existing generators as templates; quality > quantity |
| HF Space won't load | Submit GitHub repo URL with clear deploy instructions in README; keynote allows GitHub for source |

---

## 11. Definition of done

The submission is "done" when **all** of:

1. `main` is green on CI.
2. HF Space loads, env responds to `reset` and `step` with valid scenarios.
3. README.md has every `[FILL POST-TRAIN]` block populated.
4. Training reward curve PNG is committed to repo and embedded in README.
5. Multi-baseline comparison table is committed and embedded in README.
6. At least one ablation plot is committed.
7. Demo video is published, link is in README.
8. Colab notebook runs end-to-end on a fresh runtime (Sahil verifies on a separate Google account).
9. `openenv.yaml` validates against the latest OpenEnv schema.
10. Submission URL is posted to the hackathon submission portal.

If you can check 8/10 by submission deadline, submit anyway with what you have. **A submitted v0.8 beats a polished v1.0 missed deadline.**

---

## 12. Order of operations summary (the one-page version)

```
DAY 0 SAT
  Prasham: Phase 0 → tag phase-0-complete, push to main
  Priyanshi & Sahil: clone, branch, start their phase 1
  All: progress through phases independently
  Smoke-test: Branch C's mock-env tests pass

DAY 1 SUN AFTERNOON  ← GATE 1
  Branch A: A1, A2, A3 done
  Branch B: B1–B5 done (200 scenarios published to HF dataset)
  Branch C: C1, C2 done (composite reward works on mock data)
  → All three PR into main, rebase, integrate, tag gate-1

DAY 2 SUN ONSITE → MON
  Branch A: A5 (visualizer)         [A4 counterfactual is deferred to v2]
  Branch C: C3 (SFT data gen) → C4 (GRPO smoke test)
  → Gate-2 merge

DAY 3 MON
  Full GRPO run (~30 hours wall but parallel-safe to monitor)
  Branch C: C5 (eval), C6 (ablations)
  README population
  Demo video record
  Submit
```

---

## 13. Standing rules

- **No `[TODO]` left on `main`.** Either land it or remove the placeholder.
- **No tests skipped without justification in PR description.**
- **No schema changes without team OK.**
- **No `git push --force` to `main`. `--force-with-lease` to your own branch only.**
- **Commit messages: `<type>(<branch>): <imperative>`** e.g. `feat(branch-a): add MCP tool registry`.
- **Update BUDGET-LOG.md after every paid API/compute run.**
- **If you change a schema, you bump the schema version constant and update all callers.**
- **No LLM-as-judge in any reward. Ever. Hard rule from the FAQ.**
- **All Python invocations through `uv run`.** Never `pip install` directly. Never run `python` against system Python. If a teammate adds a dep, `git pull && uv sync`.
- **Commit `uv.lock` on every change** that adds/removes/upgrades a dependency. Treat it as source.
