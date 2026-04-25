# Phase B3 — Failure Clustering

**Owner:** Branch B.
**Prerequisite:** B1 + B2 merged.
**Estimated time:** 2–3 hours.
**Budget:** $5 max OpenAI API.

---

## Outcome

Cluster all mined `FailureRecord`s into the 7 failure families and extract per-family archetypes (template log shapes that B4 generators will inflate). By end of phase:

1. `python -m ci_triage_env.data.cli cluster` runs the full pipeline.
2. Output: `data_artifacts/clustering/<family>/archetypes.json` — one file per family, listing 3–5 archetypal patterns with templated slots.
3. Hybrid approach: rule-based first (fast), LLM call only for residuals.
4. Coverage: each family has ≥ 3 archetypes; total OpenAI cost ≤ $5.
5. All B3 tests pass.

---

## Files to create

### `src/ci_triage_env/data/clustering/classifier.py`

```python
from ..datasets._base import FailureRecord

FAMILIES = [
    "real_bug", "race_flake", "timing_flake",
    "infra_network", "infra_resource", "dependency_drift",
    "ambiguous",
]

class RuleBasedClassifier:
    """Keyword + regex matching to classify obvious cases."""

    RULES: dict[str, list[re.Pattern]] = {
        "infra_resource": [
            re.compile(r"OOMKilled|out of memory|cannot allocate|killed by OS|137 SIGKILL"),
            re.compile(r"no space left on device|disk full|ENOSPC"),
            re.compile(r"too many open files|EMFILE"),
        ],
        "infra_network": [
            re.compile(r"DNS resolution|unable to resolve|getaddrinfo failed|connection refused"),
            re.compile(r"TLS handshake.*timeout|x509:.*certificate"),
            re.compile(r"socket: connection reset|EHOSTUNREACH|ENETUNREACH"),
        ],
        "race_flake": [
            re.compile(r"data race|race detected|WARNING: DATA RACE", re.IGNORECASE),
            re.compile(r"concurrent map writes|fatal error: concurrent"),
            re.compile(r"deadlock detected"),
        ],
        "timing_flake": [
            re.compile(r"deadline exceeded|context canceled|timeout exceeded"),
            re.compile(r"test timed out after \d+", re.IGNORECASE),
        ],
        "dependency_drift": [
            re.compile(r"Cargo\.lock|package-lock\.json|go\.sum.*conflict"),
            re.compile(r"npm ERR! peer dep|incompatible dependency"),
            re.compile(r"version (mismatch|conflict)"),
        ],
        # real_bug is the residual after others — handled below
        # ambiguous is usually multi-signal — handled below
    }

    def classify(self, record: FailureRecord) -> tuple[str, float]:
        """Returns (family, confidence). Confidence in [0, 1]."""
        text = record.log_text
        scores = {}
        for family, patterns in self.RULES.items():
            matches = sum(1 for p in patterns if p.search(text))
            if matches:
                scores[family] = matches / len(patterns)
        if not scores:
            return ("unknown", 0.0)
        # Multi-family hits = ambiguous
        hit_families = [f for f, s in scores.items() if s > 0.3]
        if len(hit_families) > 1:
            return ("ambiguous", min(scores.values()))
        family = max(scores, key=scores.get)
        return (family, scores[family])

class LLMClassifier:
    """Fallback for records the rule-based classifier marked 'unknown'."""

    SYSTEM_PROMPT = """You are a CI failure classifier. Given a failure log,
output exactly one label from: real_bug, race_flake, timing_flake,
infra_network, infra_resource, dependency_drift, ambiguous.

Choose ambiguous only if multiple causes are plausible and no single one dominates.
Respond with the label only, no explanation."""

    def __init__(self, api_key: str, model: str = "gpt-5-mini", budget_usd: float = 5.0):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.budget = budget_usd
        self.spent = 0.0

    def classify_batch(self, records: list[FailureRecord]) -> list[tuple[str, float]]:
        results = []
        for record in records:
            if self.spent >= self.budget:
                results.append(("unknown", 0.0))
                continue
            log_excerpt = record.log_text[:3000]   # cap context
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": log_excerpt},
                ],
                max_tokens=20,
            )
            label = response.choices[0].message.content.strip().lower()
            if label not in FAMILIES:
                label = "unknown"
            # Track approximate cost
            self.spent += self._estimate_cost(response)
            results.append((label, 0.7))   # LLM confidence
        return results

    def _estimate_cost(self, response) -> float:
        usage = response.usage
        # gpt-5-mini approx pricing — update with real numbers
        input_cost_per_M = 1.0
        output_cost_per_M = 4.0
        return (usage.prompt_tokens * input_cost_per_M + usage.completion_tokens * output_cost_per_M) / 1_000_000

def classify_all(records: list[FailureRecord], openai_api_key: str | None = None) -> dict[str, list[FailureRecord]]:
    rule_classifier = RuleBasedClassifier()
    by_family: dict[str, list[FailureRecord]] = {f: [] for f in FAMILIES}
    unknowns = []
    for r in records:
        family, conf = rule_classifier.classify(r)
        if family == "unknown":
            unknowns.append(r)
        else:
            by_family[family].append(r)
    if unknowns and openai_api_key:
        llm = LLMClassifier(openai_api_key)
        results = llm.classify_batch(unknowns)
        for record, (family, conf) in zip(unknowns, results):
            if family in by_family:
                by_family[family].append(record)
        print(f"LLM classified {len(unknowns)} residuals; spent ~${llm.spent:.2f}")
    return by_family
```

### `src/ci_triage_env/data/clustering/archetypes.py`

```python
class Archetype(BaseModel):
    """A templated log pattern for a family."""
    archetype_id: str
    family: str
    pattern_summary: str          # human-readable description
    log_template: str             # log text with {SLOT} placeholders
    slot_distributions: dict[str, list[str]]   # SLOT_NAME -> list of realistic values
    informative_tools_hint: list[str]
    minimal_evidence_hint: list[str]

class ArchetypeExtractor:
    """For each family, pick representative log samples and extract templates."""

    def extract(self, records: list[FailureRecord], family: str, n_archetypes: int = 4) -> list[Archetype]:
        # Strategy:
        # 1. Cluster within family by log similarity (TF-IDF + KMeans or simpler)
        # 2. For each cluster, pick the most representative log (closest to centroid)
        # 3. Extract template by replacing variable parts with slots:
        #    - PIDs, timestamps, paths, line numbers, test names
        # 4. Sample slot values from the cluster's variants
        ...
```

Templates for the `infra_resource` family example:

```json
{
  "archetype_id": "infra_resource_oom_kill_001",
  "family": "infra_resource",
  "pattern_summary": "OOM-killer terminated test process",
  "log_template": "[{TIMESTAMP}] kernel: Out of memory: Killed process {PID} ({PROCESS_NAME}) total-vm:{VM}kB, anon-rss:{RSS}kB\n[{TIMESTAMP}] systemd[1]: {SERVICE}.service: Main process exited, code=killed, status=9/KILL\nFAIL {TEST_NAME} ({DURATION}s)",
  "slot_distributions": {
    "TIMESTAMP": ["2024-..."], "PID": ["1234", "5678", "9001"],
    "PROCESS_NAME": ["pytest", "go test", "cargo test", "npm test"],
    "VM": ["2048576", "4194304"], "RSS": ["1957891", "3987234"],
    "SERVICE": ["test-runner", "ci-job"],
    "TEST_NAME": ["test_user_creation", "test_async_handler"],
    "DURATION": ["120.5", "60.0"]
  },
  "informative_tools_hint": ["read_logs:kernel", "cluster_metrics:5m"],
  "minimal_evidence_hint": ["cluster_metrics:5m"]
}
```

### Modify `src/ci_triage_env/data/cli.py`

Add `cluster` subcommand:

```python
def cmd_cluster(args):
    # Load all FailureRecords from cache
    records = load_all_cached()
    by_family = classify_all(records, openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Print classification summary
    for family, recs in by_family.items():
        print(f"{family}: {len(recs)} records")

    # Extract archetypes per family
    extractor = ArchetypeExtractor()
    out_dir = Path("data_artifacts/clustering")
    for family, recs in by_family.items():
        if not recs:
            print(f"WARNING: {family} has no records")
            continue
        archetypes = extractor.extract(recs, family, n_archetypes=4)
        family_dir = out_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "archetypes.json").write_text(
            json.dumps([a.model_dump() for a in archetypes], indent=2)
        )
        print(f"Wrote {len(archetypes)} archetypes for {family}")
```

---

## Implementation notes

- **Rule-based first.** Most failures match obvious patterns. Aim for rule-based to handle 70%+. LLM for the residual saves API budget.
- **The `ambiguous` family is hardest.** It needs records where multiple causes are plausible. Don't expect to find many from pure mined data — augment manually if needed (B4's ambiguous generator can also synthesize them from scratch).
- **Slot extraction.** Replace numeric tokens (PIDs, timestamps, durations), file paths, test names, hash-like strings. Keep structural log lines intact. A simple regex-based slotter is enough for v1.
- **LogHub integration.** LogHub records often lack the structural failure context that GitHub Actions logs have. Use them mainly for `infra_resource` and `infra_network` archetypes (kernel/system patterns) rather than as test-failure records.
- **Budget tracking.** After every 50 LLM calls, log spent. Hard stop at $5.

---

## Tests required (`tests/data/test_clustering.py`)

```python
def test_rule_based_oom():
    record = FailureRecord(log_text="kernel: Out of memory: Killed process 123", ...)
    family, conf = RuleBasedClassifier().classify(record)
    assert family == "infra_resource"
    assert conf > 0

def test_rule_based_race():
    record = FailureRecord(log_text="WARNING: DATA RACE detected", ...)
    family, _ = RuleBasedClassifier().classify(record)
    assert family == "race_flake"

def test_rule_based_unknown_returns_unknown():
    record = FailureRecord(log_text="generic test failure", ...)
    family, conf = RuleBasedClassifier().classify(record)
    assert family == "unknown"

def test_ambiguous_when_multiple_families_match():
    record = FailureRecord(log_text="OOM and DATA RACE both reported", ...)
    family, _ = RuleBasedClassifier().classify(record)
    assert family == "ambiguous"

def test_archetype_round_trip():
    """Archetype serializes and deserializes."""

def test_archetype_extraction_from_fixture():
    """Given 5 fixture FailureRecords, extract at least 1 archetype with slots."""

def test_classify_all_writes_per_family_files(tmp_path):
    """Running classify_all + extract puts files under each family dir."""

def test_llm_classifier_respects_budget(mock_openai):
    """LLMClassifier stops calling after budget exhausted."""
```

Use `pytest-mock` to stub OpenAI calls in tests.

---

## Smoke test (manual)

```bash
export OPENAI_API_KEY=sk-...
python -m ci_triage_env.data.cli cluster

# Verify
ls data_artifacts/clustering/
cat data_artifacts/clustering/infra_resource/archetypes.json | jq '.[0]'
```

Expected: 7 family directories, each with archetypes.json containing 3–5 archetypes.

---

## Open questions

1. **What if a family has zero records after clustering?** Hand-author a synthetic seed archetype using the team's CI experience. Document in `archetypes.json` metadata.
2. **TF-IDF + KMeans vs simpler hash-based bucketing for archetype clustering?** Start simple (group by first error line + token overlap). Upgrade only if archetypes look noisy.
3. **Update LLM model name.** `gpt-5-mini` is a placeholder — verify the current cheap reasoning model on OpenAI's dashboard at run time. Update pricing constants accordingly.

---

## What's NOT in this phase

- Generating full scenarios from archetypes (B4)
- Annotation of `informative_tools` / `minimal_evidence_set` as final per-scenario labels (B5; B3 produces only *hints*)
