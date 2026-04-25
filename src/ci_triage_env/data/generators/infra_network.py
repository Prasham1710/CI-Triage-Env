"""InfraNetworkGenerator — scenario family: infra_network."""

from __future__ import annotations

import random

from ci_triage_env.data.clustering.archetypes import Archetype
from ci_triage_env.data.generators._helpers import (
    ArchetypedGenerator,
    _metric_samples,
    build_base_outputs,
    fake_short_sha,
    fake_timestamp,
    fill_template,
    make_failure_summary,
    pick_test_name,
    scenario_id_for,
)
from ci_triage_env.schemas.diagnosis import DiagnosisLabel
from ci_triage_env.schemas.scenario import (
    GroundTruth,
    Scenario,
    ScenarioMetadata,
    TerminalActionSpec,
    ToolOutput,
)


class InfraNetworkGenerator(ArchetypedGenerator):
    family_name = "infra_network"
    label = DiagnosisLabel.INFRA_NETWORK

    def informative_tools(self) -> list[str]:
        return ["read_logs", "cluster_metrics", "run_diagnostic"]

    def minimal_evidence_set(self) -> list[str]:
        return ["cluster_metrics"]

    def _default_archetypes(self) -> list[Archetype]:
        return [
            Archetype(
                archetype_id="infra_network_001",
                family="infra_network",
                pattern_summary="DNS resolution failure / connection refused",
                log_template=(
                    "dial tcp IP:{NUM}: connect: connection refused\n"
                    "getaddrinfo failed: Temporary failure in name resolution\n"
                    "FAILED {TEST_MODULE}::{TEST_FUNC} — network error\n"
                    "  ConnectionRefusedError: [Errno {NUM}] Connection refused"
                ),
                slot_distributions={
                    "TEST_MODULE": ["tests/integration/test_db", "tests/integration/test_cache"],
                    "TEST_FUNC": ["test_connect", "test_ping", "test_health_check"],
                    "NUM": ["111", "5432", "6379", "8080"],
                    "IP": ["10.0.0.1", "172.16.0.1", "192.168.1.10"],
                },
                informative_tools_hint=["read_logs:full", "cluster_metrics:network_latency", "run_diagnostic:network"],
                minimal_evidence_hint=["cluster_metrics:network_latency"],
            ),
            Archetype(
                archetype_id="infra_network_002",
                family="infra_network",
                pattern_summary="TLS handshake timeout / x509 certificate error",
                log_template=(
                    "TLS handshake timeout after {DURATION}s\n"
                    "x509: certificate signed by unknown authority\n"
                    "FAILED {TEST_MODULE}::{TEST_FUNC}\n"
                    "  ssl.SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
                ),
                slot_distributions={
                    "TEST_MODULE": ["tests/integration/test_tls", "tests/e2e/test_auth"],
                    "TEST_FUNC": ["test_secure_connect", "test_mtls", "test_cert_rotation"],
                    "DURATION": ["10", "30", "60"],
                },
                informative_tools_hint=["read_logs:full", "cluster_metrics:network_latency"],
                minimal_evidence_hint=["cluster_metrics:network_latency"],
            ),
        ]

    def generate(self, seed: int, source_log_hash: str | None = None) -> Scenario:
        rng = random.Random(seed)
        archetype = self._pick_archetype(rng)
        log_text = fill_template(archetype.log_template, archetype.slot_distributions, rng)
        test_name = pick_test_name(rng)

        summary = make_failure_summary(
            self.family_name, rng, test_name=test_name, log_excerpt=log_text
        )
        branch = summary.branch

        # Rerun might pass after transient network blip
        rerun_passes = (seed % 4 != 0)
        outputs = build_base_outputs(
            test_name, branch, rng,
            log_lines=log_text.splitlines(),
            rerun_passes=rerun_passes,
        )

        # --- informative overrides ---
        # Network latency elevated during the failure window
        outputs["cluster_metrics:network_latency"] = ToolOutput(
            tool_name="cluster_metrics",
            payload={"samples": _metric_samples(rng, "network_latency", elevated=True, n=6)},
            cost_units=0.003,
        )

        # run_diagnostic:network shows the problem
        outputs["run_diagnostic:network"] = ToolOutput(
            tool_name="run_diagnostic",
            payload={
                "ok": False,
                "details": {
                    "latency_ms": rng.randint(5000, 30000),
                    "dns_ok": False,
                    "error": "getaddrinfo ENOTFOUND internal-service.cluster.local",
                },
            },
            cost_units=0.005,
        )

        # Test had 100% pass rate before — not test-specific
        outputs[f"query_flake_history:{test_name}"] = ToolOutput(
            tool_name="query_flake_history",
            payload={"failure_count": 0, "pass_count": 60, "recent_failures": []},
            cost_units=0.002,
        )

        # No code changes
        outputs[f"recent_commits:{branch}"] = ToolOutput(
            tool_name="recent_commits",
            payload={"commits": [
                {"sha": fake_short_sha(rng), "author": "@ops-bot",
                 "msg": "chore: rotate service credentials", "files": [".env.template"]},
            ]},
            cost_units=0.002,
        )

        difficulty = rng.choice(["easy", "medium", "hard"])
        rationale = (
            "cluster_metrics:network_latency shows elevated network errors during the failure window. "
            "run_diagnostic:network confirms DNS/connectivity failure to internal services. "
            "query_flake_history shows the test always passed — not test-specific. "
            "No test-touching commits in recent_commits — infra blip, not a code bug."
        )

        return Scenario(
            schema_version="1.0",
            scenario_id=scenario_id_for(self.family_name, seed),
            family=self.family_name,
            seed=seed,
            ground_truth=GroundTruth(
                label=self.label,
                rationale=rationale,
                is_ambiguous=False,
                confidence_target=1.0,
            ),
            failure_summary=summary,
            tool_outputs=outputs,
            informative_tools=self.informative_tools(),
            minimal_evidence_set=self.minimal_evidence_set(),
            correct_terminal_action=TerminalActionSpec(
                primary="submit_diagnosis",
                args={
                    "diagnosis": self.label.value,
                    "confidence": 0.95,
                    "secondary_actions": [{"name": "rerun_test"}],
                },
                acceptable_alternatives=[
                    {"primary": "submit_diagnosis",
                     "args": {"diagnosis": "infra_network", "confidence": 0.9}},
                    {"primary": "submit_diagnosis",
                     "args": {"diagnosis": "infra_network", "confidence": 0.95,
                              "secondary_actions": [{"name": "ping_owner", "owner": "@infra-team"}]}},
                ],
            ),
            metadata=ScenarioMetadata(
                generator_version="1.0",
                generated_at=fake_timestamp(rng),
                source_log_hash=source_log_hash,
                difficulty=difficulty,
            ),
        )
