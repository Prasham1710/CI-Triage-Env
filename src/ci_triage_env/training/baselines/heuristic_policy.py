"""HeuristicPolicy — rule-based classifier. The strong simple baseline."""

from __future__ import annotations

from ci_triage_env.data.clustering.classifier import RuleBasedClassifier


class HeuristicPolicy:
    """Hand-coded rule-based classifier.

    Executes a fixed investigation plan, then classifies via keyword matching on logs.
    """

    name = "heuristic"

    INVESTIGATION_PLAN: list[tuple[str, dict]] = [
        ("read_logs", {"scope": "full"}),
        ("query_flake_history", {"test_name": "failing_test"}),
        ("recent_commits", {"branch": "main"}),
        ("cluster_metrics", {"metric": "queue_depth"}),
    ]

    def act(self, obs, history: list) -> dict:
        if len(history) < len(self.INVESTIGATION_PLAN):
            tool, args = self.INVESTIGATION_PLAN[len(history)]
            return {"tool_name": tool, "args": args}
        return self._classify_from_history(history)

    def _classify_from_history(self, history: list) -> dict:
        all_text = " ".join(str(h.get("output", "")) for h in history)
        record_proxy = type("R", (), {"log_text": all_text})()
        family, conf = RuleBasedClassifier().classify(record_proxy)
        if family == "unknown":
            family = "ambiguous"
            conf = 0.4
        return {
            "action_type": "submit_diagnosis",
            "diagnosis": family,
            "confidence": conf,
            "secondary_actions": self._secondary_for(family),
        }

    def _secondary_for(self, family: str) -> list[dict]:
        if family == "real_bug":
            return [
                {
                    "name": "file_bug",
                    "args": {
                        "title": "auto",
                        "summary": "auto",
                        "owner": "auto",
                        "severity": "high",
                    },
                }
            ]
        if family in ("race_flake", "timing_flake"):
            return [
                {
                    "name": "quarantine_test",
                    "args": {"test_name": "failing_test", "reason": "flaky"},
                }
            ]
        if family.startswith("infra_"):
            return [{"name": "rerun_test", "args": {"test_name": "failing_test"}}]
        if family == "dependency_drift":
            return [
                {
                    "name": "ping_owner",
                    "args": {"owner": "deps", "message": "Dependency drift detected in CI"},
                }
            ]
        return []
