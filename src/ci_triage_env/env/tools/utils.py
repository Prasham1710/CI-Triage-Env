"""Shared helpers for Phase A2 tool implementations.

- ``deterministic_rng(seed, step, tool_name)``: reproducible RNG seeded from the
  episode seed, the step index, and the tool name. Use in any handler that
  needs randomness (e.g. log line shuffling) so the same episode always
  produces the same outputs.

- ``args_hash(args)``: short stable hash of an args dict, useful for keying
  tool-call repeats (the redundancy-penalty in the reward layer interprets
  identical hashes as the same call).

- ``SchemaValidatedHandler``: ``ToolHandler`` mixin that validates ``args`` via
  ``jsonschema`` against the frozen ``MCPToolDef.args_schema`` from
  ``schemas/tools.py``. Concrete handlers implement ``call``.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import ClassVar

import jsonschema

from ci_triage_env.env.tools.base import ToolHandler
from ci_triage_env.schemas.tools import ALL_TOOLS

_TOOL_DEFS = {t.name: t for t in ALL_TOOLS}


def deterministic_rng(seed: int, step: int, tool_name: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{step}:{tool_name}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def args_hash(args: dict) -> str:
    return hashlib.sha1(
        json.dumps(args or {}, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


class SchemaValidatedHandler(ToolHandler):
    """Concrete-handler base that validates args against the MCP schema."""

    name: ClassVar[str] = ""
    cost_unit: ClassVar[float] = 0.0

    def validate_args(self, args: dict) -> None:
        spec = _TOOL_DEFS.get(self.name)
        if spec is None:
            raise ValueError(f"unknown tool: {self.name}")
        try:
            jsonschema.validate(instance=args or {}, schema=spec.args_schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"invalid args for {self.name}: {exc.message}") from exc
