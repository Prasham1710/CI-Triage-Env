"""``python -m ci_triage_env.data.cli`` entrypoint.

Phase B1 ships ``load <dataset>``; B5 will extend this with ``generate`` and
``publish-hf`` subcommands without breaking the surface here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ci_triage_env.data.datasets import LOADER_REGISTRY


def cmd_load(args: argparse.Namespace) -> int:
    loader_cls = LOADER_REGISTRY[args.dataset]
    kwargs: dict = {}
    if args.data_path:
        kwargs["data_path"] = Path(args.data_path)
    if args.cache_dir:
        kwargs["cache_dir"] = Path(args.cache_dir)
    loader = loader_cls(**kwargs)

    if args.force and loader.cache_dir.exists():
        for path in loader.cache_dir.glob("*.json"):
            path.unlink()

    records = list(loader.fetch())
    written = loader.cache_records(records)
    print(f"loaded {len(records)} records from {args.dataset}; cached {written} into {loader.cache_dir}")
    if args.summary:
        print(json.dumps(loader.info(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ci_triage_env.data.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_load = sub.add_parser("load", help="Load a public dataset into the local cache.")
    p_load.add_argument("dataset", choices=sorted(LOADER_REGISTRY.keys()))
    p_load.add_argument(
        "--data-path",
        default=None,
        help="Local artifact path (overrides the dataset's env-var fallback).",
    )
    p_load.add_argument("--cache-dir", default=None, help="Override the cache directory.")
    p_load.add_argument(
        "--force",
        action="store_true",
        help="Wipe the cache before fetching.",
    )
    p_load.add_argument(
        "--summary",
        action="store_true",
        help="Print a JSON summary (label distribution, count) after loading.",
    )
    p_load.set_defaults(func=cmd_load)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
