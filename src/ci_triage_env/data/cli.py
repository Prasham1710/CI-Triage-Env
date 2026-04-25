"""``python -m ci_triage_env.data.cli`` entrypoint.

Phase B1 ships ``load <dataset>``; B3 adds ``cluster``; B5 will extend this
with ``generate`` and ``publish-hf`` without breaking the surface here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ci_triage_env.data.clustering import ArchetypeExtractor, classify_all
from ci_triage_env.data.datasets import LOADER_REGISTRY
from ci_triage_env.data.datasets.cache import load_all_cached
from ci_triage_env.data.mining import (
    DEFAULT_REPOS,
    GhAuthError,
    GitHubActionsLogScraper,
    check_gh_auth,
)

DEFAULT_GHA_OUT_DIR = Path("data_artifacts/datasets_cache/github_actions")


def cmd_mine(args: argparse.Namespace) -> int:
    if not args.skip_auth_check:
        try:
            check_gh_auth()
        except GhAuthError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    scraper_kwargs: dict = {"rate_limit_per_min": args.rate_limit}
    if args.cache_dir:
        scraper_kwargs["cache_dir"] = Path(args.cache_dir)
    scraper = GitHubActionsLogScraper(**scraper_kwargs)

    repos = [args.repo] if args.repo else list(DEFAULT_REPOS)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_GHA_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for repo in repos:
        repo_records = list(scraper.mine_repo(repo, count=args.count))
        for record in repo_records:
            (out_dir / f"{record.record_id}.json").write_text(record.model_dump_json())
        total += len(repo_records)
        print(f"  {repo}: {len(repo_records)} records")
    print(f"mined {total} records into {out_dir}")
    return 0


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


def cmd_cluster(args: argparse.Namespace) -> int:
    records = load_all_cached()
    if not records:
        print("warning: no cached records found — run `cli load <dataset>` first", file=sys.stderr)

    api_key: str | None = os.environ.get("OPENAI_API_KEY") if not args.no_llm else None
    by_family = classify_all(records, openai_api_key=api_key)

    for family, recs in sorted(by_family.items()):
        print(f"  {family}: {len(recs)} records")

    extractor = ArchetypeExtractor()
    out_dir = Path(args.out_dir) if args.out_dir else Path("data_artifacts/clustering")

    total_archetypes = 0
    for family, recs in by_family.items():
        if not recs:
            print(f"WARNING: {family} has no records — skipping archetype extraction")
            continue
        archetypes = extractor.extract(recs, family, n_archetypes=args.n_archetypes)
        family_dir = out_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "archetypes.json").write_text(
            json.dumps([a.model_dump() for a in archetypes], indent=2)
        )
        total_archetypes += len(archetypes)
        print(f"  wrote {len(archetypes)} archetypes → {family_dir}/archetypes.json")

    print(f"cluster complete: {total_archetypes} archetypes across {len(by_family)} families")
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

    p_mine = sub.add_parser(
        "mine",
        help="Mine failed GitHub Actions logs from public repos via the gh CLI.",
    )
    p_mine.add_argument(
        "--repo",
        default=None,
        help="owner/name (e.g. kubernetes/kubernetes). Omit to mine the default repo set.",
    )
    p_mine.add_argument("--count", type=int, default=30, help="Failed runs per repo (max).")
    p_mine.add_argument(
        "--rate-limit",
        type=int,
        default=60,
        help="Outbound gh calls per minute (default 60; cap on the 83/min auth'd limit).",
    )
    p_mine.add_argument("--cache-dir", default=None, help="Override raw-log cache directory.")
    p_mine.add_argument(
        "--out-dir",
        default=None,
        help="Override the FailureRecord output directory (default data_artifacts/datasets_cache/github_actions).",
    )
    p_mine.add_argument(
        "--skip-auth-check",
        action="store_true",
        help="Skip the `gh auth status` precheck (use only when calling against a recorded fixture).",
    )
    p_mine.set_defaults(func=cmd_mine)

    p_cluster = sub.add_parser(
        "cluster",
        help="Classify cached FailureRecords into families and extract archetypes.",
    )
    p_cluster.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for archetype JSON files (default data_artifacts/clustering).",
    )
    p_cluster.add_argument(
        "--n-archetypes",
        type=int,
        default=4,
        help="Max archetypes to extract per family (default 4).",
    )
    p_cluster.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM fallback even if OPENAI_API_KEY is set.",
    )
    p_cluster.set_defaults(func=cmd_cluster)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
