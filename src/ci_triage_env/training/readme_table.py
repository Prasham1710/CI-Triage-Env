"""Generate the Results section markdown table from eval.csv."""

from __future__ import annotations


def generate_results_table(df) -> str:
    """Return a GitHub-flavoured markdown table for README's Results section."""
    summary = df.groupby("baseline").agg(
        diagnosis_acc=("diagnosis_correct", "mean"),
        action_qual=("action_quality", "mean"),
        cost=("total_cost", "mean"),
        steps=("tool_call_count", "mean"),
        reward=("total_reward", "mean"),
    )
    return _df_to_markdown(summary, floatfmt=".3f")


def _df_to_markdown(df, floatfmt: str = ".3f") -> str:
    """Format a pandas DataFrame as a GFM table without requiring tabulate."""
    index_name = df.index.name or "baseline"
    cols = list(df.columns)
    header = [index_name] + cols
    sep = [":---"] + ["---:" for _ in cols]
    rows = []
    for idx, row in df.iterrows():
        cells = [str(idx)] + [
            format(v, floatfmt) if isinstance(v, float) else str(v)
            for v in row
        ]
        rows.append(cells)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]
    return "\n".join(lines)
