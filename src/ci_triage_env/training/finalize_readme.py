"""Populate README's [FILL POST-TRAIN] markers after training is complete.

Run after eval.csv, ablations.csv, and plots/ are all present:
    python -c "from ci_triage_env.training.finalize_readme import populate_readme; populate_readme()"
"""

from __future__ import annotations

from pathlib import Path


def populate_readme(
    template_path: Path = Path("README.md"),
    eval_csv: Path = Path("data_artifacts/results/eval.csv"),
    ablation_csv: Path = Path("data_artifacts/results/ablations.csv"),
    plots_dir: Path = Path("data_artifacts/results/plots/"),
) -> int:
    """Fill [FILL …] markers in README.md in-place.

    Args:
        template_path: Path to README.md.
        eval_csv: Path to the master eval CSV from Phase C5.
        ablation_csv: Path to the ablation results CSV from Phase C6.
        plots_dir: Directory containing PNG plots.

    Returns:
        Number of markers replaced.
    """
    import pandas as pd

    from ci_triage_env.training.readme_table import generate_results_table

    text = template_path.read_text()
    replaced = 0

    # 1. Results table
    if eval_csv.exists():
        df_eval = pd.read_csv(eval_csv)
        table_md = generate_results_table(df_eval)
        marker = "[FILL: 5-row × 6-metric table]"
        if marker in text:
            text = text.replace(marker, table_md)
            replaced += 1

    # 2. Embed plot images — replace [FILL: <stem with spaces>] with markdown img tags
    if plots_dir.exists():
        for png in sorted(plots_dir.glob("*.png")):
            stem_words = png.stem.replace("_", " ")
            marker = f"[FILL: {stem_words}]"
            rel = png.relative_to(template_path.parent)
            embed = f"![{png.stem}]({rel})"
            if marker in text:
                text = text.replace(marker, embed)
                replaced += 1

    # 3. Remove any remaining generic [FILL POST-TRAIN] or [FILL] markers
    #    by replacing them with a placeholder so the README stays valid.
    import re
    generic = re.compile(r"\[FILL[^\]]*\]")
    remaining = generic.findall(text)
    if remaining:
        print(f"WARNING: {len(remaining)} unfilled marker(s) remain: {remaining[:5]}")

    template_path.write_text(text)
    return replaced
