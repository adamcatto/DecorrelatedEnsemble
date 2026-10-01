import argparse
import json
from pathlib import Path

import pandas as pd


def make_note(run_id):
    directory = Path("results/summaries") / run_id
    records = pd.read_csv(directory / "per_split.csv")
    tasks = pd.read_csv(directory / "per_task.csv")
    comparisons = json.loads((directory / "comparisons.json").read_text())
    total = len(records[["dataset", "seed", "fold"]].drop_duplicates())
    null = (
        records[records.dataset == "null_binary"]
        .dropna(subset=["certified_count"])
        .drop_duplicates(["seed", "fold"])
    )
    screened = records[records.method == "coerror"]
    bi = int(((screened.task == "binary") & (screened.status == "complete")).sum())
    reg = int(((screened.task == "regression") & (screened.status == "complete")).sum())
    vals = {
        "ExpOneOuterFolds": str(total),
        "ExpOneBinaryFeasible": str(bi),
        "ExpOneRegressionFeasible": str(reg),
        "ExpOneNullMin": str(int(null.certified_count.min())),
        "ExpOneNullMax": str(int(null.certified_count.max())),
    }
    for suffix, method in [
        ("Coerror", "coerror_no_cert"),
        ("Forest", "random_forest"),
        ("Catboost", "catboost"),
    ]:
        row = tasks[(tasks.dataset == "additive_regression") & (tasks.method == method)].iloc[0]
        vals["ExpOneAdditive" + suffix] = f"{row.normalized_squared_loss:.3f}"
    c = next(
        c
        for c in comparisons
        if c["left"] == "coerror_no_cert" and c["right"] == "top_no_cert" and c["metric"] == "auroc"
    )
    vals["ExpOneAucGain"] = f"{c['mean_advantage']:.4f}"
    facts = Path("paper/tables") / (run_id + "_facts.tex")
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text(
        "% Generated from frozen experiment artifacts; do not edit.\n"
        + "\n".join(
            chr(92) + "newcommand{" + chr(92) + key + "}{" + value + "}"
            for key, value in vals.items()
        )
        + "\n"
    )
    counts = records.groupby("status").size().to_dict()
    lines = [
        f"# {run_id}: generated factual note",
        "",
        f"Outer folds: {total}. Status counts: {counts}.",
        "",
        f"Screened co-error feasible binary splits: {bi}; regression splits: {reg}.",
        f"Null binary survivors per 100 candidates: {null.certified_count.astype(int).tolist()}.",
        "",
        "## Paired task effects (positive = left better)",
        "",
    ]
    for result in comparisons:
        if result["paired_tasks"]:
            lines.append(
                f"- {result['left']} vs {result['right']}, {result['metric']}: "
                f"{result['mean_advantage']:.5f}, descriptive interval "
                f"[{result['ci_lower']:.5f}, {result['ci_upper']:.5f}], "
                f"{result['paired_tasks']} tasks / {result['paired_splits']} shared splits."
            )
    lines += [
        "",
        "Intervals resample observed synthetic regimes after pairing splits; these are exploratory,",
        "unadjusted intervals without a real-world task-population interpretation. Inspect coverage.csv.",
        "Interpretation and follow-up decisions belong in the research log and experiment README.",
    ]
    (directory / "research_note.md").write_text("\n".join(lines) + "\n")
    return facts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    print(make_note(parser.parse_args().run_id))
