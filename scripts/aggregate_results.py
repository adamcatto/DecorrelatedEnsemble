import argparse
import json
from pathlib import Path

import pandas as pd

from decorrelated_ensemble.evaluation.artifacts import verify_manifest, write_json
from decorrelated_ensemble.statistics import paired_task_comparison


def aggregate(run):
    verify_manifest(run)
    status = json.loads((run / "status.json").read_text())
    if status["state"] != "complete":
        raise ValueError("Refusing to aggregate an incomplete/failed run")
    records = pd.read_csv(run / "records.csv")
    output = Path("results/summaries") / run.name
    output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / "per_split.csv", index=False)
    coverage = records.groupby(["dataset", "method", "status"]).size().rename("count").reset_index()
    coverage.to_csv(output / "coverage.csv", index=False)
    completed = records[records.status == "complete"]
    numeric = completed.select_dtypes("number").columns.difference(["seed", "fold"])
    means = completed.groupby(["dataset", "task", "method"])[numeric].mean().reset_index()
    means.to_csv(output / "per_task.csv", index=False)
    comparisons = []
    pairs = [
        ("coerror", "top_quality"),
        ("coerror", "abs_correlation"),
        ("coerror", "random_forest"),
        ("coerror_no_cert", "top_no_cert"),
        ("top_quality", "top_no_cert"),
        ("shrinkage", "coerror"),
        ("coerror", "caruana"),
        ("coerror_no_cert", "random_forest"),
    ]
    for task, metric, higher in [
        ("binary", "auroc", True),
        ("regression", "rmse", False),
        ("multiclass", "log_loss", False),
    ]:
        subset = records[records.task == task]
        for left, right in pairs:
            if metric in subset and len(subset):
                comparisons.append(paired_task_comparison(subset, left, right, metric, higher))
    write_json(output / "comparisons.json", comparisons)
    pd.DataFrame(
        [{k: v for k, v in c.items() if not isinstance(v, dict)} for c in comparisons]
    ).to_csv(output / "comparisons.csv", index=False)
    primary = []
    for task, metric in [
        ("binary", "auroc"),
        ("regression", "normalized_squared_loss"),
        ("multiclass", "log_loss"),
    ]:
        frame = means[means.task == task].copy()
        if len(frame):
            frame["rank"] = frame.groupby("dataset")[metric].rank(ascending=task != "binary")
            ranks = frame.groupby("method").agg(
                mean_rank=("rank", "mean"), tasks=("dataset", "nunique")
            )
            ranks["task_type"] = task
            primary.append(ranks.reset_index())
    if primary:
        pd.concat(primary).to_csv(output / "ranks.csv", index=False)
    write_json(
        output / "provenance.json",
        {
            "run_id": run.name,
            "status": status,
            "raw_manifest": json.loads((run / "manifest.json").read_text()),
            "coverage_warning": "Ranks and paired comparisons can use different task coverage; inspect coverage.csv",
            "development_only": True,
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    print(aggregate(Path("results/runs") / args.run_id))
