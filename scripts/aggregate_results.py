import argparse
import json
from pathlib import Path

import pandas as pd

from decorrelated_ensemble.evaluation.artifacts import sha256, verify_manifest, write_json
from decorrelated_ensemble.statistics import paired_task_comparison


def aggregate(run, predictions_only=False):
    verify_manifest(run, allow_missing_models=predictions_only)
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
    counts = completed.groupby(["dataset", "task", "method"]).size().rename("completed_splits")
    means = means.merge(counts.reset_index(), on=["dataset", "task", "method"])
    expected = (
        records.groupby("dataset")
        .apply(lambda group: len(group[["seed", "fold"]].drop_duplicates()), include_groups=False)
        .rename("expected_splits")
    )
    means = means.merge(expected.reset_index(), on="dataset")
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
        ("random_subspace_affine", "random_subspace"),
        ("top_no_cert_affine", "top_no_cert"),
        ("coerror_no_cert_affine", "coerror_no_cert"),
        ("affine_profiled", "coerror_no_cert_affine"),
        ("affine_profiled", "top_no_cert_affine"),
        ("affine_profiled", "random_subspace_affine"),
        ("affine_profiled", "random_forest"),
    ]
    for task, metric, higher in [
        ("binary", "auroc", True),
        ("regression", "rmse", False),
        ("regression", "normalized_squared_loss", False),
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
            "models_verified": not predictions_only,
            "report_script_sha256": sha256(Path(__file__)),
            "comparison_pairs": pairs,
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        help="Allow explicitly omitted model.joblib files; verify all decision/prediction artifacts",
    )
    args = parser.parse_args()
    print(aggregate(Path("results/runs") / args.run_id, args.predictions_only))
