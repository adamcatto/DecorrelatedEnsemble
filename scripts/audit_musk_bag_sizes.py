"""Registered count-only explanation control; never changes the main Musk run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from decorrelated_ensemble.evaluation.artifacts import sha256, verify_manifest, write_json


def group_summary(groups, y):
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    labels = np.array([y[np.flatnonzero(inverse == j)[0]] for j in range(len(unique))])
    if not np.array_equal(labels[inverse], y):
        raise ValueError("Molecule labels must be constant")
    return unique, counts, labels


def fit_count_baseline(train_counts, train_y, seed):
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1, max_iter=3000, random_state=seed)
    )
    model.fit(np.log1p(train_counts)[:, None], train_y)
    return model


def audit_bag_sizes(run_id="exp_006_large_musk2_v1"):
    root = Path("results/runs") / run_id
    verify_manifest(root)
    cfg = json.loads((root / "config.json").read_text())
    if cfg["datasets"][0]["id"] != "musk2" or cfg.get("group_aggregation") != "max":
        raise ValueError("This registered control requires the Musk max protocol")
    if json.loads((root / "status.json").read_text())["state"] != "complete":
        raise ValueError("Main predictions must be finalized before auxiliary analysis")
    output = Path("results/summaries/exp_006_musk_bag_sizes")
    output.mkdir(parents=True, exist_ok=True)
    records, predictions, fits = [], [], []
    for split_path in sorted(root.rglob("outer_split.npz")):
        parent = split_path.parent.parent
        groups, y = np.load(parent / "groups.npy"), np.load(parent / "y.npy")
        seed = int(parent.name.split("_")[-1])
        fold = int(split_path.parent.name.split("_")[-1])
        split = np.load(split_path)
        train_groups, train_counts, train_y = group_summary(
            groups[split["train"]], y[split["train"]]
        )
        test_groups, test_counts, test_y = group_summary(groups[split["test"]], y[split["test"]])
        if np.intersect1d(train_groups, test_groups).size:
            raise ValueError("Whole-molecule holdout failed")
        model = fit_count_baseline(train_counts, train_y, seed)
        scale, logistic = model.steps[0][1], model.steps[1][1]
        fits.append(
            {
                "seed": seed,
                "fold": fold,
                "train_groups": train_groups,
                "train_counts": train_counts,
                "train_y": train_y,
                "scaler_mean": scale.mean_,
                "scaler_scale": scale.scale_,
                "coefficient": logistic.coef_,
                "intercept": logistic.intercept_,
            }
        )
        probability = model.predict_proba(np.log1p(test_counts)[:, None])[:, 1]
        records.append(
            {
                "seed": seed,
                "fold": fold,
                "group_auroc": roc_auc_score(test_y, probability),
                "train_molecules": len(train_y),
                "test_molecules": len(test_y),
            }
        )
        predictions.extend(
            {
                "seed": seed,
                "fold": fold,
                "molecule": group,
                "count": int(count),
                "y": int(label),
                "prediction": float(p),
            }
            for group, count, label, p in zip(test_groups, test_counts, test_y, probability)
        )
    pd.DataFrame(records).to_csv(output / "per_split.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "test_predictions.csv", index=False)
    write_json(output / "fits.json", fits)
    write_json(
        output / "provenance.json",
        {
            "main_run": run_id,
            "main_manifest_sha256": sha256(root / "manifest.json"),
            "script_sha256": sha256(Path(__file__)),
            "primary": "molecule AUROC",
            "protocol": "One observation per molecule; log1p count, fold-local scale, C1 logistic; unchanged main outer folds; auxiliary structural-information control",
            "mean_group_auroc": float(np.mean([r["group_auroc"] for r in records])),
            "scope": "One dataset and three overlapping development folds; no population interval or optimized MIL claim",
        },
    )
    return output


if __name__ == "__main__":
    print(audit_bag_sizes())
