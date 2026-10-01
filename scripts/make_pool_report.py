"""Audit nested candidate pools and report fixed-K search scaling."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decorrelated_ensemble.evaluation.artifacts import sha256, write_json


def verify_prefixes(roots):
    max_difference = 0.0
    checked = 0
    for small, large in pairwise(roots):
        cfg_small = json.loads((small / "config.json").read_text())
        cfg_large = json.loads((large / "config.json").read_text())
        b = cfg_small["candidates"]["B"]
        if b >= cfg_large["candidates"]["B"]:
            raise ValueError("Supply runs in increasing B order")
        cfg_small.pop("id")
        cfg_large.pop("id")
        cfg_small["candidates"].pop("B")
        cfg_large["candidates"].pop("B")
        if cfg_small != cfg_large:
            raise ValueError("Scaling configs differ beyond B and experiment ID")
        small_folds = sorted(p.relative_to(small) for p in small.rglob("oof.npz"))
        large_folds = sorted(p.relative_to(large) for p in large.rglob("oof.npz"))
        if small_folds != large_folds:
            raise ValueError("Scaling runs have different folds")
        for relative in small_folds:
            a, z = small / relative.parent, large / relative.parent
            if sha256(a.parent / "X.pkl") != sha256(z.parent / "X.pkl"):
                raise ValueError("Scaling feature data differs")
            if not np.array_equal(np.load(a.parent / "y.npy"), np.load(z.parent / "y.npy")):
                raise ValueError("Scaling labels differ")
            specs_a = json.loads((a / "candidates.json").read_text())
            specs_z = json.loads((z / "candidates.json").read_text())
            if specs_a != specs_z[:b]:
                raise ValueError("Candidate pool prefix mismatch")
            for name in ["outer_split.npz", "oof.npz"]:
                arr_a, arr_z = np.load(a / name), np.load(z / name)
                for key in arr_a.files:
                    expected = arr_z[key][:, :b] if key == "predictions" else arr_z[key]
                    if not np.array_equal(arr_a[key], expected):
                        raise ValueError(f"Non-identical pool OOF/split prefix: {name}/{key}")
            if (a / "inner_splits.json").read_text() != (z / "inner_splits.json").read_text():
                raise ValueError("Inner splits differ")
            cert_a, cert_z = np.load(a / "certification.npz"), np.load(z / "certification.npz")
            for key in cert_a.files:
                expected = cert_z[key][:, :b] if key == "bootstrap" else cert_z[key][:b]
                difference = float(np.max(np.abs(cert_a[key].astype(float) - expected)))
                max_difference = max(max_difference, difference)
                if difference > 1e-12:
                    raise ValueError("Bootstrap/quality prefix mismatch")
            checked += 1
    return {
        "adjacent_pool_fold_pairs_checked": checked,
        "oof_predictions_exactly_identical": True,
        "maximum_bootstrap_quality_difference": max_difference,
    }


def make_report(run_ids):
    roots = [Path("results/runs") / name for name in run_ids]
    summaries = [Path("results/summaries") / name for name in run_ids]
    for root in roots:
        if json.loads((root / "status.json").read_text())["state"] != "complete":
            raise ValueError("Refusing incomplete pool study")
    audit = verify_prefixes(roots)
    out = Path("results/summaries/exp_003_pool_scaling_v1")
    figures = Path("paper/figures/exp_003_pool_scaling_v1")
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    splits, tasks = [], []
    for root, summary in zip(roots, summaries):
        b = json.loads((root / "config.json").read_text())["candidates"]["B"]
        splits.append(pd.read_csv(summary / "per_split.csv").assign(pool_B=b))
        tasks.append(pd.read_csv(summary / "per_task.csv").assign(pool_B=b))
    splits, tasks = pd.concat(splits), pd.concat(tasks)
    splits.to_csv(out / "per_split.csv", index=False)
    tasks.to_csv(out / "per_task.csv", index=False)
    write_json(out / "prefix_audit.json", audit)
    plt.rcParams.update(
        {"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    methods = [
        "random_subspace",
        "top_no_cert",
        "coerror_no_cert",
        "shrinkage_no_cert",
        "coerror_no_cert_affine",
        "affine_profiled",
        "random_forest",
    ]
    labels = ["RS", "Top", "CoErr", "Shrink .5", "CoErr+aff", "APCE", "RF256"]
    colors = ["#808080", "#b15f24", "#1e77a9", "#97549f", "#268847", "#d2494e", "#161616"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.2), layout="constrained")
    for ax, (dataset, frame) in zip(axes.flat, sorted(tasks.groupby("dataset"))):
        metric = "auroc" if dataset.endswith("binary") else "normalized_squared_loss"
        for method, label, color in zip(methods, labels, colors):
            rows = frame[frame.method == method].sort_values("pool_B")
            if len(rows):
                ax.plot(rows.pool_B, rows[metric], "o-", label=label, color=color, markersize=4)
        ax.set(
            xscale="log",
            xticks=[30, 100, 300],
            title=dataset.replace("_", " "),
            ylabel="AUROC ↑" if metric == "auroc" else "MSE/null MSE ↓",
            xlabel="Candidate B; K=8",
        )
        ax.set_xticklabels(["30", "100", "300"])
    axes.flat[1].legend(fontsize=7)
    axes.flat[3].legend(fontsize=7)
    fig.savefig(figures / "performance_vs_B.pdf")
    fig.savefig(figures / "performance_vs_B.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.2), layout="constrained")
    for ax, (dataset, frame) in zip(axes.flat, sorted(tasks.groupby("dataset"))):
        test_metric = "brier" if dataset.endswith("binary") else "mse"
        for method, label, color in zip(methods[1:6], labels[1:6], colors[1:6]):
            rows = frame[frame.method == method].sort_values("pool_B")
            if not len(rows):
                continue
            oof_metric = (
                "oof_calibrated_squared_loss"
                if method.endswith("affine") or method == "affine_profiled"
                else "oof_squared_loss"
            )
            ax.plot(rows.pool_B, rows[oof_metric], "o--", color=color, alpha=0.7)
            ax.plot(rows.pool_B, rows[test_metric], "o-", color=color, label=label)
        ax.set(
            xscale="log",
            xticks=[30, 100, 300],
            title=dataset.replace("_", " "),
            ylabel="Squared loss; solid=test, dashed=OOF",
            xlabel="Candidate B; K=8",
        )
        ax.set_xticklabels(["30", "100", "300"])
    axes.flat[1].legend(fontsize=7)
    axes.flat[3].legend(fontsize=7)
    fig.savefig(figures / "oof_test_vs_B.pdf")
    plt.close(fig)
    screening = splits.dropna(subset=["certified_count"]).drop_duplicates(
        ["dataset", "seed", "fold", "pool_B"]
    )
    screening.to_csv(out / "screening.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), layout="constrained")
    for dataset, frame in screening[screening.dataset.str.startswith("null_")].groupby("dataset"):
        rows = frame.groupby("pool_B")[["certified_count", "certification_rate"]].mean()
        axes[0].plot(rows.index, rows.certified_count, "o-", label=dataset.removeprefix("null_"))
        axes[1].plot(rows.index, rows.certification_rate, "o-")
    for method, label, color in zip(methods[1:6], labels[1:6], colors[1:6]):
        rows = tasks[tasks.method == method].groupby("pool_B").train_cpu_seconds.mean()
        if len(rows):
            axes[2].plot(rows.index, rows, "o-", label=label, color=color)
    for ax in axes:
        ax.set(xscale="log", xticks=[30, 100, 300], xlabel="Candidate B; K=8")
        ax.set_xticklabels(["30", "100", "300"])
    axes[0].set_ylabel("Mean null-screen survivors")
    axes[1].set_ylabel("Mean null-screen rate")
    axes[2].set_ylabel("Standalone search/train CPU seconds")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=6)
    fig.savefig(figures / "screening_compute_vs_B.pdf")
    plt.close(fig)
    effects = []
    for task, metric in [("binary", "auroc"), ("regression", "normalized_squared_loss")]:
        for method in methods:
            frame = splits[
                (splits.task == task) & (splits.method == method) & (splits.status == "complete")
            ]
            a = frame[frame.pool_B == 30].merge(
                frame[frame.pool_B == 300], on=["dataset", "seed", "fold"], suffixes=("_30", "_300")
            )
            a["advantage_300"] = a[metric + "_300"] - a[metric + "_30"]
            if task == "regression":
                a["advantage_300"] *= -1
            for dataset, rows in a.groupby("dataset"):
                effects.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "advantage_B300_vs_B30": rows.advantage_300.mean(),
                        "paired_splits": len(rows),
                    }
                )
    pd.DataFrame(effects).to_csv(out / "endpoint_effects.csv", index=False)
    facts = {}
    pool_words = {30: "Thirty", 100: "Hundred", 300: "ThreeHundred"}
    for task in ["binary", "regression"]:
        for b in [30, 100, 300]:
            counts = screening[
                (screening.dataset == "null_" + task) & (screening.pool_B == b)
            ].certified_count
            facts[f"ExpThreeNull{task.title()}B{pool_words[b]}Min"] = str(int(counts.min()))
            facts[f"ExpThreeNull{task.title()}B{pool_words[b]}Max"] = str(int(counts.max()))
    table = tasks.pivot(
        index=["dataset", "pool_B"], columns="method", values="normalized_squared_loss"
    )
    for b in [30, 100, 300]:
        for suffix, method in [
            ("Coerror", "coerror_no_cert"),
            ("Affine", "coerror_no_cert_affine"),
            ("Profiled", "affine_profiled"),
        ]:
            facts[f"ExpThreeAdditive{suffix}B{pool_words[b]}"] = (
                f"{table.loc[('additive_regression', b), method]:.3f}"
            )
    Path("paper/tables/exp_003_pool_scaling_v1_facts.tex").write_text(
        "% Generated from frozen artifacts.\n"
        + "\n".join(
            chr(92) + "newcommand{" + chr(92) + k + "}{" + v + "}" for k, v in facts.items()
        )
        + "\n"
    )
    write_json(
        out / "provenance.json",
        {
            "runs": run_ids,
            "report_sha256": sha256(Path(__file__)),
            "K": 8,
            "B": [30, 100, 300],
            "role": "development",
            "interpretation": "OOF/test gaps combine selection, training-size/refit, and sampling effects; no causal or significance claim",
        },
    )
    (out / "research_note.md").write_text(
        "# exp_003 pool scaling: generated factual note\n\n"
        + "## Endpoint effects (positive = B300 better than B30)\n\n"
        + pd.DataFrame(effects).to_string(index=False)
        + "\n\n"
        + "## Null screening counts/rates\n\n"
        + screening[screening.dataset.str.startswith("null_")]
        .groupby(["dataset", "pool_B"])[["certified_count", "certification_rate"]]
        .agg(["mean", "min", "max"])
        .to_string()
        + "\n\nNo independent confirmation; reused synthetic development seeds.\n"
    )
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs=3)
    print(make_report(parser.parse_args().run_ids))
