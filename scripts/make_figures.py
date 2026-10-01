import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decorrelated_ensemble.metrics import residual_matrix
from decorrelated_ensemble.selection import correlation_matrix


def make_figures(run_id):
    summary = Path("results/summaries") / run_id
    data = pd.read_csv(summary / "per_task.csv")
    splits = pd.read_csv(summary / "per_split.csv")
    out = Path("paper/figures") / run_id
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    methods = [
        "random_subspace",
        "certified_random",
        "top_quality",
        "abs_correlation",
        "quality_diversity",
        "coerror",
        "coerror_no_cert",
        "shrinkage",
        "random_forest",
        "catboost",
    ]
    methods = [m for m in methods if m in data.method.unique()]
    fig, axes = plt.subplots(2, 1, figsize=(7.3, 8.5), layout="constrained")
    for ax, task, metric in zip(
        axes, ["binary", "regression"], ["auroc", "normalized_squared_loss"]
    ):
        subset = data[data.task == task]
        if not len(subset):
            ax.set_visible(False)
            continue
        table = subset.pivot(index="dataset", columns="method", values=metric).reindex(
            columns=methods
        )
        cmap = plt.colormaps["viridis" if task == "binary" else "viridis_r"].copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(table.values, aspect="auto", cmap=cmap)
        for i in range(len(table)):
            for j in range(len(methods)):
                value = table.iloc[i, j]
                rgba = cmap(im.norm(value)) if np.isfinite(value) else (1, 1, 1, 1)
                luminance = np.dot(rgba[:3], [0.2126, 0.7152, 0.0722])
                ax.text(
                    j,
                    i,
                    "—" if np.isnan(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="black" if luminance > 0.5 else "white",
                )
        ax.set_xticks(range(len(methods)), methods, rotation=65, ha="right", fontsize=7)
        ax.set_yticks(range(len(table)), [name.replace("_" + task, "") for name in table.index])
        ax.set_title(
            "Binary AUROC (higher better)"
            if task == "binary"
            else "Regression MSE/null MSE (lower better)"
        )
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(out / "performance.pdf")
    fig.savefig(out / "performance.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for ax, task in zip(axes, ["binary", "regression"]):
        subset = splits[splits.task == task].drop_duplicates(["dataset", "seed", "fold"])
        grouped = subset.groupby("dataset").certification_rate.agg(["mean", "min", "max"])
        if len(grouped):
            ax.bar(range(len(grouped)), grouped["mean"], color="#406b96")
            ax.errorbar(
                range(len(grouped)),
                grouped["mean"],
                yerr=[grouped["mean"] - grouped["min"], grouped["max"] - grouped["mean"]],
                fmt="none",
                color="black",
            )
            ax.set_xticks(
                range(len(grouped)),
                [name.replace("_" + task, "") for name in grouped.index],
                rotation=55,
                ha="right",
            )
        ax.set(title=task, ylabel="Empirical screening survival", ylim=(0, 1))
    fig.savefig(out / "certification.pdf")
    plt.close(fig)
    selected = data[
        data.method.isin(
            ["top_quality", "abs_correlation", "signed_correlation", "coerror", "shrinkage"]
        )
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), layout="constrained")
    for ax, task in zip(axes, ["binary", "regression"]):
        for method, group in selected[selected.task == task].groupby("method"):
            ax.scatter(
                group.mean_abs_residual_correlation, group.mean_selected_quality, label=method, s=22
            )
        ax.set(
            title=task,
            xlabel="Mean absolute residual correlation",
            ylabel="Mean individual OOF skill",
        )
    axes[1].legend(fontsize=6)
    fig.savefig(out / "quality_dependence.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for method, group in data.groupby("method"):
        if method in [
            "random_subspace",
            "top_quality",
            "coerror",
            "coerror_no_cert",
            "random_forest",
            "catboost",
            "xgboost",
        ]:
            axes[0].scatter(
                group.train_wall_seconds.mean(), group.normalized_squared_loss.mean(), label=method
            )
            axes[1].scatter(
                group.model_bytes.mean(), group.normalized_squared_loss.mean(), label=method
            )
    axes[0].set(
        xscale="log", xlabel="Training/search wall seconds", ylabel="Mean normalized squared loss"
    )
    axes[1].set(
        xscale="log", xlabel="Serialized model bytes", ylabel="Mean normalized squared loss"
    )
    axes[1].legend(fontsize=6)
    fig.savefig(out / "resources.pdf")
    plt.close(fig)
    # One declared example, not a hand-selected favorable heatmap.
    run = Path("results/runs") / run_id
    example = run / "additive_binary" / "seed_11" / "fold_0"
    if example.exists():
        oof = np.load(example / "oof.npz")
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
        for ax, method in zip(axes, ["top_quality", "coerror"]):
            file = example / method / "selection.json"
            if file.exists():
                ids = json.loads(file.read_text())["ids"]
                R = correlation_matrix(
                    residual_matrix(oof["y"], oof["predictions"][:, ids], "binary")
                )
                im = ax.imshow(R, vmin=-1, vmax=1, cmap="coolwarm")
                ax.set_title(method)
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.savefig(out / "residual_heatmap.pdf")
        plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    print(make_figures(parser.parse_args().run_id))
