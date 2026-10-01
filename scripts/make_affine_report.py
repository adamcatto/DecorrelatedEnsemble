"""Generate attenuation controls directly from a frozen affine experiment run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_report(run_id):
    summary = Path("results/summaries") / run_id
    tasks = pd.read_csv(summary / "per_task.csv")
    splits = pd.read_csv(summary / "per_split.csv")
    methods = [
        "random_subspace",
        "random_subspace_affine",
        "top_no_cert",
        "top_no_cert_affine",
        "coerror_no_cert",
        "coerror_no_cert_affine",
        "affine_profiled",
    ]
    labels = ["RS", "RS+aff", "Top", "Top+aff", "CoErr", "CoErr+aff", "APCE"]
    table = tasks.pivot(index="dataset", columns="method", values="normalized_squared_loss")
    tables = Path("paper/tables")
    figures = Path("paper/figures") / run_id
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        "Regime & " + " & ".join(labels) + r" \\",
        r"\midrule",
    ]
    for dataset, row in table[methods].iterrows():
        lines.append(
            dataset.removesuffix("_regression")
            + " & "
            + " & ".join(f"{v:.3f}" for v in row)
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Experiment 002: regression MSE divided by training-mean null MSE, "
            r"averaged over six outer evaluations per regime. All subsets are unscreened. "
            r"+aff fits two aggregate parameters on training OOF predictions; APCE also changes "
            r"selection. B=100 and K=8 are fixed. Lower is better. Development evidence only.}"
        ),
        r"\end{table}",
    ]
    (tables / (run_id + "_regression.tex")).write_text("\n".join(lines) + "\n")
    parameters = []
    root = Path("results/runs") / run_id
    for path in sorted(root.rglob("calibration.json")):
        dataset, seed, fold, method = path.relative_to(root).parts[:-1]
        parameters.append(
            {
                "dataset": dataset,
                "seed": int(seed.removeprefix("seed_")),
                "fold": int(fold.removeprefix("fold_")),
                "method": method,
                **json.loads(path.read_text()),
            }
        )
    params = pd.DataFrame(parameters)
    params.to_csv(summary / "affine_parameters.csv", index=False)
    plt.rcParams.update(
        {"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.7), layout="constrained")
    datasets = table.index.tolist()
    colors = ["#377eb8", "#e07a1f", "#48935c"]
    effects = []
    for j, (raw, color) in enumerate(zip(methods[::2][:3], colors)):
        paired = splits[splits.method == raw].merge(
            splits[splits.method == raw + "_affine"],
            on=["dataset", "seed", "fold"],
            suffixes=("_raw", "_aff"),
        )
        paired["advantage"] = (
            paired.normalized_squared_loss_raw - paired.normalized_squared_loss_aff
        )
        means = paired.groupby("dataset").advantage.mean().reindex(datasets)
        x = np.arange(len(datasets)) + (j - 1) * 0.23
        axes[0].bar(x, means, width=0.21, color=color, alpha=0.8, label=labels[j * 2])
        for i, dataset in enumerate(datasets):
            values = paired[paired.dataset == dataset].advantage.to_numpy()
            axes[0].scatter(
                x[i] + np.linspace(-0.06, 0.06, len(values)),
                values,
                color=color,
                s=9,
                edgecolors="black",
                linewidths=0.3,
            )
        effects.append(paired[["dataset", "seed", "fold", "advantage"]].assign(control=raw))
        calibrated = raw + "_affine"
        gain = table[calibrated] - table.affine_profiled
        axes[1].plot(np.arange(len(datasets)), gain, "o-", color=color, label=labels[j * 2 + 1])
    pd.concat(effects).to_csv(summary / "calibration_effects.csv", index=False)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(range(len(datasets)), [d.removesuffix("_regression") for d in datasets])
        ax.legend(fontsize=8, ncols=3)
    axes[0].set(ylabel="Raw loss − calibrated loss", title="Affine correction of unchanged subsets")
    axes[1].set(
        ylabel="Calibrated control loss − APCE loss",
        title="Additional effect of profiled selection (task means)",
    )
    fig.savefig(figures / "affine_controls.pdf")
    fig.savefig(figures / "affine_controls.png", dpi=180)
    plt.close(fig)
    facts = {}
    for regime in ["additive", "mixed", "null", "dominant"]:
        for suffix, method in [
            ("Raw", "coerror_no_cert"),
            ("Affine", "coerror_no_cert_affine"),
            ("Profiled", "affine_profiled"),
            ("Forest", "random_forest"),
        ]:
            facts["ExpTwo" + regime.title() + suffix] = (
                f"{table.loc[regime + '_regression', method]:.3f}"
            )
    wins = sum(table.affine_profiled < table.coerror_no_cert_affine - 1e-12)
    facts["ExpTwoProfiledWins"] = str(wins)
    comparison = splits[splits.method == "affine_profiled"].merge(
        splits[splits.method == "coerror_no_cert_affine"],
        on=["dataset", "seed", "fold"],
        suffixes=("_apce", "_co"),
    )
    comparison["oof_advantage"] = (
        comparison.oof_calibrated_squared_loss_co - comparison.oof_calibrated_squared_loss_apce
    )
    comparison["test_advantage"] = (
        comparison.normalized_squared_loss_co - comparison.normalized_squared_loss_apce
    )
    comparison[["dataset", "seed", "fold", "oof_advantage", "test_advantage"]].to_csv(
        summary / "profiled_effects.csv", index=False
    )
    facts["ExpTwoOOFNoWorse"] = str(sum(comparison.oof_advantage >= -1e-12))
    facts["ExpTwoProfiledLosses"] = str(
        sum(table.affine_profiled > table.coerror_no_cert_affine + 1e-12)
    )
    facts["ExpTwoCalibratedWins"] = str(
        sum(table.coerror_no_cert_affine < table.coerror_no_cert - 1e-12)
    )
    (tables / (run_id + "_facts.tex")).write_text(
        "% Generated from frozen artifacts.\n"
        + "\n".join(
            chr(92) + "newcommand{" + chr(92) + key + "}{" + value + "}"
            for key, value in facts.items()
        )
        + "\n"
    )
    notes = [
        f"# {run_id}: generated affine diagnostic",
        "",
        "Effects are paired within dataset/seed/fold; positive means improvement.",
        "",
        table.to_string(float_format=lambda v: f"{v:.4f}"),
        "",
        "## OOF-fitted slopes (mean and range over folds/seeds)",
        "",
        params.groupby(["dataset", "method"]).slope.agg(["mean", "min", "max"]).to_string(),
        "",
        "Dots in the calibration figure are paired split effects, not IID confidence intervals.",
        "These are deliberately chosen development mechanisms, not a task-population benchmark.",
    ]
    (summary / "research_note.md").write_text("\n".join(notes) + "\n")
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    print(make_report(parser.parse_args().run_id))
