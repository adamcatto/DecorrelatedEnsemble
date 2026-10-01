"""Full paired loss-alignment report; no outer-test configuration selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from make_large_report import LABELS, NAMES, PRIMARY, paired_effect

from decorrelated_ensemble.evaluation.artifacts import sha256, verify_manifest, write_json

NEW = "top_squared_b6000_k64"
OLD = ["top_quality_b6000_k64", "coerror_b6000_k64"]


def report(run_id, predictions_only=False):
    root = Path("results/runs") / run_id
    verify_manifest(root, allow_missing_models=predictions_only)
    cfg = json.loads((root / "config.json").read_text())
    frames = [pd.read_csv(root / "records.csv")]
    for reference_id in cfg["reference_runs"]:
        parent = Path("results/runs") / reference_id
        verify_manifest(parent, allow_missing_models=predictions_only)
        original = pd.read_csv(parent / "records.csv")
        frames.append(original[original.method.isin(OLD)])
    records = pd.concat(frames, ignore_index=True)
    if len(records) != 54 or set(records.dataset.unique()) != set(NAMES):
        raise ValueError("Incomplete six-task control comparison")
    if records.duplicated(["dataset", "seed", "fold", "method"]).any():
        raise ValueError("Repeated split records")
    output = Path("results/summaries") / run_id
    output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / "comparison_per_split.csv", index=False)
    means = records.groupby(["dataset", "task", "method"]).mean(numeric_only=True).reset_index()
    means.to_csv(output / "comparison_per_task.csv", index=False)
    effects = []
    tables = Path("paper/tables") / run_id
    tables.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Task / metric & Original quality & Squared-loss quality & Co-error \\",
        r"\midrule",
    ]
    facts = []
    for name in NAMES:
        frame = means[means.dataset == name].set_index("method")
        primary = "group_auroc" if name == "musk2" else PRIMARY[frame.task.iloc[0]]
        measures = [primary, "brier"] if frame.task.iloc[0] == "binary" else ["rmse"]
        for metric in measures:
            lines.append(
                LABELS[name]
                + " / "
                + ("group AUC" if metric == "group_auroc" else metric.upper())
                + " & "
                + " & ".join(f"{frame.loc[m, metric]:.4f}" for m in [OLD[0], NEW, OLD[1]])
                + r" \\"
            )
            for left, right in [(NEW, OLD[0]), (OLD[1], NEW)]:
                effects.append(
                    paired_effect(
                        records[records.dataset == name],
                        left,
                        right,
                        metric,
                        metric.endswith("auroc"),
                    )
                )
        for suffix, method in [("Aligned", NEW), ("Quality", OLD[0]), ("Co", OLD[1])]:
            key = "".join(w.capitalize() for w in name.split("_")) + suffix
            facts.append(
                r"\newcommand{\ExpSeven" + key + "}{" + f"{frame.loc[method, primary]:.3f}" + "}"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Loss-aligned quality intervention at fixed $B=6,000,K=64$, equal weights and immutable experiment 006 inputs. Original classification quality is row AUROC; aligned quality is individual row squared loss. Musk group AUC uses fixed molecule max; Brier remains row loss. Means of three reused development folds; no independent confirmation.}",
        r"\end{table}",
    ]
    (tables / "quality_alignment.tex").write_text("\n".join(lines) + "\n")
    (tables / "facts.tex").write_text("\n".join(facts) + "\n")
    pd.concat(effects, ignore_index=True).to_csv(output / "paired_effects.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    classifiers = ["credit_default", "higgs", "miniboone", "musk2"]
    figure, axes = plt.subplots(2, 4, figsize=(11, 6), constrained_layout=True)
    for column, name in enumerate(classifiers):
        frame = means[means.dataset == name].set_index("method")
        for row, metric in enumerate(["group_auroc" if name == "musk2" else "auroc", "brier"]):
            ax = axes[row, column]
            methods = [OLD[0], NEW, OLD[1]]
            for position, method in enumerate(methods):
                ax.scatter(
                    [position],
                    [frame.loc[method, metric]],
                    s=40,
                    color=["#777777", "#e69f00", "#0072b2"][position],
                    zorder=3,
                )
                values = records[(records.dataset == name) & (records.method == method)][
                    metric
                ].values
                ax.scatter(
                    np.full(len(values), position),
                    values,
                    s=16,
                    marker="x",
                    color="#999999",
                    alpha=0.7,
                )
            ax.set_xticks([0, 1, 2], ["Q(AUC)", "Q(Sq)", "Co"], rotation=25)
            ax.set_title(LABELS[name] if row == 0 else "")
            ax.set_ylabel(
                ("Group " if metric == "group_auroc" else "") + "AUROC ↑"
                if row == 0
                else "Row Brier ↓"
            )
            ax.grid(alpha=0.15)
    figure.suptitle("Same library and K: quality criterion versus joint squared-loss selection")
    figures = Path("paper/figures") / run_id
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "loss_alignment.pdf")
    figure.savefig(figures / "loss_alignment.png", dpi=180)
    plt.close(figure)
    write_json(
        output / "report_provenance.json",
        {
            "run_id": run_id,
            "reference_runs": cfg["reference_runs"],
            "report_script_sha256": sha256(Path(__file__)),
            "scope": "Reused development folds; task means and paired fold effects; no population intervals or test-selected settings",
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--predictions-only", action="store_true")
    args = parser.parse_args()
    print(report(args.run_id, args.predictions_only))
