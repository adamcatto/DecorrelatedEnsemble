"""Audit unchanged pools and report the OOF-equivalence intervention."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from make_real_report import DATASET_LABELS, LABELS, PRIMARY, prediction_columns

from decorrelated_ensemble.evaluation.artifacts import sha256, verify_manifest, write_json
from decorrelated_ensemble.selection import distinct_prediction_indices


def report(run_id, reference_id, predictions_only=False):
    root, reference = [Path("results/runs") / name for name in (run_id, reference_id)]
    cfgs = []
    for folder in (root, reference):
        verify_manifest(folder, allow_missing_models=predictions_only)
        if json.loads((folder / "status.json").read_text())["state"] != "complete":
            raise ValueError("Duplicate report requires completed runs")
        cfgs.append(json.loads((folder / "config.json").read_text()))
    base_keys = set(cfgs[0]) - {"id", "hypothesis", "methods", "baselines"}
    if any(cfgs[0][key] != cfgs[1][key] for key in base_keys):
        raise ValueError("Duplicate ablation changed data/candidates/evaluation")
    controls = [
        "random_subspace",
        "top_no_cert",
        "coerror_no_cert",
        "caruana",
        "random_forest",
        "linear",
    ]
    records = pd.read_csv(root / "records.csv")
    folds = sorted(p.relative_to(root) for p in root.rglob("oof.npz"))
    if folds != sorted(p.relative_to(reference) for p in reference.rglob("oof.npz")):
        raise ValueError("Duplicate ablation changed folds")
    unique_checks, prediction_checks = [], 0
    for rel in folds:
        a, b = root / rel.parent, reference / rel.parent
        for name in ("X.pkl", "y.npy", "metadata.json"):
            if sha256(a.parent / name) != sha256(b.parent / name):
                raise ValueError("Duplicate ablation raw data/metadata changed")
        if (a / "candidates.json").read_text() != (b / "candidates.json").read_text():
            raise ValueError("Duplicate ablation specifications changed")
        if (a / "inner_splits.json").read_text() != (b / "inner_splits.json").read_text():
            raise ValueError("Duplicate ablation inner partitions changed")
        for name in ("outer_split.npz", "oof.npz", "certification.npz"):
            arrays = [np.load(folder / name) for folder in (a, b)]
            for key in arrays[0].files:
                np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
        for method in controls:
            arrays = [np.load(folder / method / "test_predictions.npz") for folder in (a, b)]
            for key in arrays[0].files:
                np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
            prediction_checks += 1
        P = np.load(a / "oof.npz")["predictions"]
        expected = distinct_prediction_indices(P)
        for path in a.glob("*/selection.json"):
            selection = json.loads(path.read_text())
            if not selection["config"].get("deduplicate_oof", False):
                continue
            np.testing.assert_array_equal(selection["trace"][0]["eligible_pool_ids"], expected)
            ids = selection["ids"]
            if len(ids) != 8 or len(np.unique(prediction_columns(P)[ids], axis=0)) != 8:
                raise ValueError("Unique variant violates fixed distinct-OOF K")
            np.testing.assert_array_equal(selection["weights"], np.full(8, 1 / 8))
            if selection["eligible_count"] != len(expected):
                raise ValueError("Deduplicated eligible count mismatch")
            unique_checks.append(
                {
                    "fold_path": str(rel.parent),
                    "method": path.parent.name,
                    "distinct_pool": len(expected),
                    "distinct_selected": len(ids),
                }
            )
    output = Path("results/summaries") / run_id
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unique_checks).to_csv(output / "uniqueness_audit.csv", index=False)
    complete = records[records.status == "complete"]
    effects = []
    pairs = [(base + "_unique", base) for base in controls[:3]] + [
        ("coerror_no_cert", "top_no_cert"),
        ("coerror_no_cert_unique", "top_no_cert_unique"),
        ("coerror_no_cert_unique", "caruana"),
    ]
    for dataset, frame in complete.groupby("dataset"):
        task = frame.task.iloc[0]
        primary, higher = PRIMARY[task]
        metrics = [
            (primary, higher),
            ("normalized_squared_loss" if task == "regression" else "brier", False),
        ]
        keys = ["dataset", "seed", "fold"]
        for metric, higher in metrics:
            for left, right in pairs:
                joined = frame[frame.method == left][keys + [metric]].merge(
                    frame[frame.method == right][keys + [metric]],
                    on=keys,
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                )
                joined["advantage"] = (1 if higher else -1) * (
                    joined[metric + "_left"] - joined[metric + "_right"]
                )
                effects.append(
                    joined[keys + ["advantage"]].assign(left=left, right=right, metric=metric)
                )
    effects = pd.concat(effects, ignore_index=True)
    effects.to_csv(output / "paired_effects.csv", index=False)
    groups = ["dataset", "metric", "left", "right"]
    effects.groupby(groups).advantage.agg(["count", "mean", "min", "max"]).to_csv(
        output / "effect_summary.csv"
    )
    effects.groupby(groups + ["seed"]).advantage.mean().to_csv(output / "seed_effects.csv")
    methods = [m["id"] for m in cfgs[0]["methods"] + cfgs[0]["baselines"]]
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Cancer AUROC $\uparrow$ & Wine log loss $\downarrow$ & Diabetes RMSE $\downarrow$ \\",
        r"\midrule",
    ]
    facts, note = {}, []
    for method in methods:
        values = []
        for dataset in DATASET_LABELS:
            subset = complete[(complete.dataset == dataset) & (complete.method == method)]
            if subset.empty:
                values.append("--")
                continue
            metric = PRIMARY[subset.task.iloc[0]][0]
            mean = subset[metric].mean()
            values.append(f"{mean:.3f}" + (r"$^{\dagger}$" if len(subset) != 6 else ""))
            prefix = {"breast_cancer": "Cancer", "wine": "Wine", "diabetes": "Diabetes"}[dataset]
            tag = {
                "top_no_cert": "Top",
                "coerror_no_cert": "Coerror",
                "random_subspace": "Random",
            }.get(method.removesuffix("_unique"))
            if tag:
                facts[
                    "ExpFive"
                    + prefix
                    + tag
                    + ("Unique" if method.endswith("_unique") else "Original")
                ] = f"{mean:.3f}"
        label = LABELS.get(
            method, LABELS.get(method.removesuffix("_unique"), method) + " (OOF-unique)"
        )
        lines.append(label + " & " + " & ".join(values) + r" \\")
        note.append(label + ": " + ", ".join(values))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Experiment 005: exact training-OOF duplicate removal at fixed K=8, B=100 and feature fraction .1. Original and OOF-unique selectors share all data, specifications, OOF predictions and splits; unique variants retain first-ID representatives. Six overlapping outer evaluations per task, unscreened. Caruana has eight steps and frequency weights. OOF equality does not imply global function equality. Static forest/linear references have unmatched budgets.}",
        r"\end{table}",
    ]
    Path("paper/tables/exp_005_oof_duplicates_v1_results.tex").write_text("\n".join(lines) + "\n")
    Path("paper/tables/exp_005_oof_duplicates_v1_facts.tex").write_text(
        "% Generated from frozen artifacts.\n"
        + "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in facts.items())
        + "\n"
    )
    (output / "research_note.md").write_text(
        "# OOF-equivalence intervention: generated facts\n\n"
        + "Primary metrics in cancer/wine/diabetes order. Means of six overlapping CV evaluations, not independent datasets.\n\n"
        + "\n".join(note)
        + "\n\nFull paired split/seed effects in CSVs. No task-population interval or superiority claim.\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))
    for ax, (dataset, label) in zip(axes, DATASET_LABELS.items()):
        frame = complete[complete.dataset == dataset]
        metric = PRIMARY[frame.task.iloc[0]][0]
        for base, color in zip(controls[:3], ["#999999", "#e07a1f", "#377eb8"]):
            means = [
                frame[frame.method == method][metric].mean() for method in (base, base + "_unique")
            ]
            ax.plot([0, 1], means, "o-", color=color, label=LABELS[base])
            for x, method in enumerate((base, base + "_unique")):
                values = frame[frame.method == method][metric].to_numpy()
                ax.scatter(
                    x + np.linspace(-0.02, 0.02, len(values)), values, s=9, alpha=0.5, color=color
                )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Original", "OOF-unique"],
            title=label,
            ylabel=metric.replace("_", " "),
            xlim=(-0.15, 1.15),
        )
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncols=3, frameon=False, fontsize=8
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    destination = Path("paper/figures") / run_id
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(destination / ("duplicate_intervention." + suffix), dpi=180)
    plt.close(fig)
    write_json(
        output / "duplicate_control_audit.json",
        {
            "run_id": run_id,
            "reference_id": reference_id,
            "identical_pool_fold_pairs": len(folds),
            "identical_control_prediction_pairs": prediction_checks,
            "unique_subsets_checked": len(unique_checks),
            "checks": [
                "identical raw data/specifications/splits/OOF/bootstrap",
                "unchanged test prediction controls",
                "first-ID representatives",
                "eight exact-distinct observed OOF columns and equal weights",
            ],
            "raw_manifests": {
                folder.name: sha256(folder / "manifest.json") for folder in (root, reference)
            },
            "report_script_sha256": sha256(Path(__file__)),
            "development_only": True,
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("reference_id")
    parser.add_argument("--predictions-only", action="store_true")
    args = parser.parse_args()
    print(report(args.run_id, args.reference_id, args.predictions_only))
