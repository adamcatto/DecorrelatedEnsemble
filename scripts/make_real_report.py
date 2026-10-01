"""Paired-width audit and dataset-specific reporting for the real development panel."""

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from decorrelated_ensemble.evaluation.artifacts import sha256, verify_manifest, write_json

LABELS = {
    "random_subspace": "Random subspace",
    "certified_random": "Screened random",
    "top_quality": "Screened top quality",
    "top_no_cert": "Top quality (unfiltered)",
    "abs_correlation": "Absolute residual correlation",
    "signed_correlation": "Signed residual correlation",
    "minimax": "Minimax residual correlation",
    "quality_diversity": "Quality + diversity",
    "prediction_correlation": "Prediction correlation",
    "coerror_greedy": "Co-error greedy",
    "direct_squared": "Direct squared loss",
    "coerror": "Screened co-error + swaps",
    "coerror_no_cert": "Co-error (unfiltered)",
    "covariance_only": "Covariance only",
    "shrinkage": "Shrinkage .5",
    "caruana": "Caruana (frequency weights)",
    "direct_auroc": "Direct AUROC",
    "coerror_no_cert_affine": "Co-error + affine",
    "affine_profiled": "APCE",
    "rf_shallow": "Shallow Random Forest",
    "et_shallow": "Shallow ExtraTrees",
    "random_patches": "Random Patches",
    "random_forest": "Random Forest (256)",
    "extra_trees": "ExtraTrees (256)",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "linear": "Logistic / ridge",
}
PRIMARY = {
    "binary": ("auroc", True),
    "multiclass": ("log_loss", False),
    "regression": ("rmse", False),
}
DATASET_LABELS = {"breast_cancer": "Breast cancer", "wine": "Wine", "diabetes": "Diabetes"}


def verify_width_controls(roots, predictions_only=False):
    configs = []
    for root in roots:
        verify_manifest(root, allow_missing_models=predictions_only)
        if json.loads((root / "status.json").read_text())["state"] != "complete":
            raise ValueError("Real panel contains an incomplete run")
        cfg = json.loads((root / "config.json").read_text())
        fraction = cfg["candidates"]["feature_fractions"]
        if len(fraction) != 1:
            raise ValueError("Each real-panel run must have one feature fraction")
        configs.append(cfg)
    fractions = [cfg["candidates"]["feature_fractions"][0] for cfg in configs]
    if fractions != [0.1, 0.5]:
        raise ValueError("Expected preregistered widths .1 then .5")
    normalized = []
    for cfg in configs:
        cfg = copy.deepcopy(cfg)
        cfg.pop("id")
        cfg["candidates"].pop("feature_fractions")
        for base in cfg["baselines"]:
            if base["family"] == "random_patches":
                base["parameters"].pop("feature_fraction")
        normalized.append(cfg)
    if normalized[0] != normalized[1]:
        raise ValueError("Configs differ beyond declared feature-width changes")
    controls = [b["id"] for b in configs[0]["baselines"] if b["family"] != "random_patches"]
    relative_folds = [sorted(p.relative_to(root) for p in root.rglob("oof.npz")) for root in roots]
    if relative_folds[0] != relative_folds[1]:
        raise ValueError("Real width runs have different folds")
    checked = 0
    dataset_metadata = {}
    for rel in relative_folds[0]:
        a, b = [root / rel.parent for root in roots]
        if sha256(a.parent / "X.pkl") != sha256(b.parent / "X.pkl"):
            raise ValueError("Width comparison raw features changed")
        np.testing.assert_array_equal(np.load(a.parent / "y.npy"), np.load(b.parent / "y.npy"))
        meta_a, meta_b = [json.loads((p.parent / "metadata.json").read_text()) for p in (a, b)]
        if meta_a != meta_b:
            raise ValueError("Width comparison dataset metadata changed")
        dataset_metadata[meta_a["id"]] = meta_a
        arrays = [np.load(p / "outer_split.npz") for p in (a, b)]
        for key in arrays[0].files:
            np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
        if (a / "inner_splits.json").read_text() != (b / "inner_splits.json").read_text():
            raise ValueError("Width comparison inner splits changed")
        oofs = [np.load(p / "oof.npz") for p in (a, b)]
        for key in ("fold_ids", "null", "y"):
            np.testing.assert_array_equal(oofs[0][key], oofs[1][key])
        for method in controls:
            predictions = [np.load(p / method / "test_predictions.npz") for p in (a, b)]
            for key in predictions[0].files:
                np.testing.assert_array_equal(predictions[0][key], predictions[1][key])
            checked += 1
    return configs, {
        "width_fold_pairs": len(relative_folds[0]),
        "unchanged_baseline_prediction_pairs": checked,
        "checks": [
            "artifact hashes and completed status",
            "configs differ only in declared widths",
            "identical raw data, metadata, outer/inner partitions and null",
            "unchanged baseline predictions",
        ],
        "dataset_metadata": dataset_metadata,
        "scope": "Mechanism controls; wider candidate feature masks are not nested subsets",
    }


def paired_effects(records):
    completed = records[records.status == "complete"]
    within, width = [], []
    pairs = [
        ("coerror_no_cert", "top_no_cert"),
        ("coerror", "top_quality"),
        ("coerror", "abs_correlation"),
        ("top_quality", "top_no_cert"),
        ("coerror_no_cert", "random_forest"),
        ("coerror_no_cert", "caruana"),
        ("coerror_no_cert_affine", "coerror_no_cert"),
        ("affine_profiled", "coerror_no_cert_affine"),
    ]
    for dataset, data in completed.groupby("dataset"):
        task = data.task.iloc[0]
        primary, higher = PRIMARY[task]
        metrics = [(primary, higher)] + (
            [("normalized_squared_loss", False)] if task == "regression" else [("brier", False)]
        )
        keys = ["dataset", "seed", "fold"]
        for metric, higher in metrics:
            sign = 1 if higher else -1
            for fraction, frame in data.groupby("feature_fraction"):
                for left, right in pairs:
                    joined = frame[frame.method == left][keys + [metric]].merge(
                        frame[frame.method == right][keys + [metric]],
                        on=keys,
                        suffixes=("_left", "_right"),
                        validate="one_to_one",
                    )
                    joined["advantage"] = sign * (
                        joined[metric + "_left"] - joined[metric + "_right"]
                    )
                    within.append(
                        joined[keys + ["advantage"]].assign(
                            left=left, right=right, metric=metric, feature_fraction=fraction
                        )
                    )
            for method, frame in data.groupby("method"):
                joined = frame[frame.feature_fraction == 0.5][keys + [metric]].merge(
                    frame[frame.feature_fraction == 0.1][keys + [metric]],
                    on=keys,
                    suffixes=("_wide", "_narrow"),
                    validate="one_to_one",
                )
                joined["advantage"] = sign * (joined[metric + "_wide"] - joined[metric + "_narrow"])
                width.append(joined[keys + ["advantage"]].assign(method=method, metric=metric))
    return pd.concat(within, ignore_index=True), pd.concat(width, ignore_index=True)


def make_report(run_ids, predictions_only=False):
    roots = [Path("results/runs") / name for name in run_ids]
    configs, controls = verify_width_controls(roots, predictions_only)
    frames = []
    for root, cfg in zip(roots, configs):
        frame = pd.read_csv(root / "records.csv")
        frames.append(
            frame.assign(
                feature_fraction=cfg["candidates"]["feature_fractions"][0], run_id=root.name
            )
        )
    records = pd.concat(frames, ignore_index=True)
    output = Path("results/summaries/exp_004_real_v1")
    output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / "per_split.csv", index=False)
    records.groupby(["dataset", "feature_fraction", "method", "status"]).size().rename(
        "count"
    ).to_csv(output / "coverage.csv")
    completed = records[records.status == "complete"]
    numeric = completed.select_dtypes("number").columns.difference(
        ["seed", "fold", "feature_fraction"]
    )
    keys = ["dataset", "task", "feature_fraction", "method"]
    means = completed.groupby(keys)[numeric].mean().reset_index()
    counts = completed.groupby(keys).size().rename("completed_splits").reset_index()
    means = means.merge(counts, on=keys)
    means.to_csv(output / "per_task.csv", index=False)
    within, width = paired_effects(records)
    within.to_csv(output / "selection_effects.csv", index=False)
    width.to_csv(output / "width_effects.csv", index=False)
    for label, effects, grouping in [
        ("selection", within, ["dataset", "feature_fraction", "metric", "left", "right"]),
        ("width", width, ["dataset", "metric", "method"]),
    ]:
        effects.groupby(grouping).advantage.agg(["count", "mean", "min", "max"]).to_csv(
            output / (label + "_effect_summary.csv")
        )
        effects.groupby(grouping + ["seed"]).advantage.mean().to_csv(
            output / (label + "_seed_effects.csv")
        )
    screen = records.drop_duplicates(["dataset", "feature_fraction", "seed", "fold"])
    screen[
        ["dataset", "feature_fraction", "seed", "fold", "certified_count", "certification_rate"]
    ].to_csv(output / "screening.csv", index=False)
    table_root, figure_root = Path("paper/tables"), Path("paper/figures/exp_004_real_v1")
    figure_root.mkdir(parents=True, exist_ok=True)
    methods = [m["id"] for m in configs[0]["methods"] + configs[0]["baselines"]]
    columns = []
    for name in DATASET_LABELS:
        task = controls["dataset_metadata"][name]["task"]
        columns += [(name, fraction, PRIMARY[task][0]) for fraction in (0.1, 0.5)]
    lines = [
        r"\begin{table}[p]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Cancer AUROC $\uparrow$} & \multicolumn{2}{c}{Wine log loss $\downarrow$} & \multicolumn{2}{c}{Diabetes RMSE $\downarrow$} \\",
        r"Method & .1 & .5 & .1 & .5 & .1 & .5 \\",
        r"\midrule",
    ]
    for method in methods:
        if method == configs[0]["baselines"][0]["id"]:
            lines.append(r"\midrule")
        values = []
        for dataset, fraction, metric in columns:
            row = means[
                (means.dataset == dataset)
                & (means.feature_fraction == fraction)
                & (means.method == method)
            ]
            if row.empty:
                values.append("--")
            else:
                item = row.iloc[0]
                values.append(
                    f"{item[metric]:.3f}" + (r"$^{\dagger}$" if item.completed_splits != 6 else "")
                )
        lines.append(LABELS[method] + " & " + " & ".join(values) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Experiment 004: three real development datasets, B=100 and K=8, depth 3, six outer evaluations per dataset/width. Each pair of columns changes only feature fraction (.1/.5) and its Random Patches control. Means use completed splits; $\dagger$ marks partial feasibility, -- indicates infeasible or task-restricted. Baselines use static settings and unmatched resources. Wine co-error optimizes Brier, not its primary log loss. Repeated CV folds are not independent datasets.}",
        r"\end{table}",
    ]
    (table_root / "exp_004_real_v1_results.tex").write_text("\n".join(lines) + "\n")
    plt.rcParams.update(
        {"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    plotted = [
        "random_subspace",
        "top_no_cert",
        "coerror_no_cert",
        "caruana",
        "random_forest",
        "linear",
    ]
    colors = dict(zip(plotted, ["#999999", "#e07a1f", "#377eb8", "#48935c", "#b43a36", "#8b5ba6"]))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, (dataset, title) in zip(axes, DATASET_LABELS.items()):
        task = controls["dataset_metadata"][dataset]["task"]
        metric, _ = PRIMARY[task]
        for method in plotted:
            subset = means[(means.dataset == dataset) & (means.method == method)].sort_values(
                "feature_fraction"
            )
            ax.plot(
                subset.feature_fraction,
                subset[metric],
                "o-",
                color=colors[method],
                label=LABELS[method],
                markersize=4,
            )
            raw = completed[(completed.dataset == dataset) & (completed.method == method)]
            for i, fraction in enumerate((0.1, 0.5)):
                observations = raw[raw.feature_fraction == fraction][metric].to_numpy()
                ax.scatter(
                    fraction + np.linspace(-0.008, 0.008, len(observations)),
                    observations,
                    s=9,
                    alpha=0.5,
                    color=colors[method],
                )
        ax.set(
            title=title,
            xlabel="Feature fraction",
            ylabel=metric.replace("_", " "),
            xticks=[0.1, 0.5],
            xlim=(0.05, 0.55),
        )
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncols=3, frameon=False, fontsize=8
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(figure_root / ("performance_vs_fraction." + suffix), dpi=180)
    plt.close(fig)
    # Standalone costs preserve the charged search cost, rather than sharing it away.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (dataset, title) in zip(axes, DATASET_LABELS.items()):
        task = controls["dataset_metadata"][dataset]["task"]
        metric, _ = PRIMARY[task]
        for method in plotted + ["catboost"]:
            subset = means[(means.dataset == dataset) & (means.method == method)]
            for _, row in subset.iterrows():
                ax.scatter(
                    row.train_cpu_seconds,
                    row[metric],
                    marker="o" if row.feature_fraction == 0.1 else "s",
                    color=colors.get(method, "#111111"),
                    s=25,
                )
                if row.feature_fraction == 0.5:
                    ax.annotate(
                        LABELS[method].replace(" (unfiltered)", ""),
                        (row.train_cpu_seconds, row[metric]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=6,
                    )
        ax.set(
            xscale="log",
            xlabel="Charged training CPU seconds",
            ylabel=metric.replace("_", " "),
            title=title,
        )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(figure_root / ("compute_performance." + suffix), dpi=180)
    plt.close(fig)
    facts = {}
    for dataset, prefix in [
        ("breast_cancer", "Cancer"),
        ("wine", "Wine"),
        ("diabetes", "Diabetes"),
    ]:
        task = controls["dataset_metadata"][dataset]["task"]
        metric = PRIMARY[task][0]
        for fraction, fraction_name in [(0.1, "Narrow"), (0.5, "Wide")]:
            for method, label in [
                ("coerror_no_cert", "Coerror"),
                ("top_no_cert", "Top"),
                ("random_forest", "Forest"),
                ("extra_trees", "Extra"),
                ("catboost", "Cat"),
                ("linear", "Linear"),
            ]:
                item = means[
                    (means.dataset == dataset)
                    & (means.feature_fraction == fraction)
                    & (means.method == method)
                ].iloc[0]
                facts["ExpFour" + prefix + label + fraction_name] = f"{item[metric]:.3f}"
    (table_root / "exp_004_real_v1_facts.tex").write_text(
        "% Generated from frozen real-data artifacts.\n"
        + "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in facts.items())
        + "\n"
    )
    notes = [
        "# Experiment 004: generated real development report",
        "",
        "No task-level interval is estimated: there is one dataset per task type.",
        "Dots/min/max are overlapping split observations, not independent confidence intervals.",
        "",
        "## Task-specific primary metrics",
        "",
    ]
    for dataset in DATASET_LABELS:
        task = controls["dataset_metadata"][dataset]["task"]
        subset = means[means.dataset == dataset]
        notes += [
            "### " + dataset,
            "",
            subset.pivot(
                index="method", columns="feature_fraction", values=PRIMARY[task][0]
            ).to_string(),
            "",
        ]
    notes += [
        "## Screen survival",
        "",
        screen.groupby(["dataset", "feature_fraction"])
        .certified_count.agg(["mean", "min", "max"])
        .to_string(),
        "",
        "All per-split secondary metrics, effects, seed effects and resource measurements are stored in CSV files.",
        "No pooled ranking or statistical superiority claim is warranted.",
    ]
    (output / "research_note.md").write_text("\n".join(notes) + "\n")
    write_json(
        output / "provenance.json",
        {
            "runs": run_ids,
            "control_audit": controls,
            "raw_manifests": {root.name: sha256(root / "manifest.json") for root in roots},
            "report_script_sha256": sha256(Path(__file__)),
            "development_only": True,
            "uncertainty": "No IID fold intervals or task-population inference from three hand-named tasks",
            "metrics": PRIMARY,
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs=2)
    parser.add_argument("--predictions-only", action="store_true")
    args = parser.parse_args()
    print(make_report(args.run_ids, args.predictions_only))
