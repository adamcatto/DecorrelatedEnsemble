"""Frozen-anchor and full-sweep reporting for the registered exp_006 panel."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from decorrelated_ensemble.candidates import CandidateSpec, generate_candidates
from decorrelated_ensemble.certification import Certification
from decorrelated_ensemble.evaluation.artifacts import sha256, verify_manifest, write_json
from decorrelated_ensemble.evaluation.runner import candidate_pool_mask, eligibility
from decorrelated_ensemble.selection import distinct_prediction_indices

NAMES = ["credit_default", "higgs", "miniboone", "superconductivity", "california_housing", "musk2"]
LABELS = dict(
    zip(
        NAMES,
        [
            "Credit default",
            "HIGGS",
            "MiniBooNE",
            "Superconductivity",
            "California housing",
            "Musk v2 (molecules)",
        ],
    )
)
ANCHOR = "coerror_b6000_k64"
CORE = ["random", "top_quality", "coerror"]
METHOD_LABELS = {"random": "Random", "top_quality": "Top quality", "coerror": "Co-error"}
PRIMARY = {"binary": "auroc", "regression": "rmse"}
TABLE_METHODS = [
    ANCHOR,
    "top_quality_b6000_k64",
    "caruana_b6000_k64",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
    "linear",
]
TABLE_LABELS = ["Co-error", "Top", "Caruana", "RF", "ET", "XGB", "LGB", "Cat", "Linear"]


def paired_effect(frame, left, right, metric, higher):
    keys = ["dataset", "seed", "fold"]
    a = frame[(frame.method == left) & (frame.status == "complete")][keys + [metric]]
    b = frame[(frame.method == right) & (frame.status == "complete")][keys + [metric]]
    joined = a.merge(b, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    joined["effect_favors_left"] = (joined[metric + "_left"] - joined[metric + "_right"]) * (
        1 if higher else -1
    )
    joined["left"], joined["right"], joined["metric"] = left, right, metric
    return joined


def normalized_protocol(cfg):
    """Permit only the registered molecule-versus-vector grouping difference."""
    c = copy.deepcopy(cfg)
    name = c["datasets"][0]["id"]
    expected = (
        {
            "source_groups": True,
            "sampling_structure": "clustered",
            "group_exact_duplicates": False,
            "group_aggregation": "max",
        }
        if name == "musk2"
        else {
            "source_groups": False,
            "sampling_structure": "iid",
            "group_exact_duplicates": True,
            "group_aggregation": None,
        }
    )
    if name not in NAMES or len(c["datasets"]) != 1:
        raise ValueError("Unregistered dataset in panel")
    for key, value in expected.items():
        if c.get(key, value) != value:
            raise ValueError(f"Registered group protocol changed: {name} {key}")
        c.pop(key, None)
    c.pop("id")
    c.pop("datasets")
    return c


def verify_sweep(root, predictions_only=False):
    verify_manifest(root, allow_missing_models=predictions_only)
    if json.loads((root / "status.json").read_text())["state"] != "complete":
        raise ValueError("Only completed registered runs can enter the report")
    cfg = json.loads((root / "config.json").read_text())
    checks, screening = [], []
    for path in sorted(root.rglob("oof.npz")):
        fold = path.parent
        seed = int(fold.parent.name.split("_")[-1])
        number = int(fold.name.split("_")[-1])
        metadata = json.loads((fold.parent / "metadata.json").read_text())
        definitions = json.loads((fold / "candidates.json").read_text())
        specs = [CandidateSpec(**{**d, "features": tuple(d["features"])}) for d in definitions]
        expected = generate_candidates(metadata["p"], cfg["candidates"], seed + 1000 + number)
        if specs != expected:
            raise ValueError("Stored library differs from deterministic registered generator")
        oof = np.load(path)
        P = oof["predictions"]
        distinct_pool = distinct_prediction_indices(P)
        saved = np.load(fold / "certification.npz")
        cert = Certification(saved["quality"], saved["lower"], saved["passed"], saved["bootstrap"])
        # Selected-only exact equality diagnostics: avoids hashing all 6,000 vectors repeatedly.
        for selection_path in fold.glob("*/selection.json"):
            selection = json.loads(selection_path.read_text())
            method = selection["config"]
            pool = candidate_pool_mask(specs, method)
            eligible = eligibility(cert, metadata["task"], cfg["certification"], method) & pool
            np.testing.assert_array_equal(
                selection["trace"][0]["eligible_pool_ids"], np.flatnonzero(eligible)
            )
            ids = np.array(selection["ids"])
            if not eligible[ids].all():
                raise ValueError("Selection exceeds config-defined pool")
            if method["selector"] != "caruana_coerror":
                if len(ids) != method["K"] or len(np.unique(ids)) != method["K"]:
                    raise ValueError("Subset cardinality differs from registered K")
                np.testing.assert_array_equal(
                    selection["weights"], np.full(method["K"], 1 / method["K"])
                )
            distinct = len(distinct_prediction_indices(P[:, ids]))
            checks.append(
                {
                    "dataset": metadata["id"],
                    "seed": seed,
                    "fold": number,
                    "method": selection_path.parent.name,
                    "pool_B": int(pool.sum()),
                    "eligible_B": int(eligible.sum()),
                    "retained_count": len(ids),
                    "distinct_selected_oof": distinct,
                    "generated_B": len(specs),
                }
            )
        for fraction in [0.05, 0.2, 0.5]:
            for depth in [3, 6]:
                pool = candidate_pool_mask(
                    specs, {"candidate_pool": {"feature_fraction": fraction, "max_depth": depth}}
                )
                if pool.sum() != 1000:
                    raise ValueError("Homogeneous cell is not the registered matched B=1000")
                screening.append(
                    {
                        "dataset": metadata["id"],
                        "seed": seed,
                        "fold": number,
                        "fraction": fraction,
                        "depth": depth,
                        "B": int(pool.sum()),
                        "survivors": int(cert.passed[pool].sum()),
                        "distinct_oof_patterns": len(distinct_prediction_indices(P[:, pool])),
                        "mean_quality": float(cert.quality[pool].mean()),
                        "mean_lower": float(cert.lower[pool].mean()),
                    }
                )
        for prefix in [300, 1000, 3000]:
            if (
                generate_candidates(
                    metadata["p"], {**cfg["candidates"], "B": prefix}, seed + 1000 + number
                )
                != specs[:prefix]
            ):
                raise ValueError("Candidate pool prefix changed with B")
        for prefix in [300, 1000, 3000, 6000]:
            screening.append(
                {
                    "dataset": metadata["id"],
                    "seed": seed,
                    "fold": number,
                    "fraction": None,
                    "depth": None,
                    "B": prefix,
                    "survivors": int(cert.passed[:prefix].sum()),
                    "distinct_oof_patterns": int(np.sum(distinct_pool < prefix)),
                    "mean_quality": float(cert.quality[:prefix].mean()),
                    "mean_lower": float(cert.lower[:prefix].mean()),
                }
            )
    return cfg, checks, screening


def plot_sweeps(means, figure_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    colors = {"random": "#777777", "top_quality": "#e69f00", "coerror": "#0072b2"}
    for variable in ["B", "K", "fraction_depth"]:
        fig, axes = plt.subplots(2, 3, figsize=(11.3, 6.2), constrained_layout=True)
        axes = axes.ravel()
        # All six registered tasks retained.
        for ax, name in zip(axes, NAMES):
            frame = means[means.dataset == name].set_index("method")
            metric = "group_auroc" if name == "musk2" else PRIMARY[frame.task.iloc[0]]
            for selector in CORE if variable != "fraction_depth" else ["top_quality", "coerror"]:
                if variable == "B":
                    xs = [300, 1000, 3000, 6000] if selector != "random" else [1000, 3000, 6000]
                    ys = [frame.loc[f"{selector}_b{b}_k64", metric] for b in xs]
                    ax.plot(
                        xs, ys, "o-", color=colors[selector], label=METHOD_LABELS[selector], ms=4
                    )
                    ax.set_xscale("log")
                    ax.set_xlabel("Available pool B (K = 64)")
                    ax.set_xticks([300, 1000, 3000, 6000], ["300", "1k", "3k", "6k"])
                elif variable == "K":
                    xs = [16, 64, 128] + ([256] if selector == "coerror" else [])
                    ys = [frame.loc[f"{selector}_b6000_k{k}", metric] for k in xs]
                    ax.plot(
                        xs, ys, "o-", color=colors[selector], label=METHOD_LABELS[selector], ms=4
                    )
                    ax.set_xscale("log", base=2)
                    ax.set_xlabel("Retained K (B = 6,000)")
                    ax.set_xticks([16, 64, 128, 256], ["16", "64", "128", "256"])
                else:
                    for depth, style in [(3, "--"), (6, "-")]:
                        xs = [0.05, 0.2, 0.5]
                        ys = [
                            frame.loc[f"{selector}_f{int(f * 100):02}_d{depth}_b1000_k64", metric]
                            for f in xs
                        ]
                        ax.plot(
                            xs,
                            ys,
                            style + "o",
                            color=colors[selector],
                            label=f"{METHOD_LABELS[selector]}, d={depth}",
                            ms=4,
                        )
                    ax.set_xlabel("Feature fraction (B = 1,000, K = 64)")
                    ax.set_xticks([0.05, 0.2, 0.5], [".05", ".20", ".50"])
            ax.axhline(frame.loc["random_forest", metric], color="#009e73", ls=":", label="RF256")
            ax.axhline(
                frame.loc["linear", metric], color="#cc79a7", ls="-.", label="Logistic / ridge"
            )
            ax.set_title(LABELS[name])
            ax.set_ylabel(
                "AUROC (higher better)" if metric.endswith("auroc") else "RMSE (lower better)"
            )
            ax.grid(alpha=0.15)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
        fig.suptitle("Registered development sweep: one seed, three group-separated outer folds")
        fig.savefig(figure_dir / f"performance_vs_{variable}.pdf")
        fig.savefig(figure_dir / f"performance_vs_{variable}.png", dpi=180)
        plt.close(fig)


def control_equivalences(roots):
    rows = []
    pairs = [
        ("coerror_b6000_k64", "coerror_cert_b6000_k64"),
        ("top_quality_b6000_k64", "top_quality_cert_b6000_k64"),
        ("abs_correlation_b6000_k64", "signed_correlation_b6000_k64"),
    ]
    for root in roots:
        for oof_path in sorted(root.rglob("oof.npz")):
            fold = oof_path.parent
            metadata = json.loads((fold.parent / "metadata.json").read_text())
            for left, right in pairs:
                a_path, b_path = fold / left / "selection.json", fold / right / "selection.json"
                if not a_path.exists() or not b_path.exists():
                    continue
                a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
                with (
                    np.load(fold / left / "test_predictions.npz") as x,
                    np.load(fold / right / "test_predictions.npz") as y,
                ):
                    same_test = np.array_equal(x["prediction"], y["prediction"])
                rows.append(
                    {
                        "dataset": metadata["id"],
                        "seed": int(fold.parent.name.split("_")[-1]),
                        "fold": int(fold.name.split("_")[-1]),
                        "left": left,
                        "right": right,
                        "same_ids": a["ids"] == b["ids"],
                        "same_weights": a["weights"] == b["weights"],
                        "same_test_predictions": same_test,
                    }
                )
    return pd.DataFrame(rows)


def report(run_ids, predictions_only=False):
    roots = [Path("results/runs") / r for r in run_ids]
    frames, checks, screening, configs, resources = [], [], [], [], []
    for root in roots:
        cfg, c, s = verify_sweep(root, predictions_only)
        checks.extend(c)
        screening.extend(s)
        configs.append(cfg)
        frames.append(pd.read_csv(root / "records.csv"))
        resources.append(
            {"run_id": root.name, **json.loads((root / "status.json").read_text())["resources"]}
        )
    normalized = [normalized_protocol(cfg) for cfg in configs]
    if any(c != normalized[0] for c in normalized):
        raise ValueError("Panel methodology differs across task configs")
    records = pd.concat(frames, ignore_index=True)
    if set(records.dataset.unique()) != set(NAMES):
        raise ValueError("Report must retain every registered dataset")
    if records.duplicated(["dataset", "seed", "fold", "method"]).any():
        raise ValueError("Repeated paired-split records")
    for cfg in configs:
        name = cfg["datasets"][0]["id"]
        expected_keys = {
            (seed, fold, method["id"])
            for seed in cfg["seeds"]
            for fold in range(cfg["outer_folds"])
            for method in cfg["methods"] + cfg["baselines"]
        }
        actual_keys = set(
            records[records.dataset == name][["seed", "fold", "method"]].itertuples(
                index=False, name=None
            )
        )
        if actual_keys != expected_keys:
            raise ValueError("Panel coverage differs from all registered settings")
    output = Path("results/summaries/exp_006_large_v1")
    output.mkdir(parents=True, exist_ok=True)
    records.to_csv(output / "per_split.csv", index=False)
    records.groupby(["dataset", "method", "status"]).size().rename("count").reset_index().to_csv(
        output / "coverage.csv", index=False
    )
    complete = records[records.status == "complete"]
    numeric = complete.select_dtypes(include="number").columns.difference(["seed", "fold"])
    means = complete.groupby(["dataset", "task", "method"])[numeric].mean().reset_index()
    means.to_csv(output / "per_task.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "pool_selection_audit.csv", index=False)
    pd.DataFrame(screening).to_csv(output / "screening_by_cell.csv", index=False)
    pd.DataFrame(resources).to_csv(output / "run_resources.csv", index=False)
    control_equivalences(roots).to_csv(output / "control_equivalences.csv", index=False)
    effects = []
    comparisons = [
        (ANCHOR, x)
        for x in [
            "top_quality_b6000_k64",
            "random_b6000_k64",
            "caruana_b6000_k64",
            "random_forest",
            "extra_trees",
            "xgboost",
            "lightgbm",
            "catboost",
            "rf_leaf5",
            "rf_shallow64",
            "random_patches64",
            "linear",
            "abs_correlation_b6000_k64",
            "signed_correlation_b6000_k64",
            "quality_diversity_l002_b6000_k64",
            "quality_diversity_l01_b6000_k64",
            "quality_diversity_l05_b6000_k64",
            "coerror_cert_b6000_k64",
            "coerror_b1000_k64",
            "coerror_b3000_k64",
            "coerror_b6000_k16",
            "coerror_b6000_k128",
            "coerror_b6000_k256",
            "coerror_swap_b6000_k64",
        ]
    ]
    for name, frame in records.groupby("dataset"):
        metric = "group_auroc" if name == "musk2" else PRIMARY[frame.task.iloc[0]]
        for left, right in comparisons:
            for measure in [
                metric,
                "brier" if metric.endswith("auroc") else "normalized_squared_loss",
            ]:
                effects.append(
                    paired_effect(frame, left, right, measure, measure.endswith("auroc"))
                )
    effects = pd.concat(effects, ignore_index=True)
    effects.to_csv(output / "paired_effects.csv", index=False)
    effects.groupby(["dataset", "left", "right", "metric"]).effect_favors_left.agg(
        ["mean", "min", "max", "count"]
    ).reset_index().to_csv(output / "paired_effect_summary.csv", index=False)
    figures = Path("paper/figures/exp_006_large_v1")
    figures.mkdir(parents=True, exist_ok=True)
    tables = Path("paper/tables/exp_006_large_v1")
    tables.mkdir(parents=True, exist_ok=True)
    plot_sweeps(means, figures)
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l" + "r" * len(TABLE_METHODS) + "}",
        r"\toprule",
        "Task / metric & " + " & ".join(TABLE_LABELS) + r" \\",
        r"\midrule",
    ]
    for name in NAMES:
        frame = means[means.dataset == name].set_index("method")
        metric = "group_auroc" if name == "musk2" else PRIMARY[frame.task.iloc[0]]
        lines.append(
            LABELS[name]
            + (" / AUC" if metric.endswith("auroc") else " / RMSE")
            + " & "
            + " & ".join(f"{frame.loc[m, metric]:.3f}" for m in TABLE_METHODS)
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Predesignated $B=6,000,K=64$ co-error anchor and static references. Means of three outer folds and one seed. Exact feature-vector groups for five tasks; molecule groups and molecule-max AUC for Musk. Other sweep cells are reported separately; no test-based winner selection or resource matching.}",
        r"\end{table}",
    ]
    (tables / "anchor.tex").write_text("\n".join(lines) + "\n")
    facts = []
    for name in NAMES:
        frame = means[means.dataset == name].set_index("method")
        metric = "group_auroc" if name == "musk2" else PRIMARY[frame.task.iloc[0]]
        for label, method in [
            ("Co", ANCHOR),
            ("RF", "random_forest"),
            ("Linear", "linear"),
            ("Top", "top_quality_b6000_k64"),
            ("Caruana", "caruana_b6000_k64"),
            ("SmallB", "coerror_b1000_k64"),
        ]:
            key = "".join(w.capitalize() for w in name.split("_")) + label
            facts.append(
                r"\newcommand{\ExpSix" + key + "}{" + f"{frame.loc[method, metric]:.3f}" + "}"
            )
    (tables / "facts.tex").write_text("\n".join(facts) + "\n")
    anchor = means[means.method.isin(TABLE_METHODS)][
        [
            "dataset",
            "task",
            "method",
            "auroc",
            "group_auroc",
            "rmse",
            "brier",
            "log_loss",
            "train_cpu_seconds",
            "train_wall_seconds",
            "model_bytes",
            "leaves",
            "retained_count",
        ]
    ]
    anchor.to_csv(output / "anchor_and_references.csv", index=False)
    note = "# Experiment 006: larger-data sweep\n\nAll five original tasks and the registered Musk extension retained; one seed and three group-separated outer folds. Molecule-max AUROC is primary only for Musk.\nNo calibrated intervals or best-test tuned method; static baselines and unmatched budgets.\n\n"
    for name in NAMES:
        frame = means[means.dataset == name].set_index("method")
        metric = "group_auroc" if name == "musk2" else PRIMARY[frame.task.iloc[0]]
        note += f"## {LABELS[name]} ({metric})\n\nAnchor {frame.loc[ANCHOR, metric]:.6f}; top {frame.loc['top_quality_b6000_k64', metric]:.6f}; Caruana {frame.loc['caruana_b6000_k64', metric]:.6f}; RF {frame.loc['random_forest', metric]:.6f}; linear {frame.loc['linear', metric]:.6f}.\n\n"
        note += (
            "Co-error pool curve (K64): "
            + ", ".join(
                f"B{b}={frame.loc[f'coerror_b{b}_k64', metric]:.6f}"
                for b in [300, 1000, 3000, 6000]
            )
            + ".\n\n"
        )
        note += (
            "Co-error retained-size curve (B6000): "
            + ", ".join(
                f"K{k}={frame.loc[f'coerror_b6000_k{k}', metric]:.6f}" for k in [16, 64, 128, 256]
            )
            + ".\n\n"
        )
    (output / "research_note.md").write_text(note)
    write_json(
        output / "provenance.json",
        {
            "runs": run_ids,
            "report_script_sha256": sha256(Path(__file__)),
            "anchor": ANCHOR,
            "configs": configs,
            "sweep_selection_checks": len(checks),
            "interval_scope": "Descriptive paired folds only; one seed; no task-population inference",
            "resource_scope": "Every setting charged full shared 6000-candidate library; no matched standalone prefix CPU budget",
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--predictions-only", action="store_true")
    args = parser.parse_args()
    print(report(args.run_ids, args.predictions_only))
