"""Explicit partial-panel report at dataset-level audit barriers."""

import json
from pathlib import Path

import pandas as pd
from make_large_report import control_equivalences

from decorrelated_ensemble.evaluation.artifacts import sha256, verify_manifest, write_json

names = ["credit_default", "higgs", "miniboone"]
labels = {"credit_default": "Credit default", "higgs": "HIGGS", "miniboone": "MiniBooNE"}
frames = []
for name in names:
    root = Path("results/runs") / f"exp_006_large_{name}_v1"
    verify_manifest(root)
    if json.loads((root / "status.json").read_text())["state"] != "complete":
        raise ValueError("Incomplete classification task")
    frames.append(pd.read_csv(root / "records.csv"))
records = pd.concat(frames, ignore_index=True)
output = Path("results/summaries/exp_006_classification_partial")
output.mkdir(parents=True, exist_ok=True)
records.to_csv(output / "per_split.csv", index=False)
means = (
    records[records.status == "complete"]
    .groupby(["dataset", "method"])
    .mean(numeric_only=True)
    .reset_index()
)
means.to_csv(output / "per_task.csv", index=False)
control_equivalences([Path("results/runs") / f"exp_006_large_{n}_v1" for n in names]).to_csv(
    output / "control_equivalences.csv", index=False
)
tables = Path("paper/tables/exp_006_classification_partial")
tables.mkdir(parents=True, exist_ok=True)
methods = [
    "coerror_b6000_k64",
    "top_quality_b6000_k64",
    "random_forest",
    "linear",
    "catboost",
    "lightgbm",
]
lines = [
    r"\begin{table}[t]",
    r"\centering\small",
    r"\begin{tabular}{lrrrrrrr}",
    r"\toprule",
    r"Task & Co-error & Top & RF & RF leaf5 & Logistic & CatBoost & LightGBM \\",
    r"\midrule",
]
facts = []
for name in names:
    frame = means[means.dataset == name].set_index("method")
    lines.append(
        labels[name]
        + " & "
        + " & ".join(
            f"{frame.loc[m, 'auroc']:.3f}" for m in methods[:3] + ["rf_leaf5"] + methods[3:]
        )
        + r" \\"
    )
    for suffix, method in [
        ("Co", "coerror_b6000_k64"),
        ("RF", "random_forest"),
        ("Linear", "linear"),
        ("Top", "top_quality_b6000_k64"),
        ("LeafFive", "rf_leaf5"),
    ]:
        key = "".join(w.capitalize() for w in name.split("_")) + suffix
        facts.append(
            r"\newcommand{\ExpSixPartial" + key + "}{" + f"{frame.loc[method, 'auroc']:.3f}" + "}"
        )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Experiment 006 classification checkpoint: fixed $B=6,000,K=64$ co-error anchor and static references, three outer feature-group folds and one seed. Regression tasks are reported separately. No test-based configuration selection or resource matching.}",
    r"\end{table}",
]
(tables / "anchor.tex").write_text("\n".join(lines) + "\n")
(tables / "facts.tex").write_text("\n".join(facts) + "\n")
write_json(
    output / "provenance.json",
    {
        "datasets": names,
        "outside_this_classification_report": ["superconductivity", "california_housing", "musk2"],
        "report_script_sha256": sha256(Path(__file__)),
        "scope": "Partial registered development panel; not final confirmation; task means of three overlapping outer folds; no intervals or best-test tuning",
    },
)
print(output)

# Optional separate regression checkpoint; never overwrites the historical class panel.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--regression-progress", action="store_true")
args = parser.parse_args()
if args.regression_progress:
    reg_names = ["superconductivity", "california_housing"]
    reg_labels = {
        "superconductivity": "Superconductivity",
        "california_housing": "California housing",
    }
    reg_frames = []
    for name in reg_names:
        root = Path("results/runs") / f"exp_006_large_{name}_v1"
        verify_manifest(root)
        if json.loads((root / "status.json").read_text())["state"] != "complete":
            raise ValueError("Incomplete regression task")
        reg_frames.append(pd.read_csv(root / "records.csv"))
    reg_records = pd.concat(reg_frames, ignore_index=True)
    reg_means = reg_records.groupby(["dataset", "method"]).mean(numeric_only=True).reset_index()
    reg_output = Path("results/summaries/exp_006_regression_partial")
    reg_output.mkdir(parents=True, exist_ok=True)
    reg_records.to_csv(reg_output / "per_split.csv", index=False)
    reg_means.to_csv(reg_output / "per_task.csv", index=False)
    reg_tables = Path("paper/tables/exp_006_regression_partial")
    reg_tables.mkdir(parents=True, exist_ok=True)
    reg_lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Task & Co-error & Top & RF & Ridge & CatBoost & LightGBM \\",
        r"\midrule",
    ]
    reg_facts = []
    for name in reg_names:
        frame = reg_means[reg_means.dataset == name].set_index("method")
        reg_lines.append(
            reg_labels[name]
            + " & "
            + " & ".join(f"{frame.loc[m, 'rmse']:.3f}" for m in methods)
            + r" \\"
        )
        for suffix, method in [
            ("Co", methods[0]),
            ("Top", methods[1]),
            ("RF", methods[2]),
            ("Linear", methods[3]),
        ]:
            key = "".join(w.capitalize() for w in name.split("_")) + suffix
            reg_facts.append(
                r"\newcommand{\ExpSixReg" + key + "}{" + f"{frame.loc[method, 'rmse']:.3f}" + "}"
            )
    reg_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Experiment 006 regression checkpoint, RMSE (lower is better). Fixed $B=6,000,K=64$ anchor; three exact-feature-group outer folds, one seed. Musk remains pending. Static references, unmatched resources.}",
        r"\end{table}",
    ]
    (reg_tables / "anchor.tex").write_text("\n".join(reg_lines) + "\n")
    (reg_tables / "facts.tex").write_text("\n".join(reg_facts) + "\n")
    write_json(
        reg_output / "provenance.json",
        {
            "datasets": reg_names,
            "remaining": ["musk2"],
            "script_sha256": sha256(Path(__file__)),
            "scope": "Separate regression partial report; no population interval or test-based settings",
        },
    )
    print(reg_output)
