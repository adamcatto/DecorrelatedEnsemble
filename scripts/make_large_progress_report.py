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
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r"Task & Co-error & Top & RF & Logistic & CatBoost & LightGBM \\",
    r"\midrule",
]
facts = []
for name in names:
    frame = means[means.dataset == name].set_index("method")
    lines.append(
        labels[name] + " & " + " & ".join(f"{frame.loc[m, 'auroc']:.3f}" for m in methods) + r" \\"
    )
    for suffix, method in [
        ("Co", "coerror_b6000_k64"),
        ("RF", "random_forest"),
        ("Linear", "linear"),
        ("Top", "top_quality_b6000_k64"),
    ]:
        key = "".join(w.capitalize() for w in name.split("_")) + suffix
        facts.append(
            r"\newcommand{\ExpSixPartial" + key + "}{" + f"{frame.loc[method, 'auroc']:.3f}" + "}"
        )
lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Partial experiment 006 classification panel: fixed $B=6,000,K=64$ co-error anchor and static references, three outer feature-group folds and one seed. Remaining registered tasks are pending. No test-based configuration selection or resource matching.}",
    r"\end{table}",
]
(tables / "anchor.tex").write_text("\n".join(lines) + "\n")
(tables / "facts.tex").write_text("\n".join(facts) + "\n")
write_json(
    output / "provenance.json",
    {
        "datasets": names,
        "remaining": ["superconductivity", "california_housing", "musk2"],
        "report_script_sha256": sha256(Path(__file__)),
        "scope": "Partial registered development panel; not final confirmation; task means of three overlapping outer folds; no intervals or best-test tuning",
    },
)
print(output)
