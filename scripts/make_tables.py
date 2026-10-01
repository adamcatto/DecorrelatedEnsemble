import argparse
from pathlib import Path

import pandas as pd


def escape(text):
    return str(text).replace("_", r"\_")


def make_tables(run_id):
    data = pd.read_csv(Path("results/summaries") / run_id / "per_task.csv")
    out = Path("paper/tables")
    out.mkdir(parents=True, exist_ok=True)
    methods = [
        "random_subspace",
        "top_quality",
        "abs_correlation",
        "coerror",
        "coerror_no_cert",
        "random_forest",
        "catboost",
    ]
    methods = [m for m in methods if m in data.method.unique()]
    for task, metric in [("binary", "auroc"), ("regression", "normalized_squared_loss")]:
        table = (
            data[data.task == task]
            .pivot(index="dataset", columns="method", values=metric)
            .reindex(columns=methods)
        )
        if not len(table):
            continue
        labels = {
            "random_subspace": "RS",
            "top_quality": "Top",
            "abs_correlation": "AbsCorr",
            "coerror": "CoErr",
            "coerror_no_cert": "CoErr-all",
            "random_forest": "RF",
            "catboost": "CatBoost",
        }
        text = [
            r"\begin{table}[t]",
            r"\centering\small",
            r"\begin{tabular}{l" + "r" * len(methods) + "}",
            r"\toprule",
            "Regime & " + " & ".join(labels[m] for m in methods) + r" \\",
            r"\midrule",
        ]
        for name, row in table.iterrows():
            text.append(
                escape(name.replace("_" + task, ""))
                + " & "
                + " & ".join("--" if pd.isna(v) else f"{v:.3f}" for v in row)
                + r" \\"
            )
        text.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\caption{"
                + escape(run_id)
                + ": "
                + task
                + ", "
                + escape(metric)
                + ". Means over completed outer folds and seeds. Missing entries indicate fixed-K infeasibility; full coverage is stored separately. Static baselines, unmatched compute; development only.}",
                r"\end{table}",
            ]
        )
        (out / (run_id + "_" + task + ".tex")).write_text("\n".join(text) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    print(make_tables(parser.parse_args().run_id))
