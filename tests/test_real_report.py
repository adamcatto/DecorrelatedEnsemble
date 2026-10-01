import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def test_real_effect_directions_and_shared_feasible_support():
    path = Path(__file__).parents[1] / "scripts/make_real_report.py"
    spec = importlib.util.spec_from_file_location("real_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for task, metric, value, direction in [
        ("binary", "auroc", 0.7, 1),
        ("multiclass", "log_loss", 0.5, -1),
        ("regression", "rmse", 2.0, -1),
    ]:
        for fraction in [0.1, 0.5]:
            for fold in [0, 1]:
                for method in ["coerror_no_cert", "top_no_cert"]:
                    improvement = 0.1 * (method == "coerror_no_cert") + 0.2 * (fraction == 0.5)
                    rows.append(
                        {
                            "dataset": task,
                            "task": task,
                            "feature_fraction": fraction,
                            "seed": 11,
                            "fold": fold,
                            "method": method,
                            "status": "infeasible"
                            if method == "top_no_cert" and fold == 1
                            else "complete",
                            metric: value + direction * improvement,
                            "brier": 1 - improvement,
                            "normalized_squared_loss": 1 - improvement,
                        }
                    )
    within, width = module.paired_effects(pd.DataFrame(rows))
    core = within[within.metric.isin(["auroc", "log_loss", "rmse"])]
    assert len(core) == 6  # One shared feasible split per task and width.
    assert core.advantage.to_numpy() == pytest.approx([0.1] * 6)
    core = width[width.metric.isin(["auroc", "log_loss", "rmse"])]
    assert len(core) == 9  # Co-error has two pairs; top quality only one per task.
    assert core.advantage.to_numpy() == pytest.approx([0.2] * 9)
