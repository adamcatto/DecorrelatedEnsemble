import pandas as pd
import pytest

from decorrelated_ensemble.statistics import paired_task_comparison


def test_dataset_not_row_or_fold_is_unit():
    rows = []
    for dataset, delta in [("a", 0.1), ("b", -0.1)]:
        for fold in range(5):
            for method, value in [("left", 0.5 + delta), ("right", 0.5)]:
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": 1,
                        "fold": fold,
                        "method": method,
                        "status": "complete",
                        "auc": value,
                    }
                )
    result = paired_task_comparison(pd.DataFrame(rows), "left", "right", "auc", True)
    assert result["paired_tasks"] == 2 and result["paired_splits"] == 10
    assert result["mean_advantage"] == pytest.approx(0)
    assert result["win"] == result["loss"] == 1
