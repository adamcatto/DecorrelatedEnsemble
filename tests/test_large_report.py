import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("large_report", Path("scripts/make_large_report.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_large_report_rejects_changed_group_protocol():
    import yaml

    cfg = yaml.safe_load(Path("configs/experiments/exp_006_large_musk2.yaml").read_text())
    normalized = module.normalized_protocol(cfg)
    assert "source_groups" not in normalized
    cfg["group_aggregation"] = "mean"
    with pytest.raises(ValueError, match="Registered group protocol changed"):
        module.normalized_protocol(cfg)


def test_large_paired_effect_directions_and_infeasible_support():
    frame = pd.DataFrame(
        [
            {
                "dataset": "d",
                "seed": 11,
                "fold": 0,
                "method": "left",
                "status": "complete",
                "auroc": 0.75,
                "rmse": 2,
            },
            {
                "dataset": "d",
                "seed": 11,
                "fold": 1,
                "method": "left",
                "status": "infeasible",
                "auroc": None,
                "rmse": None,
            },
            {
                "dataset": "d",
                "seed": 11,
                "fold": 0,
                "method": "right",
                "status": "complete",
                "auroc": 0.7,
                "rmse": 3,
            },
            {
                "dataset": "d",
                "seed": 11,
                "fold": 1,
                "method": "right",
                "status": "complete",
                "auroc": 0.9,
                "rmse": 1,
            },
        ]
    )
    a = module.paired_effect(frame, "left", "right", "auroc", True)
    b = module.paired_effect(frame, "left", "right", "rmse", False)
    assert len(a) == len(b) == 1
    assert a.effect_favors_left.iloc[0] == pytest.approx(0.05)
    assert b.effect_favors_left.iloc[0] == 1


def test_lossless_oof_repacking_preserves_every_array(tmp_path):
    import numpy as np

    spec = importlib.util.spec_from_file_location("export_run", Path("scripts/export_run.py"))
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    original, exported = tmp_path / "original.npz", tmp_path / "exported.npz"
    arrays = {
        "predictions": np.arange(60, dtype=float).reshape(10, 6),
        "null": np.zeros(10),
        "fold_ids": np.arange(10) % 2,
        "y": np.arange(10),
    }
    np.savez_compressed(original, **arrays)
    exporter.repack_oof(original, exported)
    with np.load(exported) as result:
        for key, value in arrays.items():
            np.testing.assert_array_equal(result[key], value)
            assert result[key].dtype == value.dtype
        assert result["predictions"].flags.f_contiguous
