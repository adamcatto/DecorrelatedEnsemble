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
    cfg["group_aggregation"] = "max"
    cfg.pop("source_groups")
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


def test_followup_array_digest_is_storage_independent_and_value_sensitive():
    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "alignment", Path("scripts/run_quality_alignment.py")
    )
    aligned = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aligned)
    values = np.arange(40, dtype=np.float64).reshape(10, 4)
    assert aligned.array_digest(values) == aligned.array_digest(np.asfortranarray(values))
    altered = values.copy()
    altered[0, 0] = np.nextafter(0.0, 1.0)
    assert aligned.array_digest(values) != aligned.array_digest(altered)
    assert aligned.array_digest(values) != aligned.array_digest(values.astype(np.float32))


def test_large_metadata_separates_sample_and_full_source_counts(tmp_path):
    import json

    import numpy as np

    root = tmp_path / "run"
    sample = root / "musk2" / "seed_11"
    sample.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"datasets": [{"id": "musk2"}], "seeds": [11]}))
    (sample / "metadata.json").write_text(
        json.dumps(
            {
                "id": "musk2",
                "task": "binary",
                "n_full": 100,
                "missing_values": 12,
                "exact_duplicate_feature_rows": 10,
            }
        )
    )
    pd.DataFrame({"x": [1.0, 1.0, np.nan, 3.0]}).to_pickle(sample / "X.pkl")
    np.save(sample / "y.npy", [0, 0, 1, 1])
    np.save(sample / "groups.npy", ["a", "a", "b", "c"])
    row = module.dataset_metadata([root]).iloc[0]
    assert row.sample_rows == 4 and row.source_full_rows == 100
    assert row.sample_missing_cells == 1 and row.source_catalog_missing_cells == 12
    assert row.sample_repeated_feature_rows == 1
    assert row.source_catalog_repeated_feature_rows == 10
    assert row.split_groups == 3 and row.max_rows_per_group == 2
    assert row.sample_positive_rows == 2 and row.positive_molecules == 2
    assert row.primary_metric == "group_auroc"
    fold = sample / "fold_0"
    fold.mkdir()
    np.savez(fold / "outer_split.npz", train=[2, 1, 0], test=[3])
    partitions = module.partition_metadata([root]).set_index("role")
    assert partitions.loc["train", "rows"] == 3
    assert partitions.loc["train", "split_groups"] == 2
    assert partitions.loc["train", "positive_rows"] == 1
    assert partitions.loc["train", "positive_molecules"] == 1
    assert partitions.loc["test", "split_groups"] == 1
