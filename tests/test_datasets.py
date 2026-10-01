import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes

from decorrelated_ensemble.datasets import load_dataset


@pytest.mark.parametrize(
    "name,shape,task",
    [
        ("breast_cancer", (569, 30), "binary"),
        ("wine", (178, 13), "multiclass"),
        ("diabetes", (442, 10), "regression"),
    ],
)
def test_builtin_provenance_and_seed_invariant_data(name, shape, task):
    config = {"id": name, "source": "sklearn", "name": name}
    first, second = load_dataset(config, 11), load_dataset(config, 29)
    assert first.X.shape == shape and first.task == task
    assert first.metadata["role"] == "development"
    assert first.metadata["missing_values"] == 0
    assert first.metadata["content_sha256"] == second.metadata["content_sha256"]
    pd.testing.assert_frame_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert first.metadata["feature_names"] == list(first.X.columns)
    if task != "regression":
        assert sum(first.metadata["class_counts"].values()) == shape[0]


def test_diabetes_loader_returns_raw_not_full_data_scaled_features():
    data = load_dataset({"id": "diabetes", "source": "sklearn", "name": "diabetes"}, 11)
    raw = load_diabetes(as_frame=True, scaled=False)
    pd.testing.assert_frame_equal(data.X, raw.data)
    assert data.metadata["loader_arguments"] == {"as_frame": True, "scaled": False}
    assert not np.allclose(data.X, load_diabetes(as_frame=True).data)
