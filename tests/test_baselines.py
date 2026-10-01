import numpy as np
import pandas as pd
import pytest

from decorrelated_ensemble.baselines import baseline, capacity
from decorrelated_ensemble.evaluation.runner import predict_baseline


@pytest.mark.parametrize("task", ["binary", "regression", "multiclass"])
@pytest.mark.parametrize(
    "family", ["random_forest", "extra_trees", "random_patches", "xgboost", "lightgbm", "catboost"]
)
def test_baseline_probability_shapes_and_capacity(task, family):
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(45, 8)))
    y = np.arange(45) % (3 if task == "multiclass" else 2)
    if task == "regression":
        y = y.astype(float)
    params = {"iterations": 3, "depth": 2} if family == "catboost" else {"n_estimators": 3}
    if family == "random_patches":
        params.update(max_depth=2, feature_fraction=0.5)
    model = baseline(
        {"family": family, "parameters": params, "classes": np.unique(y)}, X, task, 5
    ).fit(X, y)
    prediction = predict_baseline(model, X, task)
    assert np.isfinite(prediction).all()
    assert prediction.shape == ((45, 3) if task == "multiclass" else (45,))
    if task == "multiclass":
        np.testing.assert_allclose(prediction.sum(axis=1), 1)
    measured = capacity(model)
    assert measured["nodes"] >= measured["leaves"] > 0
