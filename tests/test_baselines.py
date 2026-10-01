import numpy as np
import pandas as pd
import pytest

from decorrelated_ensemble.baselines import baseline, capacity
from decorrelated_ensemble.evaluation.runner import predict_baseline


@pytest.mark.parametrize("task", ["binary", "regression", "multiclass"])
@pytest.mark.parametrize(
    "family",
    ["random_forest", "extra_trees", "random_patches", "xgboost", "lightgbm", "catboost", "linear"],
)
def test_baseline_probability_shapes_and_capacity(task, family):
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(45, 8)))
    y = np.arange(45) % (3 if task == "multiclass" else 2)
    if task == "regression":
        y = y.astype(float)
    params = {"iterations": 3, "depth": 2} if family == "catboost" else {"n_estimators": 3}
    if family == "linear":
        params = {}
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
    if family == "linear":
        assert measured["coefficient_count"] > 0
        assert measured["nodes"] is None and measured["leaves"] is None
    else:
        assert measured["nodes"] >= measured["leaves"] > 0


def test_linear_scaler_uses_only_fitting_rows():
    X = pd.DataFrame({"a": [0.0, 2.0, 4.0, 6.0]})
    model = baseline({"family": "linear"}, X, "regression", 11).fit(X, np.arange(4.0))
    np.testing.assert_allclose(model["scale"].mean_, [3.0])
    prediction = predict_baseline(model, pd.DataFrame({"a": [1000.0]}), "regression")
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(model["scale"].mean_, [3.0])
