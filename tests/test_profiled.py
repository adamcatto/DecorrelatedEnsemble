import numpy as np
import pandas as pd
import pytest

from decorrelated_ensemble.candidates import generate_candidates
from decorrelated_ensemble.evaluation.runner import fit_selection
from decorrelated_ensemble.selection import Selection
from decorrelated_ensemble.selection.profiled import ProfiledLoss, fit_affine, profiled_greedy


def test_affine_known_slope_intercept_and_nonnegative_constraint():
    p = np.arange(8, dtype=float)
    parameters = fit_affine(2 * p + 3, p)
    assert parameters["slope"] == 2 and parameters["intercept"] == 3
    assert fit_affine(-p, p)["slope"] == 0
    assert fit_affine(p, np.ones(8))["slope"] == 0


def test_profiled_greedy_matches_explicit_affine_fit_and_swap_objective():
    rng = np.random.default_rng(41)
    y = rng.normal(size=35)
    P = rng.normal(size=(35, 9)) + 0.4 * y[:, None]
    oracle = ProfiledLoss(y, P)
    selected = []
    for _ in range(4):
        possibilities = []
        for j in set(range(9)) - set(selected):
            p = P[:, selected + [j]].mean(axis=1)
            parameters = fit_affine(y, p)
            loss = np.mean((y - parameters["slope"] * p - parameters["intercept"]) ** 2)
            possibilities.append((loss, j))
        selected.append(min(possibilities)[1])
    greedy = profiled_greedy(y, P, 4)
    assert greedy.ids.tolist() == selected
    optimized = profiled_greedy(y, P, 4, 100)
    assert optimized.objective <= greedy.objective + 1e-12
    for out in optimized.ids:
        for add in set(range(9)) - set(optimized.ids):
            assert (
                oracle.objective([i for i in optimized.ids if i != out] + [add])
                >= optimized.objective - 1e-12
            )


def test_profiled_loss_invariant_to_aggregate_positive_scale_and_offset():
    rng = np.random.default_rng(8)
    y = rng.normal(size=30)
    P = rng.normal(size=(30, 4))
    assert ProfiledLoss(y, P).objective([0, 2]) == pytest.approx(
        ProfiledLoss(y, 5 * P + 10).objective([0, 2])
    )


def test_calibration_fitted_from_oof_not_refit_training_predictions():
    X = pd.DataFrame({"x": np.arange(20, dtype=float)})
    y = 2 * np.arange(20) + 3.0
    specs = generate_candidates(1, {"B": 1, "feature_fractions": [1.0], "depths": [2]}, 5)
    chosen = Selection(np.array([0]), np.array([1.0]), 0, [])
    oof = np.arange(20, dtype=float)[:, None]
    model = fit_selection(X, y, specs, chosen, "regression", calibration=True, oof_predictions=oof)
    assert model.parameters["slope"] == 2 and model.parameters["intercept"] == 3
    np.testing.assert_allclose(model.predict(X), 2 * model.base.predict(X) + 3)
