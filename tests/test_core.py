import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from decorrelated_ensemble.candidates import CandidateSpec, MaskedTree, generate_candidates
from decorrelated_ensemble.certification import bootstrap_indices, certify
from decorrelated_ensemble.datasets import synthetic
from decorrelated_ensemble.evaluation.oof import build_oof, make_splits, verify_splits
from decorrelated_ensemble.evaluation.runner import fit_selection, nested_tune
from decorrelated_ensemble.metrics import (
    binary_auc_columns,
    evaluate_metrics,
    quality_columns,
    residual_matrix,
    squared_loss,
)
from decorrelated_ensemble.selection import (
    CoError,
    Selection,
    coerror_greedy,
    correlation_matrix,
    direct_greedy,
    exact_coerror,
    pair_greedy,
    pair_objective,
    select,
)


def spec(features=(0,), row_fraction=1):
    return CandidateSpec(0, features, 0.1, 3, 2, row_fraction, 123, "default")


def config(B=8):
    return {"B": B, "feature_fractions": [0.1, 0.2], "depths": [2, 3], "min_samples_leaf": [2]}


def test_feature_mask_and_fold_local_preprocessing():
    X = pd.DataFrame({"kept": [0.0, 1, 2, np.nan, 4, 5], "forbidden": [0, 1, 0, 1, 0, 1]})
    y = np.array([0, 0, 0, 1, 1, 1])
    model = MaskedTree(spec(), "binary").fit(X, y)
    changed = X.copy()
    changed["forbidden"] = np.nan
    np.testing.assert_array_equal(model.predict(X), model.predict(changed))
    assert list(model.pipeline_["preprocess"].feature_names_in_) == ["kept"]
    assert model.pipeline_["preprocess"].named_transformers_["numerical"].statistics_[0] == 2


def test_categorical_unknown_not_seen_during_fit():
    X = pd.DataFrame({"a": ["x", "y", "x", "y", "x", "y"]})
    model = MaskedTree(spec(), "binary").fit(X, np.array([0, 1, 0, 1, 0, 1]))
    encoded = model.pipeline_["preprocess"].named_transformers_["categorical"]["encode"]
    assert "heldout" not in encoded.categories_[0]
    assert np.isfinite(model.predict(pd.DataFrame({"a": ["heldout"]}))).all()


def test_masks_nested_prefix_and_small_p():
    small, large = generate_candidates(3, config(8), 7), generate_candidates(3, config(16), 7)
    assert small == large[:8]
    assert all(len(s.features) >= 1 and max(s.features) < 3 for s in large)
    assert small == generate_candidates(3, config(8), 7)


def test_row_subsampling_enforces_training_rows():
    X = pd.DataFrame(np.arange(40).reshape(20, 2))
    model = MaskedTree(spec(row_fraction=0.5), "regression").fit(X, np.arange(20))
    assert len(model.fit_rows_) == 10 and model.fit_rows_.max() < 20


def test_split_integrity_and_oof_refit_manual():
    data = synthetic({"regime": "additive", "task": "binary", "n": 90, "p": 10}, 5)
    specs = generate_candidates(10, config(3), 1)
    result = build_oof(data.X, data.y, specs, data.task, 3, 8)
    for fold, (train, valid) in enumerate(result.splits):
        assert set(train).isdisjoint(valid)
        assert np.all(result.fold_ids[valid] == fold)
        expected = (
            MaskedTree(specs[0], data.task)
            .fit(data.X.iloc[train], data.y[train])
            .predict(data.X.iloc[valid])
        )
        np.testing.assert_array_equal(result.predictions[valid, 0], expected)
        assert np.all(result.null[valid] == np.mean(data.y[train]))
    repeated = build_oof(data.X, data.y, specs, data.task, 3, 8)
    np.testing.assert_array_equal(result.predictions, repeated.predictions)


def test_heldout_label_cannot_change_its_oof_prediction():
    X = pd.DataFrame(np.arange(60).reshape(30, 2))
    y = np.arange(30, dtype=float)
    result = build_oof(X, y, [spec()], "regression", 3, 42)
    train, valid = result.splits[0]
    modified = y.copy()
    modified[valid] += 10000
    refit = MaskedTree(spec(), "regression").fit(X.iloc[train], modified[train])
    np.testing.assert_array_equal(refit.predict(X.iloc[valid]), result.predictions[valid, 0])


@pytest.mark.parametrize(
    "splits,n",
    [([(np.array([0, 1]), np.array([1, 2]))], 3), ([(np.array([0, 1]), np.array([2]))], 3)],
)
def test_invalid_splits_rejected(splits, n):
    with pytest.raises(ValueError):
        verify_splits(splits, n)


def test_auc_ties_and_metrics_analytic():
    y = np.array([0, 0, 1, 1])
    P = np.array([[0.1, 0.5], [0.2, 0.5], [0.8, 0.5], [0.9, 0.5]])
    np.testing.assert_allclose(binary_auc_columns(y, P), [1, 0.5])
    np.testing.assert_allclose(
        binary_auc_columns(y, P), [roc_auc_score(y, P[:, j]) for j in range(2)]
    )
    metrics = evaluate_metrics(y, P[:, 0], "binary")
    assert metrics["brier"] == pytest.approx(0.025)
    reg = evaluate_metrics(np.array([1.0, 2]), np.array([1.0, 3]), "regression")
    assert reg["rmse"] == pytest.approx(np.sqrt(0.5))
    assert reg["mae"] == 0.5


def test_skill_crossfitted_null():
    y = np.array([0.0, 2, 4])
    P = np.column_stack([y, np.array([2.0, 2, 2])])
    np.testing.assert_allclose(quality_columns(y, P, np.full(3, 2.0), "regression"), [1, 0])


def test_stratified_joint_bootstrap():
    y = np.array([0] * 17 + [1] * 3)
    rows = bootstrap_indices(y, "binary", np.random.default_rng(4))
    np.testing.assert_array_equal(np.bincount(y[rows]), [17, 3])
    P = np.column_stack([y * 0.8 + 0.1, y * 0.8 + 0.1, np.full(20, 0.5)])
    certificate = certify(y, P, np.full(20, 0.15), "binary", {"bootstrap_reps": 100}, 4)
    np.testing.assert_array_equal(certificate.bootstrap[:, 0], certificate.bootstrap[:, 1])
    assert certificate.lower[0] == 1 and certificate.lower[2] == 0.5
    assert certificate.passed.tolist() == [True, True, False]
    with pytest.raises(ValueError):
        certify(y, P, np.full(20, 0.15), "binary", {"bootstrap_reps": 10}, 4)


def test_covariance_and_bias_identity():
    E = np.array([[1, 2, 3], [2, -1, 4], [0, 0, 5.0]], float)
    mu, G = E.mean(axis=0), E.T @ E / len(E)
    centered = E - mu
    np.testing.assert_allclose(G, centered.T @ centered / len(E) + np.outer(mu, mu))
    np.testing.assert_allclose(CoError(E).dense(), G)
    assert CoError(E).objective([0, 1]) == pytest.approx(np.mean(E[:, :2].mean(axis=1) ** 2))
    np.testing.assert_allclose(
        CoError(E, 1).dense(), np.diag(np.diag(G) - mu**2) + np.outer(mu, mu)
    )


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0])
def test_incremental_greedy_and_swaps(alpha):
    oracle = CoError(np.random.default_rng(0).normal(size=(15, 9)), alpha)
    dense = oracle.dense()
    chosen = []
    for _ in range(4):
        options = [
            (dense[np.ix_(chosen + [j], chosen + [j])].sum(), j)
            for j in range(9)
            if j not in chosen
        ]
        chosen.append(min(options)[1])
    greedy = coerror_greedy(oracle, 4)
    assert greedy.ids.tolist() == chosen
    swap = coerror_greedy(oracle, 4, 100)
    assert swap.objective <= greedy.objective + 1e-12
    for remove in swap.ids:
        for add in set(range(9)) - set(swap.ids):
            trial = [i for i in swap.ids if i != remove] + [add]
            assert oracle.objective(trial) >= swap.objective - 1e-12


def test_exact_analytic_complementarity():
    oracle = CoError(np.array([[1, -1, 2], [1, -1, 2.0]]))
    chosen = exact_coerror(oracle, 2)
    assert chosen.ids.tolist() == [0, 1] and chosen.objective == 0
    with pytest.raises(ValueError):
        exact_coerror(oracle, 2, max_combinations=2)


def test_direct_squared_equals_coerror_greedy():
    y = np.random.default_rng(3).normal(size=20)
    P = np.random.default_rng(4).normal(size=(20, 10))
    greedy = coerror_greedy(CoError(y[:, None] - P), 4)
    direct = direct_greedy(y, P, "regression", 4)
    assert set(greedy.ids) == set(direct.ids)
    assert greedy.objective == pytest.approx(direct.objective)


def test_binary_coerror_sign_and_bias_cancellation_counterexample():
    y = np.array([0, 1])
    P = np.array([[0.7, 0.1], [0.9, 0.3]])
    E = residual_matrix(y, P, "binary")
    G = E.T @ E / len(y)
    np.testing.assert_allclose(G, [[0.25, 0.07], [0.07, 0.25]])
    np.testing.assert_allclose(correlation_matrix(E), np.ones((2, 2)))
    np.testing.assert_allclose(E.mean(axis=0), [-0.3, 0.3])
    assert squared_loss(y, P.mean(axis=1), "binary") == pytest.approx(0.16)
    rng = np.random.default_rng(912)
    y_random = rng.integers(0, 2, 100)
    residuals = residual_matrix(y_random, rng.random((100, 20)), "binary")
    assert np.all(residuals.T @ residuals >= 0)


def test_negative_correlation_and_constant_convention():
    E = np.array([[1, -1, 1], [-1, 1, 1], [2, -2, 1.0]])
    R = correlation_matrix(E)
    assert R[0, 1] == pytest.approx(-1)
    assert R[0, 2] == R[1, 2] == 1


@pytest.mark.parametrize("kind", ["mean", "minimax", "quality_diversity"])
def test_pair_greedy_objective_and_local_search(kind):
    R = np.abs(correlation_matrix(np.random.default_rng(9).normal(size=(20, 7))))
    q = np.linspace(0.5, 0.8, 7)
    g = pair_greedy(R, q, 3, kind, 0.2, 1, 0)
    optimized = pair_greedy(R, q, 3, kind, 0.2, 1, 20)
    assert optimized.objective <= g.objective + 1e-12
    assert optimized.objective == pytest.approx(pair_objective(optimized.ids, R, q, kind, 0.2))


def test_multiclass_brier_and_class_bias_shrinkage():
    y = np.array([0, 1, 2, 0])
    P = np.random.default_rng(0).uniform(size=(4, 3, 3))
    P /= P.sum(axis=2, keepdims=True)
    E = residual_matrix(y, P, "multiclass")
    oracle = CoError(E)
    assert oracle.objective([0, 1]) == pytest.approx(
        squared_loss(y, P[:, :2].mean(axis=1), "multiclass")
    )
    bias = (np.eye(3)[y][:, None, :] - P).mean(axis=0).T
    raw = E.T @ E / len(E)
    # Pooled sample/class residual means vanish; this convention is normalized G.
    np.testing.assert_allclose(E.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(
        correlation_matrix(E), raw / np.sqrt(np.outer(np.diag(raw), np.diag(raw)))
    )
    expected = np.diag(np.diag(raw) - np.sum(bias**2, axis=0)) + bias.T @ bias
    np.testing.assert_allclose(CoError(E, 1, n_classes=3).dense(), expected)


def test_selected_specs_refit_full_training():
    data = synthetic({"regime": "sparse", "task": "regression", "n": 60, "p": 10}, 6)
    specs = generate_candidates(10, config(3), 9)
    chosen = Selection(np.array([0, 2]), np.array([0.5, 0.5]), 0, [])
    model = fit_selection(data.X, data.y, specs, chosen, data.task)
    assert all(len(m.fit_rows_) == 60 for m in model.models_)
    manual = np.mean(
        [MaskedTree(specs[i], data.task).fit(data.X, data.y).predict(data.X) for i in [0, 2]],
        axis=0,
    )
    np.testing.assert_array_equal(model.predict(data.X), manual)


def test_infeasible_fixed_K_no_fallback():
    y = np.array([0, 0, 1, 1])
    P = np.column_stack([y, y])
    with pytest.raises(ValueError, match="infeasible"):
        select(
            y,
            P,
            np.ones(4) * 0.5,
            "binary",
            np.array([1.0, 1]),
            np.array([True, False]),
            {"K": 2, "selector": "random"},
            1,
        )


def test_nested_tuning_rebuilds_inside_validation_boundary(monkeypatch):
    from decorrelated_ensemble.evaluation import runner

    data = synthetic({"regime": "additive", "task": "binary", "n": 90, "p": 10}, 5)
    specs = generate_candidates(10, config(4), 9)
    seen = []
    original = runner.build_oof

    def tracked(X, y, *args, **kwargs):
        seen.append(set(X.index))
        return original(X, y, *args, **kwargs)

    monkeypatch.setattr(runner, "build_oof", tracked)
    method = {
        "K": 2,
        "selector": "coerror",
        "certified": False,
        "tuning_grid": [{"shrinkage": 0.0}, {"shrinkage": 0.5}],
        "tuning_folds": 3,
    }
    _, trace = nested_tune(data.X, data.y, specs, "binary", {"bootstrap_reps": 100}, method, 2, 7)
    assert len(seen) == 3 and all(len(indices) == 60 for indices in seen)
    for indices, (_, valid) in zip(seen, make_splits(data.y, "binary", 3, 7)):
        assert indices.isdisjoint(valid)
    assert "chosen_option" in trace[-1]


def test_top_squared_is_quality_independent_and_stable_on_ties():
    y = np.array([0, 1])
    P = np.array([[0.4, 0.1, 0.1], [0.6, 0.9, 0.9]])
    quality = np.array([0.99, 0.7, 0.6])
    selection = select(
        y,
        P,
        np.array([0.5, 0.5]),
        "binary",
        quality,
        np.array([True, True, True]),
        {"selector": "top_squared", "K": 2},
        11,
    )
    np.testing.assert_array_equal(selection.ids, [1, 2])
    assert selection.objective == pytest.approx(0.01)
    selected_only = select(
        y,
        P,
        np.array([0.5, 0.5]),
        "binary",
        quality,
        np.array([True, False, True]),
        {"selector": "top_squared", "K": 1},
        11,
    )
    np.testing.assert_array_equal(selected_only.ids, [2])


def test_pooled_oof_auc_can_distort_a_grouped_null_predictor():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    X = pd.DataFrame({"constant": np.zeros(8)})
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    spec = CandidateSpec(0, (0,), 1.0, 1, 1, 1.0, 11, "default")
    oof = build_oof(X, y, [spec], "binary", 2, 11, groups=groups)
    np.testing.assert_array_equal(oof.predictions[:, 0], oof.null)
    assert roc_auc_score(y, oof.null) == pytest.approx(0.25)
    for _, valid in oof.splits:
        assert roc_auc_score(y[valid], oof.null[valid]) == pytest.approx(0.5)
