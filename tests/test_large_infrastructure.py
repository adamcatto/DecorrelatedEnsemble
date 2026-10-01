import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from decorrelated_ensemble.candidates import generate_candidates
from decorrelated_ensemble.certification import (
    bootstrap_indices,
    certify,
    weighted_bootstrap_quality,
)
from decorrelated_ensemble.datasets import load_dataset, synthetic
from decorrelated_ensemble.evaluation.artifacts import sha256
from decorrelated_ensemble.evaluation.oof import build_oof, make_splits
from decorrelated_ensemble.evaluation.runner import candidate_pool_mask
from decorrelated_ensemble.selection import (
    CoError,
    CorrelationColumns,
    caruana_coerror,
    correlation_matrix,
    direct_greedy,
    pair_greedy,
)


def test_balanced_library_nested_prefix_and_cells():
    cfg = {"B": 60, "feature_fractions": [0.05, 0.2, 0.5], "depths": [3, 6], "balanced_grid": True}
    specs = generate_candidates(81, cfg, 9)
    assert specs[:12] == generate_candidates(81, {**cfg, "B": 12}, 9)
    for f in cfg["feature_fractions"]:
        for d in cfg["depths"]:
            mask = candidate_pool_mask(
                specs, {"candidate_pool": {"feature_fraction": f, "max_depth": d, "limit": 7}}
            )
            assert mask.sum() == 7
            assert all(
                specs[i].feature_fraction == f and specs[i].max_depth == d
                for i in np.flatnonzero(mask)
            )
    assert candidate_pool_mask(specs, {"candidate_pool": {"prefix": 17}}).sum() == 17


@pytest.mark.parametrize("task", ["binary", "regression"])
def test_threaded_oof_exactly_matches_serial_and_groups(task):
    data = synthetic({"regime": "additive", "task": task, "n": 90, "p": 10}, 5)
    specs = generate_candidates(10, {"B": 8, "feature_fractions": [0.2], "depths": [3]}, 1)
    groups = np.repeat(np.arange(30), 3)
    a = build_oof(data.X, data.y, specs, task, 3, 7, groups=groups)
    b = build_oof(data.X, data.y, specs, task, 3, 7, n_jobs=2, groups=groups)
    np.testing.assert_array_equal(a.predictions, b.predictions)
    for train, valid in a.splits:
        assert set(groups[train]).isdisjoint(groups[valid])
    # Both levels use group partitions; rows within a group never straddle them.
    for train, test in make_splits(data.y, task, 3, 11, groups):
        inner = make_splits(data.y[train], task, 2, 12, groups[train])
        assert set(groups[train]).isdisjoint(groups[test])
        for fit, valid in inner:
            assert set(groups[train][fit]).isdisjoint(groups[train][valid])


@pytest.mark.parametrize("task", ["binary", "regression"])
@pytest.mark.parametrize("grouped", [False, True])
def test_fast_bootstrap_equivalent_to_resampling(task, grouped):
    rng = np.random.default_rng(88)
    y = np.arange(60) % 2 if task == "binary" else rng.normal(size=60)
    P = np.round(rng.random((60, 7)), 1)  # Lots of ties, including all-constant column.
    P[:, 0] = 0.5
    groups = np.repeat(np.arange(20), 3) if grouped else None
    null = np.full(60, 0.5 if task == "binary" else 0)
    cfg = {"bootstrap_reps": 100}
    a = certify(y, P, null, task, cfg, 8, groups=groups)
    b = certify(y, P, null, task, {**cfg, "engine": "weighted"}, 8, groups=groups)
    np.testing.assert_allclose(a.bootstrap, b.bootstrap, atol=1e-14)
    np.testing.assert_array_equal(a.passed, b.passed)
    if groups is not None:
        rows = bootstrap_indices(y, task, np.random.default_rng(3), groups)
        counts = np.bincount(rows, minlength=len(y)).reshape(-1, 3)
        assert np.all(counts == counts[:, :1])


def test_weighted_auc_known_ties_and_arbitrary_multiplicities():
    y = np.array([0, 0, 1, 1])
    P = np.array([[0.2, 0.5], [0.5, 0.5], [0.5, 0.5], [0.9, 0.5]])
    counts = np.array([[2.0, 0, 1, 3], [1, 2, 3, 1]])
    out = weighted_bootstrap_quality(y, P, y * 0, "binary", counts)
    for r in range(2):
        rows = np.repeat(np.arange(4), counts[r].astype(int))
        for j in range(2):
            assert out[r, j] == pytest.approx(roc_auc_score(y[rows], P[rows, j]))
    assert out[0, 0] == 1 and np.all(out[:, 1] == 0.5)


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("kind", ["mean", "minimax", "quality_diversity"])
def test_matrix_free_correlation_and_objectives_match_dense(absolute, kind):
    E = np.random.default_rng(8).normal(size=(50, 20))
    E[:, 0] = 0
    q = np.arange(20) / 20
    R = correlation_matrix(E)
    if absolute:
        R = np.abs(R)
    columns = CorrelationColumns(E, absolute)
    for j in range(20):
        np.testing.assert_allclose(columns[j], R[j], atol=1e-14)
    a, b = [pair_greedy(x, q, 5, kind, 0.1, 9, 1) for x in [R, columns]]
    np.testing.assert_array_equal(a.ids, b.ids)
    assert a.objective == pytest.approx(b.objective)


def test_fast_caruana_and_cached_coerror_match_direct():
    rng = np.random.default_rng(99)
    y, P = rng.normal(size=80), rng.normal(size=(80, 17))
    oracle = CoError(y[:, None] - P, cache_columns=True)
    assert oracle.column(3) is oracle.column(3)
    a = caruana_coerror(oracle, 30)
    b = direct_greedy(y, P, "regression", 30, replacement=True)
    np.testing.assert_array_equal(a.ids, b.ids)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.objective == pytest.approx(b.objective)


def test_offline_loader_pinning_seed_independent_subset_and_target_exclusion(tmp_path):
    path = tmp_path / "fixture.pkl"
    X = pd.DataFrame({"signal": np.arange(60), "category": ["a", "b"] * 30})
    y = np.arange(60) % 2
    pd.to_pickle({"X": X, "y": y}, path)
    path.with_suffix(".json").write_text(json.dumps({"task": "binary"}))
    cfg = {
        "id": "fixture",
        "source": "cached_public",
        "path": str(path),
        "sha256": sha256(path),
        "max_samples": 30,
        "subsample_seed": 123,
    }
    a, b = [load_dataset(cfg, seed) for seed in [11, 29]]
    pd.testing.assert_frame_equal(a.X, b.X)
    assert a.X.shape == (30, 2) and np.bincount(a.y).tolist() == [15, 15]
    assert a.metadata["source_row_ids"] == a.X.index.tolist()
    with pytest.raises(ValueError, match="hash"):
        load_dataset({**cfg, "sha256": "invalid"}, 11)
