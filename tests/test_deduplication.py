import numpy as np
import pytest

from decorrelated_ensemble.selection import distinct_prediction_indices, select


@pytest.mark.parametrize("multiclass", [False, True])
def test_exact_oof_equivalence_first_representative_and_signed_zero(multiclass):
    P = np.array([[0.0, -0.0, 1.0, 1.0 + 1e-10], [1.0, 1.0, 2.0, 2.0]])
    if multiclass:
        P = np.stack([P, -P], axis=2)
    np.testing.assert_array_equal(distinct_prediction_indices(P), [0, 2, 3])


def test_hash_collision_resolved_by_exact_equality(monkeypatch):
    from decorrelated_ensemble import selection

    class ConstantHash:
        def digest(self):
            return b"collision"

    monkeypatch.setattr(selection.hashlib, "sha256", lambda _: ConstantHash())
    np.testing.assert_array_equal(distinct_prediction_indices(np.array([[1.0, 2.0, 1.0]])), [0, 1])


def test_deduplication_uses_only_eligible_training_columns_and_fixed_K():
    P = np.array([[0.0, 0.0, 1.0, 2.0], [1.0, 1.0, 2.0, 3.0]])
    y = np.array([0.0, 1.0])
    quality = np.array([1.0, 1.0, 0.0, -1.0])
    chosen = select(
        y,
        P,
        y,
        "regression",
        quality,
        np.array([False, True, True, True]),
        {"K": 2, "selector": "top_quality", "deduplicate_oof": True},
        11,
    )
    np.testing.assert_array_equal(chosen.ids, [1, 2])
    assert chosen.trace[0]["eligible_pool_ids"] == [1, 2, 3]
    with pytest.raises(ValueError, match="infeasible: 3 eligible < K=4"):
        select(
            y,
            P,
            y,
            "regression",
            quality,
            np.ones(4, dtype=bool),
            {"K": 4, "selector": "coerror", "deduplicate_oof": True},
            11,
        )
    with pytest.raises(ValueError, match="finite"):
        distinct_prediction_indices(np.array([[np.nan, 1.0]]))
