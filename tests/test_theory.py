import numpy as np
import pytest
from scipy.linalg import hadamard

from decorrelated_ensemble.selection import CoError


def test_oracle_additive_attenuation_on_orthogonal_design():
    # Centered orthogonal columns have unit variance and no cross terms.
    design = hadamard(8).astype(float)
    signals = design[:, 1:5]
    noise = design[:, 5]
    target = signals.sum(axis=1) + noise
    # Each conditional-mean oracle has one of d=4 informative features.
    oracle = CoError(target[:, None] - signals)
    expected = 1 + 4 * (1 - 1 / 4) ** 2
    assert oracle.objective([0, 1, 2, 3]) == pytest.approx(expected)
    assert np.mean((target - signals.mean(axis=1)) ** 2) == pytest.approx(expected)
