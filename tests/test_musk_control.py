import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "musk_control", Path("scripts/audit_musk_bag_sizes.py")
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def test_molecule_counts_preserve_one_label_per_group():
    groups, counts, labels = control.group_summary(
        np.array(["b", "a", "b", "b"]), np.array([1, 0, 1, 1])
    )
    np.testing.assert_array_equal(groups, ["a", "b"])
    np.testing.assert_array_equal(counts, [1, 3])
    np.testing.assert_array_equal(labels, [0, 1])
    with pytest.raises(ValueError, match="constant"):
        control.group_summary(np.array(["a", "a"]), np.array([0, 1]))


def test_count_control_scaling_uses_training_molecules_only():
    counts, labels = np.array([1, 2, 3, 5]), np.array([0, 0, 1, 1])
    model = control.fit_count_baseline(counts, labels, 11)
    mean = model.steps[0][1].mean_.copy()
    np.testing.assert_allclose(mean, [np.log1p(counts).mean()])
    model.predict_proba(np.log1p(np.array([1000, 2000]))[:, None])
    np.testing.assert_array_equal(model.steps[0][1].mean_, mean)
