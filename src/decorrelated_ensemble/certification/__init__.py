from dataclasses import dataclass

import numpy as np

from decorrelated_ensemble.metrics import quality_columns


@dataclass
class Certification:
    quality: np.ndarray
    lower: np.ndarray
    passed: np.ndarray
    bootstrap: np.ndarray


def bootstrap_indices(y, task, rng):
    if task == "regression":
        return rng.choice(len(y), len(y), replace=True)
    return np.concatenate(
        [rng.choice(np.flatnonzero(y == c), np.sum(y == c), replace=True) for c in np.unique(y)]
    )


def certify(y, P, null, task, config, seed):
    """Empirical percentile screen. Does not claim OOF or multiplicity validity."""
    alpha, reps = config.get("alpha", 0.05), config.get("bootstrap_reps", 100)
    if not 0 < alpha < 0.5 or reps * alpha < 5:
        raise ValueError("Require >=5 draws in the nominal lower tail; increase bootstrap_reps")
    rng = np.random.default_rng(seed)
    samples = np.empty((reps, P.shape[1]))
    for b in range(reps):
        rows = bootstrap_indices(y, task, rng)
        samples[b] = quality_columns(y[rows], P[rows], null[rows], task)
    quality = quality_columns(y, P, null, task)
    lower = np.quantile(samples, alpha, axis=0)
    threshold = (
        config.get("regression_threshold", 0.01)
        if task == "regression"
        else config.get("auc_threshold", 0.51)
    )
    return Certification(quality, lower, lower > threshold, samples)
