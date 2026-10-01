from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from decorrelated_ensemble.metrics import quality_columns


@dataclass
class Certification:
    quality: np.ndarray
    lower: np.ndarray
    passed: np.ndarray
    bootstrap: np.ndarray


def bootstrap_indices(y, task, rng, groups=None):
    if groups is not None:
        unique, inverse = np.unique(groups, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        members = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
        return np.concatenate(
            [members[i] for i in rng.choice(len(unique), len(unique), replace=True)]
        )
    if task == "regression":
        return rng.choice(len(y), len(y), replace=True)
    return np.concatenate(
        [rng.choice(np.flatnonzero(y == c), np.sum(y == c), replace=True) for c in np.unique(y)]
    )


def weighted_bootstrap_quality(y, P, null, task, counts):
    """Exact resampling multiplicities; binary AUROC uses ordered tie groups."""
    if task == "regression":
        denominator = counts @ ((y - null) ** 2)
        if np.any(denominator <= 0):
            raise ValueError("Null squared error is zero in a bootstrap sample")
        return 1 - (counts @ ((y[:, None] - P) ** 2)) / denominator[:, None]
    if task != "binary":
        raise ValueError("Weighted bootstrap engine supports binary/regression only")
    n1, n0 = counts[:, y == 1].sum(axis=1), counts[:, y == 0].sum(axis=1)
    if np.any(n1 == 0) or np.any(n0 == 0):
        raise ValueError("Bootstrap AUROC requires both classes")
    result = np.empty((len(counts), P.shape[1]))
    for j in range(P.shape[1]):
        _, inverse = np.unique(P[:, j], return_inverse=True)
        # Sparse aggregation avoids sorting each of hundreds of repeated samples.
        n_groups = int(inverse.max()) + 1
        sums = []
        for label in [0, 1]:
            rows = np.flatnonzero(y == label)
            assignment = csr_matrix(
                (np.ones(len(rows)), (inverse[rows], rows)), shape=(n_groups, len(y))
            )
            sums.append((assignment @ counts.T).T)
        negative, positive = sums
        below = np.cumsum(negative, axis=1) - negative
        result[:, j] = np.sum(positive * (below + 0.5 * negative), axis=1) / (n1 * n0)
    return result


def certify(y, P, null, task, config, seed, groups=None):
    """Empirical percentile screen. Does not claim OOF or multiplicity validity."""
    alpha, reps = config.get("alpha", 0.05), config.get("bootstrap_reps", 100)
    if not 0 < alpha < 0.5 or reps * alpha < 5:
        raise ValueError("Require >=5 draws in the nominal lower tail; increase bootstrap_reps")
    rng = np.random.default_rng(seed)
    samples = np.empty((reps, P.shape[1]))
    if config.get("engine", "resample") == "weighted":
        counts = np.empty((reps, len(y)))
        for b in range(reps):
            counts[b] = np.bincount(bootstrap_indices(y, task, rng, groups), minlength=len(y))
        samples = weighted_bootstrap_quality(y, P, null, task, counts)
    else:
        for b in range(reps):
            rows = bootstrap_indices(y, task, rng, groups)
            samples[b] = quality_columns(y[rows], P[rows], null[rows], task)
    quality = quality_columns(y, P, null, task)
    lower = np.quantile(samples, alpha, axis=0)
    threshold = (
        config.get("regression_threshold", 0.01)
        if task == "regression"
        else config.get("auc_threshold", 0.51)
    )
    return Certification(quality, lower, lower > threshold, samples)
