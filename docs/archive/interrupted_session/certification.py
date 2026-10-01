from dataclasses import dataclass
import numpy as np
from .metrics import candidate_quality


@dataclass
class Certification:
    quality: np.ndarray
    lower: np.ndarray
    passed: np.ndarray
    bootstrap: np.ndarray
    indices: np.ndarray


def bootstrap_screen(y, P, null, task, cfg, seed):
    """Conditional empirical OOF screen; not simultaneous statistical certification."""
    n_boot = cfg.get("n_bootstrap", 200)
    alpha = cfg.get("alpha", .05)
    if n_boot < 20 or not 0 < alpha < 1:
        raise ValueError("At least 20 bootstrap draws and alpha in (0,1) required")
    if n_boot * alpha < 5:
        raise ValueError("Insufficient bootstrap tail resolution; require n_boot*alpha>=5")
    if cfg.get("adjustment", "none") != "none":
        raise ValueError("Adjusted OOF bootstrap guarantees are not implemented")
    rng = np.random.default_rng(seed)
    indices = np.empty((n_boot, len(y)), dtype=int)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)] if task != "regression" else [np.arange(len(y))]
    for r in range(n_boot):
        indices[r] = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
    quality = candidate_quality(y, P, null, task)
    draws = np.array([candidate_quality(y[ix], P[ix], null[ix], task) for ix in indices])
    lower = np.quantile(draws, alpha, axis=0)
    threshold = cfg.get("threshold", .01 if task == "regression" else .51)
    return Certification(quality, lower, lower > threshold, draws, indices)

