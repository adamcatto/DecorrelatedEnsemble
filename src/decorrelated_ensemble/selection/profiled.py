"""Affine-profiled squared loss: a diagnostic for aggregate attenuation."""

import numpy as np

from decorrelated_ensemble.selection import CoError, Selection


def fit_affine(y, prediction):
    centered_y = y - np.mean(y)
    centered_p = prediction - np.mean(prediction)
    variance = float(np.mean(centered_p**2))
    covariance = float(np.mean(centered_y * centered_p))
    # Nonnegative slope preserves ranking and prevents sign reversal of weak fits.
    slope = max(0.0, covariance / variance) if variance > 1e-14 else 0.0
    intercept = float(np.mean(y) - slope * np.mean(prediction))
    return {
        "slope": slope,
        "intercept": intercept,
        "oof_variance": variance,
        "oof_covariance": covariance,
        "fit_scope": "outer-training OOF aggregate only",
    }


class ProfiledLoss:
    def __init__(self, y, P):
        self.centered = P - P.mean(axis=0)
        self.moments = CoError(self.centered)
        self.c = np.mean((y - y.mean())[:, None] * self.centered, axis=0)
        self.target_variance = float(np.mean((y - y.mean()) ** 2))
        self.b = P.shape[1]

    def value(self, covariance_sum, variance_sum):
        covariance_sum = np.maximum(covariance_sum, 0)
        benefit = np.divide(
            covariance_sum**2,
            variance_sum,
            out=np.zeros_like(np.asarray(variance_sum, dtype=float)),
            where=np.asarray(variance_sum) > 1e-14,
        )
        return np.maximum(self.target_variance - benefit, 0)

    def objective(self, ids):
        variance = np.mean(np.sum(self.centered[:, ids], axis=1) ** 2)
        return float(self.value(self.c[ids].sum(), variance))


def profiled_greedy(y, P, k, max_swaps=0):
    oracle = ProfiledLoss(y, P)
    if not 1 <= k <= oracle.b:
        raise ValueError("Invalid cardinality")
    selected = []
    sums = np.zeros(oracle.b)
    total_variance, total_covariance = 0.0, 0.0
    available = np.ones(oracle.b, dtype=bool)
    trace = []
    for step in range(k):
        values = oracle.value(
            total_covariance + oracle.c, total_variance + 2 * sums + oracle.moments.diag
        )
        values[~available] = np.inf
        j = int(np.argmin(values))
        total_variance += 2 * sums[j] + oracle.moments.diag[j]
        total_covariance += oracle.c[j]
        sums += oracle.moments.column(j)
        selected.append(j)
        available[j] = False
        trace.append({"step": "add", "id": j, "objective": float(values[j])})
    current = oracle.objective(selected)
    for _ in range(max_swaps):
        best, pair = current - 1e-12, None
        for out in selected:
            variance = (
                total_variance
                - 2 * sums[out]
                + oracle.moments.diag[out]
                + 2 * (sums - oracle.moments.column(out))
                + oracle.moments.diag
            )
            covariance = total_covariance - oracle.c[out] + oracle.c
            values = oracle.value(covariance, variance)
            values[~available] = np.inf
            j = int(np.argmin(values))
            if values[j] < best:
                best, pair = float(values[j]), (out, j)
        if pair is None:
            break
        out, add = pair
        total_variance = (
            total_variance
            - 2 * sums[out]
            + oracle.moments.diag[out]
            + 2 * (sums[add] - oracle.moments.column(out)[add])
            + oracle.moments.diag[add]
        )
        total_covariance += oracle.c[add] - oracle.c[out]
        sums += oracle.moments.column(add) - oracle.moments.column(out)
        selected[selected.index(out)] = add
        available[out], available[add] = True, False
        current = best
        trace.append({"step": "swap", "out": out, "in": add, "objective": best})
    return Selection(np.array(selected), np.full(k, 1 / k), oracle.objective(selected), trace)
