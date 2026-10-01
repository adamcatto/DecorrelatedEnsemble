import hashlib
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np

from decorrelated_ensemble.metrics import residual_matrix, selection_loss


@dataclass
class Selection:
    ids: np.ndarray
    weights: np.ndarray
    objective: float
    trace: list


class CoError:
    """Matrix-free second moment with optional bias-preserving shrinkage."""

    def __init__(self, E, shrinkage=0.0, n_classes=None, cache_columns=False):
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must lie in [0,1]")
        self.E = np.asarray(E, dtype=float)
        self.n, self.b = self.E.shape
        self.alpha = shrinkage
        self.cache = {} if cache_columns else None
        self.mu = self.E.mean(axis=0)
        self.bias = (
            self.E.reshape(-1, n_classes, self.b).mean(axis=0) / np.sqrt(n_classes)
            if n_classes
            else self.mu[None, :]
        )
        self.bias_diag = np.sum(self.bias**2, axis=0)
        self.diag = np.mean(self.E**2, axis=0)

    def column(self, j):
        if self.cache is not None and j in self.cache:
            return self.cache[j]
        raw = self.E.T @ self.E[:, j] / self.n
        col = (1 - self.alpha) * raw + self.alpha * (self.bias.T @ self.bias[:, j])
        col[j] = self.diag[j]
        if self.cache is not None:
            self.cache[j] = col
        return col

    def dense(self):
        raw = self.E.T @ self.E / self.n
        out = (1 - self.alpha) * raw + self.alpha * (self.bias.T @ self.bias)
        np.fill_diagonal(out, self.diag)
        return out

    def objective(self, ids):
        if len(ids) == 0:
            return 0.0
        avg = self.E[:, ids].mean(axis=1)
        return float(
            (1 - self.alpha) * np.mean(avg**2)
            + self.alpha
            * (
                np.sum(self.bias[:, ids].mean(axis=1) ** 2)
                + np.sum(self.diag[ids] - self.bias_diag[ids]) / len(ids) ** 2
            )
        )


def coerror_greedy(oracle, k, max_swaps=0):
    if not 1 <= k <= oracle.b:
        raise ValueError("Invalid cardinality")
    selected = []
    available = np.ones(oracle.b, dtype=bool)
    sums = np.zeros(oracle.b)
    total = 0.0
    trace = []
    for t in range(k):
        values = (total + 2 * sums + oracle.diag) / (t + 1) ** 2
        values[~available] = np.inf
        j = int(np.argmin(values))
        total += 2 * sums[j] + oracle.diag[j]
        selected.append(j)
        available[j] = False
        sums += oracle.column(j)
        trace.append({"step": "add", "id": j, "objective": float(total / (t + 1) ** 2)})
    for _ in range(max_swaps):
        best_delta, best_pair = -1e-12, None
        for a in selected:
            delta = 2 * (sums - sums[a]) + oracle.diag + oracle.diag[a] - 2 * oracle.column(a)
            delta[~available] = np.inf
            b = int(np.argmin(delta))
            if delta[b] < best_delta:
                best_delta, best_pair = float(delta[b]), (a, b)
        if best_pair is None:
            break
        a, b = best_pair
        selected[selected.index(a)] = b
        available[a], available[b] = True, False
        sums += oracle.column(b) - oracle.column(a)
        total += best_delta
        trace.append({"step": "swap", "out": a, "in": b, "objective": float(total / k**2)})
    selected = np.asarray(selected, dtype=int)
    return Selection(selected, np.full(k, 1 / k), oracle.objective(selected), trace)


def exact_coerror(oracle, k, max_combinations=200_000):
    if not 1 <= k <= oracle.b or comb(oracle.b, k) > max_combinations:
        raise ValueError("Exact reference exceeds explicit combinatorial budget")
    G = oracle.dense()
    value, ids = min(
        (float(G[np.ix_(s, s)].sum() / k**2), s) for s in combinations(range(oracle.b), k)
    )
    return Selection(
        np.asarray(ids),
        np.full(k, 1 / k),
        value,
        [{"enumerated": comb(oracle.b, k), "globally_optimal": True}],
    )


def correlation_matrix(E):
    E = np.asarray(E, dtype=float)
    centered = E - E.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    normalized = centered / np.maximum(norm, 1e-15)
    R = np.clip(normalized.T @ normalized, -1, 1)
    constant = norm <= 1e-12
    # Undefined dependence is handled conservatively, not mislabeled independence.
    R[constant, :] = 1.0
    R[:, constant] = 1.0
    np.fill_diagonal(R, 1.0)
    return R


class CorrelationColumns:
    """Exact column access without allocating or forming a B-by-B matrix."""

    def __init__(self, E, absolute=False):
        centered = E - E.mean(axis=0)
        norms = np.linalg.norm(centered, axis=0)
        self.normalized = centered / np.maximum(norms, 1e-15)
        self.constant = norms <= 1e-12
        self.absolute, self.cache = absolute, {}

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, cols = key
            return np.column_stack([self[int(c)][rows.ravel()] for c in cols.ravel()])
        j = int(key)
        if j not in self.cache:
            column = np.clip(self.normalized.T @ self.normalized[:, j], -1, 1)
            column[self.constant] = 1
            if self.constant[j]:
                column[:] = 1
            column[j] = 1
            self.cache[j] = np.abs(column) if self.absolute else column
        return self.cache[j]


def caruana_coerror(oracle, k):
    """Squared-loss forward selection with replacement, using Gram columns."""
    sums, total, sequence, trace = np.zeros(oracle.b), 0.0, [], []
    for t in range(k):
        values = (total + 2 * sums + oracle.diag) / (t + 1) ** 2
        j = int(np.argmin(values))
        total += 2 * sums[j] + oracle.diag[j]
        sequence.append(j)
        sums += oracle.column(j)
        trace.append({"step": "add", "id": j, "objective": float(total / (t + 1) ** 2)})
    ids, counts = np.unique(sequence, return_counts=True)
    return Selection(ids, counts / k, oracle.objective(sequence), trace)


def pair_objective(ids, R, quality, kind, lam):
    if len(ids) < 2:
        dependence = 0.0
    else:
        pair = R[np.ix_(ids, ids)][np.triu_indices(len(ids), 1)]
        dependence = float(np.max(pair) if kind == "minimax" else np.mean(pair))
    return (
        dependence
        if kind != "quality_diversity"
        else -float(np.mean(quality[ids])) + lam * dependence
    )


def pair_greedy(R, quality, k, kind, lam, seed, max_swaps):
    rng = np.random.default_rng(seed)
    # Pure diversity does not seed with the best individual model.
    first = (
        int(np.argmax(quality)) if kind == "quality_diversity" else int(rng.integers(len(quality)))
    )
    selected = [first]
    sums = R[first].copy()
    maxima = R[first].copy()
    pair_sum, pair_max = 0.0, -np.inf
    qsum = quality[first]
    trace = []
    for t in range(1, k):
        dependence = (
            np.maximum(pair_max, maxima)
            if kind == "minimax"
            else (pair_sum + sums) / (t * (t + 1) / 2)
        )
        score = (
            dependence
            if kind != "quality_diversity"
            else -(qsum + quality) / (t + 1) + lam * dependence
        )
        score = score.copy()
        score[selected] = np.inf
        j = int(np.argmin(score))
        pair_sum += sums[j]
        pair_max = max(pair_max, maxima[j])
        qsum += quality[j]
        selected.append(j)
        sums += R[j]
        maxima = np.maximum(maxima, R[j])
        trace.append({"step": "add", "id": j, "objective": float(score[j])})
    current = pair_objective(selected, R, quality, kind, lam)
    for _ in range(max_swaps):
        best, pair = current - 1e-12, None
        unselected = np.setdiff1d(np.arange(len(quality)), selected)
        for pos in range(k):
            for j in unselected:
                trial = selected.copy()
                trial[pos] = int(j)
                score = pair_objective(trial, R, quality, kind, lam)
                if score < best:
                    best, pair = score, (pos, int(j))
        if pair is None:
            break
        pos, j = pair
        trace.append({"step": "swap", "out": selected[pos], "in": j, "objective": best})
        selected[pos] = j
        current = best
    return Selection(np.array(selected), np.full(k, 1 / k), current, trace)


def direct_greedy(y, P, task, k, metric="squared", replacement=False):
    running = np.zeros_like(P[:, 0])
    sequence, trace = [], []
    for t in range(k):
        scores = np.array(
            [
                selection_loss(y, (running + P[:, j]) / (t + 1), task, metric)
                for j in range(P.shape[1])
            ]
        )
        if not replacement:
            scores[sequence] = np.inf
        j = int(np.argmin(scores))
        sequence.append(j)
        running += P[:, j]
        trace.append({"step": "add", "id": j, "objective": float(scores[j])})
    ids, counts = np.unique(sequence, return_counts=True)
    return Selection(ids, counts / k, float(trace[-1]["objective"]), trace)


def distinct_prediction_indices(P):
    """First representative per exact OOF vector; streaming scratch O(N*C).

    Hashing only indexes buckets. Equality is checked to resolve collisions.
    Signed zeros compare equal; observed equality is not global function equality.
    """
    buckets, keep = {}, []
    for i in range(P.shape[1]):
        column = np.asarray(P[:, i], dtype=np.float64).reshape(-1)
        if not np.isfinite(column).all():
            raise ValueError("OOF deduplication requires finite predictions")
        canonical = np.where(column == 0, 0.0, column)
        digest = hashlib.sha256(canonical.tobytes()).digest()
        matches = buckets.setdefault(digest, [])
        if any(np.array_equal(column, P[:, j].reshape(-1)) for j in matches):
            continue
        keep.append(i)
        matches.append(i)
    return np.asarray(keep, dtype=int)


def select(y, P, null, task, quality, eligible, cfg, seed):
    ids = np.flatnonzero(eligible)
    original_ids = ids.copy()
    if cfg.get("deduplicate_oof", False):
        ids = ids[distinct_prediction_indices(P[:, ids])]
    k = cfg["K"]
    if k < 1 or len(ids) < k:
        raise ValueError(f"infeasible: {len(ids)} eligible < K={k}")
    Psmall, q = P[:, ids], quality[ids]
    E = residual_matrix(y, Psmall, task)
    kind = cfg["selector"]
    swaps = cfg.get("max_swaps", 0)
    if kind == "random":
        chosen = np.random.default_rng(seed).choice(len(ids), k, replace=False)
        result = Selection(chosen, np.full(k, 1 / k), CoError(E).objective(chosen), [])
    elif kind == "top_quality":
        chosen = np.argsort(-q, kind="stable")[:k]
        result = Selection(chosen, np.full(k, 1 / k), -float(q[chosen].mean()), [])
    elif kind in {
        "coerror",
        "shrinkage_coerror",
        "exact_coerror",
        "covariance_only",
        "caruana_coerror",
    }:
        if kind == "covariance_only":
            if task == "multiclass":
                structured = E.reshape(-1, P.shape[2], E.shape[1])
                E = (structured - structured.mean(axis=0)).reshape(E.shape)
            else:
                E = E - E.mean(axis=0)
        oracle = CoError(
            E,
            cfg.get("shrinkage", 0.0),
            P.shape[2] if task == "multiclass" else None,
            cfg.get("cache_columns", False),
        )
        result = (
            exact_coerror(oracle, k, cfg.get("max_combinations", 200_000))
            if kind == "exact_coerror"
            else caruana_coerror(oracle, k)
            if kind == "caruana_coerror"
            else coerror_greedy(oracle, k, swaps)
        )
    elif kind in {
        "abs_correlation",
        "signed_correlation",
        "minimax",
        "quality_diversity",
        "prediction_correlation",
    }:
        if not cfg.get("matrix_free", False) and len(ids) > cfg.get("max_dense_candidates", 4000):
            raise ValueError("Dense correlation selector exceeds explicit memory budget")
        if kind == "prediction_correlation":
            source = (
                Psmall if task != "multiclass" else Psmall.transpose(0, 2, 1).reshape(-1, len(ids))
            )
        else:
            source = E
        if cfg.get("matrix_free", False):
            R = CorrelationColumns(source, kind != "signed_correlation")
        else:
            R = correlation_matrix(source)
            if kind != "signed_correlation":
                R = np.abs(R)
        objective_kind = kind if kind in {"minimax", "quality_diversity"} else "mean"
        result = pair_greedy(R, q, k, objective_kind, cfg.get("lambda", 0.1), seed, swaps)
    elif kind in {"direct_squared", "direct_auroc", "caruana"}:
        metric = "auroc" if kind == "direct_auroc" else "squared"
        result = direct_greedy(y, Psmall, task, k, metric, kind == "caruana")
    elif kind == "affine_profiled":
        if task != "regression":
            raise ValueError("Affine-profiled diagnostic supports regression only")
        from decorrelated_ensemble.selection.profiled import profiled_greedy

        result = profiled_greedy(y, Psmall, k, swaps)
    else:
        raise ValueError(f"Unknown selector {kind}")
    result.ids = ids[result.ids]
    # Trace IDs above index the eligible pool; record mapping explicitly.
    result.trace.insert(0, {"eligible_pool_ids": ids.tolist()})
    if cfg.get("deduplicate_oof", False):
        result.trace[0]["eligible_pool_ids_before_deduplication"] = original_ids.tolist()
        result.trace[0]["equivalence"] = "exact training OOF equality; first eligible ID retained"
    return result
