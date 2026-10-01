from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from .candidates import SparseTree


def make_folds(y, task, n_splits, seed):
    if task != "regression" and np.bincount(y).min() < n_splits:
        raise ValueError("Insufficient examples per class for requested folds")
    cls = KFold if task == "regression" else StratifiedKFold
    splitter = cls(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y))
    validate_folds(folds, len(y))
    return folds


def validate_folds(folds, n):
    seen = np.zeros(n, dtype=int)
    for train, valid in folds:
        if len(np.unique(train)) != len(train) or len(np.unique(valid)) != len(valid):
            raise ValueError("Duplicate indices within a fold")
        if np.any(np.asarray(train) < 0) or np.any(np.asarray(valid) < 0):
            raise ValueError("Negative split index")
        if np.intersect1d(train, valid).size:
            raise ValueError("Train/validation overlap")
        if len(np.union1d(train, valid)) != n:
            raise ValueError("Fold does not partition all rows")
        seen[valid] += 1
    if not np.all(seen == 1):
        raise ValueError("Each row must be held out exactly once")


@dataclass
class OOFResult:
    predictions: np.ndarray
    null: np.ndarray
    folds: list
    audit: list


def cross_fit(X, y, specs, task, folds):
    validate_folds(folds, len(y))
    c = len(np.unique(y)) if task != "regression" else 0
    shape = (len(y), len(specs), c) if task == "multiclass" else (len(y), len(specs))
    P = np.full(shape, np.nan)
    null = np.zeros((len(y), c)) if task == "multiclass" else np.zeros(len(y))
    audit = []
    for f, (train, valid) in enumerate(folds):
        if task == "multiclass":
            null[valid] = np.bincount(y[train], minlength=c) / len(train)
        else:
            null[valid] = np.mean(y[train])
        for j, spec in enumerate(specs):
            model = SparseTree(spec, task).fit(X.iloc[train], y[train])
            P[valid, j] = model.predict(X.iloc[valid], c)
            audit.append({"fold": f, "candidate": spec.id,
                          "fit_rows": train[model.fit_rows_].tolist(), "valid_rows": valid.tolist()})
    if not np.all(np.isfinite(P)):
        raise ValueError("Non-finite/incomplete OOF predictions")
    return OOFResult(P, null, folds, audit)

