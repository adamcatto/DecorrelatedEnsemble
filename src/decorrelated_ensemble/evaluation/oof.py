from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from decorrelated_ensemble.candidates import MaskedTree


def make_splits(y, task, n_splits, seed, groups=None):
    if groups is not None:
        cls = GroupKFold if task == "regression" else StratifiedGroupKFold
        splitter = cls(n_splits, shuffle=True, random_state=seed)
    elif task == "regression":
        splitter = KFold(n_splits, shuffle=True, random_state=seed)
    else:
        if np.bincount(y).min() < n_splits:
            raise ValueError("Insufficient observations per class for stratified folds")
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    verify_splits(splits, len(y))
    if groups is not None:
        for train, valid in splits:
            if np.intersect1d(groups[train], groups[valid]).size:
                raise ValueError("Feature-group overlap between training and validation")
    return splits


def verify_splits(splits, n):
    coverage = np.zeros(n, dtype=int)
    for train, valid in splits:
        if len(np.unique(train)) != len(train) or len(np.unique(valid)) != len(valid):
            raise ValueError("Duplicate rows in a fold")
        if np.intersect1d(train, valid).size or set(train) | set(valid) != set(range(n)):
            raise ValueError("Fold leakage or incomplete partition")
        coverage[valid] += 1
    if not np.all(coverage == 1):
        raise ValueError("Each row must be validated exactly once")


@dataclass
class OOFResult:
    predictions: np.ndarray
    null: np.ndarray
    fold_ids: np.ndarray
    splits: list


def build_oof(X, y, specs, task, n_splits, seed, n_jobs=1, groups=None):
    n_classes = len(np.unique(y)) if task != "regression" else 0
    shape = (len(y), len(specs), n_classes) if task == "multiclass" else (len(y), len(specs))
    P = np.full(shape, np.nan)
    null = np.full((len(y), n_classes) if task == "multiclass" else (len(y),), np.nan)
    fold_ids = np.full(len(y), -1)
    splits = make_splits(y, task, n_splits, seed, groups)
    for fold, (train, valid) in enumerate(splits):
        fold_ids[valid] = fold
        if task == "multiclass":
            null[valid] = np.bincount(y[train], minlength=n_classes) / len(train)
        else:
            null[valid] = np.mean(y[train])
        Xtrain, Xvalid, ytrain = X.iloc[train], X.iloc[valid], y[train]

        def fit_predict(spec, Xtrain=Xtrain, ytrain=ytrain, Xvalid=Xvalid):
            model = MaskedTree(spec, task, n_classes).fit(Xtrain, ytrain)
            return model.predict(Xvalid)

        outputs = Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            delayed(fit_predict)(spec) for spec in specs
        )
        for j, prediction in enumerate(outputs):
            P[valid, j] = prediction
    if not np.isfinite(P).all() or not np.isfinite(null).all():
        raise ValueError("OOF predictions incomplete or nonfinite")
    return OOFResult(P, null, fold_ids, splits)
