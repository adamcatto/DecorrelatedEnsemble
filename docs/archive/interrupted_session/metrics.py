import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             log_loss, mean_absolute_error, r2_score, roc_auc_score)


def binary_auc_columns(y, P):
    """Exact tie-aware AUROC for all columns without Python estimator loops."""
    P = np.asarray(P)
    if P.ndim == 1:
        P = P[:, None]
    positive = np.asarray(y) == 1
    n1, n0 = positive.sum(), (~positive).sum()
    if min(n1, n0) == 0:
        raise ValueError("AUROC requires both classes")
    ranks = rankdata(P, axis=0, method="average")
    return (ranks[positive].sum(axis=0) - n1*(n1+1)/2) / (n1*n0)


def candidate_quality(y, P, null, task):
    if task == "binary":
        return binary_auc_columns(y, P)
    if task == "regression":
        denominator = np.sum((y-null)**2)
        if denominator <= 0:
            raise ValueError("Zero null SSE makes skill undefined")
        return 1 - np.sum((y[:, None]-P)**2, axis=0) / denominator
    return np.array([roc_auc_score(y, P[:, j, :], multi_class="ovr", average="macro")
                     for j in range(P.shape[1])])


def residual_matrix(y, P, task):
    if task != "multiclass":
        return y[:, None] - P
    # sqrt(C) scaling makes E.T E/(N*C) equal sum-over-class Brier / N.
    c = P.shape[2]
    error = np.eye(c)[y][:, None, :] - P
    return (np.sqrt(c) * error.transpose(0, 2, 1)).reshape(len(y)*c, P.shape[1])


def evaluate(y, p, task):
    if task == "regression":
        return {"rmse": float(np.sqrt(np.mean((y-p)**2))),
                "mse": float(np.mean((y-p)**2)), "mae": float(mean_absolute_error(y, p)),
                "r2": float(r2_score(y, p))}
    if task == "binary":
        p = np.clip(p, 0, 1)
        return {"auroc": float(roc_auc_score(y, p)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(np.mean((y-p)**2)),
                "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5)),
                "auprc": float(average_precision_score(y, p))}
    return {"log_loss": float(log_loss(y, p, labels=np.arange(p.shape[1]))),
            "auroc_macro_ovr": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
            "brier": float(np.mean(np.sum((np.eye(p.shape[1])[y]-p)**2, axis=1))),
            "balanced_accuracy": float(balanced_accuracy_score(y, p.argmax(axis=1)))}


def selection_loss(y, p, task, metric="squared"):
    if metric == "squared":
        if task == "multiclass":
            return np.mean(np.sum((np.eye(p.shape[1])[y]-p)**2, axis=1))
        return np.mean((y-p)**2)
    if metric == "auroc":
        if task != "binary":
            raise ValueError("Direct AUROC selector is binary only")
        return -roc_auc_score(y, p)
    if metric == "log_loss" and task != "regression":
        return log_loss(y, p)
    raise ValueError("Unknown selection loss")

