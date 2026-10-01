import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)


def binary_auc_columns(y, P):
    """Tie-aware vectorized AUROC, also used in joint stratified bootstraps."""
    P = np.asarray(P)
    if P.ndim == 1:
        P = P[:, None]
    pos = np.asarray(y) == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if not n1 or not n0:
        raise ValueError("AUROC requires both classes")
    ranks = rankdata(P, axis=0, method="average")
    return (ranks[pos].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def quality_columns(y, P, null, task):
    if task == "binary":
        return binary_auc_columns(y, P)
    if task == "multiclass":
        return np.mean(
            [binary_auc_columns((y == c).astype(int), P[:, :, c]) for c in range(P.shape[2])],
            axis=0,
        )
    denominator = np.sum((y - null) ** 2)
    if denominator <= 0:
        raise ValueError("Null squared error is zero; regression skill undefined")
    return 1 - np.sum((y[:, None] - P) ** 2, axis=0) / denominator


def residual_matrix(y, P, task):
    if task == "multiclass":
        target = np.eye(P.shape[2])[y]
        # Coordinate ordering n,c then b; divide second moments by sample count.
        return (
            (np.sqrt(P.shape[2]) * (target[:, None, :] - P))
            .transpose(0, 2, 1)
            .reshape(-1, P.shape[1])
        )
    return y[:, None] - P


def squared_loss(y, prediction, task):
    target = np.eye(prediction.shape[1])[y] if task == "multiclass" else y
    errors = (target - prediction) ** 2
    return float(np.mean(np.sum(errors, axis=1))) if task == "multiclass" else float(errors.mean())


def evaluate_metrics(y, prediction, task):
    prediction = np.asarray(prediction)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite test prediction")
    if task == "regression":
        return {
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
            "mse": squared_loss(y, prediction, task),
            "mae": float(mean_absolute_error(y, prediction)),
            "r2": float(r2_score(y, prediction)),
        }
    if task == "binary":
        clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
        return {
            "auroc": float(roc_auc_score(y, prediction)),
            "auprc": float(average_precision_score(y, prediction)),
            "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
            "brier": squared_loss(y, prediction, task),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction >= 0.5)),
            "ece_10": ece(y, prediction),
        }
    clipped = np.maximum(prediction, 1e-7)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return {
        "auroc_macro_ovr": float(roc_auc_score(y, prediction, multi_class="ovr", average="macro")),
        "log_loss": float(log_loss(y, clipped, labels=list(range(prediction.shape[1])))),
        "brier": squared_loss(y, prediction, task),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction.argmax(axis=1))),
    }


def ece(y, p, bins=10):
    indices = np.minimum((p * bins).astype(int), bins - 1)
    return float(
        sum(
            np.mean(indices == b) * abs(np.mean(y[indices == b]) - np.mean(p[indices == b]))
            for b in range(bins)
            if np.any(indices == b)
        )
    )


def selection_loss(y, prediction, task, metric="squared"):
    if metric == "squared":
        return squared_loss(y, prediction, task)
    if metric == "auroc" and task == "binary":
        return -float(binary_auc_columns(y, prediction)[0])
    if metric == "log_loss" and task != "regression":
        return evaluate_metrics(y, prediction, task)["log_loss"]
    raise ValueError(f"Selection metric {metric} unavailable for task {task}")


def evaluate_binary_groups(y, prediction, groups, aggregation="max"):
    """Fixed multiple-instance diagnostic; group IDs never enter the predictors."""
    if aggregation not in {"max", "mean"}:
        raise ValueError("Unknown binary group aggregation")
    targets, scores = [], []
    for group in np.unique(groups):
        mask = groups == group
        labels = np.unique(y[mask])
        if len(labels) != 1:
            raise ValueError("Group metric requires one target label per group")
        targets.append(labels[0])
        scores.append(
            np.max(prediction[mask]) if aggregation == "max" else np.mean(prediction[mask])
        )
    return {
        "group_" + k: v
        for k, v in evaluate_metrics(np.asarray(targets), np.asarray(scores), "binary").items()
    }
