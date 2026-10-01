import numpy as np


def paired_task_comparison(
    records,
    left,
    right,
    metric,
    higher_is_better=False,
    bootstrap_reps=5000,
    seed=123,
    tolerance=1e-4,
):
    """Pair splits first, then average within tasks, then bootstrap task units."""
    keys = ["dataset", "seed", "fold"]
    a = records[(records.method == left) & (records.status == "complete")][keys + [metric]]
    b = records[(records.method == right) & (records.status == "complete")][keys + [metric]]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError("Duplicate method records on one dataset/seed/fold")
    paired = a.merge(b, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
    sign = 1 if higher_is_better else -1
    paired["advantage"] = sign * (paired[metric + "_left"] - paired[metric + "_right"])
    means = paired.groupby("dataset")["advantage"].mean()
    if len(means) == 0:
        return {
            "left": left,
            "right": right,
            "metric": metric,
            "paired_tasks": 0,
            "paired_splits": 0,
        }
    values = means.to_numpy()
    interval = (None, None)
    if len(values) > 1:
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(0, len(values), size=(bootstrap_reps, len(values)))].mean(
            axis=1
        )
        interval = tuple(float(np.quantile(draws, q)) for q in (0.025, 0.975))
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "paired_tasks": len(values),
        "paired_splits": len(paired),
        "mean_advantage": float(values.mean()),
        "ci_lower": interval[0],
        "ci_upper": interval[1],
        "win": int(np.sum(values > tolerance)),
        "tie": int(np.sum(np.abs(values) <= tolerance)),
        "loss": int(np.sum(values < -tolerance)),
        "task_advantages": means.to_dict(),
        "interval_scope": "task bootstrap over observed development tasks; no population guarantee"
        if len(values) > 1
        else "not estimated: only one paired dataset; repeated folds do not supply new task units",
        "multiplicity": "exploratory intervals, unadjusted; no significance declarations",
    }
