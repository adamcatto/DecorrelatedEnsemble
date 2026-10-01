import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.pipeline import Pipeline
from .candidates import preprocessor


def baseline(cfg, X, task, seed):
    family = cfg["family"]
    regression = task == "regression"
    params = cfg.get("parameters", {}).copy()
    if family in {"random_forest", "extra_trees"}:
        cls = ({"random_forest": RandomForestRegressor, "extra_trees": ExtraTreesRegressor} if regression
               else {"random_forest": RandomForestClassifier, "extra_trees": ExtraTreesClassifier})[family]
        model = cls(random_state=seed, n_jobs=1, **params)
    elif family == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBRegressor if regression else XGBClassifier
        model = cls(random_state=seed, n_jobs=1, tree_method="hist", **params)
    elif family == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMRegressor if regression else LGBMClassifier
        model = cls(random_state=seed, n_jobs=1, verbosity=-1, deterministic=True,
                    force_col_wise=True, **params)
    elif family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostRegressor if regression else CatBoostClassifier
        model = cls(random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False, **params)
    else:
        raise ValueError("Unsupported baseline family")
    return Pipeline([("preprocess", preprocessor(X)), ("model", model)])


def capacity(model):
    estimator = model["model"]
    if hasattr(estimator, "estimators_"):
        trees = np.asarray(estimator.estimators_, dtype=object).ravel()
        return {"retained_count": len(trees), "nodes": int(sum(t.tree_.node_count for t in trees)),
                "leaves": int(sum(t.tree_.n_leaves for t in trees))}
    if hasattr(estimator, "get_booster"):
        frame = estimator.get_booster().trees_to_dataframe()
        return {"retained_count": int(frame.Tree.nunique()), "nodes": len(frame),
                "leaves": int((frame.Feature == "Leaf").sum())}
    if hasattr(estimator, "booster_"):
        info = estimator.booster_.dump_model()["tree_info"]
        leaves = sum(t["num_leaves"] for t in info)
        return {"retained_count": len(info), "nodes": int(2*leaves-len(info)), "leaves": int(leaves)}
    if hasattr(estimator, "get_tree_leaf_counts"):
        leaves = estimator.get_tree_leaf_counts()
        # Logical oblivious-tree nodes; serialized representation can be more compact.
        return {"retained_count": len(leaves), "nodes": int(np.sum(2*leaves-1)),
                "leaves": int(np.sum(leaves))}
    return {"retained_count": None, "nodes": None, "leaves": None}

