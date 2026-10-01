import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decorrelated_ensemble.candidates import generate_candidates
from decorrelated_ensemble.ensembles import SelectedEnsemble
from decorrelated_ensemble.preprocessing import make_preprocessor as preprocessor


def baseline(cfg, X, task, seed):
    family = cfg["family"]
    regression = task == "regression"
    params = cfg.get("parameters", {}).copy()
    if family == "random_patches":
        count = params.pop("n_estimators", 8)
        specs = generate_candidates(
            X.shape[1],
            {
                "B": count,
                "feature_fractions": [params.pop("feature_fraction", 0.1)],
                "depths": [params.pop("max_depth", 3)],
                "min_samples_leaf": [params.pop("min_samples_leaf", 5)],
                "row_fractions": [params.pop("row_fraction", 0.5)],
            },
            seed,
        )
        if params:
            raise ValueError(f"Unknown random patches parameters: {params}")
        return SelectedEnsemble(
            specs, np.full(count, 1 / count), task, len(np.unique(cfg.get("classes", [0, 1])))
        )
    if family == "linear":
        if not regression:
            params.setdefault("max_iter", 3000)
        model = Ridge(**params) if regression else LogisticRegression(random_state=seed, **params)
        return Pipeline(
            [("preprocess", preprocessor(X)), ("scale", StandardScaler()), ("model", model)]
        )
    if family in {"random_forest", "extra_trees"}:
        cls = (
            {"random_forest": RandomForestRegressor, "extra_trees": ExtraTreesRegressor}
            if regression
            else {"random_forest": RandomForestClassifier, "extra_trees": ExtraTreesClassifier}
        )[family]
        model = cls(random_state=seed, n_jobs=1, **params)
    elif family == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBRegressor if regression else XGBClassifier
        model = cls(random_state=seed, n_jobs=1, tree_method="hist", **params)
    elif family == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        cls = LGBMRegressor if regression else LGBMClassifier
        model = cls(
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            **params,
        )
    elif family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostRegressor if regression else CatBoostClassifier
        model = cls(
            random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False, **params
        )
    else:
        raise ValueError("Unsupported baseline family")
    return Pipeline([("preprocess", preprocessor(X)), ("model", model)])


def capacity(model):
    if isinstance(model, SelectedEnsemble):
        return model.capacity()
    estimator = model["model"]
    if hasattr(estimator, "coef_"):
        return {
            "retained_count": 1,
            "nodes": None,
            "leaves": None,
            "coefficient_count": int(np.size(estimator.coef_) + np.size(estimator.intercept_)),
        }
    if hasattr(estimator, "estimators_"):
        trees = np.asarray(estimator.estimators_, dtype=object).ravel()
        return {
            "retained_count": len(trees),
            "nodes": int(sum(t.tree_.node_count for t in trees)),
            "leaves": int(sum(t.tree_.n_leaves for t in trees)),
        }
    if hasattr(estimator, "get_booster"):
        frame = estimator.get_booster().trees_to_dataframe()
        return {
            "retained_count": int(frame.Tree.nunique()),
            "nodes": len(frame),
            "leaves": int((frame.Feature == "Leaf").sum()),
        }
    if hasattr(estimator, "booster_"):
        info = estimator.booster_.dump_model()["tree_info"]
        leaves = sum(t["num_leaves"] for t in info)
        return {
            "retained_count": len(info),
            "nodes": int(2 * leaves - len(info)),
            "leaves": int(leaves),
        }
    if hasattr(estimator, "get_tree_leaf_counts"):
        leaves = estimator.get_tree_leaf_counts()
        # Logical oblivious-tree nodes; serialized representation can be more compact.
        return {
            "retained_count": len(leaves),
            "nodes": int(np.sum(2 * leaves - 1)),
            "leaves": int(np.sum(leaves)),
        }
    return {"retained_count": None, "nodes": None, "leaves": None}
