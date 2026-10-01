from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from sklearn import datasets as skdata


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    task: str
    metadata: dict


def load_dataset(cfg: dict, seed: int) -> Dataset:
    source = cfg.get("source", "synthetic")
    task = cfg.get("task", "binary")
    if source == "sklearn":
        loaders = {"breast_cancer": skdata.load_breast_cancer,
                   "diabetes": skdata.load_diabetes, "wine": skdata.load_wine}
        if cfg["name"] not in loaders:
            raise ValueError("Unknown development dataset")
        raw = loaders[cfg["name"]](as_frame=True)
        X, y = raw.data, np.asarray(raw.target)
        metadata = {"source": "sklearn_builtin", "version": "package-version-in-environment",
                    "note": "development only; not representative benchmark evidence"}
    elif source == "synthetic":
        rng = np.random.default_rng(seed)
        n, m = cfg.get("n", 900), cfg.get("m", 40)
        regime = cfg["regime"]
        q = min(cfg.get("informative", 8), m)
        X = rng.normal(size=(n, m))
        if regime == "additive":
            signal = X[:, :q].sum(axis=1) / np.sqrt(q)
        elif regime == "sparse":
            signal = X[:, :min(q, 2)].sum(axis=1) / np.sqrt(min(q, 2))
        elif regime == "dominant":
            signal = X[:, 0]
        elif regime in {"xor", "parity"}:
            order = 2 if regime == "xor" else cfg.get("order", 3)
            if order > m:
                raise ValueError("Interaction order exceeds dimension")
            signal = np.prod(np.where(X[:, :order] > 0, 1., -1.), axis=1)
        elif regime == "mixed":
            if m < 4:
                raise ValueError("Mixed regime requires four features")
            signal = (X[:, 0] + X[:, 1]) / 2 + np.sign(X[:, 2] * X[:, 3])
        elif regime == "redundant":
            latent = rng.normal(size=n)
            X[:, :q] = latent[:, None] + cfg.get("feature_noise", .5) * rng.normal(size=(n, q))
            signal = latent
        elif regime == "blocks":
            n_blocks = min(4, m)
            latent = rng.normal(size=(n, n_blocks))
            rho = cfg.get("rho", .8)
            if not 0 <= rho < 1:
                raise ValueError("rho must lie in [0,1)")
            for j in range(m):
                X[:, j] = np.sqrt(rho) * latent[:, j % n_blocks] + np.sqrt(1-rho) * X[:, j]
            signal = latent.sum(axis=1) / np.sqrt(n_blocks)
        elif regime == "null":
            signal = np.zeros(n)
        else:
            raise ValueError(f"Unknown synthetic regime {regime}")
        noise = cfg.get("noise", .7)
        if task == "regression":
            y = signal + noise * rng.normal(size=n)
        elif task == "binary":
            logit = cfg.get("signal_scale", 2.) * signal + cfg.get("intercept", 0.)
            probability = 1 / (1 + np.exp(-logit))
            y = (rng.random(n) < probability).astype(int)
            flips = rng.random(n) < cfg.get("label_noise", .05)
            y[flips] = 1 - y[flips]
        else:
            raise ValueError("Synthetic generator currently supports binary/regression")
        X = pd.DataFrame(X, columns=[f"x{j}" for j in range(m)])
        metadata = {"source": "synthetic", "generator_version": 1, "seed": seed,
                    "parameters": cfg, "signal": signal.tolist()}
    else:
        raise ValueError("Unsupported data source: use an explicit audited adapter")
    if task not in {"binary", "multiclass", "regression"}:
        raise ValueError("Unsupported task")
    if task != "regression":
        _, y = np.unique(y, return_inverse=True)
        if task == "binary" and len(np.unique(y)) != 2:
            raise ValueError("Binary task must have two classes")
    digest = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()
                            + np.ascontiguousarray(y).tobytes()).hexdigest()
    metadata.update({"n": len(y), "m": X.shape[1], "sha256": digest,
                     "task": task, "split_structure": "iid",
                     "categorical_columns": [str(c) for c in X if not pd.api.types.is_numeric_dtype(X[c])],
                     "missing_fraction": float(X.isna().to_numpy().mean()),
                     "class_counts": None if task == "regression" else np.bincount(y).tolist()})
    return Dataset(cfg.get("name", cfg.get("regime")), X, y, task, metadata)

