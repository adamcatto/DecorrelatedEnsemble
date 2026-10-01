import hashlib
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn import datasets as skdata


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    task: str
    metadata: dict


def synthetic(config: dict, seed: int) -> Dataset:
    """Known signal mechanisms; no sample-dependent normalization of the signal."""
    rng = np.random.default_rng(seed)
    n, p = config.get("n", 600), config.get("p", 40)
    regime, task = config["regime"], config["task"]
    if p < 8 or n < 30:
        raise ValueError("Synthetic mechanisms require p>=8 and n>=30")
    X = rng.normal(size=(n, p))
    signal_features = list(range(min(16, p)))
    if regime == "additive":
        q = min(16, p)
        signal = X[:, :q].sum(axis=1) / np.sqrt(q)
    elif regime == "redundant":
        latent = rng.normal(size=(n, 4))
        for j in range(p):
            X[:, j] = (latent[:, j % 4] + 0.3 * rng.normal(size=n)) / np.sqrt(1.09)
        signal = latent.sum(axis=1) / 2
        signal_features = list(range(p))
    elif regime == "correlated_blocks":
        latent = rng.normal(size=(n, 4))
        rho = config.get("rho", 0.8)
        for j in range(p):
            X[:, j] = np.sqrt(rho) * latent[:, j % 4] + np.sqrt(1 - rho) * X[:, j]
        signal = (X[:, 0] + X[:, 1] - X[:, 2] - X[:, 3]) / 2
        signal_features = list(range(4))
    elif regime == "sparse":
        signal = (X[:, 0] + X[:, 1] - X[:, 2]) / np.sqrt(3)
        signal_features = [0, 1, 2]
    elif regime == "dominant":
        signal = X[:, 0]
        signal_features = [0]
    elif regime in {"xor", "parity"}:
        order = 2 if regime == "xor" else config.get("interaction_order", 4)
        if not 2 <= order <= p:
            raise ValueError("Invalid interaction order")
        signal = np.prod(np.where(X[:, :order] > 0, 1.0, -1.0), axis=1)
        signal_features = list(range(order))
    elif regime == "mixed":
        signal = (X[:, :4].sum(axis=1) / 2 + np.where(X[:, 4] * X[:, 5] > 0, 1, -1)) / np.sqrt(2)
        signal_features = list(range(6))
    elif regime == "null":
        signal = np.zeros(n)
        signal_features = []
    else:
        raise ValueError(f"Unknown regime: {regime}")
    noise = config.get("noise", 1.0)
    if task == "regression":
        y = signal + noise * rng.normal(size=n)
    elif task == "binary":
        # Logistic link: noise controls inverse signal strength; null is balanced chance.
        intercept = config.get("intercept", 0.0)
        logits = signal * config.get("signal_scale", 2.5) / max(noise, 0.05) + intercept
        probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        y = (rng.random(n) < probability).astype(int)
        flip = config.get("label_flip", 0.0)
        y = np.where(rng.random(n) < flip, 1 - y, y)
    else:
        raise ValueError("Synthetic task must be binary or regression")
    return Dataset(
        pd.DataFrame(X, columns=[f"x{j}" for j in range(p)]),
        y,
        task,
        {
            **config,
            "source": "synthetic-v1",
            "seed": seed,
            "signal_features": signal_features,
            "n": n,
            "p": p,
            "role": "development",
            "sampling_structure": "iid",
        },
    )


def load_dataset(config: dict, seed: int) -> Dataset:
    if config.get("source", "synthetic") == "synthetic":
        return synthetic(config, seed)
    if config["source"] == "sklearn":
        loaders = {
            "breast_cancer": (skdata.load_breast_cancer, "binary"),
            "diabetes": (skdata.load_diabetes, "regression"),
            "wine": (skdata.load_wine, "multiclass"),
        }
        loader, task = loaders[config["name"]]
        # The default diabetes loader centers/scales on all rows before splitting.
        # Load raw values so any sample-dependent transform belongs to the fit fold.
        arguments = {"as_frame": True}
        if config["name"] == "diabetes":
            arguments["scaled"] = False
        data = loader(**arguments)
        sources = {
            "breast_cancer": {
                "original_source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
                "citation": "Wolberg, Mangasarian, Street, and Street (1993), doi:10.24432/C5DW2B",
                "license": "CC-BY-4.0 (UCI source)",
                "target_definition": "Diagnosis: 0=malignant, 1=benign (sklearn convention)",
                "adaptation": "sklearn bundled copy; ID removed, diagnosis encoded, 30 numerical features",
            },
            "wine": {
                "original_source": "https://archive.ics.uci.edu/dataset/109/wine",
                "citation": "Aeberhard and Forina (1992), doi:10.24432/C5PC7J",
                "license": "CC-BY-4.0 (UCI source)",
                "target_definition": "Cultivar class, encoded 0/1/2 instead of original 1/2/3",
                "adaptation": "sklearn bundled copy, 13 numerical features",
            },
            "diabetes": {
                "original_source": "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html",
                "citation": "Efron, Hastie, Johnstone, and Tibshirani (2004), Least Angle Regression",
                "license": "Original dataset license not stated in loader documentation; sklearn package BSD-3-Clause does not establish a separate data license",
                "target_definition": "Quantitative disease progression one year after baseline",
                "adaptation": "sklearn bundled raw feature copy, scaled=False; original collection transforms remain",
            },
        }
        target = np.asarray(data.target)
        digest = hashlib.sha256()
        digest.update(pd.util.hash_pandas_object(data.data, index=True).values.tobytes())
        digest.update(target.tobytes())
        digest.update(repr(list(data.data.columns)).encode())
        return Dataset(
            data.data,
            target,
            task,
            {
                **config,
                "task": task,
                "n": len(data.target),
                "p": data.data.shape[1],
                "role": "development",
                "sampling_structure": "iid",
                **sources[config["name"]],
                "loader": loader.__name__,
                "loader_arguments": arguments,
                "dataset_version": "bundled in scikit-learn " + version("scikit-learn"),
                "loader_documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets."
                + loader.__name__
                + ".html",
                "content_sha256": digest.hexdigest(),
                "feature_names": list(data.data.columns),
                "feature_dtypes": [str(t) for t in data.data.dtypes],
                "missing_values": int(data.data.isna().sum().sum()),
                "class_counts": {str(c): int(np.sum(target == c)) for c in np.unique(target)}
                if task != "regression"
                else None,
                "sampling_assumption": "IID rows assumed; no group/time identifiers in bundled feature table",
                "description": data.DESCR,
                "seed": seed,
            },
        )
    raise ValueError("Unsupported source; no silent dataset exclusion")
