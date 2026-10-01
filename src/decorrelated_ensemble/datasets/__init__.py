from dataclasses import dataclass

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
        data = loader(as_frame=True)
        return Dataset(
            data.data,
            np.asarray(data.target),
            task,
            {
                **config,
                "task": task,
                "n": len(data.target),
                "p": data.data.shape[1],
                "role": "development",
                "sampling_structure": "iid",
                "license": "see sklearn dataset documentation",
                "seed": seed,
            },
        )
    raise ValueError("Unsupported source; no silent dataset exclusion")
