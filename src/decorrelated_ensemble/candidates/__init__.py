from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from decorrelated_ensemble.preprocessing import make_preprocessor


@dataclass(frozen=True)
class CandidateSpec:
    id: int
    features: tuple[int, ...]
    feature_fraction: float
    max_depth: int
    min_samples_leaf: int
    row_fraction: float
    seed: int
    criterion: str

    def to_dict(self):
        return asdict(self)


def generate_candidates(p: int, config: dict, seed: int) -> list[CandidateSpec]:
    """Each ID has its own RNG; changing B preserves a nested pool prefix."""
    if p < 1 or config["B"] < 1:
        raise ValueError("Positive feature count and candidate count required")
    specs = []
    grid = list(product(config["feature_fractions"], config["depths"]))
    for b in range(config["B"]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, b]))
        if config.get("balanced_grid", False):
            fraction, depth = grid[b % len(grid)]
        else:
            fraction, depth = float(rng.choice(config["feature_fractions"])), None
        fraction = float(fraction)
        row_fraction = float(rng.choice(config.get("row_fractions", [1.0])))
        if not 0 < fraction <= 1 or not 0 < row_fraction <= 1:
            raise ValueError("Feature and row fractions must be in (0,1]")
        size = max(1, min(p, int(np.ceil(p * fraction))))
        features = tuple(sorted(int(j) for j in rng.choice(p, size=size, replace=False)))
        specs.append(
            CandidateSpec(
                b,
                features,
                fraction,
                int(depth if depth is not None else rng.choice(config["depths"])),
                int(rng.choice(config.get("min_samples_leaf", [5]))),
                row_fraction,
                int(rng.integers(0, 2**31 - 1)),
                config.get("criterion", "default"),
            )
        )
    return specs


class MaskedTree:
    def __init__(self, spec: CandidateSpec, task: str, n_classes: int = 2):
        self.spec, self.task, self.n_classes = spec, task, n_classes

    def fit(self, X, y):
        masked = X.iloc[:, list(self.spec.features)]
        rows = np.arange(len(y))
        if self.spec.row_fraction < 1:
            rng = np.random.default_rng(self.spec.seed)
            rows = rng.choice(rows, max(2, int(len(y) * self.spec.row_fraction)), replace=False)
        tree_class = DecisionTreeRegressor if self.task == "regression" else DecisionTreeClassifier
        criterion = self.spec.criterion
        if criterion == "default":
            criterion = "squared_error" if self.task == "regression" else "gini"
        self.pipeline_ = Pipeline(
            [
                ("preprocess", make_preprocessor(masked.iloc[rows])),
                (
                    "tree",
                    tree_class(
                        max_depth=self.spec.max_depth,
                        min_samples_leaf=self.spec.min_samples_leaf,
                        criterion=criterion,
                        random_state=self.spec.seed,
                    ),
                ),
            ]
        )
        self.fit_rows_ = rows
        self.feature_names_ = tuple(masked.columns)
        self.pipeline_.fit(masked.iloc[rows], np.asarray(y)[rows])
        return self

    def predict(self, X):
        masked = X.iloc[:, list(self.spec.features)]
        if tuple(masked.columns) != self.feature_names_:
            raise ValueError("Raw feature order differs from fit")
        if self.task == "regression":
            return self.pipeline_.predict(masked)
        raw = self.pipeline_.predict_proba(masked)
        out = np.zeros((len(X), self.n_classes))
        for j, label in enumerate(self.pipeline_["tree"].classes_):
            out[:, int(label)] = raw[:, j]
        return out[:, 1] if self.task == "binary" else out

    def capacity(self):
        tree = self.pipeline_["tree"].tree_
        return {"nodes": int(tree.node_count), "leaves": int(np.sum(tree.children_left == -1))}
