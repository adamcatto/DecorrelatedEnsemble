from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass(frozen=True)
class CandidateSpec:
    id: int
    features: tuple[int, ...]
    depth: int
    min_leaf: int
    row_fraction: float
    criterion: str
    seed: int

    def to_dict(self):
        return asdict(self)


def generate_candidates(m: int, cfg: dict, seed: int) -> list[CandidateSpec]:
    """The first B specs are identical when only B changes."""
    if m < 1 or cfg["B"] < 1:
        raise ValueError("Positive m and B required")
    specs = []
    for b in range(cfg["B"]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, b]))
        fraction = float(rng.choice(cfg.get("fractions", [.1])))
        row_fraction = float(rng.choice(cfg.get("row_fractions", [1.])))
        if not 0 < fraction <= 1 or not 0 < row_fraction <= 1:
            raise ValueError("Feature/row fractions must lie in (0,1]")
        k = max(1, min(m, int(np.ceil(m * fraction))))
        specs.append(CandidateSpec(b, tuple(sorted(rng.choice(m, k, replace=False).tolist())),
                                   int(rng.choice(cfg.get("depths", [3]))),
                                   int(rng.choice(cfg.get("min_leaves", [10]))),
                                   row_fraction, cfg.get("criterion", "default"),
                                   int(rng.integers(0, 2**31-1))))
    return specs


def preprocessor(X: pd.DataFrame):
    numeric = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X if c not in numeric]
    parts = []
    if numeric:
        parts.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric))
    if categorical:
        parts.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0)


class SparseTree:
    """Select raw mask before preprocessing; no other columns enter the pipeline."""
    def __init__(self, spec: CandidateSpec, task: str):
        self.spec, self.task = spec, task

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        rng = np.random.default_rng(self.spec.seed)
        n = max(2, int(np.ceil(len(y) * self.spec.row_fraction)))
        if n > len(y):
            raise ValueError("At least two training rows required")
        rows = np.sort(rng.choice(len(y), n, replace=False))
        self.fit_rows_ = rows
        masked = X.iloc[rows, list(self.spec.features)]
        tree_type = DecisionTreeRegressor if self.task == "regression" else DecisionTreeClassifier
        criterion = self.spec.criterion
        if criterion == "default":
            criterion = "squared_error" if self.task == "regression" else "gini"
        self.pipeline_ = Pipeline([
            ("preprocess", preprocessor(masked)),
            ("model", tree_type(max_depth=self.spec.depth, min_samples_leaf=self.spec.min_leaf,
                                criterion=criterion, random_state=self.spec.seed)),
        ])
        self.pipeline_.fit(masked, y[rows])
        return self

    def predict(self, X: pd.DataFrame, n_classes: int = 2):
        X = X.iloc[:, list(self.spec.features)]
        if self.task == "regression":
            return self.pipeline_.predict(X)
        small = self.pipeline_.predict_proba(X)
        full = np.zeros((len(X), n_classes))
        full[:, self.pipeline_["model"].classes_.astype(int)] = small
        return full[:, 1] if self.task == "binary" else full


def model_predict(model, X, task, n_classes=2):
    if isinstance(model, SparseTree):
        return model.predict(X, n_classes)
    if task == "regression":
        return model.predict(X)
    p = model.predict_proba(X)
    full = np.zeros((len(X), n_classes))
    full[:, model.classes_.astype(int)] = p
    return full[:, 1] if task == "binary" else full

