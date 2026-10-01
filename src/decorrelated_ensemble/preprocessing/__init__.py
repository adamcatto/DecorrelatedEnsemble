import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_preprocessor(X):
    """Called on training rows after applying a candidate's raw feature mask."""
    numerical = list(X.select_dtypes(include=np.number).columns)
    categorical = [c for c in X.columns if c not in numerical]
    return ColumnTransformer(
        [
            ("numerical", SimpleImputer(strategy="median", keep_empty_features=True), numerical),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
