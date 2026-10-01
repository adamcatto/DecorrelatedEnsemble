import numpy as np
import pandas as pd

from decorrelated_ensemble.preprocessing import make_preprocessor


def test_nullable_and_category_dtypes():
    X = pd.DataFrame(
        {
            "number": pd.Series([1, None, 3], dtype="Int64"),
            "category": pd.Series(["a", "b", "a"], dtype="category"),
        }
    )
    transformed = make_preprocessor(X).fit_transform(X)
    assert transformed.shape == (3, 3)
    assert np.isfinite(transformed).all()
