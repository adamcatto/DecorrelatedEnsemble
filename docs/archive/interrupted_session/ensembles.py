import numpy as np
from .candidates import SparseTree


class SelectedEnsemble:
    def __init__(self, specs, weights, task, n_classes=2):
        self.specs = specs
        self.weights = np.asarray(weights, dtype=float)
        if len(specs) != len(weights) or np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1):
            raise ValueError("Weights must be nonnegative and sum to one")
        self.task, self.n_classes = task, n_classes

    def fit(self, X, y):
        self.models_ = [SparseTree(s, self.task).fit(X, y) for s in self.specs]
        return self

    def predict(self, X):
        return sum(w*m.predict(X, self.n_classes) for w, m in zip(self.weights, self.models_))

    def capacity(self):
        trees = [m.pipeline_["model"].tree_ for m in self.models_]
        return {"retained_count": len(trees), "nodes": sum(t.node_count for t in trees),
                "leaves": sum(t.n_leaves for t in trees)}

