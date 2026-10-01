import argparse
import json
from pathlib import Path

import numpy as np

from decorrelated_ensemble.evaluation.artifacts import verify_manifest, write_json
from decorrelated_ensemble.metrics import evaluate_metrics, residual_matrix


def audit(run_id):
    root = Path("results/runs") / run_id
    verify_manifest(root)
    checks = []
    max_difference = 0.0
    for path in sorted(root.rglob("oof.npz")):
        fold = path.parent
        oof = np.load(path)
        y, P = oof["y"], oof["predictions"]
        task = json.loads((fold.parent / "metadata.json").read_text())["task"]
        indices = np.load(fold / "outer_split.npz")
        if set(indices["train"]) & set(indices["test"]):
            raise ValueError("Outer split overlap")
        for selected in fold.glob("*/selection.json"):
            config = json.loads(selected.read_text())
            ids, weights = np.array(config["ids"]), np.array(config["weights"])
            prediction = np.tensordot(P[:, ids], weights, axes=(1, 0))
            E = residual_matrix(y, P[:, ids], task)
            G = E.T @ E / len(E)
            from decorrelated_ensemble.metrics import squared_loss

            difference = abs(squared_loss(y, prediction, task) - weights @ G @ weights)
            max_difference = max(max_difference, float(difference))
            if difference > 1e-10:
                raise ValueError("Co-error loss identity failed")
            test = np.load(selected.parent / "test_predictions.npz")
            if not np.array_equal(test["rows"], indices["test"]):
                raise ValueError("Prediction rows do not match outer test split")
            actual = evaluate_metrics(test["y"], test["prediction"], task)
            saved = json.loads((selected.parent / "test_metrics.json").read_text())
            for metric, value in actual.items():
                if not np.isclose(value, saved[metric], atol=1e-12):
                    raise ValueError("Saved metric differs from stored prediction")
        greedy, direct = (
            fold / "coerror_greedy/selection.json",
            fold / "direct_squared/selection.json",
        )
        if greedy.exists() and direct.exists():
            a, b = json.loads(greedy.read_text()), json.loads(direct.read_text())
            if not np.isclose(a["objective"], b["objective"], atol=1e-10):
                raise ValueError("Greedy direct/co-error objectives disagree")
        checks.append(str(fold.relative_to(root)))
    result = {
        "run_id": run_id,
        "folds_audited": len(checks),
        "maximum_squared_loss_identity_error": max_difference,
        "checks": [
            "artifact hashes",
            "outer disjointness",
            "test prediction row mapping",
            "stored metric reconstruction",
            "weighted co-error identity",
            "direct/greedy squared objective identity",
        ],
        "scope": "Engineering audit; does not prove statistical validity or absence of every leakage channel",
    }
    write_json(Path("results/summaries") / run_id / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    print(audit(parser.parse_args().run_id))
