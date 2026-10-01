import argparse
import json
from pathlib import Path

import numpy as np

from decorrelated_ensemble.evaluation.artifacts import verify_manifest, write_json
from decorrelated_ensemble.metrics import evaluate_metrics, residual_matrix, squared_loss
from decorrelated_ensemble.selection.profiled import ProfiledLoss, fit_affine


def audit(run_id, predictions_only=False):
    root = Path("results/runs") / run_id
    verify_manifest(root, allow_missing_models=predictions_only)
    checks = []
    max_difference = 0.0
    test_predictions_audited, affine_fits_audited = 0, 0
    for path in sorted(root.rglob("oof.npz")):
        fold = path.parent
        oof = np.load(path)
        y, P = oof["y"], oof["predictions"]
        task = json.loads((fold.parent / "metadata.json").read_text())["task"]
        indices = np.load(fold / "outer_split.npz")
        if set(indices["train"]) & set(indices["test"]):
            raise ValueError("Outer split overlap")
        data_y = np.load(fold.parent / "y.npy")
        if sorted(np.r_[indices["train"], indices["test"]].tolist()) != list(range(len(data_y))):
            raise ValueError("Outer split is not a partition")
        if not np.array_equal(y, data_y[indices["train"]]):
            raise ValueError("OOF label row mapping mismatch")
        coverage = np.zeros(len(y), dtype=int)
        for j, split in enumerate(json.loads((fold / "inner_splits.json").read_text())):
            train, valid = np.array(split["train"]), np.array(split["valid"])
            if set(train) & set(valid) or sorted(np.r_[train, valid].tolist()) != list(
                range(len(y))
            ):
                raise ValueError("Inner split is not a disjoint partition")
            if not np.all(oof["fold_ids"][valid] == j):
                raise ValueError("OOF fold ID mismatch")
            coverage[valid] += 1
        if not np.all(coverage == 1):
            raise ValueError("OOF coverage differs from exactly once")
        for selected in fold.glob("*/selection.json"):
            config = json.loads(selected.read_text())
            ids, weights = np.array(config["ids"]), np.array(config["weights"])
            prediction = np.tensordot(P[:, ids], weights, axes=(1, 0))
            E = residual_matrix(y, P[:, ids], task)
            G = E.T @ E / len(E)
            difference = abs(squared_loss(y, prediction, task) - weights @ G @ weights)
            max_difference = max(max_difference, float(difference))
            if difference > 1e-10:
                raise ValueError("Co-error loss identity failed")
            calibration = selected.parent / "calibration.json"
            if calibration.exists():
                params = json.loads(calibration.read_text())
                reconstructed = fit_affine(y, prediction)
                for key in ["slope", "intercept", "oof_variance", "oof_covariance"]:
                    if not np.isclose(params[key], reconstructed[key], atol=1e-12, rtol=1e-12):
                        raise ValueError("Affine fit differs from training OOF reconstruction")
                loss = squared_loss(y, params["slope"] * prediction + params["intercept"], task)
                diagnostic = json.loads((selected.parent / "diagnostics.json").read_text())
                if not np.isclose(loss, diagnostic["oof_calibrated_squared_loss"], atol=1e-12):
                    raise ValueError("Calibrated OOF diagnostic mismatch")
                if config["config"]["selector"] == "affine_profiled":
                    if not np.isclose(loss, config["objective"], atol=1e-12):
                        raise ValueError("Profiled selection objective mismatch")
                    if not np.isclose(loss, ProfiledLoss(y, P).objective(ids), atol=1e-12):
                        raise ValueError("Profiled moment identity mismatch")
                affine_fits_audited += 1
        for prediction_path in fold.glob("*/test_predictions.npz"):
            test = np.load(prediction_path)
            if not np.array_equal(test["rows"], indices["test"]):
                raise ValueError("Prediction rows do not match outer test split")
            if not np.array_equal(test["y"], data_y[indices["test"]]):
                raise ValueError("Test label row mapping mismatch")
            actual = evaluate_metrics(test["y"], test["prediction"], task)
            null = (
                np.tile(np.mean(y), len(test["y"]))
                if task != "multiclass"
                else np.tile(np.bincount(y) / len(y), (len(test["y"]), 1))
            )
            actual["normalized_squared_loss"] = squared_loss(
                test["y"], test["prediction"], task
            ) / max(squared_loss(test["y"], null, task), 1e-15)
            saved = json.loads((prediction_path.parent / "test_metrics.json").read_text())
            for metric, value in actual.items():
                if not np.isclose(value, saved[metric], atol=1e-12):
                    raise ValueError("Saved metric differs from stored prediction")
            test_predictions_audited += 1
        for raw in ["random_subspace", "top_no_cert", "coerror_no_cert"]:
            a, b = fold / raw / "selection.json", fold / (raw + "_affine") / "selection.json"
            if a.exists() and b.exists():
                a, b = json.loads(a.read_text()), json.loads(b.read_text())
                if a["ids"] != b["ids"] or a["weights"] != b["weights"]:
                    raise ValueError("Raw and calibrated controls selected different subsets")
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
        "test_predictions_audited": test_predictions_audited,
        "affine_fits_audited": affine_fits_audited,
        "maximum_squared_loss_identity_error": max_difference,
        "checks": [
            "artifact hashes",
            "outer disjointness",
            "outer/inner partitions and OOF coverage",
            "OOF and test label row mapping",
            "test prediction row mapping",
            "stored metric reconstruction",
            "weighted co-error identity",
            "direct/greedy squared objective identity",
            "training OOF affine fit and profiled loss reconstruction where applicable",
            "same selected subsets for raw/calibrated controls where applicable",
        ],
        "scope": "Engineering audit; does not prove statistical validity or absence of every leakage channel",
    }
    write_json(Path("results/summaries") / run_id / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--predictions-only", action="store_true")
    args = parser.parse_args()
    print(audit(args.run_id, args.predictions_only))
