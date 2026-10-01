"""Independent frozen-input/decision/metric audit for experiment 007."""

import argparse
import json
from pathlib import Path

import numpy as np
from run_quality_alignment import array_digest

from decorrelated_ensemble.evaluation.artifacts import sha256, verify_manifest, write_json
from decorrelated_ensemble.metrics import evaluate_binary_groups, evaluate_metrics, squared_loss


def audit(run_id, predictions_only=False):
    root = Path("results/runs") / run_id
    verify_manifest(root, allow_missing_models=predictions_only)
    if json.loads((root / "status.json").read_text())["state"] != "complete":
        raise ValueError("Only completed interventions can enter the audit")
    resume_path = root / "resume.json"
    if resume_path.exists():
        resume = json.loads(resume_path.read_text())
        previous = Path("results/runs") / resume["run_id"]
        verify_manifest(previous, allow_missing_models=predictions_only)
        if sha256(previous / "manifest.json") != resume["manifest_sha256"]:
            raise ValueError("Immutable failed resume source changed")
        for reused in resume["reused_folds"]:
            for name, digest in reused["sha256"].items():
                old, new = previous / reused["fold"] / name, root / reused["fold"] / name
                if (
                    predictions_only
                    and name == "model.joblib"
                    and not old.exists()
                    and not new.exists()
                ):
                    continue
                if sha256(old) != digest or sha256(new) != digest:
                    raise ValueError("Resumed artifact changed or was rescored")
    rows, checked = [], set()
    for pointer in sorted(root.rglob("reference.json")):
        fold = pointer.parent
        reference = json.loads(pointer.read_text())
        parent = Path("results/runs") / reference["run_id"]
        if parent not in checked:
            verify_manifest(parent, allow_missing_models=predictions_only)
            checked.add(parent)
        manifests = [parent / "manifest.json", parent / "original_manifest.json"]
        if reference["manifest_sha256"] not in [sha256(p) for p in manifests if p.exists()]:
            raise ValueError("Frozen parent manifest changed")
        for name, expected in reference["inputs"].items():
            if sha256(parent / name) != expected:
                raise ValueError("Frozen parent input changed")
        oof_path = parent / reference["oof_path"]
        with np.load(oof_path) as oof:
            if set(oof.files) != set(reference["oof_array_digests"]):
                raise ValueError("OOF array schema changed")
            for key in oof.files:
                if array_digest(oof[key]) != reference["oof_array_digests"][key]:
                    raise ValueError("OOF numerical values/dtype/shape changed")
            y, P = oof["y"], oof["predictions"]
        source = oof_path.parent
        metadata = json.loads((source.parent / "metadata.json").read_text())
        cfg = json.loads((parent / "config.json").read_text())
        split = np.load(source / "outer_split.npz")
        data_y = np.load(source.parent / "y.npy")
        groups = np.load(source.parent / "groups.npy")
        np.testing.assert_array_equal(y, data_y[split["train"]])
        if np.intersect1d(groups[split["train"]], groups[split["test"]]).size:
            raise ValueError("Parent group holdout failed")
        selection = json.loads((fold / "selection.json").read_text())
        expected = np.argsort(
            np.mean(np.ascontiguousarray(y[:, None] - P) ** 2, axis=0), kind="stable"
        )[:64]
        np.testing.assert_array_equal(selection["ids"], expected)
        np.testing.assert_array_equal(selection["weights"], np.full(64, 1 / 64))
        original = json.loads((source / "top_quality_b6000_k64" / "selection.json").read_text())
        same = original["ids"] == selection["ids"]
        test = np.load(fold / "test_predictions.npz")
        np.testing.assert_array_equal(test["rows"], split["test"])
        np.testing.assert_array_equal(test["y"], data_y[split["test"]])
        if same:
            np.testing.assert_array_equal(
                test["prediction"],
                np.load(source / "top_quality_b6000_k64" / "test_predictions.npz")["prediction"],
            )
        metrics = evaluate_metrics(test["y"], test["prediction"], metadata["task"])
        if cfg.get("group_aggregation"):
            metrics.update(
                evaluate_binary_groups(
                    test["y"], test["prediction"], groups[test["rows"]], cfg["group_aggregation"]
                )
            )
        metrics["normalized_squared_loss"] = squared_loss(
            test["y"], test["prediction"], metadata["task"]
        ) / max(squared_loss(test["y"], np.full(len(test["y"]), y.mean()), metadata["task"]), 1e-15)
        saved = json.loads((fold / "test_metrics.json").read_text())
        for key, value in metrics.items():
            np.testing.assert_allclose(value, saved[key], atol=1e-12, rtol=1e-12)
        E = y[:, None] - P[:, expected]
        weights = np.full(64, 1 / 64)
        gram_loss = weights @ (E.T @ E / len(y)) @ weights
        actual_loss = squared_loss(y, P[:, expected] @ weights, metadata["task"])
        if abs(gram_loss - actual_loss) > 1e-10:
            raise ValueError("Squared loss identity failed")
        rows.append(
            {
                "dataset": metadata["id"],
                "fold_path": str(fold.relative_to(root)),
                "same_original_quality_ids": same,
                "same_original_quality_subset": set(original["ids"]) == set(selection["ids"]),
                "max_original_quality_prediction_difference": float(
                    np.max(
                        np.abs(
                            test["prediction"]
                            - np.load(source / "top_quality_b6000_k64" / "test_predictions.npz")[
                                "prediction"
                            ]
                        )
                    )
                ),
                "gram_identity_error": abs(gram_loss - actual_loss),
            }
        )
    if len(rows) != 18 or len(checked) != 6:
        raise ValueError("Incomplete six-task intervention coverage")
    output = Path("results/summaries") / run_id
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "audit.json",
        {
            "run_id": run_id,
            "folds": rows,
            "predictions_audited": len(rows),
            "checks": [
                "parent manifest identity",
                "input byte hashes",
                "OOF value/dtype/shape hashes independent of C/F archive order",
                "stable loss-diagonal ranking and fixed K/weights",
                "group and label/prediction row mapping",
                "metric reconstruction",
                "Gram loss identity",
                "observed original quality selection/refit equivalence; finite-precision ties recorded",
                "byte-exact reused decisions/refits/scores linked to immutable failed attempt where present",
            ],
            "scope": "Engineering audit on reused development folds; no independent confirmation or population inference",
        },
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--predictions-only", action="store_true")
    args = parser.parse_args()
    print(audit(args.run_id, args.predictions_only))
