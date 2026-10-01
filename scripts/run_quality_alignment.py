"""Loss-aligned quality intervention on immutable exp_006 libraries."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from decorrelated_ensemble.candidates import CandidateSpec
from decorrelated_ensemble.evaluation.artifacts import (
    ResourceTimer,
    environment,
    finalize_manifest,
    sha256,
    snapshot,
    verify_manifest,
    write_json,
)
from decorrelated_ensemble.evaluation.runner import diagnostics, fit_selection, measure_inference
from decorrelated_ensemble.metrics import evaluate_binary_groups, evaluate_metrics, squared_loss
from decorrelated_ensemble.selection import select


def array_digest(value):
    """Exact values/dtype/shape, independent of archive C/F storage order."""
    value = np.asarray(value)
    h = hashlib.sha256(
        json.dumps({"dtype": value.dtype.str, "shape": value.shape}, sort_keys=True).encode()
    )
    for start in range(0, len(value), 256):
        h.update(value[start : start + 256].tobytes(order="C"))
    return h.hexdigest()


def squared_quality_selection(y, P, null, task, method, seed):
    return select(
        y, P, null, task, np.zeros(P.shape[1]), np.ones(P.shape[1], dtype=bool), method, seed
    )


def reuse_completed_fold(previous, target):
    """Reuse frozen decisions, refits and scores without another test evaluation."""
    for name in ["reference.json", "selection.json"]:
        saved = json.loads((previous / name).read_text())
        current = json.loads((target / name).read_text())
        if saved != current:
            raise ValueError("Resumed fold decisions or frozen inputs changed")
    names = [
        "reference.json",
        "selection.json",
        "model.joblib",
        "resources.json",
        "diagnostics.json",
        "test_predictions.npz",
        "test_metrics.json",
    ]
    copied = {}
    for name in names:
        if not (previous / name).exists():
            raise ValueError("Resumed completed fold lacks required artifacts")
        shutil.copy2(previous / name, target / name)
        copied[name] = sha256(target / name)
    return copied


def run(config_path, run_id, predictions_only=False, resume_from=None):
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(Path(config_path).read_text())
    if cfg["role"] != "development" or cfg["method"] != {
        "id": "top_squared_b6000_k64",
        "selector": "top_squared",
        "K": 64,
        "certified": False,
    }:
        raise ValueError("This registered intervention has fixed scope")
    references = [repo / "results" / "runs" / name for name in cfg["reference_runs"]]
    names = {
        "credit_default",
        "higgs",
        "miniboone",
        "superconductivity",
        "california_housing",
        "musk2",
    }
    if len(references) != 6 or {r.name for r in references} != {
        f"exp_006_large_{name}_v1" for name in names
    }:
        raise ValueError("Retain the entire registered six-task panel")
    for reference in references:
        verify_manifest(reference, allow_missing_models=predictions_only)
        if json.loads((reference / "status.json").read_text())["state"] != "complete":
            raise ValueError("All reference tasks must be complete")
    resume = (repo / "results" / "runs" / resume_from) if resume_from else None
    resume_info, reused = {}, []
    if resume is not None:
        verify_manifest(resume)
        if json.loads((resume / "config.json").read_text()) != cfg:
            raise ValueError("Resume config differs from the registered intervention")
        old_status = json.loads((resume / "status.json").read_text())
        if old_status["state"] != "failed":
            raise ValueError("Only an immutable failed attempt can supply resume artifacts")
        resume_info = {
            "run_id": resume_from,
            "manifest_sha256": sha256(resume / "manifest.json"),
            "status": old_status,
            "scope": "Reuse completed decisions/refits/test outcomes byte-exact; no repeated test scoring. Total run time measures this continuation only; per-method resources retain inherited and original refit costs.",
        }
    destination = repo / "results" / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "config.json", cfg)
    write_json(destination / "environment.json", environment(repo))
    snapshot(repo, destination / "source.zip")
    write_json(destination / "status.json", {"state": "running", "run_id": run_id})
    if resume is not None:
        write_json(destination / "resume.json", {**resume_info, "reused_folds": []})
    records, audits = [], []
    try:
        with threadpool_limits(limits=1), ResourceTimer() as total:
            for reference in references:
                parent_cfg = json.loads((reference / "config.json").read_text())
                for oof_path in sorted(reference.rglob("oof.npz")):
                    source = oof_path.parent
                    data = source.parent
                    metadata = json.loads((data / "metadata.json").read_text())
                    name, task = metadata["id"], metadata["task"]
                    seed, fold = int(data.name.split("_")[-1]), int(source.name.split("_")[-1])
                    target = destination / name / data.name / source.name
                    target.mkdir(parents=True)
                    X, y_all = pd.read_pickle(data / "X.pkl"), np.load(data / "y.npy")
                    groups = (
                        np.load(data / "groups.npy") if (data / "groups.npy").exists() else None
                    )
                    with np.load(source / "outer_split.npz") as split:
                        train, test = split["train"].copy(), split["test"].copy()
                    with np.load(oof_path) as oof:
                        oof_arrays = {key: oof[key] for key in oof.files}
                        P, y, null = oof_arrays["predictions"], oof_arrays["y"], oof_arrays["null"]
                    np.testing.assert_array_equal(y, y_all[train])
                    if P.shape[1] != 6000:
                        raise ValueError("Reference library is not B6000")
                    if groups is not None and np.intersect1d(groups[train], groups[test]).size:
                        raise ValueError("Reference groups overlap")
                    definitions = json.loads((source / "candidates.json").read_text())
                    specs = [
                        CandidateSpec(**{**d, "features": tuple(d["features"])})
                        for d in definitions
                    ]
                    with ResourceTimer() as selection_timer:
                        chosen = squared_quality_selection(
                            y, P, null, task, cfg["method"], seed + 8000 + fold
                        )
                    expected = np.argsort(
                        np.mean(np.ascontiguousarray(y[:, None] - P) ** 2, axis=0), kind="stable"
                    )[:64]
                    np.testing.assert_array_equal(chosen.ids, expected)
                    if task not in {"binary", "regression"}:
                        raise ValueError("Registered panel only supports scalar outputs")
                    inputs = [
                        source / "candidates.json",
                        source / "outer_split.npz",
                        data / "X.pkl",
                        data / "y.npy",
                    ]
                    if groups is not None:
                        inputs.append(data / "groups.npy")
                    write_json(
                        target / "reference.json",
                        {
                            "run_id": reference.name,
                            "manifest_sha256": sha256(reference / "manifest.json"),
                            "inputs": {str(p.relative_to(reference)): sha256(p) for p in inputs},
                            "oof_path": str(oof_path.relative_to(reference)),
                            "oof_array_digests": {
                                key: array_digest(value) for key, value in oof_arrays.items()
                            },
                            "refit_seed": "Fixed candidate specifications; original full outer training rows",
                        },
                    )
                    write_json(
                        target / "selection.json",
                        {
                            "config": cfg["method"],
                            "ids": chosen.ids,
                            "weights": chosen.weights,
                            "objective": chosen.objective,
                            "trace": chosen.trace,
                        },
                    )
                    previous = (resume / name / data.name / source.name) if resume else None
                    if previous is not None and (previous / "test_metrics.json").exists():
                        copied = reuse_completed_fold(previous, target)
                        reused.append(
                            {"fold": str(target.relative_to(destination)), "sha256": copied}
                        )
                        write_json(
                            destination / "resume.json", {**resume_info, "reused_folds": reused}
                        )
                        resource = json.loads((target / "resources.json").read_text())
                        diag = json.loads((target / "diagnostics.json").read_text())
                        metrics = json.loads((target / "test_metrics.json").read_text())
                        with np.load(target / "test_predictions.npz") as saved:
                            prediction = saved["prediction"]
                    else:
                        with ResourceTimer() as refit_timer:
                            model = fit_selection(
                                X.iloc[train], y, specs, chosen, task, n_jobs=cfg["refit_jobs"]
                            )
                        joblib.dump(model, target / "model.joblib", compress=0)
                        resource = {
                            "selection": selection_timer.to_dict(),
                            "refit": refit_timer.to_dict(),
                            "model_bytes": (target / "model.joblib").stat().st_size,
                            **model.capacity(),
                            **measure_inference(model, X.iloc[train], lambda m, x: m.predict(x)),
                        }
                        inherited = json.loads(
                            (source / "coerror_b6000_k64" / "resources.json").read_text()
                        )
                        for kind in ["cpu", "wall"]:
                            key = kind + "_seconds"
                            resource["train_" + key] = (
                                inherited["search"][key]
                                + inherited["screen"][key]
                                + getattr(selection_timer, kind)
                                + getattr(refit_timer, kind)
                            )
                        resource["search"] = inherited["search"]
                        resource["screen"] = inherited["screen"]
                        resource["resource_scope"] = (
                            "Inherited full shared-library cost plus incremental selector/refit; cached-input reads separate in total run time; no matched budget"
                        )
                        quality = np.load(source / "certification.npz")["quality"]
                        diag = diagnostics(y, P, task, specs, chosen, quality)
                        write_json(target / "resources.json", resource)
                        write_json(target / "diagnostics.json", diag)
                        # Every decision and refit is saved before accessing test outcomes here.
                        prediction = model.predict(X.iloc[test])
                        metrics = evaluate_metrics(y_all[test], prediction, task)
                        if parent_cfg.get("group_aggregation"):
                            metrics.update(
                                evaluate_binary_groups(
                                    y_all[test],
                                    prediction,
                                    groups[test],
                                    parent_cfg["group_aggregation"],
                                )
                            )
                        metrics["normalized_squared_loss"] = squared_loss(
                            y_all[test], prediction, task
                        ) / max(
                            squared_loss(y_all[test], np.full(len(test), y.mean()), task), 1e-15
                        )
                        np.savez_compressed(
                            target / "test_predictions.npz",
                            rows=test,
                            y=y_all[test],
                            prediction=prediction,
                        )
                        write_json(target / "test_metrics.json", metrics)
                    original_top = json.loads(
                        (source / "top_quality_b6000_k64" / "selection.json").read_text()
                    )
                    same_ids = original_top["ids"] == chosen.ids.tolist()
                    audit = {
                        "dataset": name,
                        "seed": seed,
                        "fold": fold,
                        "stable_top_diagonal_verified": True,
                        "same_ids_as_original_quality": same_ids,
                        "same_subset_as_original_quality": set(original_top["ids"])
                        == set(chosen.ids.tolist()),
                        "squared_objective_identity_error": abs(
                            diag["oof_squared_loss"]
                            - squared_loss(y, P[:, chosen.ids] @ chosen.weights, task)
                        ),
                    }
                    if same_ids:
                        with np.load(
                            source / "top_quality_b6000_k64" / "test_predictions.npz"
                        ) as original:
                            np.testing.assert_array_equal(prediction, original["prediction"])
                        audit["same_test_predictions"] = True
                    with np.load(
                        source / "top_quality_b6000_k64" / "test_predictions.npz"
                    ) as original:
                        audit["max_original_quality_prediction_difference"] = float(
                            np.max(np.abs(prediction - original["prediction"]))
                        )
                    audits.append(audit)
                    records.append(
                        {
                            "dataset": name,
                            "task": task,
                            "seed": seed,
                            "fold": fold,
                            "method": cfg["method"]["id"],
                            "status": "complete",
                            "B": 6000,
                            "generated_B": 6000,
                            **metrics,
                            **diag,
                            **{k: v for k, v in resource.items() if not isinstance(v, dict)},
                        }
                    )
                    pd.DataFrame(records).to_csv(destination / "records.csv", index=False)
                    print(f"{name} fold={fold} loss-aligned quality complete", flush=True)
        write_json(
            destination / "audit.json",
            {
                "folds": audits,
                "scope": "Frozen-input hashes, stable diagonal ranking, observed regression subset/prediction equality; finite precision may change ties, engineering controls; no population inference",
            },
        )
        write_json(
            destination / "status.json",
            {
                "state": "complete",
                "run_id": run_id,
                "records": len(records),
                "resources": total.to_dict(),
            },
        )
    except BaseException as error:
        write_json(
            destination / "status.json",
            {"state": "failed", "run_id": run_id, "records": len(records), "error": repr(error)},
        )
        finalize_manifest(destination)
        raise
    finalize_manifest(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--run-id", default="exp_007_quality_alignment_v1")
    parser.add_argument("--predictions-only", action="store_true")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    print(run(args.config, args.run_id, args.predictions_only, args.resume_from))
