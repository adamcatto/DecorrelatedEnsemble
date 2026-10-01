import copy
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from decorrelated_ensemble.baselines import baseline, capacity
from decorrelated_ensemble.candidates import generate_candidates
from decorrelated_ensemble.certification import certify
from decorrelated_ensemble.datasets import load_dataset
from decorrelated_ensemble.ensembles import SelectedEnsemble
from decorrelated_ensemble.evaluation.artifacts import (
    ResourceTimer,
    environment,
    finalize_manifest,
    snapshot,
    write_json,
)
from decorrelated_ensemble.evaluation.oof import build_oof, make_splits
from decorrelated_ensemble.metrics import evaluate_metrics, residual_matrix, squared_loss
from decorrelated_ensemble.selection import correlation_matrix, select


def predict_baseline(model, X, task):
    if isinstance(model, SelectedEnsemble):
        return model.predict(X)
    if task == "regression":
        return model.predict(X)
    p = model.predict_proba(X)
    return p[:, 1] if task == "binary" else p


def aggregate_predictions(P, selection):
    return np.tensordot(P[:, selection.ids], selection.weights, axes=(1, 0))


def fit_selection(X, y, specs, selection, task, calibration=False, oof_predictions=None, n_jobs=1):
    n_classes = len(np.unique(y)) if task != "regression" else 0
    base = SelectedEnsemble(
        [specs[i] for i in selection.ids], selection.weights, task, n_classes
    ).fit(X, y, n_jobs=n_jobs)
    if not calibration:
        return base
    if task != "regression" or oof_predictions is None:
        raise ValueError("Affine calibration requires regression and training OOF predictions")
    from decorrelated_ensemble.ensembles.affine import AffineAggregate
    from decorrelated_ensemble.selection.profiled import fit_affine

    return AffineAggregate(base, fit_affine(y, aggregate_predictions(oof_predictions, selection)))


def nested_tune(X, y, specs, task, cert_config, method, inner_folds, seed, path=None, groups=None):
    """Regenerate base fits inside each tuning fold; never slice global OOF fits."""
    options = method.get("tuning_grid", [])
    if not options:
        return method, []
    candidates = [{**method, **option} for option in options]
    scores = np.zeros(len(options))
    trace = []
    for fold, (train, valid) in enumerate(
        make_splits(y, task, method.get("tuning_folds", 3), seed, groups)
    ):
        inner_groups = None if groups is None else groups[train]
        oof = build_oof(
            X.iloc[train],
            y[train],
            specs,
            task,
            inner_folds,
            seed + 100 + fold,
            groups=inner_groups,
        )
        cert = certify(
            y[train],
            oof.predictions,
            oof.null,
            task,
            cert_config,
            seed + 200 + fold,
            groups=inner_groups,
        )
        if path is not None:
            np.savez_compressed(
                path / f"tuning_fold_{fold}.npz",
                train=train,
                valid=valid,
                P=oof.predictions,
                null=oof.null,
                folds=oof.fold_ids,
                quality=cert.quality,
                lower=cert.lower,
                bootstrap=cert.bootstrap,
            )
        for j, candidate in enumerate(candidates):
            eligible = eligibility(cert, task, cert_config, candidate) & candidate_pool_mask(
                specs, candidate
            )
            try:
                chosen = select(
                    y[train],
                    oof.predictions,
                    oof.null,
                    task,
                    cert.quality,
                    eligible,
                    candidate,
                    seed + 300 + fold,
                )
                model = fit_selection(
                    X.iloc[train],
                    y[train],
                    specs,
                    chosen,
                    task,
                    candidate.get("affine_calibration", False),
                    oof.predictions,
                )
                prediction = model.predict(X.iloc[valid])
                metric = candidate.get("tuning_metric", "squared")
                loss = (
                    squared_loss(y[valid], prediction, task)
                    if metric == "squared"
                    else -evaluate_metrics(y[valid], prediction, task)["auroc"]
                )
                trace.append(
                    {
                        "fold": fold,
                        "option": j,
                        "loss": loss,
                        "ids": chosen.ids,
                        "weights": chosen.weights,
                    }
                )
                scores[j] += loss / len(y) * len(valid)
            except ValueError as error:
                if not str(error).startswith("infeasible:"):
                    raise
                scores[j] = np.inf
                trace.append({"fold": fold, "option": j, "error": str(error)})
    if not np.isfinite(scores).any():
        raise ValueError("infeasible: all tuning settings failed at fixed K")
    best = int(np.argmin(scores))
    chosen = copy.deepcopy(candidates[best])
    chosen.pop("tuning_grid", None)
    trace.append({"chosen_option": best, "scores": scores})
    return chosen, trace


def eligibility(cert, task, cert_config, method):
    if not method.get("certified", True):
        return np.ones(len(cert.quality), dtype=bool)
    threshold = method.get("certification_threshold")
    if threshold is None:
        return cert.passed
    return cert.lower > threshold


def candidate_pool_mask(specs, method):
    """Fixed config-defined restriction, evaluated before seeing test scores."""
    pool = method.get("candidate_pool", {})
    mask = np.ones(len(specs), dtype=bool)
    if "prefix" in pool:
        mask &= np.arange(len(specs)) < pool["prefix"]
    for key in ["feature_fraction", "max_depth"]:
        if key in pool:
            mask &= np.array([getattr(s, key) == pool[key] for s in specs])
    if "limit" in pool:
        ids = np.flatnonzero(mask)
        mask[ids[pool["limit"] :]] = False
    return mask


def diagnostics(y, P, task, specs, chosen, quality):
    E = residual_matrix(y, P[:, chosen.ids], task)
    R = correlation_matrix(E)
    pairs = R[np.triu_indices(len(chosen.ids), 1)]
    eig = np.maximum(np.linalg.eigvalsh(E.T @ E / len(E)), 0)
    eig_sum = eig.sum()
    effective_rank = float(eig_sum**2 / np.sum(eig**2)) if eig_sum else 0.0
    masks = [set(specs[i].features) for i in chosen.ids]
    overlaps = [len(a & b) / len(a | b) for i, a in enumerate(masks) for b in masks[i + 1 :]]
    return {
        "mean_selected_quality": float(np.mean(quality[chosen.ids])),
        "mean_residual_correlation": float(pairs.mean()) if len(pairs) else None,
        "mean_abs_residual_correlation": float(np.abs(pairs).mean()) if len(pairs) else None,
        "max_abs_residual_correlation": float(np.abs(pairs).max()) if len(pairs) else None,
        "residual_eigenvalues": eig,
        "residual_effective_rank": effective_rank,
        "mean_feature_jaccard": float(np.mean(overlaps)) if overlaps else None,
        "feature_coverage": len(set.union(*masks)),
        "oof_squared_loss": squared_loss(y, aggregate_predictions(P, chosen), task),
    }


def measure_inference(model, X, predict):
    # Training rows only; outer test predictions are computed once after all decisions.
    batch = X.iloc[: min(256, len(X))]
    predict(model, batch)
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        predict(model, batch)
        timings.append(time.perf_counter() - start)
    seconds = float(np.median(timings))
    return {
        "inference_batch_rows": len(batch),
        "inference_batch_seconds": seconds,
        "inference_rows_per_second": len(batch) / max(seconds, 1e-12),
        "inference_timing_scope": "median 5 warm calls on training rows; end-to-end preprocessing",
    }


def run_fold(dataset, specs, cfg, seed, fold, train, test, path, groups=None):
    path.mkdir(parents=True)
    X, y, task = dataset.X.iloc[train], dataset.y[train], dataset.task
    np.savez_compressed(path / "outer_split.npz", train=train, test=test)
    write_json(path / "candidates.json", [s.to_dict() for s in specs])
    print(
        f"{dataset.metadata['id']} seed={seed} fold={fold} OOF B={len(specs)} started", flush=True
    )
    with ResourceTimer() as search_timer:
        train_groups = None if groups is None else groups[train]
        oof = build_oof(
            X,
            y,
            specs,
            task,
            cfg["inner_folds"],
            seed + 5000 + fold,
            n_jobs=cfg.get("oof_jobs", 1),
            groups=train_groups,
        )
    with ResourceTimer() as screen_timer:
        cert = certify(
            y,
            oof.predictions,
            oof.null,
            task,
            cfg["certification"],
            seed + 6000 + fold,
            groups=train_groups,
        )
    print(
        f"{dataset.metadata['id']} fold={fold} OOF={search_timer.wall:.1f}s screen={screen_timer.wall:.1f}s",
        flush=True,
    )
    np.savez_compressed(
        path / "oof.npz", predictions=oof.predictions, null=oof.null, fold_ids=oof.fold_ids, y=y
    )
    np.savez_compressed(
        path / "certification.npz",
        quality=cert.quality,
        lower=cert.lower,
        passed=cert.passed,
        bootstrap=cert.bootstrap,
    )
    write_json(path / "inner_splits.json", [{"train": a, "valid": b} for a, b in oof.splits])
    pd.DataFrame(
        {
            "id": [s.id for s in specs],
            "quality": cert.quality,
            "lower": cert.lower,
            "passed": cert.passed,
            "features": [len(s.features) for s in specs],
        }
    ).to_csv(path / "candidate_metrics.csv", index=False)
    common = {
        "B": len(specs),
        "certified_count": int(cert.passed.sum()),
        "certification_rate": float(cert.passed.mean()),
        "seed": seed,
        "fold": fold,
        "task": task,
        "dataset": dataset.metadata["id"],
    }
    plans, rows = [], []
    for method in cfg["methods"]:
        method_path = path / method["id"]
        method_path.mkdir()
        if task not in method.get("tasks", ["binary", "regression", "multiclass"]):
            rows.append(
                {
                    **common,
                    "method": method["id"],
                    "status": "not_applicable",
                    "reason": "task restriction defined in config",
                }
            )
            continue
        try:
            with ResourceTimer() as tuning_timer:
                tuned, tuning = nested_tune(
                    X,
                    y,
                    specs,
                    task,
                    cfg["certification"],
                    method,
                    cfg["inner_folds"],
                    seed + 7000 + fold,
                    method_path,
                    train_groups,
                )
            pool_mask = candidate_pool_mask(specs, tuned)
            eligible = eligibility(cert, task, cfg["certification"], tuned) & pool_mask
            with ResourceTimer() as selection_timer:
                chosen = select(
                    y,
                    oof.predictions,
                    oof.null,
                    task,
                    cert.quality,
                    eligible,
                    tuned,
                    seed + 8000 + fold,
                )
            write_json(
                method_path / "selection.json",
                {
                    "config": tuned,
                    "ids": chosen.ids,
                    "weights": chosen.weights,
                    "objective": chosen.objective,
                    "trace": chosen.trace,
                    "tuning": tuning,
                    "eligible_count": len(chosen.trace[0]["eligible_pool_ids"]),
                    "eligible_count_before_deduplication": int(eligible.sum()),
                },
            )
            with ResourceTimer() as refit_timer:
                model = fit_selection(
                    X,
                    y,
                    specs,
                    chosen,
                    task,
                    tuned.get("affine_calibration", False),
                    oof.predictions,
                    n_jobs=cfg.get("refit_jobs", 1),
                )
            diag = diagnostics(y, oof.predictions, task, specs, chosen, cert.quality)
            if tuned.get("affine_calibration", False):
                write_json(method_path / "calibration.json", model.parameters)
                aggregate = aggregate_predictions(oof.predictions, chosen)
                calibrated = model.parameters["slope"] * aggregate + model.parameters["intercept"]
                diag["oof_calibrated_squared_loss"] = squared_loss(y, calibrated, task)
            raw_model = method_path / "model.joblib"
            joblib.dump(model, raw_model, compress=0)
            search_cost = 1 if method.get("charge_search", True) else 0
            resource = {
                "requested_K": method["K"],
                "aggregation": "affine_of_equal"
                if tuned.get("affine_calibration", False)
                else "equal"
                if np.allclose(chosen.weights, 1 / len(chosen.weights))
                else "frequency_weighted",
                "search": search_timer.to_dict(),
                "screen": screen_timer.to_dict(),
                "tuning": tuning_timer.to_dict(),
                "selection": selection_timer.to_dict(),
                "refit": refit_timer.to_dict(),
                "search_charged": bool(search_cost),
                "model_bytes": raw_model.stat().st_size,
                "train_wall_seconds": search_cost * (search_timer.wall + screen_timer.wall)
                + tuning_timer.wall
                + selection_timer.wall
                + refit_timer.wall,
                "train_cpu_seconds": search_cost * (search_timer.cpu + screen_timer.cpu)
                + tuning_timer.cpu
                + selection_timer.cpu
                + refit_timer.cpu,
                **model.capacity(),
                **measure_inference(model, X, lambda m, x: m.predict(x)),
            }
            write_json(method_path / "diagnostics.json", diag)
            write_json(method_path / "resources.json", resource)
            plans.append((method["id"], model, method_path, "selected", diag, resource))
            if len(plans) % 10 == 0:
                print(
                    f"{dataset.metadata['id']} fold={fold} finalized {len(plans)} selections",
                    flush=True,
                )
        except ValueError as error:
            if not str(error).startswith("infeasible:"):
                raise
            row = {
                **common,
                "method": method["id"],
                "status": "infeasible",
                "requested_K": method["K"],
                "reason": str(error),
            }
            rows.append(row)
            write_json(method_path / "status.json", row)
    for method in cfg.get("baselines", []):
        method_path = path / method["id"]
        method_path.mkdir()
        with ResourceTimer() as timer:
            model = baseline({**method, "classes": np.unique(y)}, X, task, seed + 9000 + fold).fit(
                X, y
            )
        raw_model = method_path / "model.joblib"
        joblib.dump(model, raw_model, compress=0)
        resource = {
            **capacity(model),
            **timer.to_dict(),
            "train_wall_seconds": timer.wall,
            "train_cpu_seconds": timer.cpu,
            "model_bytes": raw_model.stat().st_size,
            **measure_inference(model, X, lambda m, x: predict_baseline(m, x, task)),
        }
        write_json(method_path / "resources.json", resource)
        plans.append((method["id"], model, method_path, "baseline", {}, resource))
    # Seal decision provenance for every method before loading test labels here.
    write_json(
        path / "finalized_plan.json",
        {
            "methods": [p[0] for p in plans],
            "outer_test_evaluations_per_method": 1,
            "infeasible": [r["method"] for r in rows],
        },
    )
    Xtest, ytest = dataset.X.iloc[test], dataset.y[test]
    for name, model, method_path, kind, diag, resource in plans:
        prediction = (
            model.predict(Xtest) if kind == "selected" else predict_baseline(model, Xtest, task)
        )
        metrics = evaluate_metrics(ytest, prediction, task)
        np.savez_compressed(
            method_path / "test_predictions.npz", prediction=prediction, y=ytest, rows=test
        )
        metrics["normalized_squared_loss"] = squared_loss(ytest, prediction, task) / max(
            squared_loss(
                ytest,
                np.tile(np.mean(y), len(ytest))
                if task != "multiclass"
                else np.tile(np.bincount(y) / len(y), (len(ytest), 1)),
                task,
            ),
            1e-15,
        )
        row = {
            **common,
            "method": name,
            "status": "complete",
            **metrics,
            **{k: v for k, v in diag.items() if k != "residual_eigenvalues"},
            **{k: v for k, v in resource.items() if not isinstance(v, dict)},
        }
        if kind == "baseline":
            row.update(B=None, certified_count=None, certification_rate=None)
        else:
            method_cfg = next(m for m in cfg["methods"] if m["id"] == name)
            pool = candidate_pool_mask(specs, method_cfg)
            row.update(
                generated_B=len(specs),
                B=int(pool.sum()),
                certified_count=int(cert.passed[pool].sum()),
                certification_rate=float(cert.passed[pool].mean()),
            )
        rows.append(row)
        write_json(method_path / "test_metrics.json", metrics)
    write_json(path / "records.json", rows)
    return rows


def run_experiment(config_path, root=None, run_id=None):
    root = Path(root or Path(__file__).resolve().parents[3])
    cfg = yaml.safe_load(Path(config_path).read_text())
    if cfg.get("role", "development") != "development":
        raise ValueError("Confirmation runner is locked until a reviewed manifest exists")
    if cfg.get("sampling_structure", "iid") != "iid":
        raise ValueError("Grouped/temporal tasks require explicit split and bootstrap support")
    run_id = run_id or cfg["id"] + "_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = root / "results" / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "config.json", cfg)
    write_json(path / "environment.json", environment(root))
    snapshot(root, path / "source.zip")
    write_json(path / "status.json", {"state": "running", "run_id": run_id})
    records = []
    try:
        with threadpool_limits(limits=1), ResourceTimer() as total:
            for data_cfg in cfg["datasets"]:
                for seed in cfg["seeds"]:
                    dataset = load_dataset(data_cfg, seed)
                    task_path = path / data_cfg["id"] / f"seed_{seed}"
                    task_path.mkdir(parents=True)
                    dataset.X.to_pickle(task_path / "X.pkl")
                    np.save(task_path / "y.npy", dataset.y)
                    write_json(task_path / "metadata.json", dataset.metadata)
                    groups = None
                    if cfg.get("group_exact_duplicates", False):
                        groups = pd.util.hash_pandas_object(dataset.X, index=False).to_numpy()
                        np.save(task_path / "groups.npy", groups)
                    splits = make_splits(dataset.y, dataset.task, cfg["outer_folds"], seed, groups)
                    for fold, (train, test) in enumerate(splits):
                        specs = generate_candidates(
                            dataset.X.shape[1], cfg["candidates"], seed + 1000 + fold
                        )
                        records.extend(
                            run_fold(
                                dataset,
                                specs,
                                cfg,
                                seed,
                                fold,
                                train,
                                test,
                                task_path / f"fold_{fold}",
                                groups,
                            )
                        )
                        pd.DataFrame(records).to_csv(path / "records.csv", index=False)
                        print(f"{data_cfg['id']} seed={seed} fold={fold} complete", flush=True)
        write_json(
            path / "status.json",
            {
                "state": "complete",
                "run_id": run_id,
                "records": len(records),
                "resources": total.to_dict(),
            },
        )
    except BaseException as error:
        write_json(
            path / "status.json",
            {"state": "failed", "run_id": run_id, "error": repr(error), "records": len(records)},
        )
        finalize_manifest(path)
        raise
    finalize_manifest(path)
    return path
