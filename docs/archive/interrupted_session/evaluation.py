import hashlib
import json
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from .baselines import baseline, capacity
from .candidates import generate_candidates, model_predict
from .certification import bootstrap_screen
from .datasets import load_dataset
from .ensembles import SelectedEnsemble
from .metrics import evaluate, residual_matrix
from .oof import cross_fit, make_folds
from .selection import correlation_matrix, select
from .storage import artifact_manifest, measured, metadata, source_snapshot, write_json


def derived_seed(seed, *tags):
    raw = json.dumps([seed, *tags], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little") % (2**31 - 1)


def eligible_for(cert, method, task):
    if not method.get("certified", True):
        return np.ones_like(cert.passed, dtype=bool)
    return (
        cert.lower > method.get("threshold", 0.01 if task == "regression" else 0.51)
        if "threshold" in method
        else cert.passed
    )


def fit_selection(X, y, specs, oof, cert, task, method, seed):
    selection = select(
        y,
        oof.predictions,
        oof.null,
        task,
        cert.quality,
        eligible_for(cert, method, task),
        method,
        seed,
    )
    c = len(np.unique(y)) if task != "regression" else 0
    model = SelectedEnsemble([specs[i] for i in selection.ids], selection.weights, task, c).fit(
        X, y
    )
    return selection, model


def tune_selector(X, y, specs, task, method, certification, n_folds, seed, output):
    """True nested selector tuning; base OOF regenerated for each tuning-train split."""
    grid = method.get("tuning_grid")
    if not grid:
        return method, {"mode": "fixed_before_run"}
    variants = []
    for override in grid:
        v = deepcopy(method)
        v.pop("tuning_grid")
        v.update(override)
        variants.append(v)
    tune_folds = make_folds(y, task, method.get("tuning_folds", 3), derived_seed(seed, "tune"))
    scores = np.full((len(tune_folds), len(variants)), np.nan)
    output.mkdir(parents=True, exist_ok=True)
    for t, (tr, va) in enumerate(tune_folds):
        folds = make_folds(y[tr], task, n_folds, derived_seed(seed, "tune_oof", t))
        oof = cross_fit(X.iloc[tr], y[tr], specs, task, folds)
        cert = bootstrap_screen(
            y[tr],
            oof.predictions,
            oof.null,
            task,
            certification,
            derived_seed(seed, "tune_boot", t),
        )
        np.savez_compressed(
            output / f"fold_{t}.npz",
            train=tr,
            valid=va,
            oof=oof.predictions,
            null=oof.null,
            quality=cert.quality,
            lower=cert.lower,
            bootstrap=cert.bootstrap,
        )
        write_json(output / f"fold_{t}_audit.json", oof.audit)
        for j, variant in enumerate(variants):
            try:
                selection, model = fit_selection(
                    X.iloc[tr],
                    y[tr],
                    specs,
                    oof,
                    cert,
                    task,
                    variant,
                    derived_seed(seed, "tune_selection", t),
                )
            except ValueError as error:
                if "infeasible:" not in str(error):
                    raise
                continue
            p = model.predict(X.iloc[va])
            metrics = evaluate(y[va], p, task)
            scores[t, j] = (
                -metrics["auroc"]
                if task == "binary"
                else metrics["rmse"]
                if task == "regression"
                else metrics["log_loss"]
            )
            np.savez_compressed(
                output / f"validation_{t}_{j}.npz",
                prediction=p,
                selected_ids=selection.ids,
                weights=selection.weights,
            )
    valid = np.all(np.isfinite(scores), axis=0)
    if not valid.any():
        raise ValueError("infeasible: no tuning variant completed every fold")
    averages = np.where(valid, np.nan_to_num(scores).mean(axis=0), np.inf)
    best = int(np.argmin(averages))
    info = {
        "mode": "nested_selector_tuning",
        "variants": variants,
        "scores": [[float(v) if np.isfinite(v) else None for v in row] for row in scores],
        "selected_variant": best,
        "folds": [{"train": tr, "valid": va} for tr, va in tune_folds],
    }
    write_json(output / "decision.json", info)
    return variants[best], info


def diagnostics(E, specs, ids, quality, weights):
    selected = E[:, ids]
    R = correlation_matrix(selected)
    pairs = R[np.triu_indices(len(ids), 1)]
    G = selected.T @ selected / len(selected)
    eig = np.linalg.eigvalsh(G)
    overlaps = []
    for t, i in enumerate(ids):
        for j in ids[t + 1 :]:
            a, b = set(specs[i].features), set(specs[j].features)
            overlaps.append(len(a & b) / len(a | b))
    return {
        "mean_quality": float(quality[ids].mean()),
        "mean_residual_correlation": float(pairs.mean()) if len(pairs) else None,
        "mean_abs_residual_correlation": float(np.abs(pairs).mean()) if len(pairs) else None,
        "max_abs_residual_correlation": float(np.abs(pairs).max()) if len(pairs) else None,
        "constant_residual_count": int((np.std(selected, axis=0) <= 1e-12).sum()),
        "residual_second_moment_eigenvalues": eig.tolist(),
        "effective_rank_participation": float(eig.sum() ** 2 / np.sum(eig**2))
        if np.sum(eig**2) > 0
        else 0.0,
        "mean_feature_jaccard": float(np.mean(overlaps)) if overlaps else None,
        "actual_oof_squared_loss": float(weights @ G @ weights),
        "individual_quality_quantiles": np.quantile(
            quality, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        ).tolist(),
    }


def inference_timing(model, X, task, n_classes):
    sample = X.iloc[: min(len(X), 256)]
    if isinstance(model, SelectedEnsemble):
        predict = model.predict
    else:
        predict = lambda frame: model_predict(model, frame, task, n_classes)
    predict(sample)  # warm-up on outer-training rows
    times = []
    for _ in range(5):
        start = time.perf_counter()
        predict(sample)
        times.append(time.perf_counter() - start)
    median = float(np.median(times))
    return {
        "inference_batch_rows": len(sample),
        "inference_median_seconds": median,
        "inference_throughput_rows_per_second": len(sample) / median,
        "inference_scope": "warm process; outer-train rows; 5 repeats; preprocessing included",
    }


def run_experiment(config_path, output_base="results/runs"):
    cfg = yaml.safe_load(Path(config_path).read_text())
    if cfg.get("suite", "development") != "development":
        raise ValueError(
            "Confirmation runner requires a locked native benchmark adapter; unavailable"
        )
    method_names = [m["name"] for m in cfg["methods"]] + [
        m["name"] for m in cfg.get("baselines", [])
    ]
    if len(set(method_names)) != len(method_names):
        raise ValueError("Method names must be unique")
    raw = json.dumps(cfg, sort_keys=True).encode()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{cfg['experiment_id']}_{stamp}_{hashlib.sha256(raw).hexdigest()[:8]}"
    root = Path(output_base) / run_id
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "config.json", cfg)
    write_json(root / "environment.json", metadata())
    source_snapshot(root)
    write_json(root / "status.json", {"state": "running", "run_id": run_id})
    records = []
    print(f"RUN {root}", flush=True)
    try:
        with threadpool_limits(limits=1), measured() as run_resource:
            for dcfg in cfg["datasets"]:
                for seed in cfg.get("seeds", [0]):
                    data = load_dataset(dcfg, seed)
                    droot = root / data.name / f"seed_{seed}"
                    droot.mkdir(parents=True)
                    joblib.dump({"X": data.X, "y": data.y}, droot / "dataset.joblib", compress=3)
                    write_json(droot / "dataset_metadata.json", data.metadata)
                    outer = make_folds(
                        data.y,
                        data.task,
                        cfg["evaluation"]["outer_folds"],
                        derived_seed(seed, data.name, "outer"),
                    )
                    for f, (train, test) in enumerate(outer):
                        print(f"  {data.name} seed={seed} outer={f}", flush=True)
                        foldroot = droot / f"outer_{f}"
                        foldroot.mkdir()
                        X, y = data.X.iloc[train], data.y[train]
                        n_classes = len(np.unique(y)) if data.task != "regression" else 0
                        foldseed = derived_seed(seed, data.name, f)
                        specs = generate_candidates(
                            X.shape[1], cfg["candidates"], derived_seed(foldseed, "pool")
                        )
                        write_json(foldroot / "candidates.json", [s.to_dict() for s in specs])
                        inner = make_folds(
                            y,
                            data.task,
                            cfg["evaluation"]["oof_folds"],
                            derived_seed(foldseed, "oof"),
                        )
                        write_json(
                            foldroot / "splits.json",
                            {
                                "outer_train": train,
                                "outer_test": test,
                                "inner_index_scope": "positions in outer_train",
                                "oof": [{"train": tr, "valid": va} for tr, va in inner],
                            },
                        )
                        with measured() as pool_resource:
                            oof = cross_fit(X, y, specs, data.task, inner)
                        with measured() as cert_resource:
                            cert = bootstrap_screen(
                                y,
                                oof.predictions,
                                oof.null,
                                data.task,
                                cfg["certification"],
                                derived_seed(foldseed, "boot"),
                            )
                        np.savez_compressed(
                            foldroot / "oof.npz",
                            predictions=oof.predictions,
                            y=y,
                            null=oof.null,
                            quality=cert.quality,
                            lower=cert.lower,
                            passed=cert.passed,
                            bootstrap=cert.bootstrap,
                            bootstrap_indices=cert.indices,
                        )
                        write_json(foldroot / "oof_fit_audit.json", oof.audit)
                        write_json(
                            foldroot / "pool_resources.json",
                            {"cross_fit": pool_resource, "screen": cert_resource},
                        )
                        E = residual_matrix(y, oof.predictions, data.task)
                        prepared = []
                        for method in cfg["methods"]:
                            name = method["name"]
                            common = {
                                "dataset": data.name,
                                "task": data.task,
                                "seed": seed,
                                "outer_fold": f,
                                "method": name,
                                "B": len(specs),
                                "requested_K": method["K"],
                                "certification_rate": float(
                                    eligible_for(cert, method, data.task).mean()
                                ),
                                "n_eligible": int(eligible_for(cert, method, data.task).sum()),
                            }
                            try:
                                with measured() as tune_resource:
                                    final, tune_info = tune_selector(
                                        X,
                                        y,
                                        specs,
                                        data.task,
                                        method,
                                        cfg["certification"],
                                        cfg["evaluation"]["oof_folds"],
                                        foldseed,
                                        foldroot / f"tuning_{name}",
                                    )
                                with measured() as selection_resource:
                                    selection = select(
                                        y,
                                        oof.predictions,
                                        oof.null,
                                        data.task,
                                        cert.quality,
                                        eligible_for(cert, final, data.task),
                                        final,
                                        derived_seed(foldseed, "selection"),
                                    )
                                with measured() as refit_resource:
                                    model = SelectedEnsemble(
                                        [specs[i] for i in selection.ids],
                                        selection.weights,
                                        data.task,
                                        n_classes,
                                    ).fit(X, y)
                                mpath = foldroot / f"model_{name}.joblib"
                                joblib.dump(model, mpath, compress=0)
                                oof_prediction = np.einsum(
                                    "b,nb...->n...",
                                    selection.weights,
                                    oof.predictions[:, selection.ids],
                                )
                                details = {
                                    "selected_ids": selection.ids,
                                    "weights": selection.weights,
                                    "selection_objective": selection.objective,
                                    "trace": selection.trace,
                                    "final_method_config": final,
                                    "tuning": tune_info,
                                    "selection": selection_resource,
                                    "refit": refit_resource,
                                    "tuning_resources": tune_resource,
                                    "oof_selection_metrics_optimistic": evaluate(
                                        y, oof_prediction, data.task
                                    ),
                                    "diagnostics": diagnostics(
                                        E, specs, selection.ids, cert.quality, selection.weights
                                    ),
                                }
                                write_json(foldroot / f"selection_{name}.json", details)
                                resource = {
                                    "search_wall_seconds": pool_resource["wall_seconds"]
                                    + cert_resource["wall_seconds"],
                                    "search_cpu_seconds": pool_resource["cpu_seconds"]
                                    + cert_resource["cpu_seconds"],
                                    "selection_wall_seconds": selection_resource["wall_seconds"],
                                    "refit_wall_seconds": refit_resource["wall_seconds"],
                                    "training_wall_seconds": sum(
                                        r["wall_seconds"]
                                        for r in [
                                            pool_resource,
                                            cert_resource,
                                            tune_resource,
                                            selection_resource,
                                            refit_resource,
                                        ]
                                    ),
                                    "training_cpu_seconds": sum(
                                        r["cpu_seconds"]
                                        for r in [
                                            pool_resource,
                                            cert_resource,
                                            tune_resource,
                                            selection_resource,
                                            refit_resource,
                                        ]
                                    ),
                                    "model_bytes": mpath.stat().st_size,
                                    **model.capacity(),
                                }
                                prepared.append((common, model, resource, details))
                            except ValueError as error:
                                if "infeasible:" not in str(error):
                                    raise
                                records.append(
                                    {**common, "status": "infeasible", "reason": str(error)}
                                )
                        for bcfg in cfg.get("baselines", []):
                            with measured() as training:
                                model = baseline(bcfg, X, data.task, foldseed).fit(X, y)
                            name = bcfg["name"]
                            mpath = foldroot / f"model_{name}.joblib"
                            joblib.dump(model, mpath, compress=0)
                            write_json(
                                foldroot / f"training_{name}.json",
                                {"config": bcfg, "resources": training},
                            )
                            resource = {
                                "training_wall_seconds": training["wall_seconds"],
                                "training_cpu_seconds": training["cpu_seconds"],
                                "model_bytes": mpath.stat().st_size,
                                **capacity(model),
                            }
                            common = {
                                "dataset": data.name,
                                "task": data.task,
                                "seed": seed,
                                "outer_fold": f,
                                "method": name,
                                "B": None,
                                "requested_K": None,
                            }
                            prepared.append((common, model, resource, {}))
                        # All choices and fits complete before the outer test is scored.
                        write_json(
                            foldroot / "finalized_methods.json", [c[0]["method"] for c in prepared]
                        )
                        for common, model, resource, details in prepared:
                            timing = inference_timing(model, X, data.task, n_classes)
                            with measured() as test_resource:
                                prediction = (
                                    model.predict(data.X.iloc[test])
                                    if isinstance(model, SelectedEnsemble)
                                    else model_predict(
                                        model, data.X.iloc[test], data.task, n_classes
                                    )
                                )
                            np.savez_compressed(
                                foldroot / f"test_{common['method']}.npz",
                                prediction=prediction,
                                y=data.y[test],
                                row_ids=test,
                            )
                            metrics = evaluate(data.y[test], prediction, data.task)
                            record = {
                                **common,
                                "status": "complete",
                                "metrics": metrics,
                                "resources": {
                                    **resource,
                                    **timing,
                                    "outer_test_predict": test_resource,
                                },
                                "diagnostics": details.get("diagnostics"),
                                "oof_metrics": details.get("oof_selection_metrics_optimistic"),
                            }
                            records.append(record)
                        write_json(root / "records.json", records)
        write_json(root / "run_resources.json", run_resource)
        write_json(
            root / "status.json", {"state": "complete", "run_id": run_id, "records": len(records)}
        )
        artifact_manifest(root)
        print(f"COMPLETE {root}", flush=True)
    except BaseException as error:
        write_json(root / "records.json", records)
        write_json(
            root / "status.json",
            {
                "state": "failed",
                "run_id": run_id,
                "exception": type(error).__name__,
                "message": str(error),
            },
        )
        artifact_manifest(root)
        raise
    return root
