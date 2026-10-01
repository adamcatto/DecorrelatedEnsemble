# Feature-sparse ensemble research

An ongoing, evidence-driven study of empirical certification, error complementarity,
and selection from weak fixed-feature-subspace trees. **No novelty or benchmark
superiority claim is established.** The [methodology](docs/methodology.md),
[research questions](docs/research_questions.md), [literature audit](docs/literature_review.md),
and [experiment registry](docs/experiment_registry.md) describe the study.

## Reproduce

```sh
uv sync --locked --python 3.12 --extra dev --extra boosting
uv run --no-sync pytest -q
uv run --no-sync python scripts/run_experiment.py configs/experiments/smoke.yaml
```

Experiment settings live in YAML. The runner builds fold-local preprocessing and
OOF predictions within each untouched outer training fold. Tuning, when configured,
regenerates fits inside another split layer. Fewer than K screened candidates is
an explicit infeasible outcome. There is no test-driven threshold or K fallback.

Runs under `results/runs/` contain raw data, fold indices, specifications, OOF and
test predictions, bootstrap draws, models, metrics, timing/capacity, environment,
source snapshot, status, and SHA256 manifests. Raw runs are ignored by Git because
they include many models; audited small experiment bundles and generated summaries
will be published separately. Joblib/pickle artifacts should only be loaded from
trusted runs. Numerical seeds are reproducible; exact bytes/timing can vary with
hardware and packages, which are recorded.

Equal-weight methods and with-replacement ensemble selection are labeled separately.
OOF percentile bootstrap bounds are heuristic screens; they do not supply valid
familywise statistical certification. Random subspaces, candidate search, ensemble
selection, and quadratic pruning all have substantial prior art.

The [living manuscript](paper/main.tex) is a development draft. Established benchmark
versions and confirmation protection are tracked in [the benchmark plan](docs/benchmark_plan.md).

## Code

`src/decorrelated_ensemble/` separates datasets, preprocessing, candidates,
certification, selection, ensembles, metrics, and evaluation. Tests include analytic
loss identities and exact tiny-pool references. The source files of an interrupted
concurrent session are retained in `docs/archive/` for auditability; canonical code
is exclusively under `src/`.

## Completed development evidence

`exp_001_mechanisms_v1` tests nine synthetic regimes for binary classification
and regression (108 outer folds). Co-error selection has modest conditional gains;
the broad performance hypothesis fails against strong static forest/boosting
baselines at these settings. Screening causes infeasibility and null false survivors.
`exp_002_affine_v1` tests aggregate attenuation (36 folds). Scalar calibration helps
unchanged co-error subsets in five of six regimes, but affine-profiled selection
loses on five of six task means despite improving its training OOF criterion.
These negative findings are retained in the manuscript and research log.
`exp_003_pool_b{30,100,300}_v1` adds 72 outer evaluations at fixed K=8. Exact
pool/OOF/bootstrap prefixes isolate B. Binary additive co-error improves, but
regression gains are non-monotone and profiled selection overfits its OOF criterion.
`exp_004_real_f{10,50}_v1` adds breast cancer, wine and diabetes (36 outer evaluations).
Co-error improves over top quality, but RF and logistic/ridge controls do better
in each task/width mean. Width effects are conditional; duplicate OOF predictions
expose effective weighting as a confound. These are tiny numerical development
tasks, not a representative confirmation suite.
`exp_005_oof_duplicates_v1` tests exact training-OOF duplicate removal (18 folds).
It removes some quality-control gaps while hurting diabetes co-error performance,
showing that repeated prediction patterns can provide useful implicit weights.
Distinct patterns do not establish independent errors or distinct refitted functions.

Bundles in `results/artifacts/` include data, predictions, decisions, config,
source snapshot, environment, and checksums. Large-run bundles explicitly omit
serialized models; export JSON files list every omission. To audit an extracted
prediction bundle, place its run folder in `results/runs/`, then use:

```sh
uv run --no-sync python scripts/aggregate_results.py exp_002_affine_v1 --predictions-only
uv run --no-sync python scripts/audit_run.py exp_002_affine_v1 --predictions-only
uv run --no-sync python scripts/make_affine_report.py exp_002_affine_v1
```

The complete runs including refitted models can be regenerated with the saved
YAML configurations. No locked confirmation suite has been evaluated.

After extracting the real-data bundles into `results/runs/`, regenerate their
paired-control audits, tables and figures with:

```sh
uv run --no-sync python scripts/make_real_report.py exp_004_real_f10_v1 exp_004_real_f50_v1 --predictions-only
uv run --no-sync python scripts/make_duplicate_report.py exp_005_oof_duplicates_v1 exp_004_real_f10_v1 --predictions-only
```

Use `audit_run.py` on each run to reconstruct all stored metrics and loss identities.

### Larger public-data development sweeps (exp_006)

Download pinned public sources once into the ignored offline cache:

```bash
python scripts/prepare_large_data.py
python scripts/run_experiment.py configs/experiments/smoke_large.yaml --run-id smoke_large_001
python scripts/run_experiment.py configs/experiments/exp_006_large_higgs.yaml --run-id exp_006_large_higgs_v1
```

Repeat the last command for credit_default, miniboone, superconductivity and
california_housing task configs. Each generates 6,000 specs per fold and evaluates
52 fixed selection settings plus nine static baseline settings on 20,000 rows.
See `experiments/exp_006_large/protocol.md` for pool/size/width/depth controls,
exact-feature group splits, source attributions, caps and resource limitations.
