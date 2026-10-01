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
