# exp_006: larger real development tasks and 6,000-tree libraries

Registered 2026-10-01 before inspecting these outer-test scores. Requested follow-up
to exp_004/005; all data and results remain DEVELOPMENT evidence. Confirmation
ledger is untouched. No claim of top-conference readiness or novel pruning objective.

## Questions and falsification

H1: At fixed K, larger balanced nested libraries improve outer performance beyond
quality-only selection. A flat/worsening pool curve, or the same gains for top-quality,
weakens a special complementarity explanation. H2: wider masks/deeper trees correct
representation limits. Compare six homogeneous cells at exactly B=1,000 and K=64;
non-monotone/negative changes falsify a universal remedy. H3: certification and
residual correlation contribute beyond co-error loss optimization. Compare controls
at B=6,000,K=64. Report Brier as well as AUC: the optimized objective is Brier, not AUC.

## Dataset selection, definitions and exclusions

The user's requested harder classification emphasis motivates credit default and
HIGGS. MiniBooNE is retained as a physics control, even if easier; observed baseline
AUC does not determine inclusion. Superconductivity adds a larger/high-dimensional
regression task, California housing a larger/lower-dimensional regression control.
This is a deliberately chosen diagnostic panel, not an unbiased benchmark sample.
The requested 0.70-0.80 RF/logistic AUC is a preference, not an inclusion filter or
promised result. No labels/features/noise are altered to force that score range.

| Task | Public source rows | Raw features | Target | Study rows |
|---|---:|---:|---|---:|
| Credit default | 30,000 | 23 | Default next month, binary | 20,000 |
| HIGGS | 98,050 | 28 | Signal vs background | 20,000 |
| MiniBooNE | 130,064 | 50 | Electron vs muon neutrinos | 20,000 |
| Superconductivity | 21,263 | 81 | Critical temperature | 20,000 |
| California housing | 20,640 | 8 | Median house value / 100,000 | 20,000 |

All sources and adaptations are pinned in configs/datasets/exp_006_public_sources.json
and API responses are archived under source_metadata/. MiniBooNE's UCI page says
130,065; actual file header and parsed rows agree at 130,064. No row was lost in parsing.
MiniBooNE's -999 sentinel rows are preserved, with counts documented, not removed or
silently imputed. HIGGS uses the published OpenML 23512 version 2 subset, not all
11 million original events or the original paper's final 500,000-example test protocol.
Credit ID and target are excluded; X2/X3/X4 are categorical, other encoded quantities
retain their numeric/ordinal representation. California features are deterministic
rowwise sklearn definitions, not globally fitted transforms. All 81 superconductivity
train-file features retained; formula file isn't a predictor. Attribution/licenses and
raw/cache SHA256 are recorded. California separate data license remains unspecified.

Uniform regression or label-stratified classification subsampling to 20,000 uses
seed 7321 once, independently of split/candidate seeds and before scoring. Selected
source row IDs are stored. This computational cap is an explicit restriction. All five
registered tasks are retained. Smaller bundled sklearn tasks aren't repeated here.
Other considered sources: Spambase (4,601 rows) and wine quality (4,898 rows,11 features)
weren't included because they add less scale than this panel; Bank Marketing would
require prospective-duration and time protocol decisions and is deferred. These are
scope decisions, not exclusions from a claimed benchmark suite.

## Split integrity and bootstrap

One generation/split seed 11; three outer folds, two OOF folds inside each outer train.
Full hyperparameter sweep is frozen before the first main run. No test-dependent
choice of methods or parameters; no per-task best outer score becomes a tuned result.
Exact feature-vector hashes (index excluded) define groups. Hash collisions can
conservatively merge groups; identical raw vectors never straddle outer or inner folds.
Superconductivity has 6,093 repeated feature rows in the full source and MiniBooNE 467;
this prevents a particularly large duplication confound. This is not a composition,
geographic, temporal, person-level or physics-systematics holdout. Residual dependence
between non-identical records remains possible. Categorical/numeric imputers/encoders
are fit only on the appropriate training fold. Original missing/sentinel policy retained.

Empirical lower percentile screening: 100 joint cluster-bootstrap draws resampling
feature groups, preserving all within-group rows. Threshold AUC>.51 or crossfit skill>.01.
Clusters may contain mixed targets, so group bootstrap is not class stratified; class
presence is checked. Weighted multiplicities produce exactly the same AUROC (including
ties)/skill as literal resampling, tested independently. This computational improvement
does not establish coverage for OOF predictions or simultaneous validity over B.

## Candidate and selector grid

6,000 specs per outer fold, fixed feature masks, leaf minimum10, full training rows.
A deterministic balanced cycle crosses feature fractions {.05,.20,.50} with depths
{3,6}; each of six cells has 1,000 candidates. All candidate IDs have independent
SeedSequence([seed,id]) streams; any prefix is unchanged when B grows. Mixed libraries
contain a single learner family with varied widths/depths, unlike earlier homogeneous
settings. Fraction/depth sweeps isolate homogeneous cells at identical B=1,000,K=64.

- Prefix B={1,000,3,000,6,000}, independently crossed with K={16,64,128}, and
  random/top-quality/co-error selection; top/co-error B=300,K=64 reference.
- Co-error B=6,000,K=256 as a larger final-size diagnostic.
- At B=6,000,K=64: certified random/top/co-error, absolute/signed residual correlation,
  quality-diversity lambda={.02,.10,.50}, one co-error swap, Caruana squared selection.
- Six homogeneous width/depth cells at B=1,000,K=64, each top-quality and co-error.
- Predesignated reported anchor: unscreened co-error B=6,000,K=64, equal weights.

Greedy construction is used for the scaling grid; one-swap full-pool variant separately
measures optimizer sensitivity. Matrix-free correlation columns and cached co-error
columns match dense/direct toy references. Squared Caruana with replacement uses the
same Gram identity, inducing frequency weights; no new statistical method claimed.
All subset methods use equal weights, except this explicit Caruana control. Nothing
is selected using outer-test AUC. More flexible weights are deferred until this evidence.

## Baselines and resources

RF256 and ExtraTrees256 defaults; RF256 leaf5; RF64 depth6 leaf10; RandomPatches64
feature.2,row.5,depth6,leaf10; logistic C1 / ridge alpha1 with fold-local standardization;
XGBoost300 depth6, LightGBM300 leaves31, CatBoost300 depth6, learning rate.05. Static
settings, not tuned competitive baselines. Boosting is deliberately stronger than the
100-round exp_004 presets. Strong modern neural/AutoML baselines remain a later gate.

OOF and selected refits use four threads in one process, BLAS limited to one;
selectors and baseline phases run sequentially. CPU timer includes all threads. Every search-dependent setting is charged
the ENTIRE 6,000-spec shared search/screen cost, including smaller prefixes/cells: a
conservative shared-library cost, not independent prefix training cost. B is the pool
available to that selector, generated_B=6,000. Report separate search/selection/refit,
observed process RSS, serialized size and end-to-end pipeline inference. Search time
and final capacity are not matched; tree counts alone do not establish fairness.
Expected base OOF fitting count is 5 tasks * 3 outer * 2 inner * 6,000 = 180,000,
plus selected refits and baselines. No concurrent unrelated CPU tests while main timing.

## Reporting

All sweep cells, feasibility and baseline results; per-task primary AUC or RMSE,
Brier/logloss/PR-AUC/balanced accuracy for classification, MAE/R2 for regression.
One seed and three folds do not justify population confidence intervals. No choosing
the best outer-test configuration and claiming a validated tuned method. Store OOF,
bootstrap, feature masks/specs, selected IDs/weights, train/test row IDs and groups,
test predictions, resources, environment/source/commit snapshots. Paper figures/tables
are generated from verified stored records. Each task is a separate resumable-by-task
run/config to keep archives manageable; completed folds are never overwritten.
