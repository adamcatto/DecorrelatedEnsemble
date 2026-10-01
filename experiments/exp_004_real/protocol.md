# Experiment 004: real development data (before results)

## Questions and falsification

Does error-complementarity selection add value beyond quality at fixed B and K
on three real datasets? Does expanding each mask from 10% to 50% help, suggesting
that restricted representation is a more important bottleneck than the selector?
Record counterexamples and every baseline gap. Failure to consistently beat
unfiltered top quality refutes a broad selector advantage at these settings;
failure of larger masks to help refutes a universal feature-width explanation.
Screened comparisons must use shared feasible splits and report coverage separately.

## Fixed design

Use all rows of sklearn's bundled breast cancer, wine, and diabetes copies.
These were named before scores were inspected, to cover binary/multiclass/regression.
No dataset is excluded. N/p: 569/30, 178/13, 442/10. Numerical dense features only,
no missing entries; this is a small development panel, not a benchmark suite.
Diabetes uses `scaled=False` to avoid the loader's full-data centering/scaling.
The raw-loader correction precedes any scored builtin study; earlier studies
were synthetic. Breast cancer uses sklearn's positive class 1=benign, so AUPRC
must be interpreted for that class. Wine's labels are 0/1/2.

Two fixed configs: `configs/experiments/exp_004_real_f{10,50}.yaml`.
B=100, K=8, max depth 3, minimum leaf 5, no candidate row subsampling;
feature count is ceil(fraction*p), giving 3/2/1 at .1 and 15/7/5 at .5.
Three outer folds and three training OOF folds; seeds 11/29 are repeated split
and candidate seeds on the **same** builtin data, not new independent datasets.
All masks remain sampled without outcomes. The wider masks are not nested
prefixes of the narrower masks: width changes the candidate population.
Require identical raw data, outer/inner splits and unchanged-control predictions
across the two runs; candidate masks/OOF predictions may change.

Use the existing v0-v4 equal-weight ablations, no-screen controls, residual versus
prediction correlation, signed/absolute correlation, covariance, shrinkage .5,
direct squared-loss selection, and with-replacement Brier/MSE Caruana selection.
Caruana uses eight forward steps and may retain fewer than eight unique members;
its unequal frequency weights/capacity are recorded as a separate reference.
Regression alone adds affine co-error and APCE as preserved diagnostics. Binary
alone adds direct AUROC selection. Bootstrap: 100 joint draws, lower .05 quantile,
AUROC or macro-OVR threshold .51; regression cross-fitted skill threshold .01.
This is an empirical screen, not simultaneous statistical certification.
No tuning grid, threshold search, or outer-test-driven config change is allowed.

Forest/ExtraTrees 256-tree references, shallow 8-tree capacity controls,
Random Patches, XGBoost, LightGBM, CatBoost use exp_001's fixed configurations.
Add default Ridge(alpha=1) or LogisticRegression(C=1,max_iter=3000), with
imputation/encoding and StandardScaler fitted on outer training only. Do not
reuse .1-run performance to adjust .5-run methods. All plans are finalized
before their outer-test predictions are scored once.

## Analysis and interpretation

Primary: binary AUROC, multiclass log loss, regression RMSE. Report all stored
secondary metrics including Brier, macro AUROC, MAE/R2 and normalized squared loss.
Report per-dataset method means over six paired outer folds, plus split/seed
effects and coverage. Do not pool these three different metrics into one rank
or present fold dispersion as an IID uncertainty estimate. No task-population
confidence interval from a single dataset per task type is meaningful.
The overlap of repeated CV folds prevents treating six folds as six independent
replicates. Across-width comparisons use paired effects and full distributions.
Co-error optimizes Brier for wine, not its primary log loss; retain this metric
mismatch in interpretation. Test candidate-survival and selected quality/diversity
as diagnostics, never as independent evidence of causal complementarity.

Report charged search CPU time, leaves/nodes, serialized bytes and warm inference
measurements. Configurations are static and budgets unmatched: comparisons do
not establish superiority under tuned or matched-compute conditions. Ridge and
logistic scaling may differ from tree preprocessing for the justified requirement
of penalized linear models. No clinical or deployed-use inference is made.
No locked confirmation access; these familiar, tiny tasks are development only.

## Provenance and next decision

UCI breast cancer: Wolberg et al. (1993), doi:10.24432/C5DW2B, CC-BY-4.0.
UCI wine: Aeberhard/Forina (1992), doi:10.24432/C5PC7J, CC-BY-4.0.
Diabetes: Efron et al. (2004), Least Angle Regression, as distributed by sklearn;
the original dataset license is not stated in loader documentation and is recorded
as unresolved rather than asserted to be the sklearn software license.
Source URLs, descriptions, version, loader arguments, target encodings, feature
names/types, class counts and content hash are saved in every dataset artifact.

First run a wine smoke to verify full multiclass artifacts. Audit all completed
runs, export frozen evidence, generate tables/figures from stored artifacts and
update the paper. Choose the next test from the complete results; this does not
replace exp_003's independently selected-library/refit mechanism experiment.
