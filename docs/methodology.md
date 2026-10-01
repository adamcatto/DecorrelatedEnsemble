# Evaluation protocol v0 (written before results)

## Data separation

Use shuffled stratified outer CV for classification and shuffled outer CV for
regression. Each outer fold has its own inner OOF folds. Masks are generated
without labels. All imputation and categorical encoding are fitted inside the
learner's training fold, after selecting its raw feature mask. Row subsampling
draws only from that training fold. OOF coverage must be exactly once per row.
The cross-fitted null uses each fold's training prevalence/mean.

Certification, quality estimates, residual dependence, and subset selection use
only outer-training OOF predictions. Selected specifications are refitted on
all outer-training rows. Final specifications for **all** compared methods must
be finalized before test predictions are scored. Selected models are serialized;
the test prediction for each method is computed once, and timing uses training
rows with repeated inference calls. Outer folds are paired across methods.

The initial runs use fixed preregistered settings. If a method has a tuning grid,
use another split layer inside outer training: regenerate OOF predictions,
certify, select, and refit using only the tuning-training portion, then evaluate
on its tuning-validation portion. Slicing one global OOF matrix for this purpose
is invalid because base fits can have seen tuning-validation labels.

## Certification

Binary: OOF AUROC lower one-sided stratified percentile bootstrap quantile above
the configured threshold. Regression: skill `1-SSE_model/SSE_crossfitted_null`.
Bootstrap resamples rows jointly across candidate columns and the null predictor;
classification resamples within classes. Multiclass uses macro one-vs-rest AUROC
and its chance threshold 0.5; primary evaluation is log loss.

These are **empirical screens**, with no population-confidence or familywise
guarantee: OOF training folds overlap, B hypotheses are searched, and bootstrap
draw counts cannot support arbitrarily extreme adjusted quantiles. Store all
draws and bounds. IID-only protocol: grouped and temporal tasks require explicit
split/bootstrap implementations and are blocked by the current runner.
If fewer than K pass, mark the method infeasible and retain that result.
No fallback or reduction of K is permitted.

## Objectives

For residual columns, `G=E.T@E/N`. Binary Brier and regression MSE equal
`1.T@G[S,S]@1/K^2`. Multiclass residuals concatenate sample/class coordinates;
the sum-over-classes Brier convention divides by N, not N*C. Internally the
flattened residual matrix is scaled by sqrt(C), so averaging over N*C coordinates
gives that convention. Multiclass shrinkage preserves the sum of outer products
of the separate class residual means; flattening must not erase class-specific bias.
Correlation centers residuals; covariance drops bias if used alone;
G includes bias. Zero-variance residual columns have undefined correlation and
receive conservative pairwise dependence 1 in correlation-based selection.
This value is a selection convention, not an estimate of correlation.
For multiclass the initial correlation selector applies Pearson correlation to
flattened sample/class residuals. Their pooled means are zero for normalized
probabilities, so this is normalized co-error G_ij/sqrt(G_ii*G_jj), not the sum
of class-centered residual correlations. It retains classwise bias. Covariance-only
and shrinkage instead center/preserve bias separately for each class. This
implementation convention must be explicit in multiclass interpretation.

Co-error greedy starts from lowest diagonal error, updates marginal objectives,
and applies improving 1-swaps. Direct squared-loss forward selection without
replacement is mathematically the same construction; do not count it as a new
mechanism. Caruana-style selection permits repeats and induces nonuniform
weights; report this separately from the equal-weight variants.
An exhaustive tiny-pool reference verifies approximation gaps; it is not scalable.

New variant: bias-preserving shrinkage,
`G_a = (1-a)C + a*diag(C) + mu*mu.T`, where C is empirical centered
residual covariance and mu is residual mean. This is a regularized selection
objective; its value is not the actual empirical ensemble loss unless a=0.

## Reporting and resources

Store raw data, split indices, every specification, OOF predictions, bootstrap
draws, selections, models, test predictions, metrics, configuration, code snapshot,
commit/worktree state, lockfile, package versions, and hardware. Every completed
run has a SHA256 artifact manifest. Incomplete runs are retained with failure state.
Training costs include OOF search, screening, selector, and refit. Shared pool
cost is charged in full to each standalone selection method; wall time and CPU
time are recorded. Memory measurements must state their scope.
The current runner retains fitted plans for all methods until scoring. Its
per-phase peak RSS includes shared OOF arrays, imported runtimes, and preceding
models; it is a process high-water measurement, not isolated standalone model
memory. A resource-matched study must use separate worker processes for that
comparison. Search-charged methods conservatively include the common bootstrap
screen even when their selection is unfiltered. Random-subspace diagnostics
piggyback on the pool, while its intrinsic no-search cost is labeled separately.

Dataset-level paired bootstrap averages repeated folds/seeds before sampling
tasks. Synthetic task means are descriptive; do not manufacture independent
datasets from folds or rows. Confidence intervals across a few deliberately
chosen synthetic regimes are not real-world generalization guarantees.
Regimes reuse generation seeds for controlled comparisons and can share raw
features/noise. Resampling these chosen regime labels does not resolve dependence
between them. The reported exploratory intervals are descriptive and must not
be interpreted as calibrated confidence coverage or significance tests.

## Confirmation protection

Synthetic and sklearn built-in datasets are development only. Benchmark policy
and current sources are recorded separately. Confirmation evaluation requires a
locked dataset/split/method manifest and an explicit milestone label; every
access is appended to a ledger before computation. No confirmation run yet.
Results from failed/partial runs are never silently deleted or pooled with
completed runs. Aggregate scripts require explicit run identifiers.
