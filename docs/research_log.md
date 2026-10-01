# Research log

## 2026-09-30 — research initialization

Repository was empty. Wrote questions and protocol before results. Main threats:
selection overfitting, OOF-bootstrap interpretation, multiplicity, loss mismatch,
candidate search compute, and inability of sparse shallow learners to capture
interactions. Core ingredients have substantial prior art; novelty unresolved.
Start with homogeneous trees and equal weights. Implement bias-preserving
covariance shrinkage as a falsifiable extension, not a novelty claim.

## 2026-10-01 — infrastructure and smoke_001

46 tests pass. Added six baseline families across binary, multiclass, and regression,
then found and fixed positional-versus-label column indexing in masked tables.
Nullable numerical and pandas categorical dtypes have an explicit regression test.
Smoke run: two tasks, two outer folds, 12 candidates, K=4; complete with infeasible
screened methods preserved. This is engineering evidence, not performance evidence.
Published the complete small smoke artifact bundle including models.

Preserved and consolidated files from a concurrent session. Verified that the old
session owned a distinct server process; stopped it and archived the thread.
No active second writer remains. Literature review found direct prior art for
quadratic ensemble pruning; added it to the novelty audit.

exp_001 is fixed before results: B=100, K=8, feature fraction=.1, depth=3,
n=600, p=40, two generation seeds, three outer and three OOF folds, nine mechanisms
for each of binary and regression. This small development design cannot establish
task-population superiority or matched-resource superiority. Strong baselines are
static configurations; tuned modern-method confirmation remains unrun.

## During exp_001, before inspecting scored results — attenuation calculation

For independent additive signal and individually population-optimal conditional
mean learners, averaging subspaces produces coefficients equal to feature inclusion
frequencies. Equal-size subspaces imply residual bias even with full union coverage.
Derived and added the precise population formula and Jensen lower bound to the
paper; a Hadamard-design analytic test verifies it. This is not a novel theorem
claim or a universal lower bound for feature-sparse functions. It motivates a
later aggregate-scale diagnostic, contingent on the first completed experiment.

## exp_001 completed — decision after inspecting all results

108 outer folds completed; engineering identity error below 1e-12. The generated
factual note and paired effects are in `results/summaries/exp_001_mechanisms_v1/`.
Co-error is modestly better than top quality but broadly worse than strong forests
and boosting; it does not consistently beat absolute correlation. Shrinkage failed
to help. Top-quality certification usually selects the same subset where feasible,
while many regression settings become infeasible. Null binary candidates pass the
unadjusted OOF bootstrap screen in every null split. Preserve these negative results.

Next experiment: isolate aggregate attenuation. Add affine calibration of the
equal-weight aggregate and a selector that profiles out the aggregate's affine
parameters. This adds two scalar degrees of freedom and is ordinary stacking-like
calibration; no novelty claim. Compare raw/calibrated random, top, and co-error
subsets plus profiled selection at fixed B/K/masks/folds. Falsification: no consistent
gain on distributed additive signal, or similar/worse gains versus calibration of
simple selection controls. Null/dominant regimes expose overfitting and unnecessary
calibration. Keep primary equal-weight results as the original reference.

## exp_002 completed — calibration helps; profiled selection fails

36 folds / 468 method evaluations completed. The extended audit reconstructs all
test metrics and 144 OOF affine fits, verifies raw/calibrated controls have the
same selected IDs, and checks profiled loss identities. Co-error affine correction
improves task-mean loss in five of six regimes (the null worsens); gains are modest
for additive/mixed signal and larger for sparse/redundant signal. Attenuation is
therefore a partial explanation, not the only failure mechanism.

APCE loses to calibrated co-error on five of six task means, while its calibrated
training OOF loss is no worse in every paired split. This pattern is consistent
with selection optimism or the smaller-fit OOF/refit distribution change. It
does not establish either cause uniquely. APCE remains a preserved negative result.
The strong baseline gap remains, and no tuned/matched-budget conclusion is justified.

Next: exp_003 varies B=30/100/300 at fixed K=8 on additive/null binary/regression,
with exact candidate/OOF prefixes and a shrinkage control. Freeze the configs and
analysis questions before launching. This reuses development seeds; it is not a
new confirmation study. Do not add algorithm complexity solely to chase scores.

Literature update: Breiman (1996) explicitly uses cross-validation predictions and
residual-product quadratic loss for stacking. Portfolio error-covariance weighting
also has direct precedent. Chen et al.'s nested honest least-squares shrinkage
theory does not apply automatically to our non-nested data-adaptive trees.

## exp_003 completed — pool scale helps conditionally and increases optimism

72 outer evaluations completed across B=30/100/300 with K=8. All 48 adjacent-pool
fold comparisons have exactly identical candidate, OOF, and bootstrap prefixes;
the forest reference has unchanged predictions. Generated summaries preserve
coverage and all paired split/seed effects.

Unfiltered co-error gains binary additive AUROC with a larger pool, but additive
regression is non-monotone with a small endpoint benefit. APCE additive OOF loss
improves strongly as B grows while its outer endpoint performance worsens. The
same flexible-search failure appears on null regression. This pattern supports
investigating selection optimism, but it does not distinguish smaller OOF fits
from refit shift or selection reuse. Alpha=.5 shrinkage provides no consistent
outer benefit. Increasing B is not a monotone improvement mechanism at fixed K.

Binary null screening survivors grow with the library; regression null screening
retains zero candidates at these settings. These distinct failure/power patterns
must not be pooled into a blanket certification conclusion. All results use two
generation seeds and reused development tasks; no confirmation access occurred.

Next discriminating design: independently fitted library with a disjoint selection
sample, comparing retained fits and refitted specifications on the same outer
test. `docs/independent_certification_design.md` derives a conservative simultaneous
binary AUROC bound for fixed functions and records what it cannot certify. Also
inspect exact tiny-pool optimization gaps before attributing all failure to statistics.

## exp_004 registered — user-requested real development panel

Before scored results, fixed breast cancer/wine/diabetes at feature fractions .1/.5,
B=100 K=8, existing selectors/static baselines and an additional ridge/logistic
control. Identical splits and unchanged baselines will be audited between widths.
Diabetes loader inspection found default scaling across the complete dataset;
fixed to raw `scaled=False` and tested before the first builtin experiment.
Metadata now records source/attribution, package copy/version, class convention,
loader arguments and content hashes. This resolves the loader issue prospectively;
no previous scored builtin run exists. The larger independent-selection study
remains pending. No confirmation access or dataset exclusion occurs here.

Wine smoke: two complete outer folds, 50 test-prediction artifacts reconstructed,
multiclass Brier/co-error discrepancy 1.11e-16. The 60 infrastructure tests pass.
The statistical reporter now suppresses a degenerate task bootstrap interval
when there is only one paired dataset; overlapping CV folds cannot repair that
missing scientific replication. Real-panel reporting preserves paired feasible
support, width effects and all secondary/resource metrics without pooled ranks.

## exp_004 completed — real data reveal width and duplicate-column confounds

36 outer evaluations, 936 test-prediction artifacts and 24 affine fits audited.
All 18 paired-width partitions/nulls and 144 unchanged baseline prediction pairs
match exactly. Applicable screened methods are feasible; top-quality screening
returns identical IDs/weights to unscreened top quality in all 36 comparisons.
Co-error improves primary means over top quality at both widths on all tasks,
but the fixed RF and logistic/ridge controls are better in each case.
Wider subspaces help wine/diabetes and hurt cancer AUROC: no universal width remedy.
Large candidate search costs are explicit; smaller retained models do not imply
faster/cheaper training. Full effects, secondary metrics and resources are generated
in `results/summaries/exp_004_real_v1/`. No confirmation data were accessed.

Exploratory OOF-equivalence audit: narrow diabetes has exactly ten distinct
prediction columns among 100 specifications in every fold. Top quality's eight
members contain only one or two distinct vectors; co-error contains two to four.
This is effective weighting/duplicate avoidance, not proof of eight independent
specialists. Wine's wide top quality gives zero probability to true labels in
three splits; AUROC screening and log-loss evaluation are different objectives.
These findings require mechanism controls, not an expanded superiority claim.

Next test: OOF-deduplicated random/top/co-error at fixed K=8 on the same raw pools,
with explicit infeasibility if fewer than eight observed prediction classes remain.
Keep original variants and rerun their predictions as reproducibility controls.
OOF equality is an observed equivalence relation, not global functional identity;
deduplication uses training OOF only and supplies no population diversity guarantee.
This complements the still-pending independent-selection/refit diagnostic.

## exp_005 completed — uniqueness is not a remedy; weighting matters

18 folds, 162 predictions, 54 unique subsets audited. All 18 candidate/OOF/bootstrap
arrays and 108 original-method/reference predictions match exp_004_f10 exactly.
No infeasible variants. The 65 tests include exact/near equality, signed zeros,
eligible-only deduplication, hash-collision resolution and fixed-K behavior.

Diabetes co-error RMSE worsens 60.078 -> 65.978; unique top quality is 65.945,
so the original quality-control gap vanishes by degrading co-error, not improving
both methods. Wine's unique primary losses nearly coincide (.237), while co-error
retains a modest Brier advantage. Cancer's top/co-error effects are unchanged.
Preserve these negative results and the full split/seed effects. This follows a
registered mechanism intervention; its explanatory interpretation is exploratory.

Equal specification weights are discrete simplex weights over repeated observed
OOF classes, bounded by pool multiplicities. Forcing K distinct classes removes
that useful concentration and forces weak patterns into the average. This is
ordinary implicit weighting, not a new independence mechanism. Exact OOF equality
does not prove global equality and may not survive full-training refit.

Next high-information test: compare explicit regularized simplex weights on the
same canonical OOF patterns against multiplicity-constrained discrete weights,
and independently selected retained fits versus refits. Separate capacity/search
cost and representation attenuation; tune only within another training split.
Do not add more benchmark tasks merely to search for favorable wins. Strong
linear synthetic/oracle controls, proper certification, tuned modern baselines,
and locked resource-matched confirmation remain open work.

### 2026-10-01 — exp_006 larger-data request, before scores

User requested larger/harder mainly classification tasks, more parameter sweeps,
and thousands of weak estimators. Registered five public development datasets;
source hashes/row caps/feature definitions and every considered exclusion documented.
Superconductivity's many repeated vectors motivate exact-feature group separation in
both CV levels and cluster bootstrap, added before scoring. MiniBooNE raw source is
130,064 rows vs web catalog130,065; header agreement verified, sentinels retained.
Weighted-multiplicity screening, threaded OOF, balanced nested candidate grids,
config-defined pool restrictions and matrix-free correlation/Caruana acceleration
match literal/dense references in unit tests. Prior variants retain default behavior.
AUC0.70-0.80 is an intended difficulty region, not an observed-score inclusion gate.

### 2026-10-01 — higher-dimensional extension registration

Musk v2 adds166 features with6598 conformations from102 molecules. Source molecule
IDs are excluded from predictors and supplied to both CV levels/bootstrap. Primary
molecule-max AUROC is a fixed source-motivated diagnostic, distinct from the row
Brier/row-quality selection objective. Added after credit default completion, before
Musk scores. Bioresponse was considered but deferred for unspecified preexisting
normalization provenance, preserving that exclusion and its original metadata.
First main five-task jobs continue with unchanged prediction algorithms; source-group
and group-metric API defaults are inert for those tasks. New APIs must pass tests
before the extension run. Final analysis will expose all52 sweep settings, not
select a best test-score setting and report it as independently tuned performance.

### 2026-10-01 — runtime scope correction before regression/Musk phases

NumPy2.5.3 on this host uses Apple Accelerate; threadpoolctl exposes no Accelerate
pool, so the runner's requested BLAS limit1 cannot establish actual concurrency.
Recorded build/backend audit and corrected protocol wording; no numerical-thread
environment changes or score-dependent reruns. Process CPU accounting already
includes all threads. This limits wall-time/thread-count claims; no matched resource
claim has been made. New environment artifacts include numerical build/pool metadata.

### 2026-10-01 — larger classification checkpoint and storage failure preserved

Credit default, HIGGS and MiniBooNE complete: 549 audited test-prediction records,
36,000 OOF base fits per task. Fixed B6000,K64 co-error AUROC means .779/.764/.970,
versus RF .767/.785/.976 and logistic .728/.689/.932. Co-error improves over top
quality on all three task means but trails a boosting reference on each. MiniBooNE
is retained despite easier baseline performance than the requested difficulty.
At K64, B6000 slightly worsens co-error versus B3000 on credit/HIGGS. Preserve the
negative scaling result; no best-test setting or matched-resource claim.

Musk smoke001 completed training but failed audit because object-typed molecule
IDs could not be read with safe NumPy loading. Preserve the original failed-audit
bundle; canonicalize nonmissing string IDs to Unicode at cache load. Smoke002
then passes group/metric/Gram audits (14 predictions). Added tests verify safe
round trips, whole-molecule holdouts and fixed max aggregation. Both smokes are
engineering checks; no primary configuration was tuned from their scores.

Lossless column-major OOF exports preserve every value/dtype and both manifests;
48MiB ordered archive parts retain all prediction/decision artifacts, omitting
only fitted models. Original complete runs/models remain local. Paper partial
results are generated from verified artifacts; remaining tasks continue unchanged.

### 2026-10-01 — regression checkpoint and analytical metric controls

Both original regressions complete/audited: 366 test predictions. Co-error K64,
B6000 RMSE is13.807 on superconductivity vs14.038 quality and10.206 RF; California
.594 vs.633 quality and.508 RF. Both show modest task-mean improvement across
B1000/3000/6000; preserve the contrasting classification negative scaling. All
original five tasks and915 predictions are now complete; Musk remains pending.

The94 tests include new top-squared diagonal selection, source-group count control,
storage-order-independent exact array hashes, and two analytic cautions. Binary
uncentered co-error is nonnegative; residual correlation1 can coexist with improved
Brier through bias cancellation. A grouped two-fold null predictor gives pooled
OOF AUROC .25 while each held-out fold AUROC is .5: different fitted calibrations
can distort pooled ranking. This is a constructed counterexample, not an observed
explanation of our gains. Preserve the initial pooled-AUC protocol; record the
limitation and investigate independent fixed-function/same-fit ranking next.
The independently registered loss-aligned control addresses a separate quality
criterion confound, not every OOF estimation issue. No confirmation access.

### 2026-10-01 — completed six-task sweep and molecule counterexample

Musk completes the six-task panel:1,098 predictions,216,000 OOF base fits,
52 selection settings and9 baselines per dataset. Three Musk folds audit with
maximum Gram identity error3.83e-15. Co-error molecule AUC.792826 loses to
quality.817151,RF.846166 and Caruana.806586, despite row Brier.123342 versus
quality.130993; Caruana row Brier.115159 is stronger still. Preserve the primary
negative result: row-loss selection does not optimize nonlinear molecule ranking.
The separately preregistered count-only molecule control averagesAUC.453927
(folds.497835/.455782/.408163). It does not support a simple count-only explanation,
without excluding pooling/cardinality interactions. Only102 molecule units exist.

Full automatically generated reports retain all61 settings, nested B/K and
homogeneous width/depth cells, actual study-sample metadata, partition counts,
per-task paired effects, descriptive reference ranks and resource measurements.
Exploratory bootstrap intervals resample task means (four binary/two regression
tasks), never rows/folds as scientific units; no coverage or significance claim.
Sample counts are separated from full-source catalog counts (sampled HIGGS has
zero missing cells although its source has9). All18 co-error/quality subsets
have64 exact distinct OOF columns; their gaps are not exact-copy weighting here.
Absolute/signed residual-correlation selections are identical onall18 folds.
Credit's RF leaf5 matches the co-error default-RF gain, while larger K worsens
both regression task means past64. No universal scaling or forest-family gain.

All95 tests pass after adding sample/partition metadata checks. The report's
generated Musk LaTeX macro names use alphabetic identifiers; paper compilation
checks the resulting tables. Main configurations remain unchanged. Exp_007 is
next, on the entire immutable panel; all six tasks remain development only.
