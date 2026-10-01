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
