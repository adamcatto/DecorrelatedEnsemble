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
