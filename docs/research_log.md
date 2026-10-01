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
