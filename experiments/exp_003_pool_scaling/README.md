# exp_003 — candidate search versus generalization at fixed K

## Decision before results

Exp_002 found that APCE improved training OOF affine loss in all 36 splits but
lost to calibration of the unchanged co-error subset on five of six task means.
Optimization failure is therefore not the immediate explanation on these splits.
Selection overfitting and the OOF-to-refit distribution change remain competing
explanations. We test the candidate-pool scaling prediction before adding complexity.

## Hypotheses and falsification

- H3.1: Increasing B improves outer performance at fixed K. Flat or worse outer
  loss despite lower training OOF loss falsifies a monotone scaling benefit here.
- H3.2: Flexible profiled selection exploits estimation noise more as B grows.
  Improved outer results with stable optimism would oppose this explanation.
- H3.3: Bias-preserving shrinkage helps at larger B despite failing at B=100 in
  exp_001. No outer improvement over unregularized co-error falsifies the fixed
  alpha=.5 variant at these settings.
- H3.4: Unadjusted null-screen survivors increase with B. This is a diagnostic
  of library multiplicity, not an estimator of familywise coverage from six splits.

## Controls and fixed settings

B in {30,100,300}; K=8 in every method. Candidate definitions at smaller B are
exact prefixes of larger pools, with the same OOF splits and bootstrap resamples.
N=600, p=40, feature fraction=.1, depth=3, leaf size=5; seeds 11/29, three outer
and three inner folds. Additive/null binary and regression: 24 outer folds per B.
All baselines retain the same settings and seed. One sequential run at a time.
Masks, OOF arrays, rows, and null predictions will be verified identical on prefixes.

Compare random subspace, top quality, co-error, shrinkage, screened top/co-error,
and (regression only) fixed-subset affine correction versus APCE. Screening at
fixed K may be infeasible and will not trigger a fallback. RF256 is an unchanged
reference. Training/search CPU and wall time are reported; budgets are unmatched.
Increasing B must not be presented as increasing final capacity K.

## Analysis fixed before results

Plot per-task outer AUROC / normalized MSE versus B, raw OOF versus test squared
loss, APCE versus affine co-error OOF/test paired differences, null screen count
and rate, and training cost. No p-values or significance declarations. Task-level
effects and full split distributions are descriptive. The four intentionally
chosen mechanisms do not represent a real-world dataset population. These are
reused development seeds/splits, not independent confirmation.

OOF fits use smaller training sets than refits, so the raw OOF/test loss gap
combines selection optimism, training-size effects, and sampling fluctuation.
This experiment can suggest an explanation, not uniquely identify its cause.
An independent selection holdout with retained (unrefitted) models would be a
subsequent test if the scaling pattern supports it.
