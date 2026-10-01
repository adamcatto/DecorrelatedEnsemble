# exp_001 — fixed homogeneous-tree mechanism study

**Hypothesis:** Loss-aligned co-error selection improves over quality or absolute
correlation; sparse shallow learners can fail on interaction structure.

**Changed:** Selection/certification method only within the shared candidate pool.
**Fixed:** N=600, p=40, B=100, K=8, depth=3, feature fraction=.1, leaf size=5,
seeds 11/29, three outer and three OOF folds. Nine mechanisms, binary and regression.
Static RF/ExtraTrees (shallow and unrestricted), random patches, XGBoost,
LightGBM, CatBoost; no tuned/resource-matched superiority comparison.

**Run:** `exp_001_mechanisms_v1`, complete, 108 outer folds, 2700 status records.
Raw artifacts are preserved locally; the Git evidence bundle includes every
decision/prediction artifact and explicitly lists omitted locally retained models.
All aggregate numbers and tables are generated from frozen artifacts.

**Result:** Co-error modestly improves over top quality, generally trails strong
baselines, and does not consistently beat absolute residual correlation. The
shrinkage extension did not help. Null learners pass the bootstrap screen in all
six binary null training splits. Screening severely limits regression feasibility
and adds essentially no benefit to top quality on paired feasible splits.

**Interpretation:** The initial mechanism is not a competitive general method at
these settings. Hypothesis remains conditional on representation, calibration,
selection uncertainty, and resources. A positive average effect can be driven by
one regime (dominant-feature Brier), so inspect task-level effects and coverage.

**Follow-up:** Test aggregate attenuation with fixed-subset affine calibration and
profiled affine selection; compare random/top-quality/co-error controls. Separately
test a genuinely independent-holdout simultaneous AUROC bound and candidate-pool
scaling. Keep all previous variants and negative findings.
