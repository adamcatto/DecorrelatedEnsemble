# Experiment 005: observed prediction duplicates (before results)

Motivation: exp_004's narrow diabetes pool has ten unique OOF columns among
100 specifications; eight top-quality members contain only one or two distinct
columns, while co-error has two to four. Equal specification weights therefore
implement nonuniform weights over observed prediction classes. This may explain
some quality-versus-co-error differences without eight independent specialists.

Keep exp_004_f10 data, masks, candidate definitions, B=100/K=8, depth/leaf size,
seeds, all outer/inner folds and OOF predictions exactly unchanged. Use the three
real development tasks. Drop certification as a confound: every variant is unscreened.
Compare original random/top/co-error and their OOF-unique counterparts, plus
Caruana with eight steps and unchanged RF/linear references.

OOF-unique: after eligibility, group candidates by **exact equality of training
OOF predictions**, including all class coordinates; retain the lowest eligible
ID per group. This uses no outer test data. Floating signed zeros are equal;
near duplicates remain distinct. A streaming hash indexes groups but explicit
equality resolves collisions. Select eight different representatives using the
unchanged selector and equal weights. Store original/retained eligible IDs.
If fewer than eight classes remain, mark infeasible; no K fallback. Observed
OOF equality does not establish global function identity or equality after refit.
The representative's full-training refit can differ from another group's member,
so representative/refit change remains an alternative explanation.

Primary metrics: cancer AUROC, wine log loss, diabetes RMSE, with Brier and
normalized regression loss as mechanism metrics. Six paired evaluations per task;
no IID fold confidence intervals or pooled ranks. Generate task-specific tables
and every paired split/seed effect; retain negative results. Audit exact candidate,
data, split, OOF, bootstrap and unchanged-control test prediction equality to
exp_004_f10 before interpretation. All test predictions are computed once after
plans are finalized. Charge unique variants for OOF search even for random
selection, because deduplication needs the OOF matrix.

Falsification: if top quality improves little after deduplication and its gap
to co-error remains, duplicate concentration alone is insufficient. If the gap
shrinks or reverses, distinguish implicit weighting from a claim of distinct
independent learners. Deduplication can hurt by forcing weak distinct predictors;
it is a mechanism intervention, not an assumed algorithmic improvement. Cancer's
almost-unique pool serves as a low-duplicate comparison. Do not claim causality
beyond this intervention or infer statistically independent errors from uniqueness.

This is a follow-up on reused development data. No confirmation access, tuning,
new datasets, or changes to previous results. The independent-selection/refit
diagnostic and tuned/resource-matched confirmation suite remain pending.
