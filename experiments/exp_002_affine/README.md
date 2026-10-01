# exp_002 — aggregate attenuation versus subset selection

**Motivation from exp_001:** Additive/mixed regression remains weak, certification
is often infeasible, and the oracle conditional-mean calculation predicts attenuation.

**Hypothesis:** A scalar affine correction improves distributed-signal regression;
selecting with affine parameters profiled out improves beyond calibrating an
unchanged subset. Both hypotheses may fail. No algorithm novelty claim.

**Named variant:** Affine-profiled co-error (APCE). For the equal-weight raw aggregate
P_S, fit a>=0 and intercept b on training OOF data. The profiled objective is
Var(y)-max(Cov(y,P_S),0)^2/Var(P_S), with constant prediction when Var(P_S)=0.
Greedy and 1-swap updates use centered prediction moments. Relative weights remain
equal; aggregate scaling means this is not a primary simplex/equal-output-weight method.

**Controls:** Raw versus affine-calibrated random, top-quality, and co-error subsets;
APCE with the same B/K/masks/splits. No hard certification in the primary comparison,
to expose attenuation without filtering infeasibility. Include additive, mixed,
redundant, sparse, dominant, and null regression; static strong baselines retained.

**Fixed:** Same N,p,B,K,depth,leaf size,seeds and fold counts as exp_001. Learned
affine parameters use only training OOF aggregate predictions. Nested tuning must
fit these parameters inside the tuning-training portion. No calibration hyperparameter
is tuned in this run. Test predictions remain single final evaluations.

**Falsification:** No gain on additive/mixed signal; no advantage over calibrating
simple top/random selection; null degradation or instability. Preserve all outcomes.
