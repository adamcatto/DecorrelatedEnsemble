# Proposed independent selection/certification diagnostic (not run)

The current OOF bootstrap is an empirical screen. A Bonferroni adjustment alone
does not fix overlapping-fit uncertainty. An alternative diagnostic must separate
fit data from certification data and distinguish fitted functions from refitted
specifications. This document proposes a test; no implementation or coverage
experiment has been completed yet.

## A conservative fixed-library binary AUROC certificate

Fit B functions using only a fit partition T. Conditional on T and generation
randomness, the functions are fixed. On an independent, IID class-conditional
selection sample with n+ positives and n− negatives, empirical AUROC averages
the pair kernel h(s+,s−)=1{s+>s−}+.5*1{s+=s−}. Its expectation is the fixed
function's population AUROC. Replacing one positive changes the average by at
most 1/n+; replacing one negative by at most 1/n−.

The bounded-differences inequality and a union bound therefore give simultaneous
lower bounds

    L_b = max(0, AUC_hat_b − sqrt(.5*(1/n+ + 1/n−)*log(B/alpha))).

Conditional on T, all B lower bounds cover their respective fixed-function
population AUROCs with probability at least 1−alpha. Candidate dependence does
not invalidate the union bound. Library generation or modification using this
selection sample does invalidate the fixed-library premise unless separately
accounted for. The result does not apply to pooled OOF scores.

This is an application of [McDiarmid's bounded differences method](https://doi.org/10.1017/CBO9781107359949.008),
not a novel concentration theorem. The derivation above specifies our use of it.
For balanced selection data of size n_sel, the radius is
sqrt(2*log(B/alpha)/n_sel). To certify an observed AUC a above threshold t, this
bound requires n_sel > 2*log(B/alpha)/(a−t)^2. It is deliberately conservative;
infeasibility for weak learners and large libraries is a meaningful outcome.

## What a subsequent experiment must distinguish

1. Compare empirical OOF screening with independent selection screening; give
   all methods the same available fit rows in the diagnostic comparison.
2. Evaluate retained fits and full-outer-training refits of the selected
   specifications as separate variants. The bound certifies only the retained
   functions; it supplies no automatic guarantee for the refits.
3. Compare unadjusted independent bootstrap screening with the simultaneous
   bound, preserving infeasible K. Null candidate survival and screening power
   are different measurements.
4. Score all finalized variants on the outer test once. Use synthetic null and
   signal controls and repeated independent generation seeds, not rows as tasks.
5. Record available training labels, selection-set size, B/K, baseline data
   access, optimizer/search compute, and refit shift. Recheck current statistical
   literature before attributing novelty to any refined certification scheme.

No analogous unbounded-Gaussian regression certificate follows from this bounded
AUROC argument. Regression needs explicit tail assumptions, truncation, or a
different procedure; do not silently reuse the binary bound.
