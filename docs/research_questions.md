# Research questions and discriminating tests

1. **Certification**: At fixed candidate pool and K, does thresholding improve over
   unfiltered random selection or top quality? Record infeasible K rather than
   silently lowering K. A percentile OOF bootstrap is a screening heuristic,
   not a familywise statistical certificate. Test label-null survival explicitly.
2. **Diversity**: Conditional on individual quality, do signed error dependence,
   absolute error correlation, prediction correlation, and co-error improve
   outer loss? Use identical candidate specifications, folds, and budgets.
3. **Loss alignment**: Co-error optimizes squared error/Brier, not AUROC or log loss.
   Compare direct AUROC selection to Brier selection; report all metrics.
4. **Candidate search**: Does increasing B at fixed K help on outer data or only on
   the selection OOF matrix? Generate nested candidate prefixes independent of B.
5. **Representation limit**: Test XOR/parity and a dominant-feature regime as
   counterexamples to the distributed-signal hypothesis. Compare shallow and
   unrestricted full-feature forests and boosting.
6. **Estimation risk**: Does covariance shrinkage reduce selection optimism as
   B/N grows? Keep residual means (bias) in the second moment. Falsified if
   shrinkage consistently worsens outer loss relative to unregularized co-error.
7. **Resource fairness**: Separate quality at fixed K/depth, final bytes/leaves,
   inference latency, and full search/refit cost. No resource superiority claim
   until genuinely matched-budget comparisons exist.

The first experiments are development evidence. Synthetic regime summaries are
not estimates of performance on a population of real-world datasets.

