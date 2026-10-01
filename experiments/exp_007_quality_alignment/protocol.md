# Experiment 007: loss-aligned quality control (registered before its results)

Motivation from the completed exp_006 classification panel: individual quality
uses AUROC, whereas co-error optimizes ensemble Brier. Their gap may partly reflect
individual quality-metric alignment rather than exploiting cross-errors. This
follow-up compares a new quality-only control, top individual squared loss,
against the unchanged AUROC/skill top-quality and co-error controls. It is a
mechanism ablation with established objectives, not a novel algorithm claim.

Keep the entire six-task exp_006 panel, immutable raw data, source groups, both
split levels, candidate masks/specifications, OOF predictions and B6000,K64.
Do not regenerate or re-screen candidates. Unscreened equal weights, no tuning,
no swaps, no count-based selection, no test-based filtering. Select the K lowest
OOF G_ii, stable candidate-ID ties, refit their specifications on the original full
outer training rows, and score once on the original outer test rows after the
selection is saved. Report Brier and AUROC jointly; Musk retains molecule-max
AUROC but the quality-only objective remains row Brier. Regression individual
skill is monotone in squared loss, so it is an unchanged-selection control.

The experiment reuses development test folds after earlier results motivated its
hypothesis. It is explicitly exploratory, not a new confirmation benchmark or
independent significance test. Retain all paired fold outcomes; do not choose
best outer-test settings. Parent manifests and byte hashes link every reused
input to the published exp_006 bundles. Store new definitions, selected IDs,
refitted models, test predictions, environment, source snapshot and resources.
CPU costs charge the inherited entire search/screen cost plus new selection/refit;
actual incremental follow-up time is also recorded. No matched-budget claim.

Falsifier: if co-error's gain over AUROC-quality disappears when individual
squared loss defines quality, that weakens a complementarity explanation for
that gap. A remaining gap is consistent with joint-loss selection benefit, but
does not distinguish effective weighting, bias cancellation or statistical
independence. The previous duplicate intervention remains relevant.
