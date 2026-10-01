# Experiment 006 extension: UCI Musk v2 (registered before its scores)

Registered 2026-10-01 after credit default completed and while HIGGS was running,
to address the user's larger-feature request directly. Selection criterion is
166 physical distance features and available molecule identities, not a measured
baseline score. No Musk score has been inspected; same52 settings and9 baselines
as the initial five-task protocol. Full6598 rows,102 molecules,39 positive molecules;
CC-BY4 UCI75 source SHA/config pinned. No identifiers become predictors. Source
features are unmodified physical descriptors. Missingness/duplicates recorded.
Bioresponse OpenML4134 was also considered; its supplied descriptor matrix is
already normalized with unspecified transformation provenance. It is deferred
rather than adding that source-preprocessing confound to this diagnostic extension.

Outer and inner folds hold out ENTIRE molecules, using their provided IDs, and
screen bootstraps resample complete molecules. Independent molecules are the
assumption; chemical scaffold dependence is not controlled. Primary diagnostic
is molecule AUROC using a fixed max over conformations AFTER averaging the selected
row predictors. This corresponds to the source's ANY-conformation prediction
motivation; max is an uncalibrated bag score. Row AUROC and all row losses also
remain recorded. Training/OOF quality and co-error selection still use inherited
row labels and row Brier, with cluster-bootstrap resampling: they do not optimize
molecule-level max loss. This deliberate protocol mismatch tests the initial
method's behavior; it is not an optimized multiple-instance learner. All baselines
use the same inherited labels, molecule holdouts and max score. The exact Gram
loss identity remains about row Brier, not the nonlinear group-max metric.

With only102 molecules, fold variability may be substantial. No population
confidence intervals or selection based on best test configuration. This extension
is development, registered later than the initial panel, and reported explicitly.
