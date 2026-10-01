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

## Auxiliary bag-size control (registered before the main Musk run)

2026-10-01, after the small engineering smokes but before the main Musk run:
max pooling may reflect bag cardinality as well as conformation signal. A separate
auxiliary diagnostic will reuse the immutable main outer partitions, fit logistic
regression (C=1, training-only standardization) on log(1+number of conformations)
with one training observation per molecule, then evaluate molecule AUROC once.
No distance features or identifier values enter this control. Record all group
sizes, predictions and fitted coefficients. This is an explanation control,
not another tuned candidate or a replacement for the predesignated anchor.
A strong count-only score would weaken claims about chemical feature signal;
a weak score would not rule out every pooling/cardinality interaction. It adds
structural bag information that max pooling also uses, so it is not a matched
row-feature learner. No main settings, scores or split definitions are changed.

The null thought experiment is exact: if conformation scores were independent
Uniform(0,1), a bag of size m has max-score CDF t^m. Larger bags then receive
stochastically larger scores without any feature skill. The independence/uniform
assumptions are illustrative, not asserted for fitted tree scores. The trained
count control tests task-specific predictive structure rather than treating that
illustration as empirical proof.
