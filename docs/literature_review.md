# Literature audit (sources checked 2026-10-01)

## Core prior art

| Source | Existing idea | Implication for this project |
|---|---|---|
| [Breiman, Bagging Predictors (1996)](https://www.stat.berkeley.edu/~breiman/papers.html) | Aggregation of unstable predictors fitted to perturbed samples | Baseline, not a contribution |
| [Ho, Random Subspace Method (1998)](https://doi.org/10.1109/34.709601) | Fixed randomly sampled feature subspaces for decision forests | Sparse fixed masks are established |
| [Louppe & Geurts, Random Patches (2012)](https://orbi.uliege.be/bitstream/2268/130099/1/glouppe12.pdf) | Joint feature and row subsampling | Row subsampling requires a random-patches baseline |
| [Tian & Feng, RaSE (2021)](https://jmlr.org/papers/v22/20-600.html) | Large random subspace search, selected weak learners, iterative signal targeting | Search over sparse subspaces and adaptive generation have direct precedent |
| [Caruana et al., Ensemble Selection (2004)](https://doi.org/10.1145/1015330.1015432) | Forward validation-loss optimization over model libraries | Direct ensemble selection is a required control |
| [Zhang, Burer & Street, Ensemble Pruning Via Semi-definite Programming (2006)](https://jmlr.org/papers/volume7/zhang06a/zhang06a.pdf) | Accuracy/diversity subset pruning as a cardinality-constrained binary quadratic program, with SDP relaxation | The quadratic cardinality formulation itself is not new; their matrix is a classifier accuracy/diversity surrogate, while ours uses probabilistic/regression squared residual moments |
| [Krogh & Vedelsby (NIPS 1994 proceedings)](https://proceedings.neurips.cc/paper_files/paper/1994/file/b8c37e33defde51cf91e1e03e51657da-Paper.pdf) | Squared-error ambiguity decomposition | The loss decomposition is an established mathematical motivation |
| [Brown & Kuncheva (2010)](https://lucykuncheva.co.uk/papers/gblkMCS10.pdf) | Majority-vote diversity can contribute both positively and negatively to error | Diversity is not synonymous with useful complementarity |
| [Liu & Yao (1999)](https://doi.org/10.1016/S0893-6080(99)00073-8) | Joint negative-correlation training of neural network ensembles | Distinguish joint training from post-hoc selection of independently generated models |
| [Reeve & Brown (2018)](https://arxiv.org/abs/1803.00314) | Diversity penalty related to effective degrees of freedom and regularization | Capacity/regularization is an alternative explanation |
| [Cannings & Samworth (2017)](https://arxiv.org/abs/1504.04595) | Validation-selected random projection ensembles with theory | Random search, selection, and aggregation are not new mechanisms |

These are source-grounded connections, not an exhaustive novelty search. Read full
texts and compare assumptions, selection matrices, certification, and resource
budgets before proposing a submission contribution. The evidence presently
supports a research study of mechanisms and estimation risk, not a novel-method claim.

Cross-validation uncertainty sources: [Bengio & Grandvalet (2004)](https://jmlr.org/papers/v5/grandvalet04a.html)
establish the absence of a universally unbiased variance estimator for K-fold CV;
[Bates, Hastie & Tibshirani](https://arxiv.org/abs/2104.00673) distinguish CV estimands
and study poor coverage of naive intervals. These support documenting OOF-bootstrap
limitations; they do not themselves supply a valid certification procedure for
this adaptive candidate library.

## Remaining search

Stacking (Wolpert 1992), portfolio-based ensemble pruning/weighting, residual
covariance shrinkage, conditional error dependence, OOF uncertainty/multiplicity,
selective inference, feature bagging, and modern ensemble selection variants.
Do not claim bias-preserving shrinkage is new before this review is complete.
