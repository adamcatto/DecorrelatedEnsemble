# exp_001_mechanisms_v1: generated factual note

Outer folds: 108. Status counts: {'complete': 2022, 'infeasible': 624, 'not_applicable': 54}.

Screened co-error feasible binary splits: 40; regression splits: 16.
Null binary survivors per 100 candidates: [3, 5, 13, 7, 4, 3].

## Paired task effects (positive = left better)

- coerror vs top_quality, auroc: 0.01148, descriptive interval [-0.00065, 0.02263], 8 tasks / 40 shared splits.
- coerror vs abs_correlation, auroc: -0.00741, descriptive interval [-0.01550, 0.00046], 8 tasks / 40 shared splits.
- coerror vs random_forest, auroc: -0.04707, descriptive interval [-0.07393, -0.02294], 8 tasks / 40 shared splits.
- coerror_no_cert vs top_no_cert, auroc: 0.01072, descriptive interval [0.00191, 0.02013], 9 tasks / 54 shared splits.
- top_quality vs top_no_cert, auroc: -0.00058, descriptive interval [-0.00173, 0.00000], 8 tasks / 40 shared splits.
- shrinkage vs coerror, auroc: -0.00276, descriptive interval [-0.00455, -0.00103], 8 tasks / 40 shared splits.
- coerror vs caruana, auroc: -0.00395, descriptive interval [-0.01297, 0.00353], 8 tasks / 40 shared splits.
- coerror_no_cert vs random_forest, auroc: -0.03853, descriptive interval [-0.06905, -0.00866], 9 tasks / 54 shared splits.
- coerror vs top_quality, rmse: 0.00826, descriptive interval [0.00049, 0.01602], 4 tasks / 16 shared splits.
- coerror vs abs_correlation, rmse: -0.00209, descriptive interval [-0.00635, 0.00112], 4 tasks / 16 shared splits.
- coerror vs random_forest, rmse: -0.07024, descriptive interval [-0.10606, -0.02027], 4 tasks / 16 shared splits.
- coerror_no_cert vs top_no_cert, rmse: 0.01307, descriptive interval [0.00657, 0.01968], 9 tasks / 54 shared splits.
- top_quality vs top_no_cert, rmse: 0.00000, descriptive interval [0.00000, 0.00000], 4 tasks / 16 shared splits.
- shrinkage vs coerror, rmse: -0.00118, descriptive interval [-0.00196, -0.00076], 4 tasks / 16 shared splits.
- coerror vs caruana, rmse: -0.00934, descriptive interval [-0.02309, 0.00215], 4 tasks / 16 shared splits.
- coerror_no_cert vs random_forest, rmse: -0.04304, descriptive interval [-0.06902, -0.01683], 9 tasks / 54 shared splits.

Intervals resample observed synthetic regimes after pairing splits; these are exploratory,
unadjusted intervals without a real-world task-population interpretation. Inspect coverage.csv.
Interpretation and follow-up decisions belong in the research log and experiment README.
