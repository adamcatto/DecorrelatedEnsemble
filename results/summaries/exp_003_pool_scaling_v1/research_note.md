# exp_003 pool scaling: generated factual note

## Endpoint effects (positive = B300 better than B30)

            dataset                 method                  metric  advantage_B300_vs_B30  paired_splits
    additive_binary        random_subspace                   auroc               0.021406              6
        null_binary        random_subspace                   auroc               0.041575              6
    additive_binary            top_no_cert                   auroc               0.003663              6
        null_binary            top_no_cert                   auroc              -0.006291              6
    additive_binary        coerror_no_cert                   auroc               0.019281              6
        null_binary        coerror_no_cert                   auroc               0.009779              6
    additive_binary      shrinkage_no_cert                   auroc               0.015178              6
        null_binary      shrinkage_no_cert                   auroc              -0.001991              6
    additive_binary          random_forest                   auroc               0.000000              6
        null_binary          random_forest                   auroc               0.000000              6
additive_regression        random_subspace normalized_squared_loss              -0.015551              6
    null_regression        random_subspace normalized_squared_loss               0.007782              6
additive_regression            top_no_cert normalized_squared_loss              -0.004411              6
    null_regression            top_no_cert normalized_squared_loss               0.000223              6
additive_regression        coerror_no_cert normalized_squared_loss               0.005743              6
    null_regression        coerror_no_cert normalized_squared_loss               0.001601              6
additive_regression      shrinkage_no_cert normalized_squared_loss               0.010031              6
    null_regression      shrinkage_no_cert normalized_squared_loss              -0.000152              6
additive_regression coerror_no_cert_affine normalized_squared_loss              -0.013282              6
    null_regression coerror_no_cert_affine normalized_squared_loss              -0.022594              6
additive_regression        affine_profiled normalized_squared_loss              -0.032123              6
    null_regression        affine_profiled normalized_squared_loss              -0.031379              6
additive_regression          random_forest normalized_squared_loss               0.000000              6
    null_regression          random_forest normalized_squared_loss               0.000000              6

## Generation-seed endpoint effects

            dataset  seed                 method                  metric  advantage_B300_vs_B30  paired_folds
    additive_binary    11        random_subspace                   auroc               0.001807             3
    additive_binary    29        random_subspace                   auroc               0.041004             3
        null_binary    11        random_subspace                   auroc               0.056451             3
        null_binary    29        random_subspace                   auroc               0.026700             3
    additive_binary    11            top_no_cert                   auroc              -0.016593             3
    additive_binary    29            top_no_cert                   auroc               0.023919             3
        null_binary    11            top_no_cert                   auroc               0.001418             3
        null_binary    29            top_no_cert                   auroc              -0.014000             3
    additive_binary    11        coerror_no_cert                   auroc              -0.010225             3
    additive_binary    29        coerror_no_cert                   auroc               0.048788             3
        null_binary    11        coerror_no_cert                   auroc               0.008207             3
        null_binary    29        coerror_no_cert                   auroc               0.011350             3
    additive_binary    11      shrinkage_no_cert                   auroc              -0.018433             3
    additive_binary    29      shrinkage_no_cert                   auroc               0.048788             3
        null_binary    11      shrinkage_no_cert                   auroc               0.019684             3
        null_binary    29      shrinkage_no_cert                   auroc              -0.023667             3
    additive_binary    11          random_forest                   auroc               0.000000             3
    additive_binary    29          random_forest                   auroc               0.000000             3
        null_binary    11          random_forest                   auroc               0.000000             3
        null_binary    29          random_forest                   auroc               0.000000             3
additive_regression    11        random_subspace normalized_squared_loss              -0.035983             3
additive_regression    29        random_subspace normalized_squared_loss               0.004881             3
    null_regression    11        random_subspace normalized_squared_loss               0.010432             3
    null_regression    29        random_subspace normalized_squared_loss               0.005131             3
additive_regression    11            top_no_cert normalized_squared_loss              -0.006248             3
additive_regression    29            top_no_cert normalized_squared_loss              -0.002574             3
    null_regression    11            top_no_cert normalized_squared_loss              -0.012678             3
    null_regression    29            top_no_cert normalized_squared_loss               0.013124             3
additive_regression    11        coerror_no_cert normalized_squared_loss               0.011880             3
additive_regression    29        coerror_no_cert normalized_squared_loss              -0.000394             3
    null_regression    11        coerror_no_cert normalized_squared_loss              -0.009547             3
    null_regression    29        coerror_no_cert normalized_squared_loss               0.012750             3
additive_regression    11      shrinkage_no_cert normalized_squared_loss               0.014836             3
additive_regression    29      shrinkage_no_cert normalized_squared_loss               0.005226             3
    null_regression    11      shrinkage_no_cert normalized_squared_loss              -0.013343             3
    null_regression    29      shrinkage_no_cert normalized_squared_loss               0.013040             3
additive_regression    11 coerror_no_cert_affine normalized_squared_loss               0.005822             3
additive_regression    29 coerror_no_cert_affine normalized_squared_loss              -0.032386             3
    null_regression    11 coerror_no_cert_affine normalized_squared_loss              -0.032605             3
    null_regression    29 coerror_no_cert_affine normalized_squared_loss              -0.012582             3
additive_regression    11        affine_profiled normalized_squared_loss              -0.045312             3
additive_regression    29        affine_profiled normalized_squared_loss              -0.018934             3
    null_regression    11        affine_profiled normalized_squared_loss              -0.042154             3
    null_regression    29        affine_profiled normalized_squared_loss              -0.020605             3
additive_regression    11          random_forest normalized_squared_loss               0.000000             3
additive_regression    29          random_forest normalized_squared_loss               0.000000             3
    null_regression    11          random_forest normalized_squared_loss               0.000000             3
    null_regression    29          random_forest normalized_squared_loss               0.000000             3

## Null screening counts/rates

                       certified_count            certification_rate                    
                                  mean  min   max               mean       min       max
dataset         pool_B                                                                  
null_binary     30            1.833333  0.0   5.0           0.061111  0.000000  0.166667
                100           5.833333  3.0  13.0           0.058333  0.030000  0.130000
                300          16.000000  7.0  39.0           0.053333  0.023333  0.130000
null_regression 30            0.000000  0.0   0.0           0.000000  0.000000  0.000000
                100           0.000000  0.0   0.0           0.000000  0.000000  0.000000
                300           0.000000  0.0   0.0           0.000000  0.000000  0.000000

No independent confirmation; reused synthetic development seeds.
