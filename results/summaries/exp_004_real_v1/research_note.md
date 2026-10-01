# Experiment 004: generated real development report

No task-level interval is estimated: there is one dataset per task type.
Dots/min/max are overlapping split observations, not independent confidence intervals.

## Task-specific primary metrics

### breast_cancer

feature_fraction             0.1       0.5
method                                    
abs_correlation         0.990330  0.985908
caruana                 0.985546  0.981341
catboost                0.993251  0.993251
certified_random        0.983868  0.980020
coerror                 0.987551  0.981006
coerror_greedy          0.987894  0.982052
coerror_no_cert         0.987551  0.981006
covariance_only         0.988035  0.981006
direct_auroc            0.991051  0.978523
direct_squared          0.987894  0.982052
et_shallow              0.982671  0.982671
extra_trees             0.993770  0.993770
lightgbm                0.990740  0.990740
linear                  0.995180  0.995180
minimax                 0.988787  0.984356
prediction_correlation  0.956612  0.985424
quality_diversity       0.987587  0.984923
random_forest           0.989336  0.989336
random_patches          0.986461  0.982186
random_subspace         0.983780  0.980020
rf_shallow              0.983842  0.983842
shrinkage               0.986545  0.980968
signed_correlation      0.990330  0.985908
top_no_cert             0.980554  0.973961
top_quality             0.980554  0.973961
xgboost                 0.991058  0.991058

### wine

feature_fraction             0.1       0.5
method                                    
abs_correlation         0.294013  0.175513
caruana                 0.225413  0.162939
catboost                0.168222  0.168222
certified_random        0.457048  0.168934
coerror                 0.230733  0.161178
coerror_greedy          0.228032  0.161687
coerror_no_cert         0.230733  0.161178
covariance_only         0.233641  0.161633
direct_squared          0.228032  0.161687
et_shallow              0.476017  0.476017
extra_trees             0.157517  0.157517
lightgbm                0.117767  0.117767
linear                  0.079228  0.079228
minimax                 0.319728  0.176131
prediction_correlation  0.612044  0.232864
quality_diversity       0.230634  0.165163
random_forest           0.151954  0.151954
random_patches          0.425840  0.246334
random_subspace         0.457048  0.168934
rf_shallow              0.254566  0.254566
shrinkage               0.226919  0.167108
signed_correlation      0.294013  0.175513
top_no_cert             0.247531  0.501979
top_quality             0.247531  0.501979
xgboost                 0.139583  0.139583

### diabetes

feature_fraction              0.1        0.5
method                                      
abs_correlation         60.694663  59.458536
affine_profiled         58.276724  58.931651
caruana                 60.077324  59.042170
catboost                55.901747  55.901747
certified_random        63.594481  59.506086
coerror                 59.826220  59.071633
coerror_greedy          59.825625  58.825177
coerror_no_cert         60.077920  59.071633
coerror_no_cert_affine  58.457968  58.553758
covariance_only         59.826220  59.071633
direct_squared          59.825625  58.825177
et_shallow              58.337211  58.337211
extra_trees             57.314700  57.314700
lightgbm                58.589359  58.589359
linear                  54.973880  54.973880
minimax                 61.502314  59.156076
prediction_correlation  61.718737  60.627811
quality_diversity       61.070919  59.642257
random_forest           58.104492  58.104492
random_patches          68.738671  59.160832
random_subspace         67.497951  59.022624
rf_shallow              57.097709  57.097709
shrinkage               59.790635  58.782791
signed_correlation      60.694663  59.458536
top_no_cert             64.453258  59.980136
top_quality             64.453258  59.980136
xgboost                 59.493443  59.493443

## Screen survival

                                      mean    min    max
dataset       feature_fraction                          
breast_cancer 0.1                99.833333   99.0  100.0
              0.5               100.000000  100.0  100.0
diabetes      0.1                40.166667   29.0   51.0
              0.5                90.500000   83.0   98.0
wine          0.1               100.000000  100.0  100.0
              0.5               100.000000  100.0  100.0

All per-split secondary metrics, effects, seed effects and resource measurements are stored in CSV files.
No pooled ranking or statistical superiority claim is warranted.

## Observed OOF-equivalence classes (exploratory diagnostic)

                                min  max
dataset       feature_fraction          
breast_cancer 0.1                96  100
              0.5                99  100
diabetes      0.1                10   10
              0.5                78   87
wine          0.1                64   73
              0.5                95  100

Equal OOF columns do not prove globally identical functions or identical full-training refits.
