# exp_002_affine_v1: generated affine diagnostic

Effects are paired within dataset/seed/fold; positive means improvement.

method                affine_profiled  catboost  coerror_no_cert  coerror_no_cert_affine  extra_trees  lightgbm  random_forest  random_subspace  random_subspace_affine  rf_shallow  top_no_cert  top_no_cert_affine  xgboost
dataset                                                                                                                                                                                                                      
additive_regression            0.9034    0.7637           0.8993                  0.8781       0.7781    0.7384         0.7838           0.9330                  0.9217      0.8971       0.9198              0.9124   0.7478
dominant_regression            0.5197    0.5142           0.5152                  0.5098       0.5025    0.5427         0.5106           0.8901                  0.8182      0.5180       0.5173              0.5125   0.5455
mixed_regression               0.9017    0.8137           0.9043                  0.8917       0.8417    0.8615         0.8460           0.9741                  0.9839      0.8915       0.9369              0.9290   0.8802
null_regression                1.0375    1.0179           1.0176                  1.0311       1.0404    1.1229         1.0376           1.0067                  1.0003      1.0478       1.0140              1.0236   1.1059
redundant_regression           0.6411    0.5803           0.6870                  0.6553       0.5929    0.6363         0.5910           0.6912                  0.6483      0.6700       0.7011              0.6865   0.6274
sparse_regression              0.6848    0.6171           0.7441                  0.6728       0.6132    0.6390         0.6387           0.9249                  0.8887      0.6608       0.7753              0.7487   0.6397

## OOF-fitted slopes (mean and range over folds/seeds)

                                                 mean       min       max
dataset              method                                              
additive_regression  affine_profiled         2.736023  2.330756  2.957787
                     coerror_no_cert_affine  2.066529  1.868782  2.405385
                     random_subspace_affine  1.388864  1.000079  1.909865
                     top_no_cert_affine      1.990989  1.030489  2.284661
dominant_regression  affine_profiled         1.756208  1.515503  1.856499
                     coerror_no_cert_affine  1.077839  1.017624  1.206311
                     random_subspace_affine  1.967458  0.000000  2.861716
                     top_no_cert_affine      0.965448  0.885297  1.200340
mixed_regression     affine_profiled         1.988173  1.448955  2.544132
                     coerror_no_cert_affine  1.577186  1.417264  1.936700
                     random_subspace_affine  0.698898  0.088966  1.441127
                     top_no_cert_affine      1.271124  0.857332  1.480278
null_regression      affine_profiled         1.471058  0.894513  1.788885
                     coerror_no_cert_affine  1.361796  0.894513  1.609393
                     random_subspace_affine  0.090353  0.000000  0.542120
                     top_no_cert_affine      1.264876  0.680863  1.717797
redundant_regression affine_profiled         1.807727  1.691862  2.025929
                     coerror_no_cert_affine  1.599692  1.461710  1.728548
                     random_subspace_affine  1.479137  1.369380  1.534106
                     top_no_cert_affine      1.408662  1.285009  1.511207
sparse_regression    affine_profiled         2.401347  2.050230  2.942251
                     coerror_no_cert_affine  1.800459  1.444359  2.000152
                     random_subspace_affine  1.471229  0.654288  2.300793
                     top_no_cert_affine      1.414517  0.914247  1.794985

Dots in the calibration figure are paired split effects, not IID confidence intervals.
These are deliberately chosen development mechanisms, not a task-population benchmark.
