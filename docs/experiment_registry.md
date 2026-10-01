# Experiment registry

| ID | Hypothesis | Config | Status | Result / next test |
|---|---|---|---|---|
| smoke_001 | Verify the complete artifact/evaluation path | `configs/experiments/smoke.yaml` | complete | Four outer folds; fixed-K infeasibility preserved; engineering check only |
| exp_001 | Co-error is more loss-aligned than absolute residual correlation; sparse trees fail on interactions | `configs/experiments/exp_001_mechanisms.yaml` | complete: exp_001_mechanisms_v1 | 108 folds; broad hypothesis unsupported at these settings; screen false survivors and infeasibility; next isolate aggregate attenuation |
| exp_002 | Aggregate attenuation explains part of regression failure; profiled selection adds value beyond calibrating a fixed subset | `configs/experiments/exp_002_affine.yaml` | complete: exp_002_affine_v1 | Fixed-subset calibration helps 5/6 task means; profiled selection loses to calibrated co-error on 5/6 despite better OOF loss in every split; test candidate-search optimism next |
| exp_003 | Increasing B at fixed K can amplify selection optimism; shrinkage may mitigate it | `configs/experiments/exp_003_pool_b{30,100,300}.yaml` | complete: exp_003_pool_b{30,100,300}_v1 | 72 folds; exact OOF/bootstrap prefixes; binary additive gains, non-monotone regression, profiled selection optimism and increased null screen survivors; shrinkage .5 gives no consistent remedy |
