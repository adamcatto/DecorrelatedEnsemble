# Experiment registry

| ID | Hypothesis | Config | Status | Result / next test |
|---|---|---|---|---|
| smoke_001 | Verify the complete artifact/evaluation path | `configs/experiments/smoke.yaml` | complete | Four outer folds; fixed-K infeasibility preserved; engineering check only |
| exp_001 | Co-error is more loss-aligned than absolute residual correlation; sparse trees fail on interactions | `configs/experiments/exp_001_mechanisms.yaml` | complete: exp_001_mechanisms_v1 | 108 folds; broad hypothesis unsupported at these settings; screen false survivors and infeasibility; next isolate aggregate attenuation |
| exp_002 | Aggregate attenuation explains part of regression failure; profiled selection adds value beyond calibrating a fixed subset | `configs/experiments/exp_002_affine.yaml` | planned from exp_001 | pending; preserve null/dominant counterexamples |
