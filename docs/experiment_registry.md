# Experiment registry

| ID | Hypothesis | Config | Status | Result / next test |
|---|---|---|---|---|
| smoke_001 | Verify the complete artifact/evaluation path | `configs/experiments/smoke.yaml` | complete | Four outer folds; fixed-K infeasibility preserved; engineering check only |
| exp_001 | Co-error is more loss-aligned than absolute residual correlation; sparse trees fail on interactions | `configs/experiments/exp_001_mechanisms.yaml` | complete: exp_001_mechanisms_v1 | 108 folds; broad hypothesis unsupported at these settings; screen false survivors and infeasibility; next isolate aggregate attenuation |
| exp_002 | Aggregate attenuation explains part of regression failure; profiled selection adds value beyond calibrating a fixed subset | `configs/experiments/exp_002_affine.yaml` | complete: exp_002_affine_v1 | Fixed-subset calibration helps 5/6 task means; profiled selection loses to calibrated co-error on 5/6 despite better OOF loss in every split; test candidate-search optimism next |
| exp_003 | Increasing B at fixed K can amplify selection optimism; shrinkage may mitigate it | `configs/experiments/exp_003_pool_b{30,100,300}.yaml` | complete: exp_003_pool_b{30,100,300}_v1 | 72 folds; exact OOF/bootstrap prefixes; binary additive gains, non-monotone regression, profiled selection optimism and increased null screen survivors; shrinkage .5 gives no consistent remedy |
| exp_004 | Co-error beats quality on real development tasks; increasing mask width separates representation restriction from selector weakness | `configs/experiments/exp_004_real_f{10,50}.yaml` | complete: exp_004_real_f{10,50}_v1 | 36 folds, 936 audited predictions; co-error improves over top-quality but trails RF/linear controls in all task/width means; wider masks help wine/diabetes but hurt cancer AUROC; duplicate OOF columns confound selection gains |
| exp_005 | Duplicate OOF columns explain part of quality-versus-co-error gains; uniqueness at fixed K isolates concentration | `configs/experiments/exp_005_oof_duplicates.yaml` | complete: exp_005_oof_duplicates_v1 | 18 folds / 162 audited predictions; exact unchanged pools and 108 controls; dedup worsens diabetes co-error RMSE and eliminates its quality gap, wine log-loss gap nearly vanishes, cancer effects unchanged; investigate explicit weights/refit shift |

## exp_006 — larger real tasks and thousand-tree libraries (registered)

- **Hypothesis:** At fixed K, library expansion adds useful complementarity beyond
  quality selection; matched B width/depth cells distinguish representation effects.
- **Protocol:** `experiments/exp_006_large/protocol.md`; five task-specific
  `configs/experiments/exp_006_large_*.yaml`, 6,000 candidates per outer fold,
  3 binary/2 regression tasks, 20,000 rows, 3 outer/2 OOF folds, one seed.
- **Status:** All six tasks complete/audited (1,098 test predictions;
  216,000 OOF base fits), full sweep and resource reports generated.
- **Anchor:** Unscreened co-error B6,000,K64; full sweep retained, no best-test tuning.
- **Result:** Anchor AUROC .779/.764/.970 on credit/HIGGS/MiniBooNE;
  exceeds top quality on each but beats default RF only on credit; RF leaf5 matches
  credit co-error at .779. Trails at least one
  boosting reference on each. Regression RMSE13.807/.594 versus RF10.206/.508.
  B6000 does not improve B3000 at K64 on credit/HIGGS; increasing K beyond64
  worsens both regression means. All18 co/quality subsets have64 unique OOF columns.
- **Interpretation/follow-up:** Conditional development evidence. These are chosen tasks with
  exact-feature group holdouts, not a locked representative confirmation benchmark.

### exp_006 Musk extension (registered before Musk scores)

`experiments/exp_006_large/musk_extension.md`:6598 rows,166 features,102 molecules;
52 same sweep settings and9 baselines; molecule groups in both folds/bootstrap;
fixed max-over-conformation molecule AUROC primary. Added after credit default
completed, to test higher dimensionality. Training/selection row objectives differ
from the group-max primary; preserve both and do not call this optimized MIL.
Bioresponse deferred because preexisting descriptor-normalization provenance is
unspecified. No primary data exclusion based on observed performance.
Complete/audited: co-error molecule AUC.793 vsquality.817,RF.846,Caruana.807;
row Brier improves from.131 quality to.123 co-error but Caruana reaches.115.
Registered count-only molecule control averagesAUC.454 (folds.498/.456/.408).
This primary-metric counterexample is preserved; no simple count-only explanation
is supported, and more complex pooling/cardinality effects remain possible.

## exp_007 — loss-aligned quality-only control (registered; pending)

- **Hypothesis:** Co-error versus AUROC-quality differences partly reflect loss
  alignment rather than cross-error optimization.
- **Protocol/config:** `experiments/exp_007_quality_alignment/protocol.md` and
  `configs/experiments/exp_007_quality_alignment.yaml`.
- **Fixed:** Entire six-task exp_006 data/OOF/specifications/partitions; B6000,K64,
  unscreened equal weights. No repeated search or new tuning.
- **Change:** Select K minimum individual OOF squared losses (G diagonal).
- **Status:** Registered after the three larger classification results, before
  this control's predictions. Development reuse is explicit; regression is an
  expected unchanged-selection check. Main exp_006 settings remain unchanged.
