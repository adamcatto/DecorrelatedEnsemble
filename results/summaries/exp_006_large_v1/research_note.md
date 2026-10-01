# Experiment 006: larger-data sweep

All five original tasks and the registered Musk extension retained; one seed and three group-separated outer folds. Molecule-max AUROC is primary only for Musk.
No calibrated intervals or best-test tuned method; static baselines and unmatched budgets. Task-bootstrap intervals resample only the four/two observed classification/regression tasks after fold averaging. They are exploratory, unadjusted, and do not justify population or significance claims; rank summaries are descriptive.

## Credit default (auroc)

Anchor 0.778678; top 0.775707; Caruana 0.777880; RF 0.766716; linear 0.727645.

Co-error pool curve (K64): B300=0.777276, B1000=0.778017, B3000=0.778827, B6000=0.778678.

Co-error retained-size curve (B6000): K16=0.776995, K64=0.778678, K128=0.778860, K256=0.778867.

## HIGGS (auroc)

Anchor 0.764240; top 0.745721; Caruana 0.763585; RF 0.784727; linear 0.689497.

Co-error pool curve (K64): B300=0.759945, B1000=0.763471, B3000=0.765307, B6000=0.764240.

Co-error retained-size curve (B6000): K16=0.762323, K64=0.764240, K128=0.764847, K256=0.764904.

## MiniBooNE (auroc)

Anchor 0.970030; top 0.963200; Caruana 0.970144; RF 0.975652; linear 0.931985.

Co-error pool curve (K64): B300=0.969097, B1000=0.969659, B3000=0.970006, B6000=0.970030.

Co-error retained-size curve (B6000): K16=0.969307, K64=0.970030, K128=0.970069, K256=0.970035.

## Superconductivity (rmse)

Anchor 13.806564; top 14.037884; Caruana 13.839816; RF 10.206069; linear 17.772847.

Co-error pool curve (K64): B300=14.139515, B1000=13.991768, B3000=13.861523, B6000=13.806564.

Co-error retained-size curve (B6000): K16=13.873863, K64=13.806564, K128=13.851352, K256=13.905825.

## California housing (rmse)

Anchor 0.593933; top 0.632712; Caruana 0.593932; RF 0.508368; linear 0.726457.

Co-error pool curve (K64): B300=0.660405, B1000=0.598080, B3000=0.594312, B6000=0.593933.

Co-error retained-size curve (B6000): K16=0.594043, K64=0.593933, K128=0.594405, K256=0.595599.

## Musk v2 (molecules) (group_auroc)

Anchor 0.792826; top 0.817151; Caruana 0.806586; RF 0.846166; linear 0.787363.

Co-error pool curve (K64): B300=0.803855, B1000=0.760256, B3000=0.778808, B6000=0.792826.

Co-error retained-size curve (B6000): K16=0.787106, K64=0.792826, K128=0.785096, K256=0.781385.

