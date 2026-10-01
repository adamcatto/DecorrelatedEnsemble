# Auditable experiment bundles

`smoke_001.tar.gz` includes the entire small run, including fitted models. SHA256
hashes and coverage are in its export metadata and original manifest.

Larger bundles may omit fitted `model.joblib` files; omissions are listed explicitly
in `<run>_export.json`. Frozen data, complete source snapshot, configuration,
environment, bootstrap/OOF/test predictions, candidate definitions, selections,
metrics, and resources remain included. Full models are retained in the original
local `results/runs/` directory and can be reproduced from the bundled inputs.
The original manifest records hashes of both included and omitted files.

Unpack a bundle under `results/runs/` to inspect decision artifacts. Aggregation
requires the full run manifest; use the stored generated summaries for an export
without models, or reproduce models/rerun to create a new full manifest.
