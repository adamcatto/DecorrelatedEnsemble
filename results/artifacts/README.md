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
verifies the full run manifest by default. For an export without models, run
`python scripts/aggregate_results.py <run_id> --predictions-only`, which verifies
every included decision/prediction file and allows only missing `model.joblib`
files. The export metadata includes the archive SHA256 for download validation.
