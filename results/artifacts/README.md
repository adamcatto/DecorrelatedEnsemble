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

Experiment 006 archives are split into at most 48 MiB parts for Git hosting.
`<run>_export.json` records part order, byte sizes and SHA256 plus the combined
archive SHA256. Reconstruct before extraction, for example:

```bash
cat results/artifacts/exp_006_large_higgs_v1.tar.gz.part* > /tmp/exp_006_large_higgs_v1.tar.gz
# Verify the reconstructed SHA256 against the export metadata, then:
tar -xzf /tmp/exp_006_large_higgs_v1.tar.gz -C results/runs
```

The complete combined archive and fitted models remain local. Only fitted models
are omitted from the parts; no predictions, unsuccessful statuses or decisions
are omitted. Use `--predictions-only` for audits/reports of restored exports.

For exp_006, the exported OOF NPZ uses column-major storage, which compresses
constant tree-leaf prediction patterns much better. **All array values and dtypes
are verified exactly equal** during export. This is lossless storage, not rounding
or prediction quantization. The original run remains unchanged. Bundles retain
`original_manifest.json`; exported `manifest.json` updates only the OOF byte hashes
and adds the original manifest hash. Export metadata records both hashes/sizes
for every repacked file. The normal audit verifies the exported manifest and
reconstructs metrics/loss identities from the same numerical predictions.

Experiment 007 reuses the six exp_006 parent runs listed in its complete config.
Restore those bundles under `results/runs/` alongside the follow-up bundle before
running `audit_quality_alignment.py --predictions-only`. Parent byte hashes and
exact array hashes link reused raw data, specifications, splits and OOF values;
the latter are independent of C/F archive storage. `verify_export.py` restores
all declared dependencies in a fresh temporary directory and performs this
audit automatically. No parent predictions are duplicated in the follow-up.
