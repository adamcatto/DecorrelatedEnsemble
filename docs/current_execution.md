# Active exp_006 execution checkpoint (temporary research operations note)

2026-10-01: Own dataset-loop process PID81143 was SIGSTOP'ed to enforce an audit
barrier after MiniBooNE. All three classification training children have completed. Tool
session56031 owns this five-task loop: credit_default, HIGGS and MiniBooNE are COMPLETE/audited, superconductivity and California are queued. Parent MUST
be SIGCONT'ed after the audit barrier; verify create_time from
/private/tmp/de-large-orchestrator.json before signaling. Do not kill training.

Classification audits, lossless exports and partial paper are complete;
new source-group and molecule-max APIs/report code pass87 tests.
Checkpoint source/protocol/paper and large artifact parts are being committed. The predeclared Musk extension is not yet scored. Resume the
original loop for remaining regressions, then run Musk as a separate job after
that loop completes (avoid concurrent timing). Main model algorithms for the
original five datasets haven't changed; added source-group/max APIs are opt-in.

Prior pushed commit74334dd; resolve git HEAD/origin for newest checkpoint. Larger source cache ignored underresults/data_sources;
6k OOF matrices are large. New export flags --column-major-oof --part-mib48
losslessly repack OOF arrays into better-compressing column order; preserve original
manifest and export hash manifest, verify all array values/dtypes, then publishparts.
Do not commit combined exp006 archives (ignored) or localmodels. Reports expect
all SIX tasks and predesignated coerror_b6000_k64, with group_auroc primary onlyforMusk.
Compile/render/inspect final paper after the full results update. No final response
has been sent for this larger-data request; continue to completed evidence/pushes.
