"""Training-only diagnosis of the failed exact regression equality assertion."""

import json
from pathlib import Path

import numpy as np

from decorrelated_ensemble.evaluation.artifacts import sha256, write_json


def audit():
    run = Path("results/runs/exp_006_large_california_housing_v1")
    fold = run / "california_housing/seed_11/fold_0"
    old = json.loads((fold / "top_quality_b6000_k64/selection.json").read_text())["ids"]
    with np.load(fold / "oof.npz") as saved:
        P, y = saved["predictions"], saved["y"]
    loss = np.mean(np.ascontiguousarray(y[:, None] - P) ** 2, axis=0)
    new = np.argsort(loss, kind="stable")[:64].tolist()
    q = np.load(fold / "certification.npz")["quality"]
    removed, added = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    comparisons = [
        {
            "removed": a,
            "added": b,
            "old_loss": loss[a],
            "new_loss": loss[b],
            "old_quality": q[a],
            "new_quality": q[b],
            "max_oof_difference": np.max(np.abs(P[:, a] - P[:, b])),
        }
        for a in removed
        for b in added
    ]
    output = Path("results/summaries/exp_007_training_ties/diagnosis.json")
    write_json(
        output,
        {
            "parent_manifest_sha256": sha256(run / "manifest.json"),
            "script_sha256": sha256(Path(__file__)),
            "old_ids": old,
            "loss_ranked_ids": new,
            "boundary_comparisons": comparisons,
            "test_outcomes_read": 0,
            "interpretation": "Strict mathematical monotonicity does not imply identical finite-precision rank/ties. Record equivalence; do not require an expected scientific control outcome as an implementation invariant.",
        },
    )
    return output


if __name__ == "__main__":
    print(audit())
