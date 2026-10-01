"""Pinned UCI molecule-level follow-up source; identifiers never become features."""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from decorrelated_ensemble.evaluation.artifacts import sha256, write_json

p = Path("results/data_sources")
p.mkdir(parents=True, exist_ok=True)
raw = p / "musk2.csv"
if not raw.exists():
    raw.write_bytes(
        urllib.request.urlopen(
            "https://archive.ics.uci.edu/static/public/75/data.csv", timeout=60
        ).read()
    )
pinned = Path("configs/datasets/exp_006_musk_source.json")
if pinned.exists() and json.loads(pinned.read_text())["raw_sha256"] != sha256(raw):
    raise ValueError("Source hash changed")
frame = pd.read_csv(raw)
print(frame.shape, frame.columns[:4].tolist(), frame.columns[-2:].tolist(), flush=True)
y = frame.pop("class").to_numpy(dtype=int)
groups = frame.pop("molecule_name").to_numpy()
frame = frame.drop(columns="conformation_name")
if frame.shape != (6598, 166) or len(np.unique(groups)) != 102:
    raise ValueError("Unexpected source dimensions")
for group in np.unique(groups):
    if len(np.unique(y[groups == group])) != 1:
        raise ValueError("Inconsistent molecule labels")
out = p / "musk2.pkl"
pd.to_pickle({"X": frame, "y": y, "groups": groups}, out)
meta = {
    "name": "musk2",
    "task": "binary",
    "n_full": len(y),
    "p": 166,
    "groups_full": 102,
    "positive_groups": 39,
    "source_url": "https://archive.ics.uci.edu/static/public/75/data.csv",
    "raw_file": raw.name,
    "raw_sha256": sha256(raw),
    "cache_sha256": sha256(out),
    "citation": "Chapman and Jain (1994), UCI Musk version 2",
    "doi": "10.24432/C51608",
    "license": "CC-BY-4.0",
    "adaptation": "Molecule/conformation names excluded from X; molecule names retained only as group IDs; physical distance features unchanged",
    "target_definition": "1=musk molecule, 0=non-musk; row targets inherited from molecule",
    "missing_values": int(frame.isna().sum().sum()),
    "exact_duplicate_feature_rows": int(frame.duplicated().sum()),
    "content_sha256": hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes() + y.tobytes()
    ).hexdigest(),
}
write_json(out.with_suffix(".json"), meta)
print(meta, flush=True)
pinned = Path("configs/datasets/exp_006_musk_source.json")
if pinned.exists() and json.loads(pinned.read_text())["raw_sha256"] != meta["raw_sha256"]:
    raise ValueError("Source hash changed")
