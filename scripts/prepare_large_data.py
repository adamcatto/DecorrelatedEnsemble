"""Download public sources once and write an offline, hashed development cache."""

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from decorrelated_ensemble.evaluation.artifacts import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results/data_sources"
SOURCES = {
    "credit_default": ("https://archive.ics.uci.edu/static/public/350/data.csv", "credit.csv"),
    "higgs": ("https://openml.org/data/v1/download/2063675/higgs.arff", "higgs.arff"),
    "miniboone": (
        "https://archive.ics.uci.edu/static/public/199/miniboone+particle+identification.zip",
        "miniboone.zip",
    ),
    "superconductivity": ("https://archive.ics.uci.edu/static/public/464/data.csv", "supercon.csv"),
    "california_housing": ("https://ndownloader.figshare.com/files/5976036", "cal_housing.tgz"),
}
ATTRIBUTION = {
    "credit_default": (
        "I-Cheng Yeh (2009)",
        "10.24432/C55S3H",
        "CC-BY-4.0",
        "Default next month, 1=default",
        "UCI 350; ID excluded; X2/X3/X4 treated as categorical",
    ),
    "higgs": (
        "Baldi, Sadowski, and Whiteson (2014)",
        "10.24432/C5V312",
        "OpenML metadata: Public; original UCI CC-BY-4.0",
        "1=signal, 0=background",
        "OpenML 23512 version 2, 98,050-row published subset; all 28 variables retained",
    ),
    "miniboone": (
        "Byron Roe (2005)",
        "10.24432/C5QC87",
        "CC-BY-4.0",
        "1=electron-neutrino signal, 0=muon-neutrino background",
        "UCI 199; labels reconstructed from first-line counts; all 50 variables retained, sentinels unchanged",
    ),
    "superconductivity": (
        "Kam Hamidieh (2018)",
        "10.24432/C53P47",
        "CC-BY-4.0",
        "Critical temperature",
        "UCI 464 train features; chemical formula file excluded; IID split is not unseen-composition evaluation",
    ),
    "california_housing": (
        "Pace and Barry (1997), Sparse Spatial Autoregressions",
        "",
        "Original separate data license unspecified in sklearn documentation",
        "Median house value, units of $100,000",
        "sklearn rowwise aggregate features; IID split is not geographic extrapolation",
    ),
}


def prepare(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    url, filename = SOURCES[name]
    raw = CACHE / filename
    if not raw.exists():
        temporary = raw.with_suffix(raw.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as out:
            while block := response.read(2**20):
                out.write(block)
        temporary.replace(raw)
    pinned_path = ROOT / "configs/datasets/exp_006_public_sources.json"
    if pinned_path.exists():
        pinned = json.loads(pinned_path.read_text())[name]
        if sha256(raw) != pinned["raw_sha256"]:
            raise ValueError("Downloaded source differs from the registered raw-source hash")
    if name in {"credit_default", "superconductivity"}:
        frame = pd.read_csv(raw)
        target = "Y" if name == "credit_default" else "critical_temp"
        y = frame.pop(target).to_numpy()
        if name == "credit_default":
            frame = frame.drop(columns="ID")
            for col in ["X2", "X3", "X4"]:
                frame[col] = frame[col].astype(str)
    elif name == "higgs":
        data, _ = arff.loadarff(raw)
        frame = pd.DataFrame(data)
        y = frame.pop("class").astype(int).to_numpy()
    elif name == "miniboone":
        with zipfile.ZipFile(raw) as archive:
            member = next(n for n in archive.namelist() if n.endswith("MiniBooNE_PID.txt"))
            with archive.open(member) as stream:
                counts = [int(x) for x in stream.readline().split()]
                values = np.loadtxt(stream)
        if sum(counts) != len(values) or values.shape[1] != 50:
            raise ValueError("MiniBooNE source counts/shape mismatch")
        frame = pd.DataFrame(values, columns=[f"pid_{j}" for j in range(50)])
        y = np.r_[np.ones(counts[0], dtype=int), np.zeros(counts[1], dtype=int)]
    else:
        expected = "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"
        if sha256(raw) != expected:
            raise ValueError("California source differs from sklearn pinned checksum")
        with tarfile.open(raw) as archive:
            values = np.loadtxt(
                io.BytesIO(archive.extractfile("CaliforniaHousing/cal_housing.data").read()),
                delimiter=",",
            )
        # Exactly sklearn's deterministic rowwise feature definitions; no fitted transform.
        values = values[:, [8, 7, 2, 3, 4, 5, 6, 1, 0]]
        y, features = values[:, 0] / 100000, values[:, 1:]
        features[:, 2] /= features[:, 5]
        features[:, 3] /= features[:, 5]
        features[:, 5] = features[:, 4] / features[:, 5]
        frame = pd.DataFrame(
            features,
            columns=[
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
            ],
        )
    task = "regression" if name in {"superconductivity", "california_housing"} else "binary"
    author, doi, license_name, target, adaptation = ATTRIBUTION[name]
    destination = CACHE / f"{name}.pkl"
    pd.to_pickle({"X": frame, "y": y}, destination)
    metadata = {
        "name": name,
        "task": task,
        "n_full": len(y),
        "p": frame.shape[1],
        "source_url": url,
        "raw_file": filename,
        "raw_sha256": sha256(raw),
        "cache_sha256": sha256(destination),
        "citation": author,
        "doi": doi,
        "license": license_name,
        "target_definition": target,
        "adaptation": adaptation,
        "missing_values": int(frame.isna().sum().sum()),
        "exact_duplicate_feature_rows": int(frame.duplicated().sum()),
        "minus_999_cells": int((frame == -999).sum().sum()),
        "class_counts": {str(c): int(np.sum(y == c)) for c in np.unique(y)}
        if task == "binary"
        else None,
    }
    metadata["content_sha256"] = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes() + y.tobytes()
    ).hexdigest()
    write_json(CACHE / f"{name}.json", metadata)
    print(json.dumps(metadata), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(SOURCES))
    for name in parser.parse_args().names:
        prepare(name)
