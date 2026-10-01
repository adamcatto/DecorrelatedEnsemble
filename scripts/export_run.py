import argparse
import copy
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from decorrelated_ensemble.evaluation.artifacts import sha256, verify_manifest, write_json


def repack_oof(source, destination):
    """Lossless column-major archive layout; preserve every array and value."""
    with np.load(source) as archive:
        arrays = {key: archive[key] for key in archive.files}
    np.savez_compressed(
        destination,
        **{
            key: np.asfortranarray(value) if key == "predictions" else value
            for key, value in arrays.items()
        },
    )
    with np.load(destination) as reconstructed:
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, reconstructed[key])
            if value.dtype != reconstructed[key].dtype:
                raise ValueError("Lossless export changed an array dtype")
    return {
        "original_sha256": sha256(source),
        "export_sha256": sha256(destination),
        "original_bytes": source.stat().st_size,
        "export_bytes": destination.stat().st_size,
        "verification": "Every array reconstructed exactly, including float dtype and values",
    }


def export(run_id, include_models=False, part_mib=None, column_major_oof=False):
    run = Path("results/runs") / run_id
    verify_manifest(run)
    out = Path("results/artifacts")
    out.mkdir(parents=True, exist_ok=True)
    destination = out / (run_id + ".tar.gz")
    included, omitted, repacked = [], [], {}
    original_manifest = json.loads((run / "manifest.json").read_text())
    export_manifest = copy.deepcopy(original_manifest)
    with tempfile.TemporaryDirectory() as temporary, tarfile.open(destination, "w:gz") as archive:
        for path in sorted(run.rglob("*")):
            if not path.is_file():
                continue
            name = str(path.relative_to(run))
            if column_major_oof and name == "manifest.json":
                continue
            if path.name == "model.joblib" and not include_models:
                omitted.append(name)
            else:
                target = path
                if column_major_oof and path.name == "oof.npz":
                    target = Path(temporary) / "repacked.npz"
                    repacked[name] = repack_oof(path, target)
                    export_manifest["sha256"][name] = sha256(target)
                archive.add(target, arcname=run_id + "/" + name)
                included.append(name)
        if column_major_oof:
            archive.add(run / "manifest.json", arcname=run_id + "/original_manifest.json")
            export_manifest["sha256"]["original_manifest.json"] = sha256(run / "manifest.json")
            manifest_path = Path(temporary) / "manifest.json"
            write_json(manifest_path, export_manifest)
            archive.add(manifest_path, arcname=run_id + "/manifest.json")
            included.extend(["original_manifest.json", "manifest.json"])
    parts = []
    if part_mib is not None:
        size = int(part_mib * 2**20)
        if size < 1:
            raise ValueError("Positive archive part size required")
        with destination.open("rb") as stream:
            number = 0
            while block := stream.read(size):
                part = destination.with_name(destination.name + f".part{number:03d}")
                part.write_bytes(block)
                parts.append({"file": part.name, "bytes": len(block), "sha256": sha256(part)})
                number += 1
    write_json(
        out / (run_id + "_export.json"),
        {
            "run_id": run_id,
            "archive": destination.name,
            "archive_sha256": sha256(destination),
            "parts": parts,
            "lossless_oof_repacking": repacked,
            "reassemble": "Concatenate parts in listed order; verify reconstructed archive SHA256 before extraction"
            if parts
            else None,
            "included": included,
            "omitted": omitted,
            "omission_reason": "Fitted models retained locally; reproduce from frozen data, config, code snapshot and lockfile. All predictions and decision artifacts included.",
            "verify": "Original manifest contains hashes of included and omitted files; verify included hashes after extraction.",
        },
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--include-models", action="store_true")
    parser.add_argument("--part-mib", type=int)
    parser.add_argument("--column-major-oof", action="store_true")
    args = parser.parse_args()
    print(export(args.run_id, args.include_models, args.part_mib, args.column_major_oof))
