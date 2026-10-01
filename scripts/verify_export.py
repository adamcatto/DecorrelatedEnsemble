"""Verify ordered export parts, restore safely, and audit a clone-style bundle."""

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from decorrelated_ensemble.evaluation.artifacts import sha256, write_json


def verify_export(run_id):
    repo = Path(__file__).resolve().parents[1]
    artifacts = repo / "results" / "artifacts"
    metadata_path = artifacts / (run_id + "_export.json")
    metadata = json.loads(metadata_path.read_text())
    with tempfile.TemporaryDirectory(prefix="de-export-audit-") as directory:
        temporary = Path(directory)
        archive = temporary / metadata["archive"]
        parts = metadata["parts"] or [
            {"file": metadata["archive"], "sha256": metadata["archive_sha256"]}
        ]
        with archive.open("wb") as destination:
            for part in parts:
                source = artifacts / part["file"]
                if sha256(source) != part["sha256"]:
                    raise ValueError("Archive part hash mismatch")
                if "bytes" in part and source.stat().st_size != part["bytes"]:
                    raise ValueError("Archive part size mismatch")
                with source.open("rb") as stream:
                    while block := stream.read(2**20):
                        destination.write(block)
        if sha256(archive) != metadata["archive_sha256"]:
            raise ValueError("Reconstructed archive hash mismatch")
        runs = temporary / "results" / "runs"
        runs.mkdir(parents=True)
        with tarfile.open(archive) as stream:
            stream.extractall(runs, filter="data")
        subprocess.run(
            [sys.executable, str(repo / "scripts" / "audit_run.py"), run_id, "--predictions-only"],
            cwd=temporary,
            check=True,
        )
        audit = json.loads(
            (temporary / "results" / "summaries" / run_id / "audit.json").read_text()
        )
        output = repo / "results" / "summaries" / run_id / "export_roundtrip.json"
        write_json(
            output,
            {
                "run_id": run_id,
                "archive_sha256": metadata["archive_sha256"],
                "export_metadata_sha256": sha256(metadata_path),
                "verification_script_sha256": sha256(Path(__file__)),
                "parts_verified": len(parts),
                "audit": audit,
                "scope": "Fresh temporary reconstruction from published ordered parts; exported manifest and every stored test metric/OOF squared-loss identity verified; only fitted models may be absent",
            },
        )
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    print(verify_export(parser.parse_args().run_id))
