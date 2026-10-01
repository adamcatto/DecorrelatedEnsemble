import argparse
import tarfile
from pathlib import Path

from decorrelated_ensemble.evaluation.artifacts import sha256, verify_manifest, write_json


def export(run_id, include_models=False):
    run = Path("results/runs") / run_id
    verify_manifest(run)
    out = Path("results/artifacts")
    out.mkdir(parents=True, exist_ok=True)
    destination = out / (run_id + ".tar.gz")
    included, omitted = [], []
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(run.rglob("*")):
            if not path.is_file():
                continue
            name = str(path.relative_to(run))
            if path.name == "model.joblib" and not include_models:
                omitted.append(name)
            else:
                archive.add(path, arcname=run_id + "/" + name)
                included.append(name)
    write_json(
        out / (run_id + "_export.json"),
        {
            "run_id": run_id,
            "archive": destination.name,
            "archive_sha256": sha256(destination),
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
    args = parser.parse_args()
    print(export(args.run_id, args.include_models))
