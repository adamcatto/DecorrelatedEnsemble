import hashlib
import importlib.metadata
import json
import platform
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import psutil


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(root):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "git_diff": git("diff", "HEAD"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "resource_scope": "one process, configured OOF/refit threads; baselines sequential; process CPU includes threads; RSS sampled every 20ms is an observed lower bound",
    }


def snapshot(root, destination):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for directory in ["src", "configs", "scripts", "tests", "docs", "paper"]:
            for path in sorted((root / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(root))
        for name in ["pyproject.toml", "uv.lock", "README.md"]:
            if (root / name).exists():
                archive.write(root / name, name)


class ResourceTimer:
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()

        def sample():
            while not self.stop.wait(0.02):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        self.wall, self.cpu = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *_):
        self.wall = time.perf_counter() - self.wall
        self.cpu = time.process_time() - self.cpu
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)

    def to_dict(self):
        return {
            "wall_seconds": self.wall,
            "cpu_seconds": self.cpu,
            "observed_peak_rss_bytes": self.peak,
        }


def finalize_manifest(path):
    files = {
        str(p.relative_to(path)): sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(path / "manifest.json", {"schema_version": 1, "sha256": files})


def verify_manifest(path, allow_missing_models=False):
    recorded = json.loads((path / "manifest.json").read_text())["sha256"]
    for name, expected in recorded.items():
        if allow_missing_models and name.endswith("/model.joblib") and not (path / name).exists():
            continue
        if sha256(path / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return True
