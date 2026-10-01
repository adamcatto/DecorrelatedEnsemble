import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def metadata():
    return {
        "utc": datetime.now(UTC).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "thread_policy": "one BLAS/OpenMP thread; estimators n_jobs=1",
    }


def source_snapshot(root):
    paths = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    with zipfile.ZipFile(root / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            if Path(p).is_file() and not p.startswith("results/"):
                archive.write(p, p)
    (root / "worktree.patch").write_text(git("diff", "HEAD"))
    if Path("uv.lock").exists():
        shutil.copy("uv.lock", root / "uv.lock")


def artifact_manifest(root):
    data = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            data[str(p.relative_to(root))] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
    write_json(root / "manifest.json", {"schema_version": 1, "files": data})


@contextmanager
def measured():
    """Absolute process RSS sampled every 20ms; no GPU or child-process accounting."""
    stats = {"rss_start_bytes": psutil.Process().memory_info().rss}
    peak = [stats["rss_start_bytes"]]
    stop = threading.Event()

    def monitor():
        process = psutil.Process()
        while not stop.wait(0.02):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    start, cpu = time.perf_counter(), time.process_time()
    thread.start()
    try:
        yield stats
    finally:
        stop.set()
        thread.join()
        stats.update(
            {
                "wall_seconds": time.perf_counter() - start,
                "cpu_seconds": time.process_time() - cpu,
                "peak_rss_bytes": max(peak[0], psutil.Process().memory_info().rss),
                "memory_scope": "absolute process RSS sampled 20ms; shared allocations included",
            }
        )
