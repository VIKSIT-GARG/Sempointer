"""Provenance recording for every experiment run.

Every run directory gets:
  environment.json - python/torch/CUDA/GPU/driver/package versions + git commit
  config.json      - the exact run configuration (no hidden constants)
  predictions.jsonl- per-example raw model output (prompt, answer, gold, metrics)

MEASURED facts go here from the running process; nothing is asserted.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional


def git_commit(repo_dir: Optional[str] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir or os.path.dirname(__file__), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def environment_record() -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "git_commit": git_commit(),
    }
    try:
        import torch

        rec["torch_version"] = torch.__version__
        rec["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rec["cuda_version"] = torch.version.cuda
            rec["gpu_name"] = torch.cuda.get_device_name(0)
            rec["gpu_total_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2, 1)
            try:
                driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], stderr=subprocess.DEVNULL).decode().strip()
                rec["gpu_driver"] = driver.splitlines()[0]
            except Exception:
                rec["gpu_driver"] = None
    except ImportError:
        rec["torch_version"] = None
    for pkg in ["transformers", "sentence_transformers", "datasets", "bitsandbytes", "kvpress", "numpy", "rank_bm25"]:
        try:
            mod = __import__(pkg)
            rec[f"{pkg}_version"] = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
        except Exception:
            rec[f"{pkg}_version"] = None
    return rec


def init_run_dir(output_dir: str, config: Dict[str, Any]) -> str:
    """Creates the run directory and writes environment.json + config.json."""
    os.makedirs(output_dir, exist_ok=True)
    env = environment_record()
    cfg = dict(config)
    cfg["git_commit"] = env["git_commit"]
    with open(os.path.join(output_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    return output_dir


def append_prediction(output_dir: str, row: Dict[str, Any]) -> None:
    with open(os.path.join(output_dir, "predictions.jsonl"), "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def finalize_metrics(output_dir: str, metrics: Dict[str, Any]) -> str:
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path
