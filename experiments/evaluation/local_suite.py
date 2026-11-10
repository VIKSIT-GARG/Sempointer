"""Offline local GPU execution of balanced subsets of the frozen suite."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time

from .cloud_policy import CandidatePool, FAMILIES, atomic_json, digest

HERE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = HERE / "results/local_ten_qwen35_nf4_100_v1"


def select_subset(rows, cap, seed):
    if cap < 1:
        raise ValueError("The per-family cap must be positive")
    selected = []
    for family in FAMILIES:
        pool = CandidatePool(seed, cap)
        for row in rows:
            if row["family"] == family:
                pool.add(row)
        selected.extend(pool.selected())
    return selected


def prepare_local(source, root, model, cap=100, prompt_cap=8192):
    from prepare_modal import code_files, file_hash, verify_bundle
    source, root, model = map(lambda p: Path(p).resolve(), (source, root, model))
    if (root / "input/manifest.json").exists():
        manifest = verify_bundle(root / "input")
        if manifest.get("execution_backend") != "local_cuda" or manifest["per_family"] != cap:
            raise ValueError("Existing local configuration differs; use another directory")
        if manifest["limits"]["prompt_tokens"] != prompt_cap or manifest["model_local_path"] != str(model):
            raise ValueError("Existing model/prompt settings differ; use another directory")
        return manifest
    original = verify_bundle(source)
    if file_hash(model / "tokenizer.json") != original["tokenizer_file_sha256"]:
        raise ValueError("Local tokenizer differs from the pinned source model")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_names = sorted(set(index["weight_map"].values()))
    for name in weight_names:
        metadata = model / ".cache/huggingface/download" / (name + ".metadata")
        if not metadata.exists() or metadata.read_text().splitlines()[0] != original["model_revision"]:
            raise ValueError("Cannot verify the pinned revision of local weights: " + name)
    weights = {name: file_hash(model / name) for name in weight_names}
    rows = select_subset(original["rows"], cap, original["seed"])
    if set(r["family"] for r in rows) != set(original["families"]):
        raise ValueError("A selected benchmark family is missing")
    stage = root / "input.preparing"
    if stage.exists():
        raise ValueError("Incomplete preparation is preserved at " + str(stage))
    stage.mkdir(parents=True)
    for row in rows + original["development"]:
        target = stage / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / row["path"], target)
    evidence = original.get("configuration_evidence")
    if evidence:
        shutil.copy2(source / evidence["path"], stage / evidence["path"])
    snapshot = code_files()
    for name, checksum in snapshot.items():
        target = stage / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / name, target)
        if file_hash(target) != checksum:
            raise ValueError("Source changed while freezing the local runner")
    manifest = {k: v for k, v in original.items() if k not in {
        "run_id", "budget_account_run_id", "checkpoint_parent_run_id", "precision_pilot_id"}}
    manifest.update(execution_backend="local_cuda", parent_run_id=original["run_id"],
                    per_family=cap, rows=rows, dtype="nf4", inference_batch_size=1,
                    code_files=snapshot, model_local_path=str(model), model_weights_sha256=weights,
                    pilot_ids=[], cwe_audit_ids=[r["id"] for r in rows if r["task"] == "cwe" and r["family"] == "ruler"][:30],
                    limits={**original["limits"], "prompt_tokens":prompt_cap})
    manifest["sampling_report"] = {
        family: dict(selected=sum(r["family"] == family for r in rows),
                     selected_strata=dict(Counter(r["stratum"] for r in rows if r["family"] == family)),
                     selection_basis="Balanced deterministic subset of the prior frozen 300; no scores used")
        for family in original["families"]}
    manifest["notes"] = [
        "Local CUDA only; no Modal client, calls, or cost pilot.",
        "NF4 and batch one fit the local 8 GiB GPU; task-specific output caps are unchanged.",
        "Routing development is rerun under NF4; BF16 results are retained separately.",
        "Full-context inputs above the local prompt cap are explicit skips; sources are not truncated.",
        "Prior precision comparison is retained as reference, not evidence of local GPU performance."]
    manifest["run_id"] = digest(manifest)[:20]
    atomic_json(stage / "manifest.json", manifest)
    verify_bundle(stage)
    if snapshot != code_files():
        raise ValueError("Runtime changed during local preparation")
    stage.rename(root / "input")
    return manifest


def advance_local(root, model, run_phase, cancelled):
    root = Path(root)
    state_path = root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"phase":0, "status":"running", "timings":{}, "partial_times":{}}
    while state["phase"] < 6:
        if cancelled():
            state["status"] = "paused_local"
            atomic_json(state_path, state)
            return state
        # Phase 3 is a paid-cloud affordability pilot. Local execution goes
        # directly to all selected test rows, with the same stored predictions.
        if state["phase"] == 3:
            state.update(phase=4, status="running")
            atomic_json(state_path, state)
            continue
        state = run_phase(root, model, time.monotonic() + 1200, lambda: None, cancelled)
    return state


def write_local_report(root):
    from .cloud_worker import report
    result = report(root)
    path = root / "results/REPORT.md"
    text = path.read_text().replace("# Modal ten-benchmark evaluation", "# Local GPU ten-benchmark evaluation", 1)
    text += "\nExecution: local CUDA, Qwen3.5-4B NF4, batch one. No cloud inference.\n"
    path.write_text(text)
    return result


def run_local(root):
    from prepare_modal import file_hash, verify_bundle
    root = Path(root).resolve()
    with (root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_bundle(root / "input")
        if manifest.get("execution_backend") != "local_cuda":
            raise ValueError("This is not a local CUDA bundle")
        model = Path(manifest["model_local_path"])
        for name, checksum in manifest["model_weights_sha256"].items():
            if file_hash(model / name) != checksum:
                raise ValueError("Local model weights changed")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Local CUDA is unavailable; no CPU/cloud fallback is allowed")
        torch.set_num_threads(8)
        torch.set_num_interop_threads(2)
        stop = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())
        cancelled = lambda: stop.is_set() or (root / "STOP_REQUESTED").exists()
        atomic_json(root / "worker.json", dict(pid=os.getpid(), started=time.time(),
                    run_id=manifest["run_id"], gpu=torch.cuda.get_device_name(0)))
        print(json.dumps(dict(status="starting_local_gpu", run_id=manifest["run_id"],
              samples=len(manifest["rows"]), dtype=manifest["dtype"], gpu=torch.cuda.get_device_name(0))), flush=True)
        from .cloud_worker import run_phase
        try:
            state = advance_local(root, str(model), run_phase, cancelled)
            print(json.dumps(dict(status=state["status"], phase=state["phase"])), flush=True)
        except BaseException as exc:
            path = root / "state.json"
            state = json.loads(path.read_text()) if path.exists() else {"phase":0}
            state.update(status="paused_error", error=f"{type(exc).__name__}: {exc}")
            atomic_json(path, state)
            raise
        finally:
            write_local_report(root)


def process_running(root):
    path = root / "worker.json"
    if not path.exists():
        return False
    pid = json.loads(path.read_text())["pid"]
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        return b"evaluation.local_suite" in command and str(root).encode() in command
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "start", "run", "status", "stop", "report"])
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=HERE / "modal_runs/ten_qwen35_recovered_v1/bundle")
    parser.add_argument("--model", type=Path, default=HERE.parent / "models/Qwen3.5-4B")
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--prompt-cap", type=int, default=8192)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.mode in {"prepare", "start"}:
        manifest = prepare_local(args.source, root, args.model, args.per_family, args.prompt_cap)
        if args.mode == "prepare":
            print(json.dumps(dict(run_id=manifest["run_id"], samples=len(manifest["rows"]))))
            return
        if process_running(root):
            print("Local worker is already running; no duplicate started")
            return
        (root / "STOP_REQUESTED").unlink(missing_ok=True)
        environment = {**os.environ, "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
                       "HF_DEACTIVATE_ASYNC_LOAD":"1", "HF_HUB_DISABLE_PROGRESS_BARS":"1",
                       "OMP_NUM_THREADS":"8", "TOKENIZERS_PARALLELISM":"true", "RAYON_NUM_THREADS":"8"}
        with (root / "worker.log").open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", "-m", "evaluation.local_suite", "run", "--output", str(root)],
                cwd=root / "input/code", env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(status="local_worker_launched", pid=child.pid, output=str(root))))
    elif args.mode == "run":
        run_local(root)
    elif args.mode == "status":
        path = root / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        state.pop("partial_times", None)
        print(json.dumps(dict(worker_running=process_running(root), state=state, output=str(root)), indent=2))
    elif args.mode == "stop":
        (root / "STOP_REQUESTED").touch()
        print("Local stop requested at the next checkpoint; saved results are retained")
    else:
        print(json.dumps(write_local_report(root)))


if __name__ == "__main__":
    main()
