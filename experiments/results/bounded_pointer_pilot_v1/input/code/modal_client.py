#!/usr/bin/env python3
"""Upload, submit and inspect the detached suite without exposing credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import modal
from evaluation.cloud_policy import digest

VOLUME_NAME="sempointer-ten-benchmarks-v1"
LOCK_NAME="sempointer-ten-suite-leases-v1"


def read_remote(volume,path):
    try:return json.loads(b"".join(volume.read_file(path)))
    except (FileNotFoundError,modal.exception.NotFoundError):return None


def reclaim_completed(leases,run_id):
    """Only release a stale lease after the provider reports its call terminal."""
    active=leases.get("active")
    if not active:return
    if active["run_id"]!=run_id:raise ValueError("Another suite owns the workspace lease: "+active["run_id"])
    sequence=active.get("sequence",0)
    ticket=leases.get(f"ticket:{run_id}:{sequence}") if sequence else active.get("setup_call")
    if not ticket:raise ValueError("Dispatch state is uncertain. Inspect the Modal app before retrying; no duplicate worker was started.")
    try:modal.FunctionCall.from_id(ticket).get(timeout=0)
    except modal.exception.FunctionTimeoutError:pass  # execution itself timed out
    except modal.exception.TimeoutError as exc:raise ValueError("The existing cloud call is still running. Use status instead.") from exc
    except (modal.exception.ExecutionError,modal.exception.RemoteError):pass
    # Compare again: a completed chunk may already have dispatched its successor.
    if leases.get("active")!=active:raise ValueError("The job advanced while checking; use status instead.")
    leases.pop("active",None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["start","resume","status","download","cancel"])
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--workspace-budget-confirmed",action="store_true")
    args=p.parse_args();bundle=args.bundle.resolve()
    manifest=json.loads((bundle/"manifest.json").read_text());run_id=manifest["run_id"]
    if digest({k:v for k,v in manifest.items() if k!="run_id"})[:20]!=run_id:p.error("Manifest checksum mismatch")
    volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=args.mode in {"start","resume"},version=2)
    leases=modal.Dict.from_name(LOCK_NAME,create_if_missing=args.mode in {"start","resume"})
    remote=f"runs/{run_id}"
    if args.mode in {"start","resume"}:
        if not args.workspace_budget_confirmed:
            p.error("Launch blocked: set this workspace's total usage limit to at most $25 and net spend limit to $0 in Modal Usage & Billing, then pass --workspace-budget-confirmed. This flag records your confirmation; it does not read or set the server-side limits.")
        for row in manifest["rows"]+manifest["development"]:
            if hashlib.sha256((bundle/row["path"]).read_bytes()).hexdigest()!=row["sha256"]:p.error("Prepared sample changed")
        for name,sha in manifest["code_files"].items():
            if hashlib.sha256((bundle/"code"/name).read_bytes()).hexdigest()!=sha:p.error("Frozen code changed")
        if args.mode=="resume":
            reclaim_completed(leases,run_id)
            leases.pop("cancel:"+run_id,None)
        elif leases.get("active"):
            p.error("A suite lease already exists. Use status, or resume after the previous call has stopped.")
        # Upload only the frozen sample/code bundle, never local credentials,
        # model weights, virtualenvs, old results or the complete dataset tree.
        if read_remote(volume,remote+"/input/manifest.json") is None:
            with volume.batch_upload() as batch:batch.put_directory(bundle,"/"+remote+"/input")
        command=[sys.executable,"-m","modal","run","--detach",str(bundle/"code/modal_suite.py"),"--run-id",run_id]
        print("Starting detached run",run_id,flush=True)
        raise SystemExit(subprocess.call(command,cwd=bundle/"code"))
    if args.mode=="status":
        from evaluation.inference_policy import shared_allowance_path
        account=shared_allowance_path(".",manifest).as_posix()
        print(json.dumps({"run_id":run_id,"active":leases.get("active"),
            "state":read_remote(volume,remote+"/state.json"),
            "allowance":read_remote(volume,account),
            "budget_account_run_id":manifest.get("budget_account_run_id",run_id),
            "pilot_projection":read_remote(volume,remote+"/results/pilot_cost_projection.json"),
            "setup_error":read_remote(volume,remote+"/setup_error.json")},indent=2))
    elif args.mode=="cancel":
        leases.put("cancel:"+run_id,True)
        print("Stop requested at the next checkpoint. Committed results will be retained.")
    else:
        destination=bundle.parent/"downloaded_results"
        command=[sys.executable,"-m","modal","volume","get",VOLUME_NAME,remote+"/results",str(destination),"--force"]
        raise SystemExit(subprocess.call(command))


if __name__=="__main__":main()
