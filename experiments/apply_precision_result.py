#!/usr/bin/env python3
"""Derive a new frozen configuration from completed measured precision results."""
import argparse
import json
from pathlib import Path
import shutil

from evaluation.cloud_policy import atomic_json,digest
from evaluation.precision import choose_precision,choose_batch
from prepare_modal import HERE,code_files,file_hash,verify_bundle


def apply_result(source,report_path,pilot_path,destination):
    source,report_path,pilot_path,destination=map(Path,[source,report_path,pilot_path,destination])
    original=verify_bundle(source)
    report=json.loads(report_path.read_text());pilot=json.loads(pilot_path.read_text())
    if report.get("status")!="complete":raise ValueError("Precision pilot has not completed; no configuration applied")
    if digest({k:v for k,v in pilot.items() if k!="pilot_id"})[:20]!=pilot["pilot_id"]:raise ValueError("Pilot input changed")
    if report["pilot_id"]!=pilot["pilot_id"] or report["source_run_id"]!=original["run_id"] or pilot["source_run_id"]!=original["run_id"]:
        raise ValueError("Precision result belongs to another frozen sample bundle")
    if any(pilot[key]!=original[key] for key in ["model_id","model_revision"]):
        raise ValueError("Precision pilot used a different model or revision")
    precision=choose_precision(report["fixed_work"],pilot["fixed_cases"])
    batch=choose_batch(precision["selected_dtype"],report["batches"])
    if precision["selected_dtype"]!=report["precision_decision"]["selected_dtype"] or batch["batch_size"]!=report["batch_decision"]["batch_size"]:
        raise ValueError("Recorded configuration decision does not match raw measurements")
    if destination.exists():raise ValueError("Destination already exists; immutable bundles are never overwritten")
    destination.mkdir(parents=True)
    shutil.copytree(source/"examples",destination/"examples")
    snapshot=code_files()
    for name,sha in snapshot.items():
        target=destination/"code"/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(HERE/name,target)
        if file_hash(target)!=sha:raise ValueError("Runtime source changed while deriving configuration")
    if snapshot!=code_files():raise ValueError("Runtime source changed while deriving configuration")
    atomic_json(destination/"precision_comparison.json",dict(report=report,pilot_input=pilot))
    manifest={k:v for k,v in original.items() if k!="run_id"}
    manifest.update(dtype=precision["selected_dtype"],inference_batch_size=batch["batch_size"],
        code_files=snapshot,parent_run_id=original["run_id"],
        budget_account_run_id=original.get("budget_account_run_id",original["run_id"]),
        configuration_evidence=dict(path="precision_comparison.json",sha256=file_hash(destination/"precision_comparison.json")),
        precision_pilot_id=pilot["pilot_id"])
    manifest["notes"]=list(original.get("notes",[]))+[
        "Inference configuration selected by paired GPU timing; correctness scores did not select precision.",
        "Batched timing, when enabled, is throughput evidence and is not isolated request latency."]
    manifest["run_id"]=digest(manifest)[:20]
    atomic_json(destination/"manifest.json",manifest)
    verify_bundle(destination)
    atomic_json(HERE/"modal_runs/selected_inference_bundle.json",dict(run_dir=str(destination.parent.resolve()),run_id=manifest["run_id"]))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=HERE/"modal_runs/ten_qwen35_v1/bundle")
    parser.add_argument("--report",type=Path,default=HERE/"modal_runs/precision_pilot/results/comparison_warmed.json")
    parser.add_argument("--pilot",type=Path,default=HERE/"modal_runs/precision_pilot/input_v2/pilot.json")
    parser.add_argument("--output",type=Path,default=HERE/"modal_runs/ten_qwen35_measured_v1/bundle")
    args=parser.parse_args()
    manifest=apply_result(args.source,args.report,args.pilot,args.output)
    print(json.dumps(dict(run_id=manifest["run_id"],dtype=manifest["dtype"],batch_size=manifest["inference_batch_size"],
                         rows=len(manifest["rows"]),budget_account_run_id=manifest["budget_account_run_id"]),indent=2))


if __name__=="__main__":main()
