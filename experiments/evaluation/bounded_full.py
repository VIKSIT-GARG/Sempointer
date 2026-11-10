"""Freeze and run the 100-per-family extension without changing pilot settings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

from .cloud_policy import FAMILIES,atomic_json,digest
from .bounded_experiment import _sample

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE/"results/bounded_pointer_full_v1"


def full_rows(rows,excluded,development_groups,cap=100,seed=20261118):
    if cap<1:raise ValueError("Per-family cap must be positive")
    selected=[]
    for family in FAMILIES:
        available=[r for r in rows if r["family"]==family]
        if not available:continue
        available=[r for r in available if r["id"] not in excluded and r["group_id"] not in development_groups]
        choice=_sample(available,cap,seed)
        if len(choice)!=cap:raise ValueError("Insufficient fresh questions for "+family)
        selected.extend(choice)
    return selected


def prepare_full(root):
    from prepare_modal import code_files,file_hash,verify_bundle
    root=Path(root).resolve()
    if (root/"input/manifest.json").exists():
        manifest=verify_bundle(root/"input")
        if manifest.get("schema")!="bounded_pointer_full_v1" or len(manifest["rows"])!=1000:
            raise ValueError("Destination contains another experiment")
        return manifest
    pilot_root=HERE/"results/bounded_pointer_pilot_v1"
    original_root=HERE/"results/local_ten_qwen35_nf4_100_v1"
    source=HERE/"modal_runs/ten_qwen35_recovered_v1/bundle"
    pilot=verify_bundle(pilot_root/"input")
    original=verify_bundle(original_root/"input");pool=verify_bundle(source)
    if json.loads((pilot_root/"state.json").read_text()).get("status")!="complete":
        raise ValueError("The pilot must be complete before starting the larger evaluation")
    selection_path=pilot_root/"results/pointer_selection.json"
    choices=json.loads(selection_path.read_text())
    if set(choices)!={"1024","2048","4096"} or any(
            c.get("source")!="development_only" or c.get("pointer_cap")!=128 for c in choices.values()):
        raise ValueError("Unexpected pilot development choices; no settings were changed")
    excluded={r["id"] for r in original["rows"]+original["development"]+pilot["rows"]+pilot["development"]}|set(pool["pilot_ids"])
    development_groups={r["group_id"] for r in pilot["development"]}
    selected=full_rows(pool["rows"],excluded,development_groups)
    if Counter(r["family"] for r in selected)!=dict.fromkeys(pilot["families"],100):
        raise ValueError("Expected exactly 100 fresh questions from every family")
    snapshot=code_files()
    for name in ["evaluation/bounded.py","evaluation/controls.py","evaluation/runner.py",
                 "evaluation/qwen35.py","sempointer/v2_pipeline.py","sempointer/pipeline.py"]:
        if snapshot[name]!=pilot["code_files"][name]:
            raise ValueError("The inference algorithm changed since the pilot: "+name)
    staging=root/"input.preparing"
    if staging.exists():raise ValueError("Incomplete preparation preserved at "+str(staging))
    staging.mkdir(parents=True)
    for row in selected:
        target=staging/row["path"];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/row["path"],target)
    for name,checksum in snapshot.items():
        target=staging/"code"/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(HERE/name,target)
        if file_hash(target)!=checksum:raise ValueError("Runtime changed while freezing")
    shutil.copy2(selection_path,staging/"pointer_selection_source.json")
    previous_groups={r["group_id"] for r in original["rows"]+pilot["rows"]}
    manifest={k:v for k,v in pilot.items() if k not in {"run_id","notes"}}
    manifest.update(schema="bounded_pointer_full_v1",evaluation_only=True,parent_run_id=pilot["run_id"],
        rows=selected,development=[],per_family=100,sampling_seed=20261118,
        frozen_pointer_selection=choices,selection_source_run_id=pilot["run_id"],
        configuration_evidence=dict(path="pointer_selection_source.json",sha256=file_hash(staging/"pointer_selection_source.json")),
        excluded_ids=sorted(excluded),excluded_development_groups=sorted(development_groups),code_files=snapshot,
        overlap_with_prior_groups={"test":dict(Counter(r["family"] for r in selected if r["group_id"] in previous_groups))},
        sampling_report={f:dict(selected=100,selected_strata=dict(Counter(r["stratum"] for r in selected if r["family"]==f)),
            eligible_after_exclusions=sum(r["family"]==f and r["id"] not in excluded and r["group_id"] not in development_groups for r in pool["rows"])) for f in pilot["families"]},
        notes=["Exactly 100 fresh question IDs per family; pilot settings and inference algorithms remain fixed.",
               "Pilot development groups are excluded, but documents may overlap older test/diagnostic evaluations.",
               "Three budgets evaluate the same 1,000 questions; they are not 3,000 independent questions.",
               "No new development, parameter selection, model training or cloud execution.",
               "Report metric-specific paired comparisons and coverage; do not pool unrelated score scales."])
    manifest["run_id"]=digest(manifest)[:20]
    atomic_json(staging/"manifest.json",manifest);verify_bundle(staging)
    if snapshot!=code_files():raise ValueError("Runtime changed during preparation")
    staging.rename(root/"input")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","start","run","status","stop","report"])
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args();root=args.output.resolve()
    if args.mode in {"prepare","start"}:
        manifest=prepare_full(root)
        if args.mode=="prepare":
            print(json.dumps(dict(run_id=manifest["run_id"],questions=len(manifest["rows"]),
                per_family=dict(Counter(r["family"] for r in manifest["rows"])),
                pointer_caps={b:c["pointer_cap"] for b,c in manifest["frozen_pointer_selection"].items()},
                prior_source_group_overlap=manifest["overlap_with_prior_groups"])))
            return
    from .bounded_experiment import main as control
    sys.argv=[sys.argv[0],args.mode,"--output",str(root)]
    control()


if __name__=="__main__":main()
