#!/usr/bin/env python3
"""Prepare prompt-only precision measurements from an immutable sample bundle."""
import argparse
import json
from pathlib import Path

from evaluation.cloud_policy import FAMILIES,atomic_json,digest
from evaluation.controls import prepare,prompts_for_example
from evaluation.data import Example
from prepare_modal import file_hash,verify_bundle

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,default=HERE/"modal_runs/ten_qwen35_v1/bundle")
    parser.add_argument("--output",type=Path,default=HERE/"modal_runs/precision_pilot/input")
    args=parser.parse_args();manifest=verify_bundle(args.bundle)
    from transformers import AutoTokenizer
    from reviewer_suite import CANDIDATES
    tok=AutoTokenizer.from_pretrained(HERE.parent/"models/Qwen3.5-4B",local_files_only=True)
    wrap=lambda prompt:tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,
                                              add_generation_prompt=True,enable_thinking=False)
    pilot=[r for r in manifest["rows"] if r["id"] in set(manifest["pilot_ids"])]
    selected=[]
    for family in FAMILIES:
        group=sorted((r for r in pilot if r["family"]==family),key=lambda r:r["source_tokens"])
        if group:selected.append(group[len(group)//2])
    cwe=next((r for r in pilot if r["task"]=="cwe"),None)
    if cwe and cwe not in selected:selected.append(cwe)
    fitting=sorted((r for r in manifest["rows"] if 4000<=r["source_tokens"]<=14000),key=lambda r:r["source_tokens"])
    for row in fitting[-2:]:
        if row not in selected:selected.append(row)
    cases=[];thresholds=next(c["thresholds"] for c in CANDIDATES if c["name"]=="paper")
    for index,row in enumerate(selected):
        ex=Example(**json.loads((args.bundle/row["path"]).read_text())["example"])
        prepared=prepare(ex.context,tok,seed=manifest["seed"],router_options=thresholds)
        variants=prompts_for_example(prepared,ex.question,tok,prompt_formatter=wrap)
        systems=["sempointer","pointer_off"] if index<len(selected)-2 else ["full_context"]
        for system in systems:
            variant=variants[system]
            if variant.get("skip_reason") or variant["prompt_tokens"]>manifest["limits"]["prompt_tokens"]:continue
            cases.append(dict(case_id=digest([ex.id,system])[:20],example_id=ex.id,system=system,
                family=ex.family,task=ex.task,metric=ex.metric,gold=ex.gold,prompt=variant["prompt"],
                prompt_tokens=variant["prompt_tokens"],output_cap=row["output_cap"]))
        print(f"Precision prompts {index+1}/{len(selected)}: {ex.family}/{ex.task}",flush=True)
    ordered=sorted(cases,key=lambda r:r["prompt_tokens"])
    fixed=list(dict.fromkeys(ordered[int(i*(len(ordered)-1)/5)]["case_id"] for i in range(6)))
    sources=["evaluation/qwen35.py","evaluation/precision.py","evaluation/precision_run.py","modal_precision.py"]
    package=dict(schema=1,source_run_id=manifest["run_id"],model_id=manifest["model_id"],
        model_revision=manifest["model_revision"],tokenizer_sha256=manifest["tokenizer_file_sha256"],
        seed=manifest["seed"],cases=cases,fixed_cases=fixed,decode_tokens=32,
        code_files={name:file_hash(HERE/name) for name in sources},
        policy="Timing selects precision; no score or answer selects the configuration",
        comparison_order=["nf4","bf16","bf16","nf4"],maximum_reserved_dollars=3)
    package["pilot_id"]=digest(package)[:20]
    atomic_json(args.output/"pilot.json",package)
    print("Prepared",len(cases),"normal-answer prompts and",len(fixed),"equal-work prompts; pilot",package["pilot_id"],flush=True)


if __name__=="__main__":main()
