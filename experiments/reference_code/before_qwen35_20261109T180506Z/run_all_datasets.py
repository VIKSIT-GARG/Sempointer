#!/usr/bin/env python3
"""Catalog, validate, evaluate, and report the downloaded dataset suite."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

from evaluation.data import discover_tasks, iter_examples

HERE=Path(__file__).resolve().parent


def main():
    # Evaluation never starts an implicit model/dataset download. Download the
    # chosen checkpoint separately before launching this offline runner.
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD","1")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["plan","validate","model-check","run","summarize","beir"])
    p.add_argument("--data-root",type=Path,default=HERE.parent/"datasets")
    p.add_argument("--output",type=Path,default=HERE/"results/downloaded_suite")
    p.add_argument("--families",default="all",help="Comma-separated catalog families; all selects every QA/generation adapter")
    p.add_argument("--tasks",default="all",help="Comma-separated exact task names from plan/catalog.json")
    p.add_argument("--systems",default="sempointer,pointer_off,bm25_2,bm25_4,bm25_matched_budget,full_context,no_context")
    local_model=HERE.parent/"models/Qwen2.5-3B-Instruct"
    p.add_argument("--model",default=str(local_model) if (local_model/"config.json").exists() else "Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dtype",choices=["nf4","bf16"],default="nf4")
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=143)
    p.add_argument("--partition",choices=["dev","test","all"],default="test")
    p.add_argument("--exclude-groups",type=Path,help="JSON list of document/conversation group IDs excluded before sampling")
    p.add_argument("--examples-per-group",type=int,default=0,help="Maximum questions per document/conversation; 0 means no group cap")
    p.add_argument("--expand-coverage",type=float,default=.8)
    p.add_argument("--ambiguity-coverage",type=float,default=.95)
    p.add_argument("--expand-score-ratio",type=float,default=.85)
    p.add_argument("--limit",type=int,default=10,help="Maximum questions per source/shard AFTER partitioning; 0 means all")
    p.add_argument("--include-train",action="store_true",help="Diagnostic only; training splits never silently included")
    p.add_argument("--validate-all",action="store_true",help="Validate every record rather than one sample per source; can take hours")
    p.add_argument("--max-new-tokens",type=int,default=0,help="0 uses task-specific caps recorded in catalog")
    p.add_argument("--max-prompt-tokens",type=int,default=8192)
    p.add_argument("--max-source-tokens",type=int,default=262144,help="0 disables source-size guard")
    p.add_argument("--max-source-chars",type=int,default=2000000,help="0 disables pre-tokenization RAM guard")
    p.add_argument("--ruler-samples",type=int,default=10)
    p.add_argument("--context-lengths",default="4096,8192,16384")
    p.add_argument("--beir-candidates",type=int,default=20)
    p.add_argument("--keep-beir-indices",action="store_true",help="Retain generated corpus indices; requires extra disk space")
    args=p.parse_args()
    if args.beir_candidates<1:p.error("--beir-candidates must be positive")
    if args.examples_per_group<0:p.error("--examples-per-group must be nonnegative")
    if not .5<=args.expand_coverage<=args.ambiguity_coverage<=1 or not 0<=args.expand_score_ratio<=1:
        p.error("Require .5 <= expand coverage <= ambiguity coverage <= 1 and score ratio in [0,1]")
    args.router_options={"expand_coverage":args.expand_coverage,"ambiguity_coverage":args.ambiguity_coverage,
                         "expand_score_ratio":args.expand_score_ratio}
    args.excluded_group_ids=[]
    if args.exclude_groups:
        try:
            groups=json.loads(args.exclude_groups.read_text())
            if not isinstance(groups,list) or any(not isinstance(x,str) for x in groups):
                raise ValueError("expected a JSON list of strings")
            args.excluded_group_ids=sorted(set(groups))
        except (OSError,ValueError) as exc:p.error(f"Invalid exclusion manifest: {exc}")
    if args.limit<0 or args.ruler_samples<1 or args.max_new_tokens<0 or args.max_prompt_tokens<1 or args.max_source_chars<0 or args.max_source_tokens<0:
        p.error("Limits must be nonnegative; prompt cap and ruler sample count must be positive")
    args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=="model-check":
        from evaluation.provenance import model_provenance
        try:
            info={"model":args.model,"tokenizer_and_config_available":True,**model_provenance(args.model)}
        except Exception as exc:
            info={"model":args.model,"tokenizer_and_config_available":False,"error":str(exc),
                  "next_step":"bash run_downloaded_suite.sh model"}
        (args.output/"model_preflight.json").write_text(json.dumps(info,indent=2)+"\n")
        print(json.dumps(info,indent=2))
        if not info["tokenizer_and_config_available"] or not info.get("weights",{}).get("available"):
            print("Model setup incomplete: bash run_downloaded_suite.sh model")
            raise SystemExit(1)
        return
    if args.mode=="summarize":
        from evaluation.runner import RunStore,summarize
        config=json.loads((args.output/"config.json").read_text())
        with RunStore(args.output,config) as store:store.export();summarize(store)
        return
    if args.mode=="beir":
        from evaluation.beir import run_beir
        run_beir(args)
        return
    tasks,coverage=discover_tasks(args.data_root,args.include_train,args.seed,args.ruler_samples,tuple(int(x) for x in args.context_lengths.split(",")))
    if args.families!="all":tasks=[t for t in tasks if t.family in args.families.split(",")]
    if args.tasks!="all":tasks=[t for t in tasks if t.task in args.tasks.split(",")]
    if not tasks:p.error("No tasks matched; inspect the plan catalog")
    catalog={"tasks":[t.to_dict() for t in tasks],"download_coverage":coverage,
             "notes":["BEIR is a separate retrieval-only mode.","Training splits excluded unless --include-train.","Source-only generators are not fixed test sets.","Missing-label and safety-limit examples are logged as skipped, not scored."]}
    # Catalog is diagnostic output only; run-specific provenance is checked
    # before anything in an existing run directory is rewritten.
    if args.mode=="plan":
        (args.output/"catalog.json").write_text(json.dumps(catalog,indent=2)+"\n")
        from collections import Counter
        print(json.dumps(dict(Counter(t.family for t in tasks)),indent=2))
        print("Catalog:",args.output/"catalog.json")
        return
    if args.mode=="validate":
        checks=[]
        for task in tasks:
            try:
                iterator=iter_examples(task)
                ex=next(iterator)
                checks.append({"task":task.key,"source":str(task.path),"member":task.member,"status":"skip" if ex.skip_reason else "ok",
                               "reason":ex.skip_reason,"metric":ex.metric,"protocol":ex.protocol,"context_chars":len(ex.context),"checked_examples":1})
                if args.validate_all:
                    reasons={ex.skip_reason:1} if ex.skip_reason else {}
                    for other in iterator:
                        checks[-1]["checked_examples"]+=1
                        if other.skip_reason:reasons[other.skip_reason]=reasons.get(other.skip_reason,0)+1
                    checks[-1]["skip_reasons"]=reasons
            except StopIteration:checks.append({"task":task.key,"status":"empty"})
            except Exception as exc:checks.append({"task":task.key,"source":str(task.path),"member":task.member,"status":"error","error":f"{type(exc).__name__}: {exc}"})
        (args.output/"validation.json").write_text(json.dumps(checks,indent=2)+"\n")
        from collections import Counter
        print(dict(Counter(x["status"] for x in checks)))
        for x in checks:
            if x["status"]=="error":print(x)
        if any(x["status"]=="error" for x in checks):raise SystemExit(1)
        return
    from evaluation.controls import SYSTEMS
    args.systems=args.systems.split(",")
    if len(set(args.systems))!=len(args.systems) or set(args.systems)-set(SYSTEMS):p.error("Unknown or duplicate system")
    from evaluation.provenance import code_fingerprint,package_versions,model_provenance
    try:resolved_model=model_provenance(args.model)
    except OSError:
        p.error("Model config/tokenizer not cached. Run 'bash run_downloaded_suite.sh model' first, or pass --model /path/to/local/HF/checkpoint")
    if not resolved_model["weights"]["available"]:p.error("Model weight shards missing. Finish 'bash run_downloaded_suite.sh model' before evaluation")
    fingerprints={}
    known={x["path"]:x.get("sha256") for x in coverage}
    source_paths=set()
    for task in tasks:
        if task.path.is_dir():source_paths.update(p for p in task.path.rglob("*") if p.is_file())
        else:source_paths.add(task.path)
    for path in sorted(source_paths,key=str):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(4*1024**2),b""):h.update(chunk)
        actual=h.hexdigest()
        if known.get(str(path)) and known[str(path)]!=actual:raise ValueError(f"Dataset checksum mismatch: {path}")
        fingerprints[str(path)]=actual
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in {"mode","output","validate_all"}}
    settings.update(source_sha256=fingerprints,code_sha256=code_fingerprint(),packages=package_versions(),
                    model_provenance=resolved_model,
                    router={"block_size":512,"pointer_tokens":16,"min_m":2,"max_m":4,**args.router_options,"resolved_cap":2560,"semantic_rescue":False,"chunking":"strict token boundaries"})
    from evaluation.runner import RunStore,run_tasks
    with RunStore(args.output,settings) as store:
        (args.output/"catalog.json").write_text(json.dumps(catalog,indent=2)+"\n")
        from engine.llm import LLMEngine
        from provenance import environment_record
        engine=LLMEngine(model_id=args.model,dtype=args.dtype,device=args.device,seed=args.seed,max_new_tokens=args.max_new_tokens or 128)
        if getattr(engine.model.config,"_commit_hash",None)!=settings["model_provenance"]["model_commit"]:
            raise ValueError("Model revision changed during loading; refusing to append results")
        info={"engine":engine.info(),"environment":environment_record(),"model_commit":getattr(engine.model.config,"_commit_hash",None)}
        with (args.output/"environments.jsonl").open("a") as f:f.write(json.dumps(info,default=str)+"\n")
        result=run_tasks(tasks,store,engine,args)
        if result["status_counts"].get("error",0):raise SystemExit(2)


if __name__=="__main__":main()
