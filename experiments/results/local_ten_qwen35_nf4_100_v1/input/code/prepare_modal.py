#!/usr/bin/env python3
"""Freeze a family-wide sample locally before any cloud computation."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import threading

from evaluation.cloud_policy import FAMILIES,SYSTEMS,CandidatePool,atomic_json,digest,balanced_counts,pilot_rows
from evaluation.data import discover_tasks,iter_examples,partition_for

HERE=Path(__file__).resolve().parent
LB_GROUPS={name:group for group,names in {
    "single_document_qa":"narrativeqa qasper multifieldqa_en multifieldqa_zh",
    "multi_document_qa":"hotpotqa 2wikimqa musique dureader",
    "summarization":"gov_report qmsum multi_news vcsum",
    "few_shot":"trec triviaqa samsum lsht",
    "synthetic":"passage_count passage_retrieval_en passage_retrieval_zh",
    "code":"lcc repobench-p"}.items() for name in names.split()}


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b""):h.update(chunk)
    return h.hexdigest()


def code_files(root=HERE):
    files=[root/name for name in ["run_all_datasets.py","reviewer_suite.py","provenance.py",
                                  "prepare_modal.py","modal_suite.py","modal_client.py"]]
    for folder in ["engine","evaluation","sempointer","metrics","benchmarks"]:
        files.extend(sorted((root/folder).glob("*.py")))
    return {str(p.relative_to(root)):file_hash(p) for p in files}


def selected_source(task):
    if task.family not in FAMILIES:return False
    if task.family=="longbench":return not task.task.endswith("_e")
    if task.family=="longmem":return task.path.name=="longmemeval_s_cleaned.json"
    if task.family=="scrolls":return task.split=="validation"  # public labels
    if task.family=="ruler":return task.task in "niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 vt cwe".split()
    if task.family=="babilong":return task.path.stem in {"4k","8k","16k","32k","64k","128k","256k"}
    return True


def stratum(task,ex):
    family=task.family
    if family=="longbench":return LB_GROUPS[task.task]
    if family=="longbench_v2":return str(ex.meta.get("domain","unknown"))
    if family=="locomo":return "category_"+str(ex.meta["category"])
    if family=="longmem":return str(ex.meta.get("question_type","unknown"))
    if family=="ruler":return task.task+"/"+str(task.extra["context_length"])
    if family=="babilong":
        size=int(task.path.stem[:-1]);bucket="4k-16k" if size<=16 else "32k-64k" if size<=64 else "128k-256k"
        return task.path.parent.name+"/"+bucket
    return task.task


def output_cap(task):
    if task.family=="ruler":return 512 if task.task=="cwe" else 128
    if task.family in {"locomo","longmem"}:return 128
    return task.max_new_tokens


def verify_bundle(root):
    root=Path(root);manifest=json.loads((root/"manifest.json").read_text())
    if digest({k:v for k,v in manifest.items() if k!="run_id"})[:20]!=manifest["run_id"]:
        raise ValueError("Bundle manifest checksum changed")
    if manifest["code_files"]!=code_files(root/"code"):raise ValueError("Frozen source snapshot changed")
    for row in manifest["rows"]+manifest["development"]:
        path=root/row["path"]
        if file_hash(path)!=row["sha256"]:raise ValueError("Prepared example changed: "+str(path))
    evidence=manifest.get("configuration_evidence")
    if evidence and file_hash(root/evidence["path"])!=evidence["sha256"]:raise ValueError("Configuration evidence changed")
    return manifest


def scan_family(tasks, tokenizer, counter, seed, cap, excluded, cache_dir, code_signature, stop_event=None):
    """Keep original within-family ordering and heap policy while batching counts."""
    family=tasks[0].family
    sources=[]
    for t in tasks:
        paths=sorted(p for p in t.path.rglob("*") if p.is_file()) if t.path.is_dir() else [t.path]
        sources.append({**t.to_dict(),"files":[[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in paths]})
    signature=digest([sources,seed,cap,sorted(excluded),code_signature,counter.fingerprint])
    checkpoint=cache_dir/(family+".json")
    if checkpoint.exists():
        cached=json.loads(checkpoint.read_text())
        if cached.get("signature")==signature:
            print(f"Sampling {family}: reused completed family ({len(cached['rows'])} questions)",flush=True)
            return cached["rows"],Counter(cached["stats"])
    pool=CandidatePool(seed,cap);stats=Counter();started=time.monotonic()
    print(f"Sampling {family}: scanning {len(tasks)} sources",flush=True)
    pending=[];pending_chars=0;last_progress=started
    def flush():
        nonlocal pending_chars
        if not pending:return
        counts=counter.count_many([context for row,context in pending])
        for (row,context),count in zip(pending,counts):
            # Earlier rows in this batch may have improved the heap threshold.
            if not pool.competitive(row):continue
            if count>262144:stats["source_token_limit"]+=1;continue
            row.update(source_tokens=count,context_sha256=hashlib.sha256(context.encode()).hexdigest())
            pool.add(row)
        pending.clear();pending_chars=0
    for i,t in enumerate(tasks):
        key=digest(t.to_dict())
        for ex in iter_examples(t,tokenizer=tokenizer):
            if stop_event is not None and stop_event.is_set():raise InterruptedError("Preparation stopped")
            stats["scanned"]+=1
            if time.monotonic()-last_progress>=10:
                print(f"Sampling {family} {i+1}/{len(tasks)}: {stats['scanned']} questions scanned; {time.monotonic()-started:.0f}s",flush=True)
                last_progress=time.monotonic()
            reason=ex.skip_reason or ("exposed_group" if ex.group_id in excluded else None)
            if not reason and partition_for(ex)!="test":reason="development_partition"
            if not reason and len(ex.context)>2_000_000:reason="source_character_limit"
            if reason:stats[reason]+=1;continue
            row={"id":ex.id,"family":ex.family,"stratum":stratum(t,ex),"source":key,
                 "semantic_id":digest([ex.family,ex.context,ex.question]),"output_cap":output_cap(t)}
            if not pool.competitive(row):continue
            pending.append((row,ex.context));pending_chars+=len(ex.context)
            if len(pending)>=32 or pending_chars>=4_000_000:flush()
        flush()
        print(f"Sampling {family} {i+1}/{len(tasks)} done: {t.task} ({stats['scanned']} scanned)",flush=True)
    rows=pool.selected()
    atomic_json(checkpoint,{"signature":signature,"rows":rows,"stats":dict(stats)})
    return rows,stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=HERE/"modal_runs/ten_qwen35_v1/bundle")
    p.add_argument("--per-family",type=int,default=300)
    p.add_argument("--seed",type=int,default=20261114)
    p.add_argument("--data-root",type=Path,default=HERE.parent/"datasets")
    cpu_count=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count() or 1
    p.add_argument("--workers",type=int,default=min(8,cpu_count),help="Concurrent family scans (default: up to 8)")
    p.add_argument("--tokenizer-threads",type=int,default=int(os.environ.get("RAYON_NUM_THREADS",cpu_count)),
                   help="Threads in the shared Rust tokenization pool (default: available CPUs)")
    args=p.parse_args()
    if not 1<=args.per_family<=300:p.error("Family cap must be between 1 and 300")
    if args.workers<1 or args.tokenizer_threads<1:p.error("Worker and thread counts must be positive")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    preparation_lock=(args.output.parent/"prepare.lock").open("a")
    try:fcntl.flock(preparation_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another preparation owns this run directory; waiting for its frozen bundle.",flush=True)
        fcntl.flock(preparation_lock,fcntl.LOCK_EX)
    if (args.output/"manifest.json").exists():
        manifest=verify_bundle(args.output)
        if manifest["per_family"]!=args.per_family or manifest["seed"]!=args.seed:p.error("Sampling settings changed; use a new output directory")
        print("Frozen bundle verified:",manifest["run_id"],len(manifest["rows"]),"test questions",flush=True)
        preparation_lock.close();return
    initial_code=code_files()
    os.environ["RAYON_NUM_THREADS"]=str(args.tokenizer_threads)
    os.environ["TOKENIZERS_PARALLELISM"]="true"
    from transformers import AutoTokenizer
    from evaluation.preparation import ExactTokenCounter
    model=HERE.parent/"models/Qwen3.5-4B"
    metadata=model/".cache/huggingface/download/config.json.metadata"
    revision=metadata.read_text().splitlines()[0] if metadata.exists() else ""
    if not re.fullmatch(r"[0-9a-f]{40}",revision):raise ValueError("Download the pinned checkpoint first: bash run_qwen35_suite.sh model")
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    if not tok.encode("tokenizer readiness check"):raise ValueError("Incomplete tokenizer download")
    from reviewer_suite import build_exclusions
    excluded,exposure=build_exclusions(args.data_root,args.output.resolve())
    excluded=set(excluded)
    tasks,_=discover_tasks(args.data_root,seed=args.seed,ruler_samples=30)
    tasks=[t for t in tasks if selected_source(t)]
    stats=defaultdict(Counter);task_by_key={digest(t.to_dict()):t for t in tasks}
    grouped={f:[t for t in tasks if t.family==f] for f in FAMILIES}
    cache_dir=args.output.parent/"preparation_cache"
    tokenizer_sha=file_hash(model/"tokenizer.json")
    print(f"Preparation: {args.workers} concurrent families; {args.tokenizer_threads} shared tokenizer CPU threads; persistent token cache",flush=True)
    started=time.monotonic();selected_by_family={};stop_event=threading.Event()
    with ExactTokenCounter(tok,cache_dir/"token_counts.sqlite",tokenizer_sha) as counter:
        with ThreadPoolExecutor(max_workers=args.workers) as workers:
            futures={workers.submit(scan_family,group,tok,counter,args.seed,args.per_family,excluded,
                                    cache_dir,digest(initial_code),stop_event):f for f,group in grouped.items() if group}
            try:
                for future in as_completed(futures):
                    family=futures[future]
                    selected_by_family[family],stats[family]=future.result()
            except BaseException:
                stop_event.set()
                for future in futures:future.cancel()
                raise
        performance={"scan_seconds":time.monotonic()-started,"workers":args.workers,
                     "tokenizer_threads":args.tokenizer_threads,"token_counts":counter.stats()}
    selected=[row for f in FAMILIES for row in selected_by_family.get(f,[])]
    atomic_json(args.output.parent/"preparation_performance.json",performance)
    wanted=defaultdict(dict)
    for row in selected:wanted[row["source"]][row["id"]]=row
    args.output.mkdir(parents=True,exist_ok=True)
    stored=[]
    def save(ex,row,stage):
        relative=f"examples/{stage}/{digest(ex.id)}.json"
        ex.meta={**ex.meta,"output_cap":row["output_cap"],"source_tokens":row["source_tokens"],
                 "sampling_stratum":row["stratum"],"original_source":str(task_by_key[row["source"]].path) if stage=="test" else "seeded local RULER generator"}
        atomic_json(args.output/relative,{"example":asdict(ex)})
        return {**row,"path":relative,"sha256":file_hash(args.output/relative),"task":ex.task,"group_id":ex.group_id}
    def materialize_family(family):
        result=[]
        for key,original in wanted.items():
            t=task_by_key[key]
            if t.family!=family:continue
            items=dict(original)
            for ex in iter_examples(t,tokenizer=tok):
                if stop_event.is_set():raise InterruptedError("Preparation stopped")
                if ex.id not in items:continue
                row=items.pop(ex.id)
                if hashlib.sha256(ex.context.encode()).hexdigest()!=row["context_sha256"]:raise ValueError("Source changed while preparing")
                result.append(save(ex,row,"test"))
                if not items:break
            if items:raise ValueError("Selected example disappeared")
        print(f"Prepared {family}: {len(result)} questions saved",flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        futures=[workers.submit(materialize_family,f) for f in FAMILIES]
        try:
            for future in as_completed(futures):stored.extend(future.result())
        except BaseException:
            stop_event.set()
            for future in futures:future.cancel()
            raise
    development=[]
    dev_tasks,_=discover_tasks(args.data_root,seed=args.seed+1,ruler_samples=3,lengths=(4096,8192))
    for t in dev_tasks:
        if t.family!="ruler" or not selected_source(t):continue
        for ex in iter_examples(t,tokenizer=tok):
            row={"id":ex.id,"family":ex.family,"stratum":stratum(t,ex),"source":"development",
                 "output_cap":output_cap(t),"source_tokens":len(tok.encode(ex.context,add_special_tokens=False))}
            development.append(save(ex,row,"development"))
    # Interleave families and strata so a budget interruption does not silently
    # leave the final benchmark families entirely untested.
    queues={f:sorted((r for r in stored if r["family"]==f),key=lambda r:digest([args.seed,r["id"]])) for f in FAMILIES}
    ordered=[]
    while any(queues.values()):
        for f in FAMILIES:
            if queues[f]:ordered.append(queues[f].pop())
    cwe=[r for r in ordered if r["family"]=="ruler" and r["task"]=="cwe"]
    counts=balanced_counts(dict(Counter(r["stratum"] for r in cwe)),30)
    audit=[]
    for r in cwe:
        if counts[r["stratum"]]:audit.append(r["id"]);counts[r["stratum"]]-=1
    if initial_code!=code_files():raise ValueError("Source code changed during preparation; restart before launching")
    for name,sha in initial_code.items():
        target=args.output/"code"/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(HERE/name,target)
        if file_hash(target)!=sha:raise ValueError("Source changed while freezing the bundle")
    manifest={"schema":1,"per_family":args.per_family,"seed":args.seed,"dtype":"nf4",
        "model_id":"Qwen/Qwen3.5-4B","model_revision":revision,
        "tokenizer_file_sha256":file_hash(model/"tokenizer.json"),"code_files":initial_code,
        "families":list(FAMILIES),"systems":list(SYSTEMS),"rows":ordered,"development":development,
        "pilot_ids":[r["id"] for r in pilot_rows(ordered)],"cwe_audit_ids":audit,
        "sampling_report":{f:{"selected":sum(r["family"]==f for r in ordered),
            "selected_strata":dict(Counter(r["stratum"] for r in ordered if r["family"]==f)),
            "scan_counts":dict(stats[f])} for f in FAMILIES},"exposure":exposure,
        "limits":{"source_chars":2_000_000,"source_tokens":262144,"prompt_tokens":16384},
        "notes":["Family-wide cap; balanced strata; labeled test partition only.",
            "LongBench base tasks only; LongMemEval-S only; SCROLLS public validation labels.",
            "BABILong 4K–256K configurations; longer releases excluded explicitly.",
            "The source history is never truncated. Over-limit sources are excluded before selection.",
            "Original/packaged overlaps remain; no claim of ten independent source corpora."]}
    manifest["run_id"]=digest(manifest)[:20]
    atomic_json(args.output/"manifest.json",manifest)
    print("Prepared",len(ordered),"test questions;",len(development),"development questions; run",manifest["run_id"],flush=True)
    print(json.dumps(manifest["sampling_report"],indent=2))
    preparation_lock.close()


if __name__=="__main__":main()
