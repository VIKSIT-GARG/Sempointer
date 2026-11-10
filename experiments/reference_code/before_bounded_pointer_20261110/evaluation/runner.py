"""Durable per-attempt evidence and per-task, paired summaries."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import shutil
import time
import uuid

import numpy as np

from .controls import prepare, prompts_for_example, SYSTEMS
from .data import iter_examples, partition_for
from .metrics import score


class RunStore:
    def __init__(self, directory, config):
        import fcntl
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=(self.directory/"run.lock").open("a")
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            path=self.directory/"config.json"
            if path.exists() and json.loads(path.read_text())!=config:
                raise ValueError("Run settings changed; choose a different --output directory")
            path.write_text(json.dumps(config,indent=2)+"\n")
            self.db=sqlite3.connect(self.directory/"results.sqlite")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("CREATE TABLE IF NOT EXISTS latest(example_id TEXT,system TEXT,status TEXT,family TEXT,task TEXT,split TEXT,metric TEXT,protocol TEXT,score REAL,active_tokens REAL,latency_ms REAL,payload TEXT,PRIMARY KEY(example_id,system))")
            self.db.execute("CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,payload TEXT)")
            self.db.commit()
        except Exception:
            self.lock.close()
            raise

    def __enter__(self):return self

    def __exit__(self,*args):
        self.db.close()
        self.lock.close()

    def done(self,example_id,system):
        row=self.db.execute("SELECT status FROM latest WHERE example_id=? AND system=?",(example_id,system)).fetchone()
        return bool(row and row[0] in {"scored","skipped"})

    def record(self,row):
        if shutil.disk_usage(self.directory).free<2*1024**3:
            raise OSError("Results stopped at the 2 GiB disk reserve; committed SQLite records are preserved")
        row={"timestamp":time.time(),**row}
        payload=json.dumps(row,ensure_ascii=False,default=str)
        values=[row.get(k) for k in ["example_id","system","status","family","task","split","metric","protocol","score","active_tokens","latency_ms"]]
        with self.db:
            self.db.execute("INSERT INTO attempts(payload) VALUES(?)",(payload,))
            self.db.execute("INSERT OR REPLACE INTO latest VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",[*values,payload])
        with (self.directory/"events.jsonl").open("a") as f:
            f.write(payload+"\n")

    def export(self):
        for table,name in [("latest","predictions.jsonl"),("attempts","attempts.jsonl")]:
            temp=self.directory/(name+".tmp")
            with temp.open("w") as f:
                for (payload,) in self.db.execute("SELECT payload FROM "+table):f.write(payload+"\n")
            temp.replace(self.directory/name)

    def record_batch(self,row):
        with (self.directory/"batch_events.jsonl").open("a") as stream:
            stream.write(json.dumps({"timestamp":time.time(),**row})+"\n")


def interval(values,seed=143):
    arr=np.asarray(values,dtype=float)
    if not len(arr):return None
    rng=np.random.default_rng(seed)
    means=[float(rng.choice(arr,size=len(arr),replace=True).mean()) for _ in range(1000)]
    return [float(x) for x in np.percentile(means,[2.5,97.5])]


def summarize(store):
    keys=["family","task","split","metric","protocol"]
    tasks=[];paired=[]
    groups=store.db.execute("SELECT DISTINCT family,task,split,metric,protocol FROM latest").fetchall()
    for group in groups:
        where=" AND ".join(k+" IS ?" for k in keys)
        rows=store.db.execute("SELECT payload FROM latest WHERE "+where,group)
        systems={}
        for (payload,) in rows:
            row=json.loads(payload);systems.setdefault(row["system"],[]).append(row)
        summary={**dict(zip(keys,group)),"systems":{}}
        for name,records in systems.items():
            scored=[x for x in records if x["status"]=="scored"]
            stat={"status_counts":dict(Counter(x["status"] for x in records)),"n_scored":len(scored)}
            if scored:
                stat.update(score_mean=float(np.mean([x["score"] for x in scored])),score_ci95=interval([x["score"] for x in scored]))
                for field in ["active_tokens","latency_ms","generation_ms","ingestion_ms","peak_vram_mb","ttft_ms","budget_slack_tokens","ndcg_at_10","recall_at_10","recall_selected","candidate_recall"]:
                    values=[x[field] for x in scored if x.get(field) is not None]
                    if values:stat[field+"_mean"]=float(np.mean(values))
            summary["systems"][name]=stat
        a={x["example_id"]:x for x in systems.get("sempointer",[]) if x["status"]=="scored"}
        for name,records in systems.items():
            if name=="sempointer":continue
            b={x["example_id"]:x for x in records if x["status"]=="scored"}
            shared=sorted(a.keys()&b.keys())
            if not shared:continue
            diff=[a[i]["score"]-b[i]["score"] for i in shared]
            comp={**dict(zip(keys,group)),"comparison":"sempointer_vs_"+name,"n_paired":len(shared),
                  "score_difference":float(np.mean(diff)),"score_difference_ci95":interval(diff)}
            for field in ["active_tokens","latency_ms","generation_ms"]:
                values=[a[i][field]-b[i][field] for i in shared if a[i].get(field) is not None and b[i].get(field) is not None]
                if values:comp[field+"_difference"]=float(np.mean(values))
            paired.append(comp)
        tasks.append(summary)
    counts=dict(store.db.execute("SELECT status,COUNT(*) FROM latest GROUP BY status"))
    result={"status_counts":counts,"tasks":tasks,"paired":paired,
            "notes":["No pooled score across different tasks or metrics.",
                     "CIs resample examples, not independent document clusters; correlated questions limit inference.",
                     "Latency is generation plus measured selector cost; suite prompt construction and shared ingestion are logged separately.",
                     "Local surrogate scores are not official judge-based benchmark results."]}
    batch_path=store.directory/"batch_events.jsonl"
    if batch_path.exists():
        batches=[json.loads(line) for line in batch_path.read_text().splitlines() if line.strip()]
        elapsed=sum(b["batch_wall_ms"] for b in batches)
        requests=sum(b["batch_size"] for b in batches)
        tokens=sum(b["batch_completion_tokens"] for b in batches)
        result["batch_throughput"]={"batches":len(batches),"generated_requests":requests,
            "generation_wall_ms":elapsed,"requests_per_second":1000*requests/elapsed if elapsed else None,
            "completion_tokens_per_second":1000*tokens/elapsed if elapsed else None,
            "scope":"Completed generation batches, including retry overhead; excludes shared prompt preparation and storage"}
        result["notes"].append("Batched predictions have no isolated request latency or per-request peak-memory claim; batch_events.jsonl records shared measurements.")
    (store.directory/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Evaluation results","",f"Recorded statuses: `{counts}`","","Results come from recorded attempts only. No score is assigned to skipped/error examples.","",
           "| Dataset | Task | Split | System | Scored | Mean score | Mean prompt tokens | Errors |","| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for task in tasks:
        for name,stat in task["systems"].items():
            lines.append(f"| {task['family']} | {task['task']} | {task['split']} | {name} | {stat['n_scored']} | {stat.get('score_mean','—')} | {stat.get('active_tokens_mean','—')} | {stat['status_counts'].get('error',0)} |")
    (store.directory/"REPORT.md").write_text("\n".join(lines)+"\n")
    with (store.directory/"summary.csv").open("w",newline="") as stream:
        columns=["family","task","split","metric","protocol","system","n_scored","n_errors","n_skipped","score_mean","active_tokens_mean","generation_ms_mean","latency_ms_mean","peak_vram_mb_mean"]
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore");writer.writeheader()
        for task in tasks:
            for name,stat in task["systems"].items():
                writer.writerow({**{k:task.get(k) for k in keys},"system":name,**stat,
                                 "n_errors":stat["status_counts"].get("error",0),"n_skipped":stat["status_counts"].get("skipped",0)})
    return result


def run_tasks(tasks,store,engine,args):
    excluded=set(getattr(args,"excluded_group_ids",[]))
    model_limit=getattr(engine.model.config,"max_position_embeddings",None)
    if model_limit is None:
        model_limit=getattr(getattr(engine.model.config,"text_config",None),"max_position_embeddings",args.max_prompt_tokens)
    for parameter in engine.model.parameters():parameter.requires_grad_(False)
    requested_batch=getattr(args,"inference_batch_size",1)
    if requested_batch>1 and not hasattr(engine,"generate_batch"):raise ValueError("Engine does not support native batched generation")
    for task in tasks:
        if getattr(args,"should_stop",lambda:False)():break
        count=0
        group_counts=Counter()
        try:
            for ex in iter_examples(task,tokenizer=engine.tokenizer):
                if args.partition!="all" and partition_for(ex)!=args.partition:continue
                if ex.group_id in excluded:continue
                per_group=getattr(args,"examples_per_group",0)
                if per_group and group_counts[ex.group_id]>=per_group:continue
                if args.limit and count>=args.limit:break
                count+=1
                group_counts[ex.group_id]+=1
                if all(store.done(ex.id,s) for s in args.systems):continue
                base={"example_id":ex.id,"family":ex.family,"task":ex.task,"split":ex.split,
                      "metric":ex.metric,"protocol":ex.protocol,"group_id":ex.group_id,"meta":ex.meta,
                      "source":str(task.path),"member":task.member,"question":ex.question,"gold":ex.gold,
                      "context_sha256":hashlib.sha256(ex.context.encode()).hexdigest(),"partition":partition_for(ex)}
                reason=ex.skip_reason
                if args.max_source_chars and len(ex.context)>args.max_source_chars:reason="source_character_safety_limit"
                if not reason:
                    source_tokens=len(engine.tokenizer.encode(ex.context,add_special_tokens=False))
                    if args.max_source_tokens and source_tokens>args.max_source_tokens:reason="source_token_safety_limit"
                if reason:
                    for system in args.systems:
                        if not store.done(ex.id,system):store.record({**base,"system":system,"status":"skipped","reason":reason})
                    continue
                try:
                    prepared=prepare(ex.context,engine.tokenizer,seed=args.seed,
                                     router_options=getattr(args,"router_options",None))
                    start=time.perf_counter()
                    variants=prompts_for_example(prepared,ex.question,engine.tokenizer,
                        prompt_formatter=getattr(engine,"format_prompt",None))
                    suite_ms=(time.perf_counter()-start)*1000
                except Exception as exc:
                    for system in args.systems:
                        if not store.done(ex.id,system):store.record({**base,"system":system,"status":"error","error":f"{type(exc).__name__}: {exc}"})
                    continue
                # Rotate order with a deterministic per-example offset. Model
                # weights and decoding settings remain identical for all arms.
                offset=int(hashlib.sha256(ex.id.encode()).hexdigest()[:8],16)%len(args.systems)
                order=args.systems[offset:]+args.systems[:offset]
                cap=args.max_new_tokens or task.max_new_tokens
                cached={};safe_batch=requested_batch
                for system in order:
                    if system not in cached and getattr(args,"should_stop",lambda:False)():break
                    if store.done(ex.id,system):continue
                    variant=variants[system]
                    row={**base,"system":system,**variant,"ingestion_ms":prepared.ingestion_ms,
                         "source_tokens":prepared.source_tokens,"n_blocks":len(prepared.blocks),
                         "prompt_suite_preparation_ms":suite_ms,"max_new_tokens":cap,
                         "generation_index_in_example":order.index(system)}
                    if variant.get("skip_reason") or variant["prompt_tokens"]+cap>min(args.max_prompt_tokens+cap,model_limit):
                        store.record({**row,"status":"skipped","reason":variant.get("skip_reason","prompt_or_model_context_limit")})
                        continue
                    try:
                        batch_meta=None
                        if requested_batch>1:
                            if system not in cached:
                                waiting=[s for s in order[order.index(system):] if not store.done(ex.id,s)
                                    and not variants[s].get("skip_reason")
                                    and variants[s]["prompt_tokens"]+cap<=min(args.max_prompt_tokens+cap,model_limit)]
                                retries=0;retry_ms=0.
                                while True:
                                    group=waiting[:safe_batch];began=time.perf_counter()
                                    try:
                                        if len(group)>1:
                                            batch=engine.generate_batch([variants[s]["prompt"] for s in group],max_new_tokens=cap)
                                            outputs=batch.results;wall=batch.latency_ms;peak=batch.peak_vram_mb
                                        else:
                                            output=engine.generate(variants[group[0]]["prompt"],max_new_tokens=cap)
                                            outputs=[output];wall=output.latency_ms;peak=output.peak_vram_mb
                                        if len(outputs)!=len(group):raise ValueError("Batched generation returned the wrong number of answers")
                                        break
                                    except Exception as exc:
                                        import torch
                                        if not isinstance(exc,torch.cuda.OutOfMemoryError) or len(group)==1:raise
                                        if getattr(engine,"device","").startswith("cuda"):torch.cuda.empty_cache()
                                        retry_ms+=(time.perf_counter()-began)*1000;retries+=1
                                        safe_batch=max(1,len(group)//2)
                                metadata={"batch_id":uuid.uuid4().hex,"batch_size":len(group),
                                    "requested_batch_size":requested_batch,"batch_wall_ms":wall+retry_ms,
                                    "batch_peak_vram_mb":peak,"batch_completion_tokens":sum(g.completion_tokens for g in outputs),
                                    "padding_tokens":max(g.prompt_tokens for g in outputs)*len(outputs)-sum(g.prompt_tokens for g in outputs),
                                    "batch_oom_retries":retries,"example_id":ex.id,"systems":group}
                                store.record_batch(metadata)
                                cached.update((s,(g,metadata)) for s,g in zip(group,outputs))
                            gen,batch_meta=cached.pop(system)
                        else:gen=engine.generate(variant["prompt"],max_new_tokens=cap)
                        measured=gen.prompt_tokens
                        if measured!=variant["prompt_tokens"]:
                            raise ValueError("Prompt token accounting differs from generation")
                        result={**row,"status":"scored","answer":gen.text,"score":score(ex.metric,gen.text,ex.gold),
                                "active_tokens":measured,"generation_ms":gen.latency_ms,
                                "latency_ms":gen.latency_ms+variant.get("selection_ms",0),"completion_tokens":gen.completion_tokens,
                                "peak_vram_mb":gen.peak_vram_mb,"ttft_ms":gen.ttft_ms,"stop_reason":gen.stop_reason}
                        if batch_meta is not None:
                            result.update(batch_meta,latency_mode="shared_batch_wall",generation_ms=None,
                                          latency_ms=None,ttft_ms=None,peak_vram_mb=None,system=system)
                        store.record(result)
                        print(f"{ex.family}/{ex.task} {system} score={result['score']:.4f} tokens={measured}",flush=True)
                    except Exception as exc:
                        store.record({**row,"status":"error","error":f"{type(exc).__name__}: {exc}"})
                        import torch
                        if torch.cuda.is_available():torch.cuda.empty_cache()
            if not count:
                store.record({"example_id":"empty-partition:"+task.key+":"+str(task.path)+":"+str(task.member),
                              "system":"loader","family":task.family,"task":task.task,"split":task.split,"status":"skipped",
                              "reason":"no_eligible_examples_after_partition_and_group_filters"})
        except Exception as exc:
            store.record({"example_id":"task-error:"+task.key+":"+str(task.path)+":"+str(task.member),
                "system":"loader","family":task.family,"task":task.task,"split":task.split,"status":"error",
                "error":f"{type(exc).__name__}: {exc}"})
        checkpoint=getattr(args,"on_checkpoint",None)
        if checkpoint is not None:checkpoint(store,task)
        # SQLite and events.jsonl protect every attempt. Re-exporting the full
        # history after every shard would cause quadratic disk traffic.
    if getattr(args,"defer_summary",False):
        return {"status_counts":dict(store.db.execute("SELECT status,COUNT(*) FROM latest GROUP BY status"))}
    store.export()
    return summarize(store)
