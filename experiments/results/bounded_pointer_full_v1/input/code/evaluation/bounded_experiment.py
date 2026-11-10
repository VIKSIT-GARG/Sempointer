"""Local, fresh-question pilot for bounded pointer prompts; no cloud client."""
import argparse
from collections import Counter,defaultdict
from contextlib import closing
import fcntl
from functools import partial
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from .cloud_policy import CandidatePool,FAMILIES,atomic_json,digest

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE/"results/bounded_pointer_pilot_v1"
BUDGETS=(1024,2048,4096)
CAPS=(128,256)
DEV_ARMS=("bounded_128","bounded_256","adaptive_no_index","bm25_budget","original_sempointer")
TEST_ARMS=("bounded_pointer","adaptive_no_index","bm25_budget","original_sempointer")


def _sample(rows,cap,seed):
    pool=CandidatePool(seed,cap)
    for row in rows:
        # Randomize stratum order deterministically when the small pilot has
        # fewer slots than categories, instead of favoring alphabetical names.
        pool.add({**row,"stratum":digest([seed,row["family"],row["stratum"]]),"_original":row})
    return [row["_original"] for row in pool.selected()]


def fresh_splits(rows,excluded,dev_cap=3,test_cap=10,seed=20261116):
    development,test=[],[]
    for family in FAMILIES:
        available=[r for r in rows if r["family"]==family and r["id"] not in excluded]
        if not available:continue
        dev=_sample(available,dev_cap,seed)
        groups={r["group_id"] for r in dev}
        heldout=_sample([r for r in available if r["group_id"] not in groups],test_cap,seed+1)
        if len(dev)!=dev_cap or len(heldout)!=test_cap:
            raise ValueError("Insufficient fresh, group-disjoint questions for "+family)
        development.extend(dev);test.extend(heldout)
    return development,test


def choose_cap(rows,role):
    if role!="development":raise ValueError("Pointer allowance may only use development results")
    by=defaultdict(dict)
    for row in rows:by[row["example_id"]][row["system"]]=row
    stats=[]
    for cap in CAPS:
        name=f"bounded_{cap}"
        if not by or any(name not in arms or "bm25_budget" not in arms or
                arms[name]["status"] not in {"scored","skipped"} or
                arms["bm25_budget"]["status"] not in {"scored","skipped"} for arms in by.values()):
            raise ValueError("Development comparisons are incomplete")
        wins=ties=losses=skips=0;tokens=[]
        for arms in by.values():
            candidate,baseline=arms[name],arms["bm25_budget"]
            if candidate["status"]=="skipped":
                skips+=1
                if baseline["status"]=="scored":losses+=1
                continue
            tokens.append(candidate["active_tokens"])
            if baseline["status"]!="scored":continue
            diff=candidate["score"]-baseline["score"]
            if diff>1e-12:wins+=1
            elif diff< -1e-12:losses+=1
            else:ties+=1
        stats.append(dict(pointer_cap=cap,wins=wins,ties=ties,losses=losses,skips=skips,
                          utility=wins-losses,mean_tokens=sum(tokens)/len(tokens) if tokens else float("inf")))
    chosen=min(stats,key=lambda r:(-r["utility"],r["skips"],r["mean_tokens"],r["pointer_cap"]))
    return dict(source="development_only",pointer_cap=chosen["pointer_cap"],candidates=stats,
                rule="Maximize question wins minus losses against budgeted BM25; ties prefer coverage, fewer tokens, smaller cap")


def prepare_experiment(root):
    from prepare_modal import code_files,file_hash,verify_bundle
    root=Path(root).resolve()
    if (root/"input/manifest.json").exists():return verify_bundle(root/"input")
    source=HERE/"modal_runs/ten_qwen35_recovered_v1/bundle"
    prior_root=HERE/"results/local_ten_qwen35_nf4_100_v1"
    prior=verify_bundle(prior_root/"input");pool=verify_bundle(source)
    excluded={r["id"] for r in prior["rows"]+prior["development"]}|set(pool["pilot_ids"])
    development,test=fresh_splits(pool["rows"],excluded)
    if len(development)!=30 or len(test)!=100:raise ValueError("Expected all ten families")
    old_groups={r["group_id"] for r in prior["rows"]+prior["development"]}
    staging=root/"input.preparing"
    if staging.exists():raise ValueError("Incomplete preparation retained at "+str(staging))
    staging.mkdir(parents=True)
    for row in development+test:
        path=staging/row["path"];path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/row["path"],path)
    snapshot=code_files()
    for name,checksum in snapshot.items():
        target=staging/"code"/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(HERE/name,target)
        if file_hash(target)!=checksum:raise ValueError("Source changed while freezing")
    selection=json.loads((prior_root/"results/threshold_selection.json").read_text())
    manifest=dict(schema="bounded_pointer_pilot_v1",execution_backend="local_cuda",dtype="nf4",
        inference_batch_size=1,seed=prior["seed"],sampling_seed=20261116,
        model_id=prior["model_id"],model_revision=prior["model_revision"],
        model_local_path=prior["model_local_path"],model_weights_sha256=prior["model_weights_sha256"],
        tokenizer_file_sha256=prior["tokenizer_file_sha256"],code_files=snapshot,
        budgets=list(BUDGETS),pointer_caps=list(CAPS),shortlist=16,
        router_options=selection["selected"]["thresholds"],limits=prior["limits"],
        development=development,rows=test,families=prior["families"],
        parent_run_id=prior["run_id"],excluded_ids=sorted(excluded),
        overlap_with_prior_groups={role:dict(Counter(r["family"] for r in values if r["group_id"] in old_groups))
                                  for role,values in [("development",development),("test",test)]},
        notes=["Small exploratory pilot; fresh question IDs do not imply independent new source documents.",
               "New development and test group IDs are disjoint; the test does not select pointer allowance.",
               "Whole-input budgets include chat framing; task output caps are unchanged.",
               "Equal upper budgets do not imply identical actual token consumption.",
               "No Modal calls, model training, manuscript edits, or prior-result overwrites."])
    manifest["run_id"]=digest(manifest)[:20]
    atomic_json(staging/"manifest.json",manifest);verify_bundle(staging)
    if snapshot!=code_files():raise ValueError("Source changed during preparation")
    staging.rename(root/"input")
    return manifest


def _read_rows(directory):
    with closing(sqlite3.connect(f"file:{directory}/results.sqlite?mode=ro",uri=True)) as db:
        return [json.loads(r[0]) for r in db.execute("SELECT payload FROM latest")]


def run_stage(root,manifest,engine,role,budget,cap,cancelled):
    from .bounded import budgeted_variants
    from .data import Task
    from .runner import RunStore,run_tasks
    from .provenance import package_versions
    name=f"{role}_{budget}"
    systems=list(DEV_ARMS if role=="development" else TEST_ARMS)
    rows=manifest["development"] if role=="development" else manifest["rows"]
    caps=CAPS if role=="development" else (cap,)

    def builder(prepared,question,tokenizer,prompt_formatter=None):
        variants=budgeted_variants(prepared,question,tokenizer,budget,pointer_caps=caps,
            shortlist=manifest["shortlist"],prompt_formatter=prompt_formatter)
        if role=="test":variants["bounded_pointer"]=variants.pop(f"bounded_{cap}")
        return variants

    config=dict(run_id=manifest["run_id"],role=role,budget=budget,pointer_caps=list(caps),
        systems=systems,model_id=manifest["model_id"],model_revision=manifest["model_revision"],
        dtype="nf4",code_files=manifest["code_files"],router_options=manifest["router_options"],
        packages=package_versions(),seed=manifest["seed"],thinking=False)
    directory=root/"results"/name
    with RunStore(directory,config) as store:
        pending=[r for r in rows if not all(store.done(r["id"],arm) for arm in systems)]
        tasks=[Task(r["family"],r["task"],root/"input"/r["path"],adapter="prepared",
                    max_new_tokens=r["output_cap"],extra={"original_id":r["id"]}) for r in pending]
        fatal=[]
        def checkpoint(store,task):
            counts=dict(store.db.execute("SELECT status,COUNT(*) FROM latest GROUP BY status"))
            if counts.get("error"):
                fatal.append("Recorded inference/loader error; inspect stage events before resuming")
            state=dict(status="running",stage=name,role=role,budget=budget,
                last_example=task.extra["original_id"],stage_counts=counts,
                stage_target=len(rows)*len(systems),updated=time.time())
            store.db.execute("PRAGMA wal_checkpoint(FULL)")
            atomic_json(root/"state.json",state)
        args=SimpleNamespace(systems=systems,partition="all",limit=0,seed=manifest["seed"],
            max_source_chars=manifest["limits"]["source_chars"],
            max_source_tokens=manifest["limits"]["source_tokens"],max_prompt_tokens=budget,
            max_new_tokens=0,router_options=manifest["router_options"],defer_summary=True,
            inference_batch_size=1,prompt_builder=builder,on_checkpoint=checkpoint,
            should_stop=lambda:cancelled() or bool(fatal))
        run_tasks(tasks,store,engine,args)
        store.export()
        if fatal:raise RuntimeError(fatal[0])
        return all(store.done(r["id"],arm) for r in rows for arm in systems)


def report(root):
    from .runner import RunStore,summarize
    manifest_path=root/"input/manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    full=manifest.get("evaluation_only",False)
    report={"stages":{},"comparisons":{}}
    for config_path in sorted((root/"results").glob("*/config.json")):
        directory=config_path.parent;config=json.loads(config_path.read_text())
        reference="bounded_128" if config["role"]=="development" else "bounded_pointer"
        with RunStore(directory,config) as store:
            store.export();summary=summarize(store,reference=reference)
        report["stages"][directory.name]=summary["status_counts"]
        if config["role"]!="test":continue
        rows=_read_rows(directory);by=defaultdict(dict)
        for row in rows:by[row["example_id"]][row["system"]]=row
        comparisons={}
        for arm in TEST_ARMS[1:]:
            pairs=[(v[reference],v[arm]) for v in by.values() if reference in v and arm in v
                   and v[reference]["status"]==v[arm]["status"]=="scored"]
            diffs=[a["score"]-b["score"] for a,b in pairs]
            comparisons[arm]=dict(n_paired=len(pairs),wins=sum(d>1e-12 for d in diffs),
                ties=sum(abs(d)<=1e-12 for d in diffs),losses=sum(d< -1e-12 for d in diffs),
                mean_input_token_difference=sum(a["active_tokens"]-b["active_tokens"] for a,b in pairs)/len(pairs) if pairs else None,
                generation_ms_difference_total=sum(a["generation_ms"]-b["generation_ms"] for a,b in pairs))
        report["comparisons"][directory.name]=comparisons
    state=json.loads((root/"state.json").read_text()) if (root/"state.json").exists() else {}
    report["status"]=state.get("status","not_started")
    report["note"]="Question win/tie/loss counts are descriptive; no mixed-metric overall accuracy is computed."
    atomic_json(root/"results/summary.json",report)
    title="Full bounded pointer local evaluation" if full else "Bounded pointer local pilot"
    description=(f"{len(manifest.get('rows',[]))} test questions; settings inherited unchanged from pilot development. No new development stage."
                 if full else "30 development questions; 100 fresh test question IDs across ten families.")
    lines=["# "+title,"",f"Status: **{report['status']}**.","",description,
        "Some source documents overlap prior evaluations; question freshness does not establish unseen-document generalization.",
        "Whole-input budgets: 1,024, 2,048 and 4,096. NF4, batch one; native non-thinking chat.",
        "Pointer caps are chosen on development data only. All stage predictions, per-task metrics and paired confidence intervals are retained.",
        "No overall score pools different metrics; report coverage and per-task paired uncertainty.","",
        "| Stage | Recorded statuses |","| --- | --- |"]
    lines += [f"| {name} | {counts} |" for name,counts in report["stages"].items()]
    lines += ["","## Paired test comparisons","",json.dumps(report["comparisons"],indent=2)]
    (root/"results/REPORT.md").write_text("\n".join(lines)+"\n")
    return report


def execute_stages(root,manifest,engine,cancelled):
    """Reuse frozen development choices for a test-only larger evaluation."""
    choices_path=root/"results/pointer_selection.json"
    if manifest.get("evaluation_only"):
        choices=manifest["frozen_pointer_selection"]
        if set(choices)!={str(b) for b in BUDGETS} or any(
                c.get("source")!="development_only" or c.get("pointer_cap") not in CAPS
                for c in choices.values()):
            raise ValueError("Invalid inherited development configuration")
    else:
        for budget in BUDGETS:
            if cancelled() or not run_stage(root,manifest,engine,"development",budget,None,cancelled):
                return False
        choices={str(b):choose_cap(_read_rows(root/"results"/f"development_{b}"),"development") for b in BUDGETS}
    if choices_path.exists() and json.loads(choices_path.read_text())!=choices:
        raise ValueError("Frozen development choice changed")
    atomic_json(choices_path,choices)
    for budget in BUDGETS:
        if cancelled() or not run_stage(root,manifest,engine,"test",budget,choices[str(budget)]["pointer_cap"],cancelled):
            return False
    return True


def run(root):
    from prepare_modal import verify_bundle,file_hash
    from .provenance import package_versions
    root=Path(root).resolve()
    with (root/"worker.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=verify_bundle(root/"input")
        model=Path(manifest["model_local_path"])
        for name,checksum in manifest["model_weights_sha256"].items():
            if file_hash(model/name)!=checksum:raise ValueError("Model weights changed")
        import torch
        if not torch.cuda.is_available():raise RuntimeError("Local CUDA required; no cloud fallback")
        torch.set_num_threads(8);torch.set_num_interop_threads(2)
        stop=threading.Event()
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.set())
        cancelled=lambda:stop.is_set() or (root/"STOP_REQUESTED").exists()
        atomic_json(root/"worker.json",dict(pid=os.getpid(),started=time.time(),run_id=manifest["run_id"]))
        atomic_json(root/"state.json",dict(status="loading_model",updated=time.time()))
        from .qwen35 import Qwen35Engine
        from provenance import environment_record
        try:
            engine=Qwen35Engine(model_id=str(model),dtype="nf4",seed=manifest["seed"])
            atomic_json(root/"environment.json",dict(engine=engine.info(),environment=environment_record(),packages=package_versions()))
            print(json.dumps(dict(status="local_gpu_ready",run_id=manifest["run_id"],gpu=torch.cuda.get_device_name(0))),flush=True)
            if execute_stages(root,manifest,engine,cancelled):
                atomic_json(root/"state.json",dict(status="complete",completed=time.time()))
                return
            state=json.loads((root/"state.json").read_text());state["status"]="paused_local"
            atomic_json(root/"state.json",state)
        except BaseException as exc:
            state=json.loads((root/"state.json").read_text());state.update(status="paused_error",error=f"{type(exc).__name__}: {exc}")
            atomic_json(root/"state.json",state);raise
        finally:
            report(root)


def running(root):
    try:
        pid=json.loads((root/"worker.json").read_text())["pid"]
        command=Path(f"/proc/{pid}/cmdline").read_bytes()
        return b"evaluation.bounded_experiment" in command and str(root).encode() in command
    except (FileNotFoundError,ProcessLookupError):return False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","start","run","status","stop","report"])
    parser.add_argument("--output",type=Path,default=ROOT)
    args=parser.parse_args();root=args.output.resolve()
    if args.mode in {"prepare","start"}:
        manifest=prepare_experiment(root)
        if args.mode=="prepare":
            print(json.dumps(dict(run_id=manifest["run_id"],development=len(manifest["development"]),test=len(manifest["rows"]),prior_document_overlap=manifest["overlap_with_prior_groups"])))
            return
        if running(root):print("Local bounded-prompt worker already running");return
        (root/"STOP_REQUESTED").unlink(missing_ok=True)
        env={**os.environ,"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
             "HF_DEACTIVATE_ASYNC_LOAD":"1","HF_HUB_DISABLE_PROGRESS_BARS":"1",
             "OMP_NUM_THREADS":"8","RAYON_NUM_THREADS":"8","TOKENIZERS_PARALLELISM":"true"}
        with (root/"worker.log").open("a") as log:
            child=subprocess.Popen([sys.executable,"-u","-m","evaluation.bounded_experiment","run","--output",str(root)],
                cwd=root/"input/code",stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        print(json.dumps(dict(status="local_worker_launched",pid=child.pid,output=str(root))))
    elif args.mode=="run":run(root)
    elif args.mode=="status":
        path=root/"state.json"
        print(json.dumps(dict(worker_running=running(root),state=json.loads(path.read_text()) if path.exists() else {},output=str(root)),indent=2))
    elif args.mode=="stop":
        (root/"STOP_REQUESTED").touch();print("Stop requested; completed calls are retained")
    else:print(json.dumps(report(root),indent=2))


if __name__=="__main__":main()
