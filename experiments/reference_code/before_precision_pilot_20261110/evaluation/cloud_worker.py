"""Checkpointed phases for the frozen ten-benchmark cloud bundle."""
from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace

from .cloud_policy import SYSTEMS,atomic_json,project_seconds,HOURLY_RATE,CHUNK_SECONDS,OVERHEAD_SECONDS

PHASES=("dev_loose","dev_paper","dev_cautious","pilot","test","cwe_cap32")
_ENGINE=None


def load_json(path,default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def stage_rows(manifest,phase):
    if phase.startswith("dev_"):return manifest["development"]
    if phase=="pilot":return [r for r in manifest["rows"] if r["id"] in set(manifest["pilot_ids"])]
    if phase=="cwe_cap32":return [r for r in manifest["rows"] if r["id"] in set(manifest["cwe_audit_ids"])]
    return manifest["rows"]


def run_phase(root,model_path,deadline,commit,cancelled):
    """Do one resumable phase/chunk. Native engine stays warm in this process."""
    from .data import Task
    from .runner import RunStore,run_tasks
    from .qwen35 import Qwen35Engine
    from .provenance import package_versions
    from reviewer_suite import CANDIDATES,select_thresholds
    from provenance import environment_record
    global _ENGINE
    root=Path(root);manifest=load_json(root/"input/manifest.json")
    state=load_json(root/"state.json",{"phase":0,"status":"running","timings":{},"partial_times":{}})
    if state["phase"]>=len(PHASES):return {**state,"status":"complete"}
    phase=PHASES[state["phase"]]
    rows=stage_rows(manifest,phase)
    systems=list(SYSTEMS[:1])+["bm25_4"] if phase.startswith("dev_") else list(SYSTEMS)
    if phase.startswith("dev_"):
        router=next(c["thresholds"] for c in CANDIDATES if phase=="dev_"+c["name"])
    else:router=load_json(root/"results/threshold_selection.json")["selected"]["thresholds"]
    engine_key=(str(model_path),manifest["dtype"],manifest["seed"])
    if _ENGINE is None or _ENGINE[0]!=engine_key:
        engine=Qwen35Engine(model_id=str(model_path),dtype=manifest["dtype"],seed=manifest["seed"])
        _ENGINE=(engine_key,engine)
    engine=_ENGINE[1]
    directory=root/"results"/("test" if phase=="pilot" else phase)
    config={"bundle_run_id":manifest["run_id"],"code_files":manifest["code_files"],
        "model_id":manifest["model_id"],"revision":manifest["model_revision"],"dtype":manifest["dtype"],
        "systems":systems,"router_options":router,"max_new_tokens":32 if phase=="cwe_cap32" else 0,
        "limits":manifest["limits"],"packages":package_versions(),"thinking":False,
        "prompt_format":"native_chat_template","seed":manifest["seed"]}
    started_question=time.monotonic()
    with RunStore(directory,config) as store:
        with (directory/"environments.jsonl").open("a") as stream:
            stream.write(json.dumps({"engine":engine.info(),"environment":environment_record()})+"\n")
        pending=[r for r in rows if not all(store.done(r["id"],s) for s in systems)]
        tasks=[Task(r["family"],r["task"],root/"input"/r["path"],adapter="prepared",
                    max_new_tokens=r["output_cap"],extra={"original_id":r["id"]}) for r in pending]
        def checkpoint(store,task):
            nonlocal started_question
            now=time.monotonic();ident=task.extra["original_id"]
            # Two failed attempts are terminal skips, not invented zero scores.
            for (payload,) in store.db.execute("SELECT payload FROM latest WHERE example_id=? AND status='error'",(ident,)).fetchall():
                row=json.loads(payload)
                attempts=store.db.execute("SELECT COUNT(*) FROM attempts WHERE json_extract(payload,'$.example_id')=? AND json_extract(payload,'$.system')=? AND json_extract(payload,'$.status')='error'",(ident,row["system"])).fetchone()[0]
                if attempts>=2:store.record({**row,"status":"skipped","reason":"error_retry_limit","last_error":row.get("error")})
            key=phase+":"+ident
            state["partial_times"][key]=state["partial_times"].get(key,0)+now-started_question
            if phase=="pilot" and all(store.done(ident,s) for s in systems):
                state["timings"].setdefault(task.family,[]).append(state["partial_times"][key])
            state.update(status="running",active_phase=phase,last_example=ident,
                stage_status_counts=dict(store.db.execute("SELECT status,COUNT(*) FROM latest GROUP BY status")))
            store.db.execute("PRAGMA wal_checkpoint(FULL)")
            atomic_json(root/"state.json",state);commit()
            started_question=time.monotonic()
        args=SimpleNamespace(systems=systems,partition="all",limit=0,seed=manifest["seed"],
            max_source_chars=manifest["limits"]["source_chars"],max_source_tokens=manifest["limits"]["source_tokens"],
            max_prompt_tokens=manifest["limits"]["prompt_tokens"],max_new_tokens=32 if phase=="cwe_cap32" else 0,
            router_options=router,defer_summary=True,on_checkpoint=checkpoint,
            should_stop=lambda:time.monotonic()>deadline-60 or cancelled())
        run_tasks(tasks,store,engine,args)
        complete=all(store.done(r["id"],s) for r in rows for s in systems)
        errors=store.db.execute("SELECT COUNT(*) FROM latest WHERE system='loader' AND status='error'").fetchone()[0]
        if errors:raise ValueError("Prepared-example loader failed; inspect recorded attempts")
        store.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    if complete:
        if phase=="dev_cautious":
            candidates=[{**c,"path":str(root/"results"/("dev_"+c["name"]))} for c in CANDIDATES]
            atomic_json(root/"results/threshold_selection.json",select_thresholds(candidates))
        state["phase"]+=1
    state["status"]="complete" if state["phase"]==len(PHASES) else "running"
    atomic_json(root/"state.json",state);commit()
    return state


def affordability(root,remaining):
    """Scores never enter the pilot decision; remaining samples stay fixed."""
    root=Path(root);manifest=load_json(root/"input/manifest.json");state=load_json(root/"state.json")
    rows=[r for r in manifest["rows"] if r["id"] not in set(manifest["pilot_ids"])]
    rows += [r for r in manifest["rows"] if r["id"] in set(manifest["cwe_audit_ids"])]
    seconds=project_seconds(rows,state["timings"])
    overhead=(1+int(seconds/max(1,CHUNK_SECONDS-60)))*OVERHEAD_SECONDS
    projected=(seconds+overhead)*HOURLY_RATE/3600
    result={"projected_remaining_dollars":projected,"remaining_allowance_dollars":remaining,
        "safety_factor":1.5,"affordable":projected<=remaining,
        "basis":"Worst observed completed pilot question per family; includes all compared systems and 50% timing headroom.",
        "warning":"A projection, not a completion guarantee. Workspace cap and allowance remain active."}
    atomic_json(root/"results/pilot_cost_projection.json",result)
    return result["affordable"]


def report(root):
    from .runner import RunStore,summarize,interval
    root=Path(root);state=load_json(root/"state.json",{})
    manifest=load_json(root/"input/manifest.json")
    counts={};all_rows={}
    for path in sorted((root/"results").glob("*/config.json")):
        with RunStore(path.parent,load_json(path)) as store:
            store.export();result=summarize(store)
            counts[path.parent.name]=result["status_counts"]
            all_rows[path.parent.name]=[json.loads(x[0]) for x in store.db.execute("SELECT payload FROM latest")]
    long={(r["example_id"],r["system"]):r for r in all_rows.get("test",[]) if r["status"]=="scored" and r["family"]=="ruler" and r["task"]=="cwe"}
    short={(r["example_id"],r["system"]):r for r in all_rows.get("cwe_cap32",[]) if r["status"]=="scored"}
    audit={}
    for system in SYSTEMS:
        ids=[k for k in short.keys() & long.keys() if k[1]==system]
        if not ids:continue
        if any(short[k]["context_sha256"]!=long[k]["context_sha256"] for k in ids):raise ValueError("CWE context mismatch")
        values=[long[k]["score"]-short[k]["score"] for k in ids]
        audit[system]={"paired_questions":len(ids),"difference_512_minus_32":sum(values)/len(values),
                       "ci95":interval(values),"cap32_hits":sum(short[k]["stop_reason"]=="max_new_tokens" for k in ids),
                       "cap512_hits":sum(long[k]["stop_reason"]=="max_new_tokens" for k in ids)}
    atomic_json(root/"results/cwe_cap_audit.json",audit)
    atomic_json(root/"results/sampling_report.json",manifest["sampling_report"])
    lines=["# Modal ten-benchmark evaluation","",f"Run: `{manifest['run_id']}`; status: **{state.get('status','unknown')}**.",
        "","Only recorded predictions are evidence. A budget stop means partial coverage, not benchmark completion.",
        "","| Family | Selected questions |","| --- | ---: |"]
    lines += [f"| {f} | {data['selected']} |" for f,data in manifest["sampling_report"].items()]
    lines += ["","## Recorded stage counts","",json.dumps(counts,indent=2),"",
        "See each stage's REPORT.md, metrics.json, raw predictions and attempts. No overall mixed-metric score is computed.",
        "","CWE cap comparisons are in cwe_cap_audit.json. The frozen development choice is in threshold_selection.json.",
        "","These are selected benchmark subsets with explicitly adapted metrics; they are not official leaderboard submissions.",
        "Confidence intervals over questions do not remove document/conversation clustering or source overlap."]
    (root/"results/REPORT.md").write_text("\n".join(lines)+"\n")
    return counts
