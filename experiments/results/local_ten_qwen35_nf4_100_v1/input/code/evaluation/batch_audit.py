"""Warm every batch shape before comparing real BF16 answer throughput."""
import gc
import hashlib
import json
from pathlib import Path
from statistics import median
import time

from .cloud_policy import atomic_json
from .precision import choose_batch


def run_audit(root,model_path,deadline,commit):
    import torch
    from .qwen35 import Qwen35Engine
    root=Path(root);package=json.loads((root/"input/pilot.json").read_text())
    previous=json.loads((root/"results/comparison.json").read_text())
    if previous.get("status")!="complete" or previous["precision_decision"]["selected_dtype"]!="bf16":
        raise ValueError("A completed BF16 precision win is required before batching")
    report=dict(status="running",pilot_id=package["pilot_id"],source_run_id=package["source_run_id"],
        protocol="Per-shape warm-up, two serial passes, two passes per batch size; identical task caps",
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        serial=[],batches=[],answers=[],warmup_groups=0)
    def save():atomic_json(root/"results/batch_audit.json",report);commit()
    def check():
        if time.monotonic()>deadline-45:raise TimeoutError("Warmed batching audit reached its reserved time limit")
    engine=None
    try:
        engine=Qwen35Engine(model_id=str(model_path),dtype="bf16",seed=package["seed"])
        cases=sorted(package["cases"],key=lambda c:(c["output_cap"],c["prompt_tokens"],c["case_id"]))
        expected={};serial_times=[]
        # Warm every serial shape, so both sides exclude compilation/startup.
        for c in cases:
            check();engine.generate(c["prompt"],max_new_tokens=c["output_cap"])
        for repeat in range(2):
            total=0.
            for c in cases:
                check();result=engine.generate(c["prompt"],max_new_tokens=c["output_cap"])
                if repeat==0:expected[c["case_id"]]=result.text
                elif result.text!=expected[c["case_id"]]:raise ValueError("Serial BF16 answers changed across repeated runs")
                total+=result.latency_ms
                report["answers"].append(dict(mode="serial",repeat=repeat,case_id=c["case_id"],
                    answer=result.text,completion_tokens=result.completion_tokens,elapsed_ms=result.latency_ms))
            serial_times.append(total);report["serial"].append(dict(repeat=repeat,elapsed_ms=total));save()
        by_cap={cap:[c for c in cases if c["output_cap"]==cap] for cap in sorted({c["output_cap"] for c in cases})}
        for size in [2,4]:
            groups=[rows[i:i+size] for rows in by_cap.values() for i in range(0,len(rows),size)]
            totals=[];matches=True;changed=set();peak=0.
            try:
                for group in groups:
                    check();engine.generate_batch([c["prompt"] for c in group],max_new_tokens=group[0]["output_cap"])
                    report["warmup_groups"]+=1;save()
                for repeat in range(2):
                    total=0.
                    for group in groups:
                        check();batch=engine.generate_batch([c["prompt"] for c in group],max_new_tokens=group[0]["output_cap"])
                        total+=batch.latency_ms;peak=max(peak,batch.peak_vram_mb)
                        for c,g in zip(group,batch.results):
                            equal=g.text==expected[c["case_id"]] and g.prompt_tokens==c["prompt_tokens"]
                            matches=matches and equal
                            if not equal:changed.add(c["case_id"])
                            report["answers"].append(dict(mode="batch",batch_size=size,repeat=repeat,
                                case_id=c["case_id"],answer=g.text,completion_tokens=g.completion_tokens,
                                matches_serial=equal))
                    totals.append(total);save()
                report["batches"].append(dict(batch_size=size,elapsed_ms=median(totals),serial_ms=median(serial_times),
                    elapsed_repeats_ms=totals,outputs_match=matches,changed_cases=sorted(changed),
                    prompts=len(cases),peak_vram_mb=peak,status="ok"))
            except torch.cuda.OutOfMemoryError:
                report["batches"].append(dict(batch_size=size,status="oom",outputs_match=False));torch.cuda.empty_cache()
            save()
        report.update(status="complete",batch_decision=choose_batch("bf16",report["batches"]))
    except Exception as exc:
        report.update(status="incomplete",error=f"{type(exc).__name__}: {exc}",
                      batch_decision=dict(batch_size=1,reason="Warmed audit incomplete; keep validated BF16 serial inference"))
    finally:
        save();engine=None;gc.collect();torch.cuda.empty_cache()
    return report
