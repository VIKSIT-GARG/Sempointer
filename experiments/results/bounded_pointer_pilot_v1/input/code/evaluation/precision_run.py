"""Real GPU precision measurements; fixed-work outputs are never scored."""
import gc
import json
from pathlib import Path
import time

from .cloud_policy import atomic_json
from .precision import choose_precision,choose_batch


def run_comparison(root,model_path,deadline,checkpoint):
    import torch
    from .qwen35 import Qwen35Engine
    from .metrics import score
    from provenance import environment_record
    root=Path(root);package=json.loads((root/"input/pilot.json").read_text())
    by_id={c["case_id"]:c for c in package["cases"]}
    fixed=[by_id[ident] for ident in package["fixed_cases"]]
    report=dict(pilot_id=package["pilot_id"],source_run_id=package["source_run_id"],status="running",
        fixed_work=[],answers=[],batches=[],loads=[],environment=environment_record(),
        gpu=dict(name=torch.cuda.get_device_name(),memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        interpretation="Pilot only; fixed-token outputs are timing probes, not benchmark answers")
    engine=None;dtype=None;rounds={"nf4":0,"bf16":0}
    def save():atomic_json(root/"results/comparison.json",report);checkpoint()
    def load(precision):
        nonlocal engine,dtype
        if dtype==precision:return
        engine=None;gc.collect();torch.cuda.empty_cache()
        began=time.monotonic()
        engine=Qwen35Engine(model_id=str(model_path),dtype=precision,seed=package["seed"])
        dtype=precision
        report["loads"].append(dict(dtype=dtype,seconds=time.monotonic()-began,engine=engine.info()))
        engine.generate(fixed[0]["prompt"],max_new_tokens=8)
    def ensure_time():
        if time.monotonic()>deadline-45:raise TimeoutError("Precision pilot reached its reserved time limit")
    try:
        for precision in package["comparison_order"]:
            ensure_time();load(precision)
            for case in fixed:
                ensure_time()
                inputs=engine.tokenizer(case["prompt"],return_tensors="pt").to(engine.model.device)
                warmup=engine.model.generate(**inputs,do_sample=False,use_cache=True,logits_to_keep=1,
                    min_new_tokens=package["decode_tokens"],max_new_tokens=package["decode_tokens"],
                    pad_token_id=engine.tokenizer.pad_token_id)
                torch.cuda.synchronize();del warmup
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
                # Force exactly the same decode work; no correctness score is attached.
                output=engine.model.generate(**inputs,do_sample=False,use_cache=True,logits_to_keep=1,
                    min_new_tokens=package["decode_tokens"],max_new_tokens=package["decode_tokens"],
                    pad_token_id=engine.tokenizer.pad_token_id)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-began)*1000
                count=output.shape[1]-inputs["input_ids"].shape[1]
                if count!=package["decode_tokens"]:raise ValueError("Fixed-token workload did not complete")
                report["fixed_work"].append(dict(dtype=precision,round=rounds[precision],case_id=case["case_id"],
                    prompt_tokens=case["prompt_tokens"],decode_tokens=count,elapsed_ms=elapsed,status="ok",
                    warmup_decode_tokens=package["decode_tokens"],
                    peak_vram_mb=torch.cuda.max_memory_allocated()/1024**2))
                del inputs,output
                save()
            if rounds[precision]==0:
                for case in package["cases"]:
                    ensure_time();result=engine.generate(case["prompt"],max_new_tokens=case["output_cap"])
                    report["answers"].append(dict(dtype=precision,case_id=case["case_id"],family=case["family"],
                        task=case["task"],system=case["system"],answer=result.text,score=score(case["metric"],result.text,case["gold"]),
                        completion_tokens=result.completion_tokens,prompt_tokens=result.prompt_tokens,
                        elapsed_ms=result.latency_ms,stop_reason=result.stop_reason,status="ok"))
                    save()
            rounds[precision]+=1
        report["precision_decision"]=choose_precision(report["fixed_work"],package["fixed_cases"])
        if report["precision_decision"]["selected_dtype"]=="bf16":
            ensure_time();load("bf16")
            # Equal caps within each batch; actual task answers must match the serial BF16 pilot.
            sample=[c for c in package["cases"] if c["output_cap"]==128][:8]
            sample=sample[:len(sample)//4*4]
            serial={r["case_id"]:r for r in report["answers"] if r["dtype"]=="bf16"}
            for size in [2,4]:
                ensure_time();elapsed=0.;matches=True;peak=0.
                try:
                    for i in range(0,len(sample),size):
                        ensure_time();group=sample[i:i+size]
                        result=engine.generate_batch([c["prompt"] for c in group],max_new_tokens=128)
                        elapsed+=result.latency_ms;peak=max(peak,result.peak_vram_mb)
                        matches=matches and all(g.text==serial[c["case_id"]]["answer"] and g.prompt_tokens==c["prompt_tokens"]
                                                for c,g in zip(group,result.results))
                    if sample:report["batches"].append(dict(batch_size=size,elapsed_ms=elapsed,
                        serial_ms=sum(serial[c["case_id"]]["elapsed_ms"] for c in sample),
                        outputs_match=matches,prompts=len(sample),peak_vram_mb=peak,status="ok"))
                except torch.cuda.OutOfMemoryError:
                    report["batches"].append(dict(batch_size=size,status="oom",outputs_match=False));torch.cuda.empty_cache()
                save()
        report["batch_decision"]=choose_batch(report["precision_decision"]["selected_dtype"],report["batches"])
        paired={}
        for row in report["answers"]:paired.setdefault(row["case_id"],{})[row["dtype"]]=row
        report["answers_changed"]=sum(r["nf4"]["answer"]!=r["bf16"]["answer"] for r in paired.values() if len(r)==2)
        report["status"]="complete"
    except Exception as exc:
        report.update(status="incomplete",error=f"{type(exc).__name__}: {exc}")
        report["precision_decision"]={"selected_dtype":"nf4","reason":"Pilot incomplete; no configuration change"}
        report["batch_decision"]={"batch_size":1,"reason":"Pilot incomplete"}
    finally:
        save();engine=None;gc.collect();torch.cuda.empty_cache()
    return report
