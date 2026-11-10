"""Conservative decisions for measured, paired precision and batching pilots."""
import math
from statistics import median


def choose_precision(rows,case_ids):
    samples={}
    for row in rows:
        if row.get("status")!="ok" or not math.isfinite(row.get("elapsed_ms",float("nan"))) or row["elapsed_ms"]<=0:
            return {"selected_dtype":"nf4","reason":"Incomplete or failed equal-work comparison"}
        samples.setdefault((row["dtype"],row["case_id"]),[]).append(row)
    ratios=[]
    for ident in case_ids:
        a=samples.get(("nf4",ident),[]);b=samples.get(("bf16",ident),[])
        if len(a)<2 or len(a)!=len(b) or len({r["decode_tokens"] for r in a+b})!=1:
            return {"selected_dtype":"nf4","reason":"Missing paired repeated fixed-token measurements"}
        ratios.append(median(r["elapsed_ms"] for r in a)/median(r["elapsed_ms"] for r in b))
    if not ratios:return {"selected_dtype":"nf4","reason":"No paired measurements"}
    speedup=median(ratios)
    wins=sum(r>=1.10 for r in ratios)
    eligible=speedup>=1.10 and wins>=math.ceil(len(ratios)*2/3)
    return {"selected_dtype":"bf16" if eligible else "nf4","speedup":speedup,
            "paired_cases":len(ratios),"cases_with_10_percent_advantage":wins,
            "reason":"BF16 meets the repeated 10% timing threshold" if eligible else "No consistent >=10% BF16 advantage",
            "selection_basis":"Equal decoded-token timing only; answers/scores do not select precision"}


def choose_batch(dtype,rows):
    valid=[r for r in rows if dtype=="bf16" and r.get("status")=="ok" and r.get("outputs_match")
           and r.get("elapsed_ms",0)>0 and r.get("serial_ms",0)/r["elapsed_ms"]>=1.05
           and len(r.get("elapsed_repeats_ms",[]))>=2
           and all(t>0 and r["serial_ms"]/t>=1.05 for t in r["elapsed_repeats_ms"])]
    if not valid:return {"batch_size":1,"reason":"No eligible measured batching advantage"}
    best=max(valid,key=lambda r:r["serial_ms"]/r["elapsed_ms"])
    return {"batch_size":best["batch_size"],"speedup":best["serial_ms"]/best["elapsed_ms"],
            "reason":"Measured throughput gain with matching pilot outputs",
            "minimum_speedup_each_repeat":1.05,
            "latency_mode":"shared_batch_wall; no per-request isolated latency claim"}
