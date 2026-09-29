#!/usr/bin/env python
"""SemPointer benchmark runner — the single entry point for all experiments.

Runs one benchmark across one or more systems under identical conditions
(same model, dtype, decoding, hardware) and writes fully-provenanced results:

    results/<run_name>/
        config.json       exact configuration of this run
        environment.json  python/torch/CUDA/GPU versions + git commit
        predictions.jsonl one raw record per (system, example): prompt, answer,
                          gold, metric score, token counts, latencies, VRAM
        metrics.json      per-system aggregate metrics + bootstrap CIs

Benchmarks:
    ruler       NVIDIA/RULER synthetic haystacks (official task templates)
    longbench   THUDM/LongBench (official HF data + official F1 metric)
    locomo      LoCoMo long-term conversation QA (official data, token-F1 adaptation)
    kv_compare  RULER haystacks + KV-compression baselines (kvpress) at equal budget
    all         ruler + longbench + locomo

Examples:
    python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa \\
        --limit 3 --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
    python run_benchmark.py --benchmark ruler --context-lengths 4096 --limit 10 \\
        --systems sempointer,full,rag,bm25
    python run_benchmark.py --dry-run --benchmark all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

from engine.llm import LLMEngine
from provenance import append_prediction, finalize_metrics, init_run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True,
                   choices=["ruler", "longbench", "locomo", "kv_compare", "all"])
    p.add_argument("--systems", default="sempointer,full,rag,bm25",
                   help="comma list: sempointer,full,rag,bm25,kvpress:<press>:<ratio>"
                        " e.g. kvpress:snapkv:0.3 (presses: snapkv,pyramidkv,knorm,tova)")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dtype", default="nf4", choices=["bf16", "nf4"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="max examples per task/conversation (None = all)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--k", type=int, default=8, help="SemPointer pointer length")
    p.add_argument("--m", type=int, default=1, help="SemPointer resolved blocks")
    p.add_argument("--rag-m", type=int, default=1, help="blocks retrieved by rag/bm25")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--context-lengths", default="4096,8192,16384",
                   help="RULER context lengths (tokens), comma list")
    p.add_argument("--ruler-tasks", default="niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,vt,cwe")
    p.add_argument("--longbench-tasks", default="hotpotqa,2wikimqa,musique,multifieldqa_en,narrativeqa,qasper")
    p.add_argument("--locomo-conversations", default="0",
                   help="LoCoMo conversation indices to use, comma list")
    p.add_argument("--locomo-categories", default=None,
                   help="LoCoMo QA categories (1-5), comma list; default all")
    p.add_argument("--output", default=None, help="results dir (default results/<benchmark>_<ts>)")
    p.add_argument("--run-name", default=None, help="stable name for the results dir")
    p.add_argument("--dry-run", action="store_true",
                   help="validate environment/deps/dataset access without running the benchmark")
    p.add_argument("--resume", action="store_true",
                   help="skip (system, example) pairs already present in predictions.jsonl")
    return p.parse_args()


# ---------------------------------------------------------------------- #
def build_examples(args) -> List[Dict[str, Any]]:
    """Loads benchmark examples. Each returns dicts with
    {id, context, question, gold (list), metric, meta}."""
    from benchmarks import get_builder

    examples: List[Dict[str, Any]] = []
    if args.benchmark in ("ruler", "kv_compare", "all"):
        builder = get_builder("ruler")
        for task in args.ruler_tasks.split(","):
            for clen in [int(x) for x in args.context_lengths.split(",")]:
                exs = builder(task=task, context_length=clen,
                              n_samples=args.limit or 5, seed=args.seed)
                examples.extend(exs)
    if args.benchmark in ("longbench", "all"):
        builder = get_builder("longbench")
        for task in args.longbench_tasks.split(","):
            exs = builder(task=task, n_samples=args.limit, seed=args.seed)
            examples.extend(exs)
    if args.benchmark in ("locomo", "all"):
        builder = get_builder("locomo")
        cats = [int(c) for c in args.locomo_categories.split(",")] if args.locomo_categories else None
        for conv in [int(c) for c in args.locomo_conversations.split(",")]:
            exs = builder(conversation_idx=conv, categories=cats, n_samples=args.limit)
            examples.extend(exs)
    if not examples:
        raise RuntimeError("No examples loaded — check benchmark/task arguments.")
    # normalize dataclass BenchExample -> dict for uniform access below
    examples = [e.to_dict() if hasattr(e, "to_dict") else dict(e) for e in examples]
    return examples


def build_systems(args, tokenizer):
    """Instantiates every requested system. Missing dependencies raise loudly."""
    from sempointer.pipeline import SemPointerPipeline

    systems: Dict[str, Any] = {}
    specs = [s.strip() for s in args.systems.split(",") if s.strip()]
    for spec in specs:
        if spec == "sempointer":
            systems["sempointer"] = SemPointerPipeline(
                tokenizer=tokenizer, k=args.k, m=args.m, block_size=args.block_size,
                seed=args.seed, device=args.device)
        elif spec == "full":
            from baselines.full_context import FullContextBaseline
            systems["full"] = FullContextBaseline(tokenizer=tokenizer, block_size=args.block_size)
        elif spec == "bm25":
            from baselines.bm25_baseline import BM25Baseline
            systems["bm25"] = BM25Baseline(tokenizer=tokenizer, m=args.rag_m, block_size=args.block_size)
        elif spec == "rag":
            from baselines.dense_rag import DenseRAGBaseline
            systems["rag"] = DenseRAGBaseline(tokenizer=tokenizer, m=args.rag_m,
                                              block_size=args.block_size, device=args.device)
        elif spec == "agentic_pointer":
            from baselines.agentic_pointer import AgenticPointerBaseline
            systems["agentic_pointer"] = AgenticPointerBaseline(
                tokenizer=tokenizer, m=args.rag_m, block_size=args.block_size)
        elif spec == "iterative_rag":
            from baselines.iterative_rag import IterativeRAGBaseline
            systems["iterative_rag"] = IterativeRAGBaseline(
                tokenizer=tokenizer, m=args.rag_m,
                block_size=args.block_size, device=args.device)
        elif spec == "hybrid":
            from baselines.hybrid import HybridRRFBaseline
            systems["hybrid"] = HybridRRFBaseline(
                tokenizer=tokenizer, m=args.rag_m,
                block_size=args.block_size, device=args.device)
        elif spec == "session_summary":
            from baselines.session_summary import SessionSummaryBaseline
            systems["session_summary"] = SessionSummaryBaseline(
                tokenizer=tokenizer, block_size=args.block_size)
        elif spec.startswith("kvpress:"):
            from baselines.kvpress_baseline import KVPressBaseline
            parts = spec.split(":")
            press = parts[1]
            ratio = float(parts[2]) if len(parts) > 2 else 0.3
            systems[f"kvpress_{press}_{ratio}"] = KVPressBaseline(
                press_name=press, compression_ratio=ratio, model_id=args.model,
                dtype=args.dtype, device=args.device, max_new_tokens=args.max_new_tokens,
                seed=args.seed)
        else:
            raise ValueError(f"Unknown system spec: {spec!r}")
    return systems


def score_example(metric: str, prediction: str, gold: List[str]) -> float:
    from metrics.official_metrics import locomo_f1, qa_em_score, qa_f1_score, ruler_containment

    if metric == "ruler_containment":
        return float(ruler_containment(prediction, gold))
    if metric == "longbench_f1":
        return float(qa_f1_score(prediction, gold))
    if metric == "longbench_em":
        return float(qa_em_score(prediction, gold))
    if metric == "locomo_f1":
        return float(locomo_f1(prediction, gold))
    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_ci(values: List[float], n_boot: int = 1000, seed: int = 42):
    if not values:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [float(arr[rng.integers(0, len(arr), len(arr))].mean()) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------- #
def dry_run(args) -> None:
    """Validates the full stack without executing the benchmark."""
    checks: List[tuple] = []

    def check(name: str, fn):
        try:
            fn()
            checks.append((name, "OK", ""))
        except Exception as e:  # noqa: BLE001
            checks.append((name, "FAIL", f"{type(e).__name__}: {e}"))

    check("python/torch import", lambda: __import__("torch"))
    check("CUDA available", lambda: (_ for _ in ()).throw(RuntimeError("CUDA not available"))
          if not __import__("torch").cuda.is_available() else None)
    if args.device == "cuda" and not args.dry_run:
        pass

    check("CUDA device", lambda: __import__("torch").cuda.get_device_name(0))
    check("benchmarks package", lambda: __import__("benchmarks"))
    check("metrics.official_metrics", lambda: __import__("metrics.official_metrics", fromlist=["x"]))
    check("baselines package", lambda: __import__("baselines"))
    check("engine.llm", lambda: __import__("engine.llm", fromlist=["LLMEngine"]))
    check("tokenizer load", lambda: __import__("transformers").AutoTokenizer.from_pretrained(args.model))
    check("dataset access (1 example)",
          lambda: build_examples(argparse.Namespace(**{**vars(args), "limit": 1,
                                                       "context_lengths": str(min(int(x) for x in args.context_lengths.split(",")))}))[0])
    check("kvpress import", lambda: __import__("kvpress"))
    check("output dir writable", lambda: os.makedirs(args.output or "results/_dryrun", exist_ok=True))

    print("\nDRY RUN — environment validation")
    failed = 0
    for name, status, msg in checks:
        print(f"  [{status:4s}] {name}" + (f"  — {msg}" if msg and status == "FAIL" else ""))
        failed += status == "FAIL"
    if failed:
        print(f"\n{failed} check(s) FAILED — fix before running the benchmark.")
        sys.exit(1)
    print("\nAll checks passed. The benchmark is ready to run (without --dry-run).")


# ---------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    if args.dry_run:
        # validate before anything else — including the `all` dispatch
        dry_run(args)
        return

    if args.benchmark == "all":
        # run the three dataset benchmarks sequentially in separate processes
        for bench in ["ruler", "longbench", "locomo"]:
            rc = os.system(
                f"{sys.executable} {__file__} --benchmark {bench} "
                f"--systems {args.systems} --model {args.model} --dtype {args.dtype} "
                f"--device {args.device} --seed {args.seed} "
                + (f"--limit {args.limit} " if args.limit else "")
                + f"--context-lengths {args.context_lengths} --ruler-tasks {args.ruler_tasks} "
                f"--longbench-tasks {args.longbench_tasks} --k {args.k} --m {args.m} "
                f"--max-new-tokens {args.max_new_tokens} --resume" +
                (f" --run-name {args.run_name}_{bench}" if args.run_name else "")
            )
            if rc != 0:
                sys.exit(rc)
        return

    t_start = time.time()
    run_name = args.run_name or f"{args.benchmark}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join("results", run_name)
    cfg = {**vars(args), "run_name": run_name}
    init_run_dir(out_dir, cfg)

    examples = build_examples(args)
    print(f"[{run_name}] {len(examples)} examples loaded.")

    llm = LLMEngine(model_id=args.model, dtype=args.dtype, device=args.device,
                    max_new_tokens=args.max_new_tokens, seed=args.seed)
    print(f"[{run_name}] model {args.model} ({args.dtype}) loaded on {args.device}.")
    engine_info = llm.info()

    systems = build_systems(args, llm.tokenizer)
    print(f"[{run_name}] systems: {list(systems)}")

    # resume support: completed (system, example) pairs are skipped, but
    # ERRORED pairs are retried — a fixed bug must be able to heal a run
    done = set()
    pred_path = os.path.join(out_dir, "predictions.jsonl")
    if args.resume and os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "score" in r:
                        done.add((r["system"], r["example_id"]))
                except Exception:
                    continue

    def run_system(sys_name: str, system, engine) -> None:
        """Runs one system over all examples (skipping completed pairs)."""
        for ex in examples:
            if (sys_name, ex["id"]) in done:
                continue
            try:
                system.ingest(ex["context"])
                out = system.answer(ex["question"], engine)
                score = score_example(ex["metric"], out["answer"], ex["gold"])
                row = {
                    "system": sys_name,
                    "example_id": ex["id"],
                    "benchmark": args.benchmark,
                    "task": ex["meta"].get("task", args.benchmark),
                    "metric": ex["metric"],
                    "gold": ex["gold"],
                    "score": score,
                    "answer": out["answer"],
                    "active_tokens": out["active_tokens"],
                    "latency_ms": out.get("generation_latency_ms", out.get("latency_ms")),
                    "peak_vram_mb": out.get("peak_vram_mb"),
                    "selected_ids": out.get("selected_ids"),
                    "n_blocks": out.get("n_blocks", out.get("n_blocks_ingested")),
                    "timestamp": time.time(),
                }
                append_prediction(out_dir, row)
                print(f"  {sys_name:24s} {ex['id'][:28]:28s} score={score:.3f} "
                      f"tok={row['active_tokens']} lat={row['latency_ms']:.0f}ms")
            except Exception as e:  # noqa: BLE001 — record failure, continue with other systems
                row = {"system": sys_name, "example_id": ex["id"], "error": f"{type(e).__name__}: {e}",
                       "timestamp": time.time()}
                append_prediction(out_dir, row)
                print(f"  {sys_name:24s} {ex['id'][:28]:28s} ERROR: {e}")

    # Two-phase execution to fit 8GB: (1) shared-engine text systems with the
    # single model resident; (2) kvpress systems ONE AT A TIME after the
    # shared engine is released, because each press carries its own model.
    shared = {n: s for n, s in systems.items() if not n.startswith("kvpress_")}
    presses = {n: s for n, s in systems.items() if n.startswith("kvpress_")}
    for sys_name, system in shared.items():
        run_system(sys_name, system, llm)
    if presses:
        llm.cleanup()
        for sys_name, system in presses.items():
            run_system(sys_name, system, None)
            if hasattr(system, "cleanup"):
                system.cleanup()

    # aggregate
    with open(pred_path) as f:
        rows = [json.loads(line) for line in f]
    scored_pairs = {(r["system"], r["example_id"]) for r in rows if "score" in r}
    metrics: Dict[str, Any] = {"benchmark": args.benchmark, "engine": engine_info,
                               "n_examples": len(examples), "systems": {}}
    for sys_name in systems:
        srows = [r for r in rows if r["system"] == sys_name and "score" in r]
        # count an error only if that pair never produced a score (retried
        # pairs may have both an old error row and a new score row)
        errs = [r for r in rows if r["system"] == sys_name and "error" in r
                and (r["system"], r["example_id"]) not in scored_pairs]
        if not srows:
            metrics["systems"][sys_name] = {"n_scored": 0, "n_errors": len(errs),
                                            "errors": [e["error"] for e in errs[:5]]}
            continue
        scores = [r["score"] for r in srows]
        lo, hi = bootstrap_ci(scores, seed=args.seed)
        metrics["systems"][sys_name] = {
            "n_scored": len(scores),
            "n_errors": len(errs),
            "metric": srows[0]["metric"],
            "score_mean": float(np.mean(scores)),
            "score_ci95": [lo, hi],
            "active_tokens_mean": float(np.mean([r["active_tokens"] for r in srows])),
            "latency_ms_mean": float(np.mean([r["latency_ms"] for r in srows if r["latency_ms"]] or [0])),
            "peak_vram_mb_mean": float(np.mean([r["peak_vram_mb"] for r in srows if r["peak_vram_mb"]] or [0])),
            "per_task": {},
        }
        for task in sorted({r["task"] for r in srows}):
            ts = [r["score"] for r in srows if r["task"] == task]
            metrics["systems"][sys_name]["per_task"][task] = {
                "n": len(ts), "score_mean": float(np.mean(ts))}
    finalize_metrics(out_dir, metrics)
    print(f"[{run_name}] done in {(time.time()-t_start)/60:.1f} min -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
