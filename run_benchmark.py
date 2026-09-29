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
                   help="comma list (see system_specs.py): full, sempointer, "
                        "sempointer[key=val;key=val] (kwargs -> SemPointerPipeline), "
                        "rag|bm25|hybrid[:m], rag_budget|bm25_budget|hybrid_budget[:tokens], "
                        "kvpress:<press>:<ratio> (presses: snapkv,pyramidkv,knorm,tova)")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dtype", default="nf4", choices=["bf16", "nf4"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="max examples per task/conversation (None = all)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--k", type=int, default=8, help="SemPointer pointer length")
    p.add_argument("--m", type=int, default=1, help="SemPointer resolved blocks")
    p.add_argument("--rag-m", default="1",
                   help="blocks retrieved by plain rag/bm25/hybrid specs; a comma list "
                        "(e.g. 1,2,4,8) expands each into rag_m1, rag_m2, ...")
    p.add_argument("--token-budget", type=int, default=None,
                   help="token-budget mode for plain rag/bm25/hybrid specs and default for "
                        "rag_budget/bm25_budget/hybrid_budget: fill top-ranked blocks until "
                        "this many block tokens")
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


def parse_specs(args):
    """Pure spec parsing (no model loading) — also used by --dry-run."""
    from system_specs import parse_system_specs

    rag_m = [int(x) for x in str(args.rag_m).split(",") if x.strip()]
    return parse_system_specs(args.systems, rag_m=rag_m, token_budget=args.token_budget)


def system_class(kind: str):
    if kind == "sempointer":
        from sempointer.pipeline import SemPointerPipeline
        return SemPointerPipeline
    if kind == "full":
        from baselines.full_context import FullContextBaseline
        return FullContextBaseline
    if kind == "bm25":
        from baselines.bm25_baseline import BM25Baseline
        return BM25Baseline
    if kind == "rag":
        from baselines.dense_rag import DenseRAGBaseline
        return DenseRAGBaseline
    if kind == "hybrid":
        from baselines.hybrid_rrf import HybridRRFBaseline
        return HybridRRFBaseline
    if kind == "kvpress":
        from baselines.kvpress_baseline import KVPressBaseline
        return KVPressBaseline
    raise ValueError(f"Unknown system kind: {kind!r}")


def default_kwargs(kind: str, args) -> Dict[str, Any]:
    """Run-level defaults; filtered to what each constructor accepts, spec kwargs win."""
    common = {"block_size": args.block_size, "device": args.device, "seed": args.seed}
    if kind == "sempointer":
        return {**common, "k": args.k, "m": args.m}
    if kind == "kvpress":
        return {"model_id": args.model, "dtype": args.dtype, "device": args.device,
                "max_new_tokens": args.max_new_tokens, "seed": args.seed}
    return common


def build_systems(args, tokenizer):
    """Instantiates every requested system. Missing deps / unknown kwargs raise loudly."""
    from system_specs import merge_kwargs

    systems: Dict[str, Any] = {}
    for spec in parse_specs(args):
        cls = system_class(spec.kind)
        kw = merge_kwargs(cls, default_kwargs(spec.kind, args), spec.kwargs, spec.raw)
        if spec.kind != "kvpress":
            kw["tokenizer"] = tokenizer
        systems[spec.name] = cls(**kw)
    return systems


# optional per-row fields recorded when a system returns them
OPTIONAL_ROW_FIELDS = ("selection_latency_ms", "ttft_ms", "n_hops", "expanded_ids", "kv_bytes",
                       "active_tokens_core", "n_selected", "completion_tokens",
                       "kv_encode_ms", "computed_tokens", "reused_tokens", "transfer_ms",
                       "rope_mixed_frequencies")


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


def gold_string_block_hit(task: str, gold: List[str], blocks: List[str],
                          selected_ids: List[int]) -> float | None:
    """Gold-string coverage for RULER NIAH, where answers occur in source text.

    Other tasks need annotated evidence spans; answer correctness is not a
    substitute for retrieval recall. A split or absent gold string is missing
    evidence, not a retrieval failure.
    """
    if not task.startswith("niah_") or not gold or not blocks or selected_ids is None:
        return None
    folded = [block.casefold() for block in blocks]
    gold_locations = [[i for i, block in enumerate(folded) if answer.casefold() in block]
                      for answer in gold if answer]
    if len(gold_locations) != len(gold) or any(not locations for locations in gold_locations):
        return None
    selected = set(selected_ids)
    return float(all(any(i in selected for i in locations) for locations in gold_locations))


def _safe_info(system) -> Dict[str, Any]:
    try:
        return system.info()
    except Exception as e:  # noqa: BLE001 - info is provenance only
        return {"info_error": f"{type(e).__name__}: {e}"}


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

    def _check_specs():
        from system_specs import validate_kwargs
        for spec in parse_specs(args):
            validate_kwargs(system_class(spec.kind), spec.kwargs, spec.raw)
        print("  systems: " + ", ".join(s.name for s in parse_specs(args)))

    check("--systems spec parse + kwarg validation", _check_specs)
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
        import shlex
        for bench in ["ruler", "longbench", "locomo"]:
            rc = os.system(
                f"{sys.executable} {__file__} --benchmark {bench} "
                f"--systems {shlex.quote(args.systems)} --model {args.model} --dtype {args.dtype} "
                f"--rag-m {shlex.quote(str(args.rag_m))} --block-size {args.block_size} "
                + (f"--token-budget {args.token_budget} " if args.token_budget else "")
                + f"--device {args.device} --seed {args.seed} "
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
    init_run_dir(out_dir, cfg, resume=args.resume)

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
                ingest_start = time.perf_counter()
                system.ingest(ex["context"])
                ingest_ms = (time.perf_counter() - ingest_start) * 1000.0
                answer_start = time.perf_counter()
                out = system.answer(ex["question"], engine)
                answer_ms = (time.perf_counter() - answer_start) * 1000.0
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
                    "latency_ms": answer_ms,
                    "ingest_ms": ingest_ms,
                    "generation_latency_ms": out.get("generation_latency_ms", out.get("latency_ms")),
                    "peak_vram_mb": out.get("peak_vram_mb"),
                    "selected_ids": out.get("selected_ids"),
                    "n_blocks": out.get("n_blocks", out.get("n_blocks_ingested")),
                    "context_length": ex["meta"].get("context_length_requested"),
                    "timestamp": time.time(),
                }
                for fld in OPTIONAL_ROW_FIELDS:
                    if out.get(fld) is not None:
                        row[fld] = out[fld]
                hit = gold_string_block_hit(row["task"], ex["gold"],
                                            getattr(system, "blocks", None),
                                            out.get("selected_ids"))
                if hit is not None:
                    row["retrieval_hit"] = hit
                    row["retrieval_basis"] = "gold string present in selected RULER NIAH block"
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
    from bench_metrics import aggregate

    agg = aggregate(rows, list(systems), seed=args.seed)
    metrics: Dict[str, Any] = {"benchmark": args.benchmark, "engine": engine_info,
                               "n_examples": len(examples),
                               "system_info": {n: _safe_info(s) for n, s in systems.items()},
                               **agg}
    finalize_metrics(out_dir, metrics)
    print(f"[{run_name}] done in {(time.time()-t_start)/60:.1f} min -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
