#!/usr/bin/env python
"""STAT worker: error taxonomy over the four final benchmark suites.

CPU-only; NEVER loads model weights or runs inference. Pure data analysis:
reconstructs benchmark contexts deterministically (seed 42, cached official
data), re-chunks them with the exact production rule
(sempointer.pipeline.chunk_text, block_size=512, Qwen2.5-3B tokenizer),
localizes gold-bearing blocks, and joins with the logged selected_ids.

Reads:
    experiments/results/final_{ruler,longbench,locomo,kvcompare}/predictions.jsonl
    experiments/benchmarks/data/{ruler_cache,longbench_cache/data.zip,locomo_cache/locomo10.json}

Classes (sparse systems: sempointer, rag, bm25):
  * error-row            full-system OOM rows (no scored row for the pair)
  * format/empty         scored row whose answer is empty/whitespace-only
  * retrieval-miss       miss AND (selected_ids ∩ gold_blocks = ∅)
  * block-hit-wrong-answer
                         miss AND (selected_ids ∩ gold_blocks ≠ ∅)
  * gold-not-localizable miss AND no block contains any gold string
                         (abstractive answers; attribution impossible)

Miss definition: score < 1.0 for containment metrics (ruler, kv_compare);
score < 0.5 for F1 metrics (longbench, locomo) -- a documented leniency
choice (mean F1 is ~0.05-0.11, so score<1.0 would label ~everything a miss);
exact-zero subcounts are reported so the threshold can be re-cut.

Per-task tables + per-hop-count breakdown for hotpotqa/2wikimqa/musique.
Hop-count labels are NOT present in official LongBench items or in
predictions.jsonl, so per-task tables ARE the hop breakdown, with the
dataset-design mapping stated as limitation (hotpotqa 2-hop, 2wikimqa 2-hop,
musique 2-4-hop by construction). LoCoMo categories ARE joined from the
official file (cat 1 = multi-hop).

Encoder-truncation exposure: prompts/block texts are NOT stored in
predictions.jsonl, so block MiniLM-token lengths are recomputed from the
regenerated blocks with the MiniLM tokenizer (tokenizer-only, no weights).
Window = 256 (embedder max_seq_length used by embed_long). The production
run used windowed mean-pooling (embed_long), so no silent truncation
occurred; we quantify how many blocks a NAIVE single-pass encoder would
have truncated.

Also writes the True-B config table (results/config_table.json): per suite
measured mean n_blocks, mean LLM-tokens per block (B_true), k=8, m=1,
kappa = k / B_true.

Writes:
    experiments/results/error_breakdown.json
    experiments/results/config_table.json
"""

from __future__ import annotations

import json
import re
import string
import sys
import zipfile
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
RESULTS = EXP / "results"

from sempointer.pipeline import chunk_text  # noqa: E402  (exact production rule)

BLOCK_SIZE = 512
K, M = 8, 1
MINILM_WINDOW = 256  # embedder max_seq_length consumed by embed_long
SPARSE_SYSTEMS = ["sempointer", "rag", "bm25"]
HOP_DESIGN_NOTE = {
    "hotpotqa": "2-hop by dataset construction (HotpotQA bridge/comparison)",
    "2wikimqa": "2-hop by dataset construction",
    "musique": "2-4-hop by dataset construction (MuSiQue compositional)",
}

SUITES = {
    "final_ruler": {"tasks": None},   # ruler regen below
    "final_longbench": {},
    "final_locomo": {},
    "final_kvcompare": {},
}


def norm(s: str) -> str:
    """SQuAD-style normalization (mirrors metrics/official_metrics.normalize_answer)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def load_predictions(suite: str):
    rows = [json.loads(line) for line in (RESULTS / suite / "predictions.jsonl").read_text().splitlines()
            if line.strip()]
    scored = {(r["system"], r["example_id"]): r for r in rows if "score" in r}
    errors = [r for r in rows if "error" in r]
    return rows, scored, errors


def regen_contexts():
    """example_id -> {context, gold, task, suite, extra} for every scored id."""
    from transformers import AutoTokenizer
    llm_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    ctx: dict[str, dict] = {}

    # RULER (final_ruler): deterministic synthetic regen, seed 42.
    from benchmarks.ruler import build_ruler
    cfg = json.loads((RESULTS / "final_ruler" / "config.json").read_text())
    for task in cfg["ruler_tasks"].split(","):
        for clen in [int(x) for x in cfg["context_lengths"].split(",")]:
            for ex in build_ruler(task=task, context_length=clen, n_samples=10, seed=42):
                d = ex.to_dict()
                ctx[d["id"]] = {"context": d["context"], "gold": d["gold"],
                                "task": task, "suite": "final_ruler"}
    # kvcompare: ruler subset at 4096, n=5.
    cfgk = json.loads((RESULTS / "final_kvcompare" / "config.json").read_text())
    for task in cfgk["ruler_tasks"].split(","):
        for clen in [int(x) for x in cfgk["context_lengths"].split(",")]:
            for ex in build_ruler(task=task, context_length=clen, n_samples=5, seed=42):
                d = ex.to_dict()
                if d["id"] not in ctx:
                    ctx[d["id"]] = {"context": d["context"], "gold": d["gold"],
                                    "task": task, "suite": "final_kvcompare"}

    # LongBench: official cached data.zip, match by official _id.
    zpath = EXP / "benchmarks" / "data" / "longbench_cache" / "data.zip"
    with zipfile.ZipFile(zpath) as z:
        for task in ["hotpotqa", "2wikimqa", "musique", "multifieldqa_en", "narrativeqa", "qasper"]:
            with z.open(f"data/{task}.jsonl") as f:
                for line in f.read().decode("utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    eid = f"longbench-{task}-{item.get('_id')}"
                    gold = item["answers"] if isinstance(item["answers"], list) else [item["answers"]]
                    ctx[eid] = {"context": item["context"], "gold": [str(a) for a in gold],
                                "task": task, "suite": "final_longbench"}

    # LoCoMo: official file, conversation 0 (sample conv-26), category join.
    from benchmarks.locomo import build_locomo
    for ex in build_locomo(conversation_idx=0, categories=None, n_samples=None):
        d = ex.to_dict()
        ctx[d["id"]] = {"context": d["context"], "gold": d["gold"],
                        "task": "locomo_qa", "suite": "final_locomo",
                        "category": d["meta"].get("category")}
    return ctx, llm_tok


def gold_blocks(blocks: list[str], golds: list[str]) -> set[int]:
    """Indices of blocks containing any gold string (raw case-insensitive,
    fallback to normalized containment for abstractive QA answers)."""
    hits: set[int] = set()
    low_blocks = [b.lower() for b in blocks]
    norm_blocks = [norm(b) for b in blocks]
    for g in golds:
        g = str(g)
        if not g.strip():
            continue
        gl, gn = g.lower(), norm(g)
        for i, (lb, nb) in enumerate(zip(low_blocks, norm_blocks)):
            if gl in lb or (gn and gn in nb):
                hits.add(i)
    return hits


def main() -> None:
    from transformers import AutoTokenizer
    ctx, llm_tok = regen_contexts()
    mini_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Chunk every context once; collect block stats for truncation + config table.
    block_cache: dict[str, list[str]] = {}
    llm_lens: dict[str, list[int]] = {}
    mini_lens: dict[str, list[int]] = {}
    for eid, c in ctx.items():
        blocks = chunk_text(c["context"], BLOCK_SIZE, llm_tok)
        block_cache[eid] = blocks
        llm_lens[eid] = [len(llm_tok.encode(b, add_special_tokens=False)) for b in blocks]
        mini_lens[eid] = [len(mini_tok.encode(b, add_special_tokens=False)) for b in blocks]

    breakdown: dict = {
        "method": ("Gold blocks recomputed (prompts/block texts are NOT stored in "
                   "predictions.jsonl): ruler regen seed-42, longbench official "
                   "data.zip, locomo official file; chunked with the exact "
                   "production rule chunk_text(block_size=512)+Qwen tokenizer. "
                   "Miss := score<1.0 (containment) or score<0.5 (F1, lenient; "
                   "exact-zero subcounts reported). retrieval-miss := miss with "
                   "selected_ids ∩ gold_blocks = ∅."),
        "miss_threshold": {"containment": "score < 1.0", "f1": "score < 0.5 (lenient)"},
        "hop_count_limitation": ("Hop-count labels are absent from official LongBench "
                                 "items and predictions.jsonl; per-task tables ARE the hop "
                                 f"breakdown. Design mapping: {HOP_DESIGN_NOTE}."),
        "suites": {},
        "encoder_truncation": {},
    }

    for suite in ["final_ruler", "final_longbench", "final_locomo", "final_kvcompare"]:
        rows, scored, errors = load_predictions(suite)
        metric_kind = "containment" if suite in ("final_ruler", "final_kvcompare") else "f1"
        miss_thr = 1.0 if metric_kind == "containment" else 0.5
        sres: dict = {"metric_kind": metric_kind, "by_system": {}, "by_task": {},
                      "multihop": {}, "locomo_categories": {},
                      "gold_not_localizable_ids": [], "unjoined_ids": []}
        # Truly-missing error rows (no scored row for the pair).
        sres["error_rows_truly_missing"] = [
            {"system": r["system"], "example_id": r["example_id"],
             "error": r["error"][:120]} for r in errors
            if (r["system"], r["example_id"]) not in scored]

        for sys in sorted({s for (s, _) in scored}):
            counts = Counter()
            per_task: dict[str, Counter] = {}
            per_hop: dict[str, Counter] = {}
            per_cat: dict[str, Counter] = {}
            exact_zero = 0
            abstain = 0
            for (s, eid), r in sorted(scored.items()):
                if s != sys:
                    continue
                c = ctx.get(eid)
                if c is None or c["suite"] not in (suite, "final_ruler" if suite == "final_kvcompare" else suite):
                    # kvcompare shares ruler-regen contexts keyed without suite split;
                    # accept ruler-regenerated ids for kvcompare too.
                    if not (suite == "final_kvcompare" and c is not None):
                        sres["unjoined_ids"].append(eid)
                        continue
                task = r.get("task", c["task"] if c else "?")
                tc = per_task.setdefault(task, Counter())
                score = float(r["score"])
                if score == 0.0:
                    exact_zero += 1
                ans = r.get("answer", "") or ""
                if sys in SPARSE_SYSTEMS or True:
                    if ans.strip().lower().startswith("you don't know"):
                        abstain += 1
                hit = score >= miss_thr
                if hit:
                    counts["hit"] += 1
                    tc["hit"] += 1
                    continue
                # ---- miss ----
                if not ans.strip():
                    cls = "format/empty"
                elif c is None:
                    cls = "unjoined-context"
                else:
                    blocks = block_cache[eid]
                    gb = gold_blocks(blocks, c["gold"])
                    if not gb:
                        cls = "gold-not-localizable"
                        sres["gold_not_localizable_ids"].append(
                            {"system": sys, "example_id": eid, "task": task})
                        # Distinguish true-abstractive answers from boundary
                        # fragmentation: gold present in the full context but
                        # split across a 512-token block boundary (no single
                        # block contains it, so even a perfect top-1 retriever
                        # fails). Verified real on RULER NIAH needles.
                        full_low = c["context"].lower()
                        if all(str(g).lower() in full_low for g in c["gold"]
                               if str(g).strip()):
                            sres.setdefault("gold_in_context_but_no_block", [])
                            if eid not in sres["gold_in_context_but_no_block"]:
                                sres["gold_in_context_but_no_block"].append(eid)
                    elif sys not in SPARSE_SYSTEMS:
                        cls = "miss-full-or-kvpress-attribution-NA"
                    else:
                        sel = set(r.get("selected_ids") or [])
                        cls = ("block-hit-wrong-answer" if (sel & gb)
                               else "retrieval-miss")
                counts[cls] += 1
                counts["miss_total"] += 1
                tc[cls] += 1
                tc["miss_total"] += 1
                if task in ("hotpotqa", "2wikimqa", "musique"):
                    hc = per_hop.setdefault(task, Counter())
                    hc[cls] += 1
                    hc["miss_total"] += 1
                if suite == "final_locomo" and c and c.get("category") is not None:
                    cc = per_cat.setdefault(f"cat{c['category']}", Counter())
                    cc[cls] += 1
                    cc["miss_total"] += 1
            sres["by_system"][sys] = {**{k: int(v) for k, v in counts.items()},
                                      "n_scored": int(sum(1 for (s, _) in scored if s == sys)),
                                      "exact_zero": int(exact_zero),
                                      "abstain_you_dont_know": int(abstain)}
            for t, cc in per_task.items():
                sres["by_task"].setdefault(t, {})[sys] = {k: int(v) for k, v in cc.items()}
            for t, cc in per_hop.items():
                sres["multihop"].setdefault(t, {})[sys] = {
                    "design_hops": HOP_DESIGN_NOTE[t],
                    **{k: int(v) for k, v in cc.items()}}
            for t, cc in per_cat.items():
                sres["locomo_categories"].setdefault(t, {})[sys] = {k: int(v) for k, v in cc.items()}
        breakdown["suites"][suite] = sres

    # ---- Encoder-truncation exposure (MiniLM 256-token window) ----
    for suite in ["final_ruler", "final_longbench", "final_locomo", "final_kvcompare"]:
        eids = [e for e, c in ctx.items()
                if c["suite"] == suite or (suite == "final_kvcompare" and c["suite"] == "final_ruler")]
        lens = [L for e in eids for L in mini_lens[e]]
        over = [L for L in lens if L > MINILM_WINDOW]
        import numpy as _np
        breakdown["encoder_truncation"][suite] = {
            "n_blocks": int(len(lens)),
            "n_over_256": int(len(over)),
            "frac_over_256": float(len(over) / len(lens)) if lens else 0.0,
            "max_minilm_tokens": int(max(lens)) if lens else 0,
            "mean_minilm_tokens": float(_np.mean(lens)) if lens else 0.0,
            "method": ("MiniLM-token lengths recomputed with the MiniLM tokenizer "
                       "(tokenizer-only, no weights) over regenerated blocks; window=256 "
                       "per embed_long. Production used windowed mean-pooling, so no "
                       "silent truncation occurred -- counts quantify NAIVE-encoder exposure. "
                       "LoCoMo: cat5 golds are adversarial non-mentions (unlocalizable by design); "
                       "cat2 temporal golds are session-date derivations, not verbatim spans. "
                       "RULER: some NIAH needles fragment across block boundaries "
                       "(gold_in_context_but_no_block) -- unretrievable at m=1 even by a perfect ranker."),
        }

    (RESULTS / "error_breakdown.json").write_text(json.dumps(breakdown, indent=2))
    print("wrote", RESULTS / "error_breakdown.json")

    # ---- True-B config table ----
    import numpy as _np
    cfg_table: dict = {
        "method": ("mean n_blocks from sempointer scored rows in predictions.jsonl "
                   "(cross-checked vs full); B_true = total LLM (Qwen) block-tokens / "
                   "total blocks over regenerated contexts; kappa = k / B_true; "
                   "k=8 pointer tokens/block, m=1 resolved block, rag_m=1, block_size=512."),
        "k": K, "m": M, "rag_m": 1, "block_size": BLOCK_SIZE,
        "suites": {},
    }
    for suite in ["final_ruler", "final_longbench", "final_locomo", "final_kvcompare"]:
        _, scored, _ = load_predictions(suite)
        sp_n = [int(r["n_blocks"]) for (s, _), r in scored.items()
                if s == "sempointer" and r.get("n_blocks")]
        full_n = [int(r["n_blocks"]) for (s, _), r in scored.items()
                  if s == "full" and r.get("n_blocks")]
        eids = [eid for (s, eid) in scored if s == "sempointer"]
        # kvcompare shares ruler contexts; dedupe via block_cache presence
        eids = [e for e in eids if e in block_cache]
        tot_tok = sum(sum(llm_lens[e]) for e in eids)
        tot_blk = sum(len(block_cache[e]) for e in eids)
        b_true = tot_tok / tot_blk if tot_blk else 0.0
        full_act = [float(r["active_tokens"]) for (s, _), r in scored.items()
                    if s == "full" and r.get("active_tokens")]
        cfg_table["suites"][suite] = {
            "n_examples": int(len(eids)),
            "mean_n_blocks_sempointer": float(_np.mean(sp_n)) if sp_n else 0.0,
            "mean_n_blocks_full": float(_np.mean(full_n)) if full_n else 0.0,
            "mean_llm_tokens_per_block_B_true": float(b_true),
            "kappa_k_over_B_true": float(K / b_true) if b_true else 0.0,
            "crosscheck_full_active_tokens_mean": float(_np.mean(full_act)) if full_act else 0.0,
        }
    (RESULTS / "config_table.json").write_text(json.dumps(cfg_table, indent=2))
    print("wrote", RESULTS / "config_table.json")

    # Console headline counts.
    for suite in ["final_ruler", "final_longbench", "final_locomo", "final_kvcompare"]:
        for sys, c in breakdown["suites"][suite]["by_system"].items():
            if sys in SPARSE_SYSTEMS:
                print(f"{suite} {sys}: " +
                      ", ".join(f"{k}={v}" for k, v in sorted(c.items()) if k != "n_scored") +
                      f" (n={c['n_scored']})")


if __name__ == "__main__":
    sys.exit(main())
