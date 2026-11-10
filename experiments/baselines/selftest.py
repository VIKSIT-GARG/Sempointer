#!/usr/bin/env python
"""Selftest for the reconstructed baselines (script, not pytest).

Runs REAL end-to-end generation for every system on one shared context and
question, prints answers / active tokens / latencies, and cleans GPU memory
between systems.

1.  Shared LLMEngine (Qwen2.5-1.5B-Instruct, bf16, greedy, max_new_tokens=48).
2.  full / bm25 / rag baselines ingest the context and answer through the
    shared engine.
3.  SemPointerPipeline (reference implementation) answers the same question
    for comparison.
4.  The shared engine is freed, then KVPressBaseline("snapkv", ratio=0.3)
    loads its own model instance (kvpress modifies attention) and answers.
5.  Every system prints its measured active_tokens.
"""

from __future__ import annotations

import gc
import sys
import traceback
from pathlib import Path

import torch

EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from engine.llm import LLMEngine  # noqa: E402
from baselines.bm25_baseline import BM25Baseline  # noqa: E402
from baselines.dense_rag import DenseRAGBaseline  # noqa: E402
from baselines.full_context import FullContextBaseline  # noqa: E402

CONTEXT = (
    "The Mars rover Perseverance landed in Jezero Crater in February 2021. "
    "Its companion helicopter Ingenuity flew the first powered flight on another planet in April 2021. "
    "The rover's primary mission lasted until 2023, when it entered an extended phase. "
    "Perseverance collected rock samples in the MCS region for later return to Earth. "
    "The mission is operated by NASA's Jet Propulsion Laboratory in Pasadena, California."
)
QUERY = "Where is NASA's Mars mission operated from?"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 48


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report(name: str, out: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"answer        : {out['answer']!r}")
    print(f"active_tokens : {out['active_tokens']}")
    if "generation_latency_ms" in out:
        print(f"latency_ms    : {out['generation_latency_ms']:.1f}")
    if "peak_vram_mb" in out:
        print(f"peak_vram_mb  : {out['peak_vram_mb']:.0f}")
    if out.get("selected_ids") is not None:
        print(f"selected_ids  : {out['selected_ids']}")
        print(f"scores        : {out.get('selection_scores')}")
    assert isinstance(out["answer"], str) and out["answer"].strip(), f"{name}: empty answer"
    assert isinstance(out["active_tokens"], int) and out["active_tokens"] > 0, (
        f"{name}: bad active_tokens {out['active_tokens']!r}"
    )


def main() -> int:
    failures: list[str] = []

    llm = LLMEngine(MODEL_ID, "bf16", "cuda", max_new_tokens=MAX_NEW_TOKENS)
    print("engine:", llm.info())

    # ---- text baselines through the SHARED engine ------------------------
    text_systems = [
        ("full_context", FullContextBaseline(llm.tokenizer)),
        ("bm25", BM25Baseline(llm.tokenizer, m=1)),
        ("rag", DenseRAGBaseline(llm.tokenizer, m=1)),
    ]
    for name, system in text_systems:
        try:
            system.ingest(CONTEXT)
            print(f"\n[{name}] info: {system.info()}")
            out = system.answer(QUERY, llm)
            report(name, out)
        except Exception:
            failures.append(name)
            print(f"\n[{name}] FAILED:\n{traceback.format_exc()}")
        finally:
            del system
            free_gpu()

    # ---- retrieval sanity on a multi-block corpus (no LLM needed) ---------
    # The mandated smoke context fits a single 512-token block, so retrieval
    # would be trivially top-1-of-1. This check verifies BM25/dense ranking
    # picks the only Mars-related block among pure-bakery distractor blocks.
    # Block 0 (tokens 0..127) fully contains the 96-token Mars context, so a
    # correct top-1 must be block 0 and must contain the JPL phrase.
    try:
        distractor = "The bakery on Elm Street sells sourdough bread every morning. "
        multi = CONTEXT + " " + distractor * 30
        bm25s = BM25Baseline(llm.tokenizer, m=1, block_size=128)
        bm25s.ingest(multi)
        ids, sc = bm25s.retrieve(QUERY)
        assert ids[0] == 0 and "Jet Propulsion Laboratory" in bm25s.blocks[0], (
            f"bm25 top-1 irrelevant: ids={ids} score={sc}"
        )
        rags = DenseRAGBaseline(llm.tokenizer, m=1, block_size=128)
        rags.ingest(multi)
        ids2, sc2 = rags.retrieve(QUERY)
        assert ids2[0] == 0 and "Jet Propulsion Laboratory" in rags.blocks[0], (
            f"rag top-1 irrelevant: ids={ids2} cosine={sc2}"
        )
        print(
            f"\n[retrieval-sanity] n_blocks={len(bm25s.blocks)} "
            f"bm25_top1={ids[0]} score={sc[0]:.3f} | rag_top1={ids2[0]} cosine={sc2[0]:.3f} -> both correct"
        )
        del bm25s, rags
        free_gpu()
    except Exception:
        failures.append("retrieval-sanity")
        print(f"\n[retrieval-sanity] FAILED:\n{traceback.format_exc()}")

    # ---- SemPointer reference run (comparison) ---------------------------
    try:
        from sempointer.pipeline import SemPointerPipeline

        sp = SemPointerPipeline(tokenizer=llm.tokenizer, m=1)
        sp.ingest(CONTEXT)
        out = sp.answer(QUERY, llm)
        report("sempointer", out)
        del sp
        free_gpu()
    except Exception:
        failures.append("sempointer")
        print(f"\n[sempointer] FAILED:\n{traceback.format_exc()}")

    # ---- kvpress baseline (own model instance; free shared engine first) --
    try:
        from baselines.kvpress_baseline import KVPressBaseline

        llm.cleanup()
        del llm
        free_gpu()

        kv = KVPressBaseline(
            "snapkv",
            compression_ratio=0.3,
            model_id=MODEL_ID,
            dtype="bf16",
            device="cuda",
            max_new_tokens=MAX_NEW_TOKENS,
        )
        kv.ingest(CONTEXT)
        print(f"\n[kvpress:snapkv] info: {kv.info()}")
        out = kv.answer(QUERY, llm=None)  # shared engine intentionally freed
        report("kvpress:snapkv", out)
        print(f"extra         : {out['extra']}")
        kv.cleanup()
        del kv
        free_gpu()
    except Exception:
        failures.append("kvpress:snapkv")
        print(f"\n[kvpress:snapkv] FAILED:\n{traceback.format_exc()}")

    print("\n==================== SUMMARY ====================")
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print("ALL SYSTEMS PASSED (real generation, non-empty answers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
