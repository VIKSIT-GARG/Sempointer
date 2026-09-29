"""Shared plumbing for the text baselines (full-context, BM25, dense RAG).

All three baselines implement the uniform `System` contract defined by
sempointer.pipeline.SemPointerPipeline:

    ingest(context) -> None
    answer(query, llm) -> {"answer", "prompt", "active_tokens", ...}
    info() -> dict

They share ONE prompt format ("Memory Context:" + numbered blocks +
"Question: {query}") and ONE answer path: a real `LLMEngine.generate` call on
the shared engine instance. Only the context mechanism differs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

try:
    # Block construction must be IDENTICAL across all evaluated systems, so the
    # baselines reuse the exact chunking helper used by SemPointer.
    from sempointer.pipeline import DEFAULT_SYSTEM_PROMPT, chunk_text
except ImportError as _exc:  # pragma: no cover - environment error, not a fallback
    raise ImportError(
        "baselines requires the `sempointer` package (the experiments/ directory "
        "must be on sys.path, e.g. run from experiments/ or add experiments/ to "
        "PYTHONPATH)."
    ) from _exc

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "chunk_text",
    "require_ingested",
    "ingest_blocks",
    "build_memory_prompt",
    "finish_answer",
]


def require_ingested(blocks: Optional[Sequence[str]], system_name: str) -> None:
    if not blocks:
        raise RuntimeError(f"{system_name}.ingest() must be called with a non-empty context before use")


def ingest_blocks(context: str, block_size: int, tokenizer) -> tuple:
    """Chunk `context` and return (blocks, ingestion_stats_dict)."""
    t0 = time.perf_counter()
    blocks = chunk_text(context, block_size, tokenizer)
    ingestion_ms = (time.perf_counter() - t0) * 1000.0
    n_context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
    block_token_counts = [len(tokenizer.encode(b, add_special_tokens=False)) for b in blocks]
    stats = {
        "ingestion_latency_ms": ingestion_ms,
        "ingestion_tokens": n_context_tokens,
        "n_blocks": len(blocks),
        "block_token_counts": block_token_counts,
    }
    return blocks, stats


def build_memory_prompt(query: str, blocks: Sequence[str], system_prompt: Optional[str]) -> str:
    """Uniform prompt: System + "Memory Context:" + numbered blocks + Question.

    Same section style as sempointer.prompt_builder.PromptBuilder (which the
    SemPointer pipeline uses), so only the context mechanism differs.
    """
    sections: List[str] = []
    if system_prompt:
        sections.append(f"System: {system_prompt.strip()}")
    numbered = [f"[MEMORY {i}]:\n{b.strip()}" for i, b in enumerate(blocks)]
    sections.append("Memory Context:\n" + "\n\n".join(numbered))
    sections.append(f"Question: {query.strip()}\nAnswer:")
    return "\n\n".join(sections)


def finish_answer(
    *,
    system: str,
    query: str,
    blocks: Sequence[str],
    llm,
    tokenizer,
    system_prompt: Optional[str],
    selected_ids: Optional[List[int]],
    selection_scores: Optional[List[float]],
    selection_latency_ms: float,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds the active prompt over `blocks` and generates the answer with the
    shared LLMEngine (real greedy generation, same as SemPointer's answer())."""
    prompt = build_memory_prompt(query, blocks, system_prompt)
    core_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    gen = llm.generate(prompt)  # real LLM generation on the shared engine

    result = {
        "answer": gen.text,
        "prompt": gen.prompt,
        "active_tokens": gen.prompt_tokens,
        "active_tokens_core": core_tokens,  # without chat-template framing (none here: raw prompt)
        "selected_ids": selected_ids,
        "selection_scores": selection_scores,
        "selection_latency_ms": selection_latency_ms,
        "generation_latency_ms": gen.latency_ms,
        "completion_tokens": gen.completion_tokens,
        "peak_vram_mb": gen.peak_vram_mb,
        "n_blocks": len(blocks),
        "stop_reason": gen.stop_reason,
        "extra": {"system": system, **extra},
    }
    return result
