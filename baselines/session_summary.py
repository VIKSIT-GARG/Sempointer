"""B9: Session-summary (MemGPT-lite) baseline (uniform `System` interface).

Maintains an incrementally summarized session memory: the running summary is
updated chunk-by-chunk with one LLM call per ingested block
("Previous summary + new chunk -> updated concise summary"), then the final
answer is generated from (summary + top-1 raw block + query).

Because the uniform `System` contract passes the LLM only to answer() — not to
ingest() — the rolling summarization runs lazily on the first answer() call
and is cached for subsequent questions in the same session. Every summarizer
call is wrapped in try/except with an extractive-truncation fallback, so a
failed summary update degrades to concatenation instead of crashing.

Top-1 raw block selection reuses BM25Baseline (lightweight, no embedder).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    ingest_blocks,
    require_ingested,
)
from .bm25_baseline import BM25Baseline

SUMMARY_CHUNK_CHARS = 1500
SUMMARY_FALLBACK_CHARS = 500

SUMMARY_UPDATE_TEMPLATE = (
    "Maintain a concise running summary of a long session. "
    "Merge the new memory chunk into the previous summary without losing facts "
    "needed to answer future questions. Reply with ONLY the updated summary.\n\n"
    "Previous summary:\n{summary}\n\nNew memory chunk ({i}/{n}):\n{chunk}\n\nUpdated summary:"
)


class SessionSummaryBaseline:
    """Incremental LLM session summary + top-1 raw block + real LLM answer."""

    def __init__(
        self,
        tokenizer,
        block_size: int = 512,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.system_prompt = system_prompt
        self.blocks: List[str] = []
        self.ingestion: Dict[str, Any] = {}
        self.summary: Optional[str] = None
        self.summary_calls: int = 0
        self.summary_prompt_tokens: int = 0
        self._bm25 = BM25Baseline(tokenizer=tokenizer, m=1, block_size=block_size)

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        self.blocks, self.ingestion = ingest_blocks(context, self.block_size, self.tokenizer)
        self._bm25.ingest(context)  # same chunking -> aligned blocks
        # Summary is (re)built lazily in answer(): ingest() receives no LLM.
        self.summary = None
        self.summary_calls = 0
        self.summary_prompt_tokens = 0

    def _ensure_summary(self, llm) -> None:
        """Rolls the summary forward over all blocks (cached per session)."""
        if self.summary is not None:
            return
        running = ""
        n = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            chunk = " ".join(blk.split())[:SUMMARY_CHUNK_CHARS]
            prompt = SUMMARY_UPDATE_TEMPLATE.format(
                summary=running or "(empty — this is the first chunk)",
                i=i + 1, n=n, chunk=chunk)
            try:
                gen = llm.generate(prompt)
                self.summary_calls += 1
                self.summary_prompt_tokens += gen.prompt_tokens
                running = gen.text.strip() or running
            except Exception:
                # Extractive fallback: never let a summary call kill the run.
                running = (running + " " + chunk[:SUMMARY_FALLBACK_CHARS]).strip()
        self.summary = running or " ".join(
            " ".join(b.split())[:SUMMARY_FALLBACK_CHARS] for b in self.blocks)

    def answer(self, query: str, llm) -> Dict[str, Any]:
        require_ingested(self.blocks, type(self).__name__)
        t0 = time.perf_counter()
        self._ensure_summary(llm)
        summary_ms = (time.perf_counter() - t0) * 1000.0

        # Top-1 raw block grounds the summary (BM25 reuse, no embedder load).
        try:
            top_ids, top_scores = self._bm25.retrieve(query)
        except Exception:
            top_ids, top_scores = ([0], [0.0]) if self.blocks else ([], [])
        top_block = self.blocks[top_ids[0]] if top_ids else ""

        sections: List[str] = []
        if self.system_prompt:
            sections.append(f"System: {self.system_prompt.strip()}")
        sections.append(f"Session Summary:\n{self.summary.strip()}")
        if top_block:
            sections.append(f"Top Raw Memory Block:\n{top_block.strip()}")
        sections.append(f"Question: {query.strip()}\nAnswer:")
        final_prompt = "\n\n".join(sections)
        final_core = len(self.tokenizer.encode(final_prompt, add_special_tokens=False))
        final_gen = llm.generate(final_prompt)

        n_calls = self.summary_calls + 1
        return {
            "answer": final_gen.text,
            "prompt": final_gen.prompt,
            "active_tokens": self.summary_prompt_tokens + final_gen.prompt_tokens,
            "active_tokens_core": final_core,
            "selected_ids": top_ids,
            "selection_scores": [float(s) for s in top_scores],
            "selection_latency_ms": summary_ms,
            "summarization_latency_ms": summary_ms,
            "generation_latency_ms": final_gen.latency_ms,
            "completion_tokens": final_gen.completion_tokens,
            "peak_vram_mb": final_gen.peak_vram_mb,
            "n_blocks": len(self.blocks),
            "n_llm_calls": n_calls,
            "stop_reason": final_gen.stop_reason,
            "extra": {
                "system": "session_summary",
                "retrieval": "incremental LLM session summary + top-1 BM25 raw block",
                "session_summary": self.summary,
                "summary_llm_calls": self.summary_calls,
                "summary_prompt_tokens": self.summary_prompt_tokens,
                "ingestion": self.ingestion,
            },
        }

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "session_summary",
            "block_size": self.block_size,
            "n_blocks": len(self.blocks),
            "summary_ready": self.summary is not None,
            "summary_llm_calls": self.summary_calls,
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
