"""B7: Iterative (two-round) dense-RAG baseline (uniform `System` interface).

Round 1 retrieves top-m blocks with the existing DenseRAG machinery and asks
the LLM for a draft answer; round 2 retrieves top-m blocks with the expanded
query (query + draft) and the final answer is generated from the UNION of both
rounds' blocks. Dense retrieval is reused via composition (no duplicated
embedding code).

2 LLM calls per question. active_tokens = both GenResults' prompt_tokens summed.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    build_memory_prompt,
    require_ingested,
)
from .dense_rag import DenseRAGBaseline

DRAFT_SUFFIX = "\nDraft (may be incomplete — used only for retrieval):"
MAX_DRAFT_CHARS = 1000


class IterativeRAGBaseline:
    """Two-round retrieve-draft-retrieve-answer loop over dense cosine retrieval."""

    def __init__(
        self,
        tokenizer,
        m: int = 1,
        block_size: int = 512,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer = tokenizer
        self.m = m
        self.block_size = block_size
        self.system_prompt = system_prompt
        self.blocks: List[str] = []
        # Reuse — never duplicate — the DenseRAG embedding/retrieval machinery.
        self._rag = DenseRAGBaseline(
            tokenizer=tokenizer,
            m=m,
            block_size=block_size,
            embedder_id=embedder_id,
            device=device,
            system_prompt=system_prompt,
        )

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        self._rag.ingest(context)
        self.blocks = self._rag.blocks

    @property
    def ingestion(self) -> Dict[str, Any]:
        return self._rag.ingestion

    def answer(self, query: str, llm) -> Dict[str, Any]:
        require_ingested(self.blocks, type(self).__name__)
        t0 = time.perf_counter()

        # ---- Round 1: retrieve top-m, draft answer -------------------------
        ids1, scores1 = self._rag.retrieve(query)
        draft_prompt = build_memory_prompt(query, [self.blocks[i] for i in ids1],
                                           self.system_prompt) + DRAFT_SUFFIX
        draft_gen = llm.generate(draft_prompt)
        retrieval_ms1 = (time.perf_counter() - t0) * 1000.0

        # ---- Round 2: retrieve with query + draft, answer from union -------
        draft_snippet = " ".join(draft_gen.text.split())[:MAX_DRAFT_CHARS]
        expanded = f"{query} {draft_snippet}".strip()
        t1 = time.perf_counter()
        ids2, scores2 = self._rag.retrieve(expanded)
        retrieval_ms2 = (time.perf_counter() - t1) * 1000.0

        union_ids = list(ids1) + [i for i in ids2 if i not in ids1]
        union_blocks = [self.blocks[i] for i in union_ids]
        final_prompt = build_memory_prompt(query, union_blocks, self.system_prompt)
        final_core = len(self.tokenizer.encode(final_prompt, add_special_tokens=False))
        final_gen = llm.generate(final_prompt)

        return {
            "answer": final_gen.text,
            "prompt": final_gen.prompt,
            "prompts": [draft_gen.prompt, final_gen.prompt],
            "active_tokens": draft_gen.prompt_tokens + final_gen.prompt_tokens,
            "active_tokens_core": final_core,
            "selected_ids": union_ids,
            "round1_ids": ids1,
            "round2_ids": ids2,
            "round1_scores": [float(s) for s in scores1],
            "round2_scores": [float(s) for s in scores2],
            "selection_scores": None,
            "selection_latency_ms": retrieval_ms1 + retrieval_ms2,
            "generation_latency_ms": draft_gen.latency_ms + final_gen.latency_ms,
            "completion_tokens": draft_gen.completion_tokens + final_gen.completion_tokens,
            "peak_vram_mb": max(draft_gen.peak_vram_mb, final_gen.peak_vram_mb),
            "n_blocks": len(self.blocks),
            "n_llm_calls": 2,
            "stop_reason": final_gen.stop_reason,
            "extra": {
                "system": "iterative_rag",
                "retrieval": "dense cosine, 2 rounds (query; query+draft), union",
                "draft_answer": draft_gen.text,
                "expanded_query": expanded,
                "ingestion": self._rag.ingestion,
            },
        }

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "iterative_rag",
            "m": self.m,
            "block_size": self.block_size,
            "embedder_id": self._rag.embedder_id,
            "embedder_device": self._rag.device,
            "n_blocks": len(self.blocks),
            "n_llm_calls": 2,
            "ingestion_tokens": self._rag.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self._rag.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
