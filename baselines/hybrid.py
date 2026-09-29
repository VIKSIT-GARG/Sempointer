"""B8: Hybrid BM25 + dense RRF baseline (uniform `System` interface).

Fuses Okapi BM25 ranks and dense-cosine ranks with reciprocal-rank fusion
(Cormack et al., SIGIR 2009):

    rrf(d) = 1 / (rrf_k + rank_bm25(d)) + 1 / (rrf_k + rank_dense(d))

with 1-indexed ranks over ALL blocks and rrf_k = 60 (standard constant).
The top-m fused blocks are answered through the shared answer path
(`finish_answer`: real LLM call on the shared engine).

Both rankers are reused from the existing modules (BM25Baseline's index and
DenseRAGBaseline's embeddings) — no duplicated scoring code beyond the rank
lookup and the RRF sum itself.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    finish_answer,
    require_ingested,
)
from .bm25_baseline import _tokenize
from .dense_rag import DenseRAGBaseline
from .bm25_baseline import BM25Baseline

RRF_K = 60


class HybridRRFBaseline:
    """RRF(BM25, dense cosine) retrieval + top-m concatenation + real LLM."""

    def __init__(
        self,
        tokenizer,
        m: int = 1,
        block_size: int = 512,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        rrf_k: int = RRF_K,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer = tokenizer
        self.m = m
        self.block_size = block_size
        self.rrf_k = rrf_k
        self.system_prompt = system_prompt
        self.device = device
        self.blocks: List[str] = []
        # Reuse existing rankers; both chunk identically so blocks align.
        self._dense = DenseRAGBaseline(
            tokenizer=tokenizer, m=m, block_size=block_size,
            embedder_id=embedder_id, device=device, system_prompt=system_prompt)
        self._bm25 = BM25Baseline(
            tokenizer=tokenizer, m=m, block_size=block_size, system_prompt=system_prompt)

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        self._dense.ingest(context)
        self._bm25.ingest(context)
        assert self._dense.blocks == self._bm25.blocks, (
            "hybrid rankers disagree on chunking — blocks must align for fusion")
        self.blocks = self._dense.blocks

    def retrieve(self, query: str) -> Tuple[List[int], List[float]]:
        """Top-m block indices and fused RRF scores for `query`."""
        require_ingested(self.blocks, type(self).__name__)
        n = len(self.blocks)
        # Dense-cosine scores over all blocks (same normalized-matmul as DenseRAG).
        with torch.no_grad():
            q = self._dense.embedder.encode(
                [query], convert_to_tensor=True, device=self._dense.device,
                show_progress_bar=False)
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            dense_scores = torch.matmul(q, self._dense.block_embeddings.T).squeeze(0)
            dense_scores = dense_scores.cpu().tolist()
        # BM25 scores over all blocks (same index as BM25Baseline).
        bm25_scores = np.asarray(
            self._bm25.bm25.get_scores(_tokenize(query)), dtype=np.float64)

        rank_dense = np.empty(n, dtype=np.int64)
        rank_dense[np.argsort(-np.asarray(dense_scores))] = np.arange(1, n + 1)
        rank_bm25 = np.empty(n, dtype=np.int64)
        rank_bm25[np.argsort(-bm25_scores, kind="stable")] = np.arange(1, n + 1)

        fused = (1.0 / (self.rrf_k + rank_dense) + 1.0 / (self.rrf_k + rank_bm25))
        top_m = min(self.m, n)
        order = np.argsort(-fused, kind="stable")[:top_m]
        return [int(i) for i in order], [float(fused[i]) for i in order]

    def answer(self, query: str, llm) -> Dict[str, Any]:
        t0 = time.perf_counter()
        selected_ids, fused_scores = self.retrieve(query)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        retrieved_blocks = [self.blocks[i] for i in selected_ids]

        out = finish_answer(
            system="hybrid_rrf",
            query=query,
            blocks=retrieved_blocks,
            llm=llm,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            selected_ids=selected_ids,
            selection_scores=fused_scores,
            selection_latency_ms=retrieval_ms,
            extra={
                "retrieval": f"reciprocal-rank-fusion(BM25, dense), rrf_k={self.rrf_k}",
                "rrf_k": self.rrf_k,
                "dense_backend": self._dense.backend,
                "bm25_backend": self._bm25.backend,
                "n_llm_calls": 1,
                "ingestion": self._dense.ingestion,
            },
        )
        out["n_llm_calls"] = 1
        return out

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "hybrid_rrf",
            "m": self.m,
            "block_size": self.block_size,
            "rrf_k": self.rrf_k,
            "n_blocks": len(self.blocks),
            "n_llm_calls": 1,
            "dense_backend": self._dense.backend,
            "bm25_backend": self._bm25.backend,
            "ingestion_tokens": self._dense.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self._dense.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
