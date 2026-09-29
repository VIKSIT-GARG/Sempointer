"""B2c: dense-retrieval + cross-encoder rerank baseline (uniform System contract).

Stage 1: DenseRAGBaseline retrieves the top-`stage1_topk` (default 8) blocks
by exact cosine similarity. Stage 2: a cross-encoder (default
cross-encoder/ms-marco-MiniLM-L-6-v2) scores each (query, block) pair and the
top-`m` reranked blocks form the active context, answered through the shared
finish_answer LLM path — same prompt format as every other text system.

Scaffolding rule: the cross-encoder import is LAZY (inside __init__). If the
dependency is missing, construction raises ImportError with install
instructions — it NEVER silently falls back to dense-only ranking, because a
silent fallback would mislabel dense results as reranked.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    finish_answer,
    require_ingested,
)

DEFAULT_RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
STAGE1_TOPK = 8


class RerankBaseline:
    """Dense top-8 retrieval, cross-encoder rerank to top-m, real LLM answer."""

    def __init__(
        self,
        tokenizer,
        m: int = 1,
        block_size: int = 512,
        embedder_id: str = DEFAULT_RERANK_EMBEDDER,
        reranker_id: str = DEFAULT_RERANKER_ID,
        stage1_topk: int = STAGE1_TOPK,
        device: str = "cuda",
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as _exc:
            raise ImportError(
                "RerankBaseline needs `sentence-transformers>=2.2` (CrossEncoder). "
                "Install with: ./warden/bin/python -m pip install -U sentence-transformers"
            ) from _exc
        if m < 1:
            raise ValueError("m must be >= 1")
        if stage1_topk < 1:
            raise ValueError("stage1_topk must be >= 1")
        if not isinstance(reranker_id, str) or not reranker_id:
            raise ValueError("reranker_id must be a non-empty model id string")

        from .dense_rag import DenseRAGBaseline

        self.tokenizer = tokenizer
        self.m = m
        self.block_size = block_size
        self.embedder_id = embedder_id
        self.reranker_id = reranker_id
        self.stage1_topk = stage1_topk
        self.device = device
        self.system_prompt = system_prompt
        self.stage1 = DenseRAGBaseline(
            tokenizer=tokenizer,
            m=stage1_topk,
            block_size=block_size,
            embedder_id=embedder_id,
            device=device,
        )
        # Loud on failure: no silent CPU/dense fallback — mislabeled results
        # are worse than a crash.
        try:
            self.reranker = CrossEncoder(reranker_id, device=device)
        except Exception as exc:  # noqa: BLE001 - re-raised loudly, never swallowed
            raise RuntimeError(
                f"RerankBaseline: could not load cross-encoder {reranker_id!r} "
                f"on device {device!r} ({type(exc).__name__}: {exc}). "
                "Check the model id, network access, and GPU memory. "
                "Refusing to fall back to dense-only ranking."
            ) from exc

    # ------------------------------------------------------------------ #
    @property
    def blocks(self) -> List[str]:
        return self.stage1.blocks

    @property
    def ingestion(self) -> Dict[str, Any]:
        return self.stage1.ingestion

    def ingest(self, context: str) -> None:
        self.stage1.ingest(context)

    def retrieve(self, query: str) -> Tuple[List[int], List[float]]:
        """Top-m block indices and cross-encoder scores for `query`."""
        require_ingested(self.blocks, type(self).__name__)
        cand_ids, cand_dense = self.stage1.retrieve(query)
        pairs = [(query, self.blocks[i]) for i in cand_ids]
        ce_scores = self.reranker.predict(pairs, show_progress_bar=False)
        top_m = min(self.m, len(cand_ids))
        order = sorted(range(len(cand_ids)), key=lambda i: -float(ce_scores[i]))[:top_m]
        self._last_stage1 = {
            "stage1_ids": [int(i) for i in cand_ids],
            "stage1_dense_scores": [float(s) for s in cand_dense],
        }
        return [int(cand_ids[i]) for i in order], [float(ce_scores[i]) for i in order]

    def answer(self, query: str, llm) -> Dict[str, Any]:
        t0 = time.perf_counter()
        selected_ids, scores = self.retrieve(query)
        selection_ms = (time.perf_counter() - t0) * 1000.0
        retrieved_blocks = [self.blocks[i] for i in selected_ids]

        return finish_answer(
            system="rerank",
            query=query,
            blocks=retrieved_blocks,
            llm=llm,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            selected_ids=selected_ids,
            selection_scores=scores,
            selection_latency_ms=selection_ms,
            extra={
                "retrieval": f"dense top-{self.stage1_topk} + cross-encoder rerank to top-{self.m}",
                "embedder_id": self.embedder_id,
                "reranker_id": self.reranker_id,
                "stage1": getattr(self, "_last_stage1", None),
                "ingestion": self.ingestion,
            },
        )

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "rerank",
            "m": self.m,
            "block_size": self.block_size,
            "embedder_id": self.embedder_id,
            "reranker_id": self.reranker_id,
            "stage1_topk": self.stage1_topk,
            "embedder_device": self.device,
            "n_blocks": len(self.blocks),
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
