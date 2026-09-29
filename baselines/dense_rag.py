"""B2: Dense retrieval-augmented generation baseline (uniform `System` interface).

ingest() chunks the context exactly like sempointer.pipeline.chunk_text and
embeds the blocks with sentence-transformers/all-MiniLM-L6-v2 (L2-normalized).
answer() retrieves the top-m blocks by exact cosine similarity (PyTorch
implementation; faiss is used only if importable and is interchangeable —
identical rankings on normalized vectors), builds a prompt containing ONLY the
retrieved blocks plus the question, and generates the answer with a REAL LLM
call on the shared engine.

No silent fallbacks: if sentence-transformers is missing, this module raises
ImportError with install instructions at import time.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    finish_answer,
    ingest_blocks,
    require_ingested,
)

try:  # faiss is optional; the exact torch cosine path is the reference
    import faiss

    RAG_BACKEND = "faiss.IndexFlatIP (normalized embeddings = cosine)"
except ImportError:
    faiss = None
    RAG_BACKEND = "torch exact cosine (faiss not installed)"


class DenseRAGBaseline:
    """Dense cosine retrieval over memory blocks + top-m concatenation + real LLM."""

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
        self.embedder_id = embedder_id
        self.device = device
        self.system_prompt = system_prompt
        self.blocks: List[str] = []
        self.ingestion: Dict[str, Any] = {}
        self.block_embeddings: Optional[torch.Tensor] = None
        self.faiss_index = None
        self.backend = RAG_BACKEND

        self.embedder = SentenceTransformer(embedder_id, device=device)

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        from sempointer.pointer_generator import embed_long

        self.blocks, self.ingestion = ingest_blocks(context, self.block_size, self.tokenizer)
        with torch.no_grad():
            embs = embed_long(self.embedder, self.blocks).to(self.device)
            self.block_embeddings = torch.nn.functional.normalize(embs, p=2, dim=-1)

        self.faiss_index = None
        if faiss is not None:
            np_embs = self.block_embeddings.cpu().numpy().astype(np.float32)
            self.faiss_index = faiss.IndexFlatIP(np_embs.shape[1])
            self.faiss_index.add(np_embs)

    def retrieve(self, query: str) -> Tuple[List[int], List[float]]:
        """Top-m block indices and cosine scores for `query`."""
        require_ingested(self.blocks, type(self).__name__)
        top_m = min(self.m, len(self.blocks))
        if self.faiss_index is not None:
            q = self.embedder.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            faiss.normalize_L2(q)
            scores, indices = self.faiss_index.search(q, top_m)
            return [int(i) for i in indices[0]], [float(s) for s in scores[0]]
        with torch.no_grad():
            q = self.embedder.encode([query], convert_to_tensor=True, device=self.device, show_progress_bar=False)
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            scores = torch.matmul(q, self.block_embeddings.T).squeeze(0)
            top_scores, top_indices = torch.topk(scores, k=top_m)
        return top_indices.cpu().tolist(), [float(s) for s in top_scores.cpu().tolist()]

    def answer(self, query: str, llm) -> Dict[str, Any]:
        t0 = time.perf_counter()
        selected_ids, scores = self.retrieve(query)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        retrieved_blocks = [self.blocks[i] for i in selected_ids]

        return finish_answer(
            system="rag",
            query=query,
            blocks=retrieved_blocks,
            llm=llm,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            selected_ids=selected_ids,
            selection_scores=scores,
            selection_latency_ms=retrieval_ms,
            extra={
                "retrieval": "dense cosine",
                "rag_backend": self.backend,
                "embedder_id": self.embedder_id,
                "embedder_device": self.device,
                "ingestion": self.ingestion,
            },
        )

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "rag",
            "m": self.m,
            "block_size": self.block_size,
            "embedder_id": self.embedder_id,
            "embedder_device": self.device,
            "rag_backend": self.backend,
            "n_blocks": len(self.blocks),
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
