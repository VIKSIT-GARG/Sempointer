"""B2b: Hybrid BM25 + dense retrieval baseline fused with Reciprocal Rank Fusion.

Both rankings are computed over ALL blocks (BM25Baseline's Okapi index and
DenseRAGBaseline's MiniLM cosine index), fused with RRF
(fused(d) = sum_r w_r / (rrf_k + rank_r(d)), rank from 1, ties -> lower
index), then selection is top-m or token-budget exactly like the other
retrieval baselines. Answer generation goes through the shared engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ._text_common import DEFAULT_SYSTEM_PROMPT, RankedRetrievalBaseline, require_ingested, rrf_fuse
from .bm25_baseline import BM25_BACKEND, _InRepoBM25Okapi, _RankBM25Okapi, _tokenize
from .dense_rag import DenseIndex

import numpy as np


class HybridRRFBaseline(RankedRetrievalBaseline):
    system_name = "hybrid"
    retrieval_desc = "bm25 + dense cosine, reciprocal rank fusion"

    def __init__(
        self,
        tokenizer,
        m: Optional[int] = 1,
        block_size: int = 512,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        rrf_k: int = 60,
        w_bm25: float = 1.0,
        w_dense: float = 1.0,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        token_budget: Optional[int] = None,
        dense_index=None,
    ):
        super().__init__(tokenizer, m=m, block_size=block_size,
                         token_budget=token_budget, system_prompt=system_prompt)
        self.embedder_id = embedder_id
        self.device = device
        self.rrf_k = int(rrf_k)
        self.w_bm25 = float(w_bm25)
        self.w_dense = float(w_dense)
        # dense_index injectable for CPU tests (anything with fit()/rank_all())
        self.dense = dense_index if dense_index is not None else DenseIndex(embedder_id, device)
        self.bm25 = None

    def _index(self) -> None:
        corpus = [_tokenize(b) for b in self.blocks]
        cls = _RankBM25Okapi if _RankBM25Okapi is not None else _InRepoBM25Okapi
        self.bm25 = cls(corpus, k1=1.5, b=0.75)
        self.dense.fit(self.blocks)

    def bm25_rank(self, query: str) -> List[int]:
        scores = np.asarray(self.bm25.get_scores(_tokenize(query)), dtype=np.float64)
        return [int(i) for i in np.lexsort((np.arange(len(scores)), -scores))]

    def rank_all(self, query: str) -> Tuple[List[int], List[float]]:
        require_ingested(self.blocks, type(self).__name__)
        dense_ids, _ = self.dense.rank_all(query)
        fused = rrf_fuse([self.bm25_rank(query), dense_ids], k=self.rrf_k,
                         weights=[self.w_bm25, self.w_dense])
        return [i for i, _ in fused], [s for _, s in fused]

    def _extra(self) -> Dict[str, Any]:
        return {"fusion": "rrf", "rrf_k": self.rrf_k,
                "weights": {"bm25": self.w_bm25, "dense": self.w_dense},
                "bm25_backend": BM25_BACKEND, "embedder_id": self.embedder_id,
                "embedder_device": self.device}
