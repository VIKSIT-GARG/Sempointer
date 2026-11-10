"""B2: Dense retrieval-augmented generation baseline (uniform `System` interface).

ingest() chunks the context exactly like sempointer.pipeline.chunk_text and
embeds the blocks with sentence-transformers/all-MiniLM-L6-v2 (L2-normalized,
long blocks window-pooled via embed_long). answer() ranks ALL blocks by exact
cosine similarity, selects either the top-m blocks or (token-budget mode) the
longest rank prefix fitting --token-budget, builds a prompt containing ONLY
the selected blocks plus the question, and generates with a REAL LLM call.

No silent fallbacks: sentence-transformers is a hard import.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer

from ._text_common import DEFAULT_SYSTEM_PROMPT, RankedRetrievalBaseline, require_ingested

RAG_BACKEND = "torch exact cosine (normalized embeddings)"

# One embedder per (id, device) per process: several rag:<m> / hybrid systems
# in one run must not each load their own copy.
_EMBEDDERS: Dict[Tuple[str, str], SentenceTransformer] = {}


def get_embedder(embedder_id: str, device: str) -> SentenceTransformer:
    key = (embedder_id, device)
    if key not in _EMBEDDERS:
        _EMBEDDERS[key] = SentenceTransformer(embedder_id, device=device)
    return _EMBEDDERS[key]


class DenseIndex:
    """Cosine index over blocks (shared by DenseRAGBaseline and HybridRRFBaseline)."""

    def __init__(self, embedder_id: str, device: str):
        self.embedder_id = embedder_id
        self.device = device
        self.embedder = get_embedder(embedder_id, device)
        self.block_embeddings: Optional[torch.Tensor] = None

    def fit(self, blocks: List[str]) -> None:
        from sempointer.pointer_generator import embed_long

        with torch.no_grad():
            embs = embed_long(self.embedder, blocks).to(self.device)
            self.block_embeddings = torch.nn.functional.normalize(embs, p=2, dim=-1)

    def rank_all(self, query: str) -> Tuple[List[int], List[float]]:
        with torch.no_grad():
            q = self.embedder.encode([query], convert_to_tensor=True, device=self.device,
                                     show_progress_bar=False)
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            scores = torch.matmul(q, self.block_embeddings.T).squeeze(0).float().cpu()
        n = scores.shape[0]
        # stable best-first order; ties -> lower index
        order = sorted(range(n), key=lambda i: (-float(scores[i]), i))
        return order, [float(scores[i]) for i in order]


class DenseRAGBaseline(RankedRetrievalBaseline):
    """Dense cosine ranking over memory blocks; top-m or token-budget selection."""

    system_name = "rag"
    retrieval_desc = "dense cosine"

    def __init__(
        self,
        tokenizer,
        m: Optional[int] = 1,
        block_size: int = 512,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        token_budget: Optional[int] = None,
    ):
        super().__init__(tokenizer, m=m, block_size=block_size,
                         token_budget=token_budget, system_prompt=system_prompt)
        self.embedder_id = embedder_id
        self.device = device
        self.backend = RAG_BACKEND
        self.dense = DenseIndex(embedder_id, device)
        self.embedder = self.dense.embedder

    def _index(self) -> None:
        self.dense.fit(self.blocks)

    @property
    def block_embeddings(self):
        return self.dense.block_embeddings

    def rank_all(self, query: str) -> Tuple[List[int], List[float]]:
        require_ingested(self.blocks, type(self).__name__)
        return self.dense.rank_all(query)

    def _extra(self) -> Dict[str, Any]:
        return {"rag_backend": self.backend, "embedder_id": self.embedder_id,
                "embedder_device": self.device}
