"""Query-conditioned block selector for SemPointer.

Rankings available per query:
  - "dense": cosine of MiniLM embeddings (block embeddings via embed_long, so
             long blocks are not truncated at the embedder's max_seq_length)
  - "bm25":  in-repo Okapi BM25 (sempointer.lexical.BM25Index)
  - "attention": caller-supplied per-block scores (e.g. LLM attention from the
             query to each block's pointer, sempointer.attention_scorer)

Modes: "dense", "bm25", "hybrid" (dense+bm25), "attention",
"hybrid+attention" (dense+bm25+attention). Rankings are combined with
weighted Reciprocal Rank Fusion: fused(i) = sum_r w_r / (rrf_k + rank_r(i)),
rank starting at 1. A single-ranking mode therefore preserves that ranking's
order exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .lexical import BM25Index

MODES = ("dense", "bm25", "hybrid", "attention", "hybrid+attention")
_MODE_COMPONENTS = {
    "dense": ("dense",),
    "bm25": ("bm25",),
    "hybrid": ("dense", "bm25"),
    "attention": ("attention",),
    "hybrid+attention": ("dense", "bm25", "attention"),
}

# Embedders are cached per (id, device) at module level so that constructing
# several selectors (or calling fit() per example) never reloads the model.
_EMBEDDER_CACHE: Dict[Tuple[str, str], object] = {}


def _resolve_device(device: str) -> str:
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return device


def get_embedder(embedder_id: str, device: str):
    device = _resolve_device(device)
    key = (embedder_id, device)
    if key not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        try:
            _EMBEDDER_CACHE[key] = SentenceTransformer(embedder_id, device=device)
        except Exception:
            if device == "cpu":
                raise
            key = (embedder_id, "cpu")
            if key not in _EMBEDDER_CACHE:
                _EMBEDDER_CACHE[key] = SentenceTransformer(embedder_id, device="cpu")
    return _EMBEDDER_CACHE[key]


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks, best (highest) score = 1; ties broken by lower index."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


class BlockSelector:
    def __init__(
        self,
        mode: str = "hybrid+attention",
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        rrf_k: int = 60,
        weights: Optional[dict] = None,
    ):
        if mode not in MODES:
            raise ValueError(f"unknown selector mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.components = _MODE_COMPONENTS[mode]
        self.embedder_id = embedder_id
        self.device = _resolve_device(device)
        self.rrf_k = rrf_k
        self.weights = {"dense": 1.0, "bm25": 1.0, "attention": 1.0}
        if weights:
            self.weights.update(weights)
        self._embedder = None
        self.blocks: List[str] = []
        self.bm25: Optional[BM25Index] = None
        self.block_emb = None  # torch (N, d), L2-normalized

    # ------------------------------------------------------------------ #
    @property
    def needs_attention(self) -> bool:
        return "attention" in self.mode

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.embedder_id, self.device)
        return self._embedder

    def fit(self, blocks: Sequence[str]) -> None:
        self.blocks = list(blocks)
        if not self.blocks:
            raise ValueError("BlockSelector.fit() needs at least one block")
        self.bm25 = BM25Index(self.blocks) if "bm25" in self.components else None
        self.block_emb = None
        if "dense" in self.components:
            import torch

            from .pointer_generator import embed_long

            with torch.no_grad():
                e = embed_long(self.embedder, self.blocks)
                self.block_emb = torch.nn.functional.normalize(e.float(), p=2, dim=-1)

    # ------------------------------------------------------------------ #
    def component_scores(self, query: str, attention_scores: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
        if not self.blocks:
            raise RuntimeError("BlockSelector.fit() must be called before rank()/select()")
        n = len(self.blocks)
        out: Dict[str, np.ndarray] = {}
        if "dense" in self.components:
            import torch

            with torch.no_grad():
                q = self.embedder.encode([query], convert_to_tensor=True, show_progress_bar=False)
                q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(self.block_emb.device)
                out["dense"] = (q @ self.block_emb.T).squeeze(0).cpu().numpy().astype(np.float64)
        if "bm25" in self.components:
            out["bm25"] = self.bm25.scores(query)
        if "attention" in self.components:
            if attention_scores is None:
                raise ValueError(f"selector mode {self.mode!r} requires attention_scores")
            a = np.asarray(attention_scores, dtype=np.float64)
            if a.shape != (n,):
                raise ValueError(f"attention_scores must have length {n}, got shape {a.shape}")
            out["attention"] = a
        return out

    def rank(self, query: str, attention_scores: Optional[Sequence[float]] = None) -> List[Tuple[int, float]]:
        comps = self.component_scores(query, attention_scores)
        n = len(self.blocks)
        fused = np.zeros(n, dtype=np.float64)
        for name, s in comps.items():
            fused += self.weights.get(name, 1.0) / (self.rrf_k + _ranks(s))
        order = np.lexsort((np.arange(n), -fused))
        return [(int(i), float(fused[i])) for i in order]

    def select(
        self,
        query: str,
        m: int,
        attention_scores: Optional[Sequence[float]] = None,
        adaptive: bool = False,
        adaptive_ratio: float = 0.5,
        m_max: Optional[int] = None,
    ) -> List[int]:
        ranked = self.rank(query, attention_scores)
        if not adaptive:
            return [i for i, _ in ranked[: max(0, min(m, len(ranked)))]]
        cap = m if m_max is None else m_max
        cap = max(1, min(cap, len(ranked)))
        top = ranked[0][1]
        chosen = [i for i, s in ranked[:cap] if s >= adaptive_ratio * top]
        return chosen or [ranked[0][0]]

    def info(self) -> dict:
        return {
            "selector_mode": self.mode,
            "components": list(self.components),
            "fusion": "weighted reciprocal rank fusion",
            "rrf_k": self.rrf_k,
            "weights": {c: self.weights.get(c, 1.0) for c in self.components},
            "embedder_id": self.embedder_id if "dense" in self.components else None,
            "embedder_device": self.device if "dense" in self.components else None,
            "bm25": {"k1": 1.5, "b": 0.75, "stopwords": True} if "bm25" in self.components else None,
        }
