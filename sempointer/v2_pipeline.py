"""V2 fast evidence router: protected lexical blocks and ambiguity expansion.

This is an experimental architecture, not a claim of measured improvement.
It reuses V1 pointer rendering and answer generation without changing V1 code.
"""

from __future__ import annotations

import time

from .lexical import tokenize
from .pipeline import SemPointerPipeline
from .selector import BlockSelector


class SemPointerV2Pipeline(SemPointerPipeline):
    def __init__(
        self, tokenizer, k: int = 16, m: int = 2, m_max: int = 4,
        block_size: int = 512, ptype: str = "keyword", substrate: str = "text",
        device: str = "cuda", seed: int = 42, show_pointers: bool = True,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        expand_coverage: float = 0.8, ambiguity_coverage: float = 0.95,
        rescue_coverage: float = 0.5,
        max_resolved_tokens: int = 2560, semantic_rescue: bool = False,
        expand_score_ratio: float = 0.85,
    ):
        if not 1 <= m <= m_max:
            raise ValueError("V2 requires 1 <= m <= m_max")
        if not 0 <= rescue_coverage <= expand_coverage <= ambiguity_coverage <= 1:
            raise ValueError("V2 requires rescue_coverage <= expand_coverage <= ambiguity_coverage <= 1")
        if max_resolved_tokens < 1:
            raise ValueError("max_resolved_tokens must be positive")
        if not 0 <= expand_score_ratio <= 1:
            raise ValueError("expand_score_ratio must be between 0 and 1")
        super().__init__(
            tokenizer=tokenizer, k=k, m=m_max, block_size=block_size,
            ptype=ptype, selector="bm25", substrate=substrate, device=device,
            seed=seed, show_pointers=show_pointers, embedder_id=embedder_id,
        )
        self.min_m = m
        self.expand_coverage = expand_coverage
        self.ambiguity_coverage = ambiguity_coverage
        self.rescue_coverage = rescue_coverage
        self.max_resolved_tokens = max_resolved_tokens
        self.semantic_rescue = semantic_rescue
        self.expand_score_ratio = expand_score_ratio
        # CPU placement prevents the embedder competing with the 3B LLM for
        # the laptop's 8 GB of GPU memory.
        self.dense_selector = (BlockSelector(mode="dense", embedder_id=embedder_id, device="cpu")
                               if semantic_rescue else None)
        self._dense_context_id = None

    def ingest(self, context: str) -> None:
        super().ingest(context)
        self._dense_context_id = None  # defer embeddings until rescue is needed

    def _coverage(self, query: str, ids: list[int]) -> float:
        terms = set(tokenize(query))
        if not terms:
            return 1.0
        present = set()
        for i in ids:
            present.update(tokenize(self.blocks[i]))
        return len(terms & present) / len(terms)

    def _weighted_coverage(self, query: str, ids: list[int]) -> float:
        # Terms absent from the corpus cannot steer lexical retrieval. Weight
        # present terms by corpus rarity so repeated prompt boilerplate does
        # not make unrelated blocks look sufficient.
        idf = self.selector.bm25.idf
        weights = {term: max(0.0, idf.get(term, 0.0)) for term in set(tokenize(query))}
        total = sum(weights.values())
        if total <= 0:
            return 0.0
        present = set()
        for i in ids:
            present.update(tokenize(self.blocks[i]))
        return sum(weight for term, weight in weights.items() if term in present) / total

    def select(self, query: str, llm=None) -> dict:
        self._require_ingested()
        start = time.perf_counter()
        scores = self.selector.bm25.scores(query)
        lexical = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        protected = lexical[:self.min_m]
        coverage = self._coverage(query, protected)
        weighted_coverage = self._weighted_coverage(query, protected)
        route = "lexical"
        target = self.min_m
        boundary = float(scores[lexical[self.min_m - 1]])
        next_score = float(scores[lexical[self.min_m]]) if len(lexical) > self.min_m else 0.0
        score_ratio = next_score / boundary if boundary > 0 else 1.0
        if weighted_coverage < self.expand_coverage or (weighted_coverage < self.ambiguity_coverage and
                                      len(lexical) > self.min_m and
                                      score_ratio >= self.expand_score_ratio):
            target = self.m
            route = "expanded_lexical"
        candidates = lexical[:target]
        rescue = None
        if self.dense_selector is not None and coverage < self.rescue_coverage:
            if self._dense_context_id != self._context_id:
                self.dense_selector.fit(self.blocks)
                self._dense_context_id = self._context_id
            dense = self.dense_selector.rank(query)
            rescue = next((i for i, _ in dense if i not in candidates), None)
            if rescue is not None:
                candidates.append(rescue)

        ids = []
        used = 0
        for i in candidates:
            cost = len(self._enc(self.blocks[i]))
            if not ids or used + cost <= self.max_resolved_tokens:
                ids.append(i)
                used += cost
        if rescue is not None and rescue in ids:
            route = "semantic_rescue"
        selection = {
            "ids": ids, "scores": [float(scores[i]) for i in ids],
            "n_hops": 1, "selection_latency_ms": (time.perf_counter() - start) * 1000,
            "router_route": route, "router_coverage": coverage,
            "router_weighted_coverage": weighted_coverage,
            "router_score_ratio": score_ratio,
            "router_resolved_tokens": used,
        }
        self._last_selection = selection
        return selection

    def answer(self, query: str, llm) -> dict:
        # V1's answer() calls this class's select() once.
        out = super().answer(query, llm)
        selection = self._last_selection
        out.update({k: selection[k] for k in
                    ("router_route", "router_coverage", "router_weighted_coverage",
                     "router_score_ratio",
                     "router_resolved_tokens")})
        return out

    def info(self) -> dict:
        out = super().info()
        out["system"] = "sempointer_v2"
        out["router"] = {
            "min_m": self.min_m, "m_max": self.m,
            "expand_coverage": self.expand_coverage,
            "ambiguity_coverage": self.ambiguity_coverage,
            "rescue_coverage": self.rescue_coverage,
            "expand_score_ratio": self.expand_score_ratio,
            "max_resolved_tokens": self.max_resolved_tokens,
            "semantic_rescue": self.semantic_rescue,
            "embedder_device": "cpu" if self.semantic_rescue else None,
        }
        return out
