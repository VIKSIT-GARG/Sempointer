"""B3: BM25 lexical-retrieval baseline (uniform `System` interface).

ingest() chunks the context exactly like sempointer.pipeline.chunk_text and
indexes the blocks with Okapi BM25. answer() retrieves the top-m blocks,
builds a prompt containing ONLY the retrieved blocks plus the question, and
generates the answer with a REAL LLM call on the shared engine.

Backend selection (not silent): `rank_bm25.BM25Okapi` is used when installed;
otherwise a faithful in-repo Okapi BM25 (`_InRepoBM25Okapi`, same formula,
k1=1.5 / b=0.75 / epsilon floor 0.25 as in rank_bm25) is used. The active
backend is always reported in info() and in the answer's `extra`.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    finish_answer,
    ingest_blocks,
    require_ingested,
)

try:  # prefer the reference implementation when available
    from rank_bm25 import BM25Okapi as _RankBM25Okapi

    BM25_BACKEND = "rank_bm25.BM25Okapi"
except ImportError:
    _RankBM25Okapi = None
    BM25_BACKEND = "baselines.bm25_baseline._InRepoBM25Okapi (rank_bm25 not installed)"

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class _InRepoBM25Okapi:
    """Faithful Okapi BM25 (Robertson et al.), mirroring rank_bm25.BM25Okapi.

    score(D, Q) = sum_{q in Q} IDF(q) * tf(q, D) * (k1 + 1)
                  / (tf(q, D) + k1 * (1 - b + b * |D| / avgdl))
    IDF(q) = ln((N - df + 0.5) / (df + 0.5)); negative IDF terms (df > N/2)
    are floored at epsilon * average_idf (epsilon = 0.25), exactly like
    rank_bm25's BM25Okapi. k1 = 1.5, b = 0.75.
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            raise ValueError("BM25 corpus must contain at least one document")
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_len: List[int] = []
        nd: Dict[str, int] = {}
        total_len = 0
        for doc in corpus:
            dl = len(doc)
            self.doc_len.append(dl)
            total_len += dl
            tf: Dict[str, int] = {}
            for w in doc:
                tf[w] = tf.get(w, 0) + 1
            self.doc_freqs.append(tf)
            for w in tf:
                nd[w] = nd.get(w, 0) + 1
        self.avgdl = total_len / self.corpus_size
        self.idf = self._calc_idf(nd)

    def _calc_idf(self, nd: Dict[str, int]) -> Dict[str, float]:
        idf = {
            word: math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            for word, freq in nd.items()
        }
        negatives = [w for w, v in idf.items() if v < 0]
        if negatives:
            eps = self.epsilon * (sum(idf.values()) / len(idf))
            for w in negatives:
                idf[w] = eps
        return idf

    def get_scores(self, query: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        for q in query:
            if q not in self.idf:
                continue
            idf_q = self.idf[q]
            for i, tf in enumerate(self.doc_freqs):
                f = tf.get(q, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf_q * f * (self.k1 + 1.0) / denom
        return scores


class BM25Baseline:
    """Okapi BM25 retrieval over memory blocks + top-m concatenation + real LLM."""

    def __init__(
        self,
        tokenizer,
        m: int = 1,
        block_size: int = 512,
        k1: float = 1.5,
        b: float = 0.75,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer = tokenizer
        self.m = m
        self.block_size = block_size
        self.k1 = k1
        self.b = b
        self.system_prompt = system_prompt
        self.blocks: List[str] = []
        self.ingestion: Dict[str, Any] = {}
        self.bm25 = None
        self.backend = BM25_BACKEND

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        self.blocks, self.ingestion = ingest_blocks(context, self.block_size, self.tokenizer)
        corpus = [_tokenize(b) for b in self.blocks]
        if _RankBM25Okapi is not None:
            self.bm25 = _RankBM25Okapi(corpus, k1=self.k1, b=self.b)
        else:
            self.bm25 = _InRepoBM25Okapi(corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str) -> Tuple[List[int], List[float]]:
        """Top-m block indices and BM25 scores for `query`."""
        require_ingested(self.blocks, type(self).__name__)
        scores = self.bm25.get_scores(_tokenize(query))
        top_m = min(self.m, len(self.blocks))
        order = np.argsort(-scores)[:top_m]
        return [int(i) for i in order], [float(scores[i]) for i in order]

    def answer(self, query: str, llm) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        selected_ids, scores = self.retrieve(query)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        retrieved_blocks = [self.blocks[i] for i in selected_ids]

        return finish_answer(
            system="bm25",
            query=query,
            blocks=retrieved_blocks,
            llm=llm,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            selected_ids=selected_ids,
            selection_scores=scores,
            selection_latency_ms=retrieval_ms,
            extra={
                "retrieval": "bm25 (Okapi)",
                "bm25_backend": self.backend,
                "bm25_params": {"k1": self.k1, "b": self.b, "m": self.m},
                "ingestion": self.ingestion,
            },
        )

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "bm25",
            "m": self.m,
            "block_size": self.block_size,
            "bm25_backend": self.backend,
            "bm25_params": {"k1": self.k1, "b": self.b},
            "n_blocks": len(self.blocks),
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
