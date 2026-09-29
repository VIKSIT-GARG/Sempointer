"""A3: Address-only retrieval-augmented generation baseline (uniform `System` interface).

Identical to DenseRAGBaseline except retrieval: blocks are selected by
AddressScorer (discrete pointer-ID overlap only — no dense cosine anywhere in
the retrieval path) instead of dense cosine similarity. Chunking
(`ingest_blocks`), the uniform prompt format (`finish_answer`), and the
`System` contract (`ingest` / `answer` / `info`) are shared, so any
address_rag-vs-rag gap isolates the retrieval mechanism.

Honest framing: this system tests whether discrete addresses carry retrieval
signal (a direct attack on M1). It is NOT claimed to beat dense retrieval —
the pilot decides, and either outcome is published.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    finish_answer,
    ingest_blocks,
    require_ingested,
)


class AddressRAGBaseline:
    """Discrete-address retrieval over memory blocks + top-m concatenation + real LLM."""

    def __init__(
        self,
        tokenizer,
        k: int = 8,
        m: int = 1,
        block_size: int = 512,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        pointer_vocab_size: int = 256,
        seed: int = 42,
        device: str = "cuda",
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        from sempointer.pointer_generator import SemanticIDPointerGenerator
        from sempointer.registry import PointerRegistry
        from sempointer.address_scorer import AddressScorer

        if k < 0:
            raise ValueError("k must be >= 0")
        self.tokenizer = tokenizer
        self.k = k
        self.m = m
        self.block_size = block_size
        self.embedder_id = embedder_id
        self.pointer_vocab_size = pointer_vocab_size
        self.seed = seed
        self.device = device
        self.system_prompt = system_prompt
        self.blocks: List[str] = []
        self.ingestion: Dict[str, Any] = {}
        self.generator = SemanticIDPointerGenerator(
            embedder_id=embedder_id,
            k=k,
            pointer_vocab_size=pointer_vocab_size,
            seed=seed,
            device=device,
        )
        self.registry: Optional[PointerRegistry] = None
        self.scorer = AddressScorer(self.generator)

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        from sempointer.registry import PointerRegistry

        self.blocks, self.ingestion = ingest_blocks(context, self.block_size, self.tokenizer)
        self.registry = PointerRegistry(
            k=self.k, B=self.block_size, max_size=max(4096, len(self.blocks) + 8)
        )
        for i, blk in enumerate(self.blocks):
            self.registry.register(blk, self.generator, metadata={"block_idx": i})

    def retrieve(self, query: str) -> Tuple[List[int], List[float]]:
        """Top-m block indices and address-overlap scores for `query`."""
        require_ingested(self.blocks, type(self).__name__)
        if self.registry is None or len(self.registry) == 0:
            raise RuntimeError(f"{type(self).__name__}.ingest() must be called with a non-empty context before use")
        top_m = min(self.m, len(self.blocks))
        t0 = time.perf_counter()
        ranked_ids, scores = self.scorer.score(query, self.registry, m=top_m)
        self._last_retrieval_ms = (time.perf_counter() - t0) * 1000.0
        return ranked_ids, [float(s) for s in scores.tolist()]

    def answer(self, query: str, llm) -> Dict[str, Any]:
        selected_ids, scores = self.retrieve(query)
        retrieval_ms = getattr(self, "_last_retrieval_ms", 0.0)
        retrieved_blocks = [self.blocks[i] for i in selected_ids]

        return finish_answer(
            system="address_rag",
            query=query,
            blocks=retrieved_blocks,
            llm=llm,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            selected_ids=selected_ids,
            selection_scores=scores,
            selection_latency_ms=retrieval_ms,
            extra={
                "retrieval": "discrete address overlap (prefix + Hamming/k), no dense cosine",
                "embedder_id": self.embedder_id,
                "embedder_device": self.device,
                "k": self.k,
                "pointer_vocab_size": self.pointer_vocab_size,
                "seed": self.seed,
                "ingestion": self.ingestion,
            },
        )

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "address_rag",
            "k": self.k,
            "m": self.m,
            "block_size": self.block_size,
            "embedder_id": self.embedder_id,
            "embedder_device": self.device,
            "pointer_vocab_size": self.pointer_vocab_size,
            "seed": self.seed,
            "n_blocks": len(self.blocks),
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
