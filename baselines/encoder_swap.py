"""B2b: encoder-swap pilot — DenseRAG protocol with a swappable embedder.

Identical System contract and identical retrieval/generation protocol to
baselines.dense_rag.DenseRAGBaseline (same chunking, same embed_long block
embeddings, same exact-cosine top-m, same finish_answer LLM path); the ONLY
difference is the sentence-transformer used, so encoder quality is isolated.
Default is BAAI/bge-small-en-v1.5 (dim 384); BGE/E5 pilots pass embedder_id.

No E5-style query/passage prefixes are applied — the protocol is deliberately
identical to the MiniLM dense baseline so the comparison is apples-to-apples
on the encoder weights alone. A prefix study is a separate follow-up.

Name-parameterized: info()["system"] and answer()["extra"]["system"] are
f"encoder_swap:{embedder_id}", so runs with different encoders never collide
in predictions.jsonl. dense_rag.py itself is untouched (subclass only).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import sentence_transformers  # noqa: F401 - loud dependency gate
except ImportError as _exc:
    raise ImportError(
        "EncoderSwapBaseline needs `sentence-transformers`. "
        "Install with: ./warden/bin/python -m pip install -U sentence-transformers"
    ) from _exc

from ._text_common import DEFAULT_SYSTEM_PROMPT
from .dense_rag import DenseRAGBaseline

DEFAULT_ENCODER_SWAP_EMBEDDER = "BAAI/bge-small-en-v1.5"


class EncoderSwapBaseline(DenseRAGBaseline):
    """DenseRAGBaseline with a pilot embedder and a name-parameterized label."""

    def __init__(
        self,
        tokenizer,
        m: int = 1,
        block_size: int = 512,
        embedder_id: str = DEFAULT_ENCODER_SWAP_EMBEDDER,
        device: str = "cuda",
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        if not isinstance(embedder_id, str) or not embedder_id:
            raise ValueError("embedder_id must be a non-empty model id string")
        super().__init__(
            tokenizer=tokenizer,
            m=m,
            block_size=block_size,
            embedder_id=embedder_id,
            device=device,
            system_prompt=system_prompt,
        )

    @property
    def system_name(self) -> str:
        return f"encoder_swap:{self.embedder_id}"

    def answer(self, query: str, llm) -> Dict[str, Any]:
        out = super().answer(query, llm)
        out["extra"]["system"] = self.system_name
        return out

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["system"] = self.system_name
        return d
