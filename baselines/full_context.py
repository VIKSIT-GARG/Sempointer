"""B1: Full-context baseline (uniform `System` interface).

ingest() chunks the context exactly like sempointer.pipeline.chunk_text and
answer() includes ALL blocks in the prompt, then generates the answer with a
REAL LLM call on the shared engine (no scoring shortcuts, no heuristics).

active_tokens = GenResult.prompt_tokens, i.e. the tokens of the prompt that
was actually sent to the LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    build_memory_prompt,
    finish_answer,
    ingest_blocks,
    require_ingested,
)


class FullContextBaseline:
    """All N memory blocks are kept and re-sent for every question."""

    def __init__(
        self,
        tokenizer,
        block_size: int = 512,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.system_prompt = system_prompt
        self.blocks: List[str] = []
        self.ingestion: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        self.blocks, self.ingestion = ingest_blocks(context, self.block_size, self.tokenizer)

    def answer(self, query: str, llm) -> Dict[str, Any]:
        require_ingested(self.blocks, type(self).__name__)
        return finish_answer(
            system="full_context",
            query=query,
            blocks=self.blocks,
            llm=llm,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            selected_ids=list(range(len(self.blocks))),  # every block is active
            selection_scores=None,
            selection_latency_ms=0.0,
            extra={
                "retrieval": "none (all blocks active)",
                "ingestion": self.ingestion,
            },
        )

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "full_context",
            "block_size": self.block_size,
            "n_blocks": len(self.blocks),
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
