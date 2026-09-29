"""B6: Agentic pointer-routing baseline (uniform `System` interface).

Two LLM calls per question:

  Call 1 (route): the prompt lists every memory block as a compact pointer
      string "[PTR_{id}: <preview...>]" plus the query, and instructs the LLM
      to reply with ONLY the needed block numbers. A robust parser accepts
      "3, 17" / "[3]" / "block 3" formats; unparseable or out-of-range output
      falls back to an empty selection (scored as a miss, never a crash).
  Call 2 (answer): the resolved full text of the routed blocks plus the query
      is sent to the LLM for the final answer.

ingest() chunks the context exactly like sempointer.pipeline.chunk_text.
active_tokens = GenResult.prompt_tokens summed over BOTH calls (measured, not
derived). Both prompts are returned for provenance logging.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    build_memory_prompt,
    ingest_blocks,
    require_ingested,
)

ROUTING_INSTRUCTION = (
    "You are a memory router. Below is a registry of memory blocks, each shown "
    "as [PTR_<number>: preview...]. Given the question, reply with ONLY the "
    "numbers of the blocks needed to answer it (e.g. \"3\" or \"3, 17\"). "
    "If no block is relevant, reply with NONE. Do not explain."
)

PREVIEW_CHARS = 300


def parse_routed_ids(text: str, n_blocks: int) -> List[int]:
    """Extracts block ids from free-form router output. Never raises.

    Accepts "3, 17" / "[3]" / "block 3" / "blocks 3 and 17" / "NONE".
    Out-of-range ids are dropped; order is preserved; duplicates removed.
    """
    try:
        if not text or not text.strip():
            return []
        if re.search(r"\bnone\b", text, re.IGNORECASE) and not re.search(r"\d", text):
            return []
        ids: List[int] = []
        for tok in re.findall(r"\d+", text):
            try:
                i = int(tok)
            except ValueError:  # pragma: no cover - defensive
                continue
            if 0 <= i < n_blocks and i not in ids:
                ids.append(i)
        return ids
    except Exception:
        return []


def _pointer_strings(blocks: List[str]) -> List[str]:
    ptrs = []
    for i, blk in enumerate(blocks):
        preview = " ".join(blk.split())[:PREVIEW_CHARS]
        ptrs.append(f"[PTR_{i}: {preview}]")
    return ptrs


class AgenticPointerBaseline:
    """LLM-routed pointer selection + resolved-block answering (2 LLM calls)."""

    def __init__(
        self,
        tokenizer,
        m: int = 4,
        block_size: int = 512,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer = tokenizer
        self.m = m  # cap on routed blocks (comparability budget)
        self.block_size = block_size
        self.system_prompt = system_prompt
        self.blocks: List[str] = []
        self.ingestion: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        self.blocks, self.ingestion = ingest_blocks(context, self.block_size, self.tokenizer)

    def build_routing_prompt(self, query: str) -> str:
        """Routing prompt (Call 1): all [PTR_id:...] strings + query."""
        require_ingested(self.blocks, type(self).__name__)
        sections: List[str] = []
        if self.system_prompt:
            sections.append(f"System: {self.system_prompt.strip()}")
        sections.append("Memory Registry Pointers:\n" + " ".join(_pointer_strings(self.blocks)))
        sections.append(f"Question: {query.strip()}\n{ROUTING_INSTRUCTION}")
        return "\n\n".join(sections)

    def answer(self, query: str, llm) -> Dict[str, Any]:
        require_ingested(self.blocks, type(self).__name__)

        # ---- Call 1: route ------------------------------------------------
        routing_prompt = self.build_routing_prompt(query)
        routing_core = len(self.tokenizer.encode(routing_prompt, add_special_tokens=False))
        t_route0 = time.perf_counter()
        route_gen = llm.generate(routing_prompt)
        routing_latency_ms = (time.perf_counter() - t_route0) * 1000.0
        routed_ids = parse_routed_ids(route_gen.text, len(self.blocks))[: max(0, self.m)]

        # ---- Call 2: answer from resolved blocks --------------------------
        resolved = [self.blocks[i] for i in routed_ids]
        final_prompt = build_memory_prompt(query, resolved, self.system_prompt)
        final_core = len(self.tokenizer.encode(final_prompt, add_special_tokens=False))
        answer_gen = llm.generate(final_prompt)

        return {
            "answer": answer_gen.text,
            "prompt": answer_gen.prompt,
            "prompts": [route_gen.prompt, answer_gen.prompt],  # both logged
            "routing_prompt": route_gen.prompt,
            "routing_raw": route_gen.text,
            "active_tokens": route_gen.prompt_tokens + answer_gen.prompt_tokens,
            "active_tokens_core": routing_core + final_core,
            "routed_ids": routed_ids,
            "selected_ids": routed_ids,
            "selection_scores": None,  # LLM-routed, no numeric scores
            "selection_latency_ms": routing_latency_ms,
            "routing_latency_ms": routing_latency_ms,
            "generation_latency_ms": route_gen.latency_ms + answer_gen.latency_ms,
            "completion_tokens": route_gen.completion_tokens + answer_gen.completion_tokens,
            "peak_vram_mb": max(route_gen.peak_vram_mb, answer_gen.peak_vram_mb),
            "n_blocks": len(self.blocks),
            "n_llm_calls": 2,
            "stop_reason": answer_gen.stop_reason,
            "extra": {
                "system": "agentic_pointer",
                "retrieval": "llm-routed pointer selection",
                "routing_instruction": ROUTING_INSTRUCTION,
                "ingestion": self.ingestion,
            },
        }

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "agentic_pointer",
            "m": self.m,
            "block_size": self.block_size,
            "n_blocks": len(self.blocks),
            "n_llm_calls": 2,
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
