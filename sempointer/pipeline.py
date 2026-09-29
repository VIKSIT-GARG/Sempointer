"""End-to-end SemPointer pipeline.

Implements the paper's five-stage lifecycle with a real causal LLM at the end:

    context -> block construction -> pointer generation (dense + semantic IDs)
            -> registry -> query-conditioned selection (top-m)
            -> resolution (M_text) -> active context construction
            -> LLMEngine.generate (REAL inference) -> answer

SemPointer genuinely determines the model's input: only the selected blocks'
full text plus the compact pointer tokens of unselected blocks are sent to the
LLM. `answer()` returns measured token counts and latencies alongside the
generated text.

The registry preamble places each unselected block's k semantic pointer tokens
in the prompt, matching the paper's active sequence
L_active = (N-m)*k + m*B_res + L_Q (framing tokens excluded from the count).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch

from .pointer_generator import SemanticIDPointerGenerator
from .prompt_builder import PromptBuilder
from .registry import PointerRegistry
from .resolver import TextResolver
from .scorer import PointerScorer

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions using only the provided memory context. "
    "If the context does not contain the answer, say you don't know. "
    "Answer concisely with only the answer itself."
)


def chunk_text(text: str, block_size: int, tokenizer) -> List[str]:
    """Chunks `text` into blocks of at most `block_size` tokens (LLM tokenizer),
    matching the block construction used by every system."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids), block_size):
        chunk_ids = ids[i : i + block_size]
        if len(chunk_ids) < block_size // 8 and chunks:
            # merge tiny trailing fragment into the previous block
            chunks[-1] = tokenizer.decode(
                tokenizer.encode(chunks[-1], add_special_tokens=False) + chunk_ids,
                skip_special_tokens=True,
            )
        else:
            chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks


class SemPointerPipeline:
    """The `System` evaluated by run_benchmark.py."""

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
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer = tokenizer
        self.k = k
        self.m = m
        self.block_size = block_size
        self.system_prompt = system_prompt
        self.generator = SemanticIDPointerGenerator(
            embedder_id=embedder_id,
            k=k,
            pointer_vocab_size=pointer_vocab_size,
            seed=seed,
            device=device,
        )
        self.registry: Optional[PointerRegistry] = None
        self.scorer: Optional[PointerScorer] = None
        self.resolver = TextResolver()
        self.prompt_builder = PromptBuilder(tokenizer=tokenizer, B_res=block_size)
        self.device = device

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        blocks = chunk_text(context, self.block_size, self.tokenizer)
        self.registry = PointerRegistry(k=self.k, B=self.block_size, max_size=max(4096, len(blocks) + 8))
        for i, blk in enumerate(blocks):
            self.registry.register(blk, self.generator, metadata={"block_idx": i})
        self.scorer = PointerScorer(self.generator, device=self.device)

    def active_prompt(self, query: str, m: Optional[int] = None) -> str:
        """Builds the active prompt without running the LLM (used by tests/dry-run)."""
        self._require_ingested()
        m = self.m if m is None else m
        ranked_ids, _ = self.scorer.score(query, self.registry, m=min(m, len(self.registry)))
        return self.prompt_builder.build_sempointer_prompt(
            query, ranked_ids, self.registry, self.resolver, system_prompt=self.system_prompt
        )

    def answer(self, query: str, llm) -> Dict[str, Any]:
        """Selection -> resolution -> active context -> REAL LLM generation.

        `llm` is the shared LLMEngine. Returns the generated answer plus
        measured retrieval/selection latencies and exact active token counts.
        """
        self._require_ingested()
        t_select0 = time.perf_counter()
        ranked_ids, scores = self.scorer.score(query, self.registry, m=min(self.m, len(self.registry)))
        select_ms = (time.perf_counter() - t_select0) * 1000.0

        prompt = self.prompt_builder.build_sempointer_prompt(
            query, ranked_ids, self.registry, self.resolver, system_prompt=self.system_prompt
        )
        prompt_token_count = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        gen = llm.generate(prompt)

        return {
            "answer": gen.text,
            "prompt": gen.prompt,
            "active_tokens": gen.prompt_tokens,
            "active_tokens_core": prompt_token_count,  # without chat-template framing
            "selected_ids": ranked_ids,
            "selection_scores": [float(s) for s in scores],
            "selection_latency_ms": select_ms,
            "generation_latency_ms": gen.latency_ms,
            "completion_tokens": gen.completion_tokens,
            "peak_vram_mb": gen.peak_vram_mb,
            "n_blocks": len(self.registry),
            "stop_reason": gen.stop_reason,
        }

    # ------------------------------------------------------------------ #
    def _require_ingested(self) -> None:
        if self.registry is None or not len(self.registry):
            raise RuntimeError("SemPointerPipeline.ingest() must be called before use")

    def info(self) -> Dict[str, Any]:
        return {
            "system": "sempointer",
            "k": self.k,
            "m": self.m,
            "block_size": self.block_size,
            **self.generator.info(),
            "substrate": "M_text",
        }
