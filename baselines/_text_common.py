"""Shared plumbing for the text baselines (full-context, BM25, dense RAG).

All three baselines implement the uniform `System` contract defined by
sempointer.pipeline.SemPointerPipeline:

    ingest(context) -> None
    answer(query, llm) -> {"answer", "prompt", "active_tokens", ...}
    info() -> dict

They share ONE prompt format ("Memory Context:" + numbered blocks +
"Question: {query}") and ONE answer path: a real `LLMEngine.generate` call on
the shared engine instance. Only the context mechanism differs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

try:
    # Block construction must be IDENTICAL across all evaluated systems, so the
    # baselines reuse the exact chunking helper used by SemPointer.
    from sempointer.pipeline import DEFAULT_SYSTEM_PROMPT, chunk_text
except ImportError as _exc:  # pragma: no cover - environment error, not a fallback
    raise ImportError(
        "baselines requires the `sempointer` package (the experiments/ directory "
        "must be on sys.path, e.g. run from experiments/ or add experiments/ to "
        "PYTHONPATH)."
    ) from _exc

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "chunk_text",
    "require_ingested",
    "ingest_blocks",
    "build_memory_prompt",
    "finish_answer",
    "rrf_fuse",
    "select_ranked",
    "RankedRetrievalBaseline",
]


def require_ingested(blocks: Optional[Sequence[str]], system_name: str) -> None:
    if not blocks:
        raise RuntimeError(f"{system_name}.ingest() must be called with a non-empty context before use")


def ingest_blocks(context: str, block_size: int, tokenizer) -> tuple:
    """Chunk `context` and return (blocks, ingestion_stats_dict)."""
    t0 = time.perf_counter()
    blocks = chunk_text(context, block_size, tokenizer)
    ingestion_ms = (time.perf_counter() - t0) * 1000.0
    n_context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
    block_token_counts = [len(tokenizer.encode(b, add_special_tokens=False)) for b in blocks]
    stats = {
        "ingestion_latency_ms": ingestion_ms,
        "ingestion_tokens": n_context_tokens,
        "n_blocks": len(blocks),
        "block_token_counts": block_token_counts,
    }
    return blocks, stats


def build_memory_prompt(query: str, blocks: Sequence[str], system_prompt: Optional[str]) -> str:
    """Uniform prompt: System + "Memory Context:" + numbered blocks + Question.

    Same section style as sempointer.prompt_builder.PromptBuilder (which the
    SemPointer pipeline uses), so only the context mechanism differs.
    """
    sections: List[str] = []
    if system_prompt:
        sections.append(f"System: {system_prompt.strip()}")
    numbered = [f"[MEMORY {i}]:\n{b.strip()}" for i, b in enumerate(blocks)]
    sections.append("Memory Context:\n" + "\n\n".join(numbered))
    sections.append(f"Question: {query.strip()}\nAnswer:")
    return "\n\n".join(sections)


def finish_answer(
    *,
    system: str,
    query: str,
    blocks: Sequence[str],
    llm,
    tokenizer,
    system_prompt: Optional[str],
    selected_ids: Optional[List[int]],
    selection_scores: Optional[List[float]],
    selection_latency_ms: float,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds the active prompt over `blocks` and generates the answer with the
    shared LLMEngine (real greedy generation, same as SemPointer's answer())."""
    prompt = build_memory_prompt(query, blocks, system_prompt)
    core_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    gen = llm.generate(prompt)  # real LLM generation on the shared engine

    result = {
        "answer": gen.text,
        "prompt": gen.prompt,
        "active_tokens": gen.prompt_tokens,
        "active_tokens_core": core_tokens,  # without chat-template framing (none here: raw prompt)
        "selected_ids": selected_ids,
        "selection_scores": selection_scores,
        "selection_latency_ms": selection_latency_ms,
        "generation_latency_ms": gen.latency_ms,
        "completion_tokens": gen.completion_tokens,
        "peak_vram_mb": gen.peak_vram_mb,
        "n_blocks": len(blocks),
        "stop_reason": gen.stop_reason,
        "extra": {"system": system, **extra},
    }
    return result


# ---------------------------------------------------------------------- #
# Equal-budget selection + rank fusion (shared by bm25 / rag / hybrid).
# ---------------------------------------------------------------------- #
def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60,
             weights: Optional[Sequence[float]] = None) -> List[tuple]:
    """Reciprocal Rank Fusion (Cormack et al. 2009).

    fused(d) = sum_r w_r / (k + rank_r(d)), rank starting at 1. Documents
    missing from a ranking contribute nothing for it. Ties are broken by the
    lower block index (deterministic). Returns [(block_id, fused_score)] best
    first.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("rrf_fuse: len(weights) must equal len(rankings)")
    fused: Dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking, start=1):
            fused[int(doc)] = fused.get(int(doc), 0.0) + float(w) / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def select_ranked(ranked_ids: Sequence[int], block_token_counts: Sequence[int],
                  m: Optional[int] = None, token_budget: Optional[int] = None) -> List[int]:
    """Chooses blocks from a best-first ranking.

    - top-m mode (token_budget is None): the first min(m, N) ids.
    - token-budget mode: the longest rank PREFIX whose summed block tokens
      (LLM tokenizer, block content only — no framing/question) is
      <= token_budget. At least one block is always returned, even if the
      top block alone exceeds the budget (callers record the used tokens).
    """
    ranked_ids = [int(i) for i in ranked_ids]
    if token_budget is None:
        if m is None or m < 1:
            raise ValueError(f"top-m selection needs m >= 1 (got {m!r})")
        return ranked_ids[: min(m, len(ranked_ids))]
    if token_budget <= 0:
        raise ValueError(f"token_budget must be > 0 (got {token_budget!r})")
    chosen: List[int] = []
    used = 0
    for i in ranked_ids:
        c = int(block_token_counts[i])
        if chosen and used + c > token_budget:
            break
        chosen.append(i)
        used += c
        if used >= token_budget:
            break
    return chosen


class RankedRetrievalBaseline:
    """Base for retrieval baselines: rank all blocks, then choose by top-m or
    by token budget, then answer through the shared engine (finish_answer).

    Subclasses implement `_index()` (after self.blocks is set) and
    `rank_all(query) -> (ids_best_first, scores)`.
    """

    system_name = "retrieval"
    retrieval_desc = ""

    def __init__(self, tokenizer, m: Optional[int] = 1, block_size: int = 512,
                 token_budget: Optional[int] = None,
                 system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT):
        if token_budget is None and (m is None or int(m) < 1):
            raise ValueError(f"{type(self).__name__}: m must be >= 1 when no token_budget is set")
        if token_budget is not None and int(token_budget) <= 0:
            raise ValueError(f"{type(self).__name__}: token_budget must be > 0")
        self.tokenizer = tokenizer
        self.m = None if m is None else int(m)
        self.token_budget = None if token_budget is None else int(token_budget)
        self.block_size = block_size
        self.system_prompt = system_prompt
        self.blocks: List[str] = []
        self.ingestion: Dict[str, Any] = {}

    @property
    def selection_mode(self) -> str:
        return "token_budget" if self.token_budget is not None else "top_m"

    # -- subclass hooks ------------------------------------------------- #
    def _index(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def rank_all(self, query: str):  # pragma: no cover - abstract
        raise NotImplementedError

    def _extra(self) -> Dict[str, Any]:
        return {}

    # -- System interface ---------------------------------------------- #
    def ingest(self, context: str) -> None:
        self.blocks, self.ingestion = ingest_blocks(context, self.block_size, self.tokenizer)
        self._index()

    def retrieve(self, query: str):
        """Selected block ids (best first) and their retrieval scores."""
        require_ingested(self.blocks, type(self).__name__)
        ids, scores = self.rank_all(query)
        score_of = {int(i): float(s) for i, s in zip(ids, scores)}
        chosen = select_ranked(ids, self.ingestion["block_token_counts"],
                               m=self.m, token_budget=self.token_budget)
        return chosen, [score_of[i] for i in chosen]

    def answer(self, query: str, llm) -> Dict[str, Any]:
        t0 = time.perf_counter()
        selected_ids, scores = self.retrieve(query)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        counts = self.ingestion["block_token_counts"]
        out = finish_answer(
            system=self.system_name,
            query=query,
            blocks=[self.blocks[i] for i in selected_ids],
            llm=llm,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            selected_ids=selected_ids,
            selection_scores=scores,
            selection_latency_ms=retrieval_ms,
            extra={
                "retrieval": self.retrieval_desc,
                "selection_mode": self.selection_mode,
                "m": self.m,
                "token_budget": self.token_budget,
                "block_tokens_selected": int(sum(counts[i] for i in selected_ids)),
                "ingestion": self.ingestion,
                **self._extra(),
            },
        )
        out["n_selected"] = len(selected_ids)
        return out

    def info(self) -> Dict[str, Any]:
        return {
            "system": self.system_name,
            "selection_mode": self.selection_mode,
            "m": self.m,
            "token_budget": self.token_budget,
            "block_size": self.block_size,
            "n_blocks": len(self.blocks),
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
            **self._extra(),
        }
