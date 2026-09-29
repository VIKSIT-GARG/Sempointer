"""B6v2: Agentic pointer-routing baseline, attempt 2 (uniform `System` interface).

HYPOTHESIS UNDER TEST (see experiments/MECH2_DIAGNOSIS.md):
    v1 routed exactly [0] on 23/30 musique pilot questions, selected a single
    block on 29/30, and never selected 2+ blocks on a benchmark that requires
    2-hop reasoning (mean F1 0.030 vs 0.054 dense at ~3.1x tokens). The top
    failure mode is therefore the ROUTER (position-collapsed, under-selecting),
    not the reader: on the 7 questions where v1 routed away from block 0, 4
    scored > 0, while block-0 routes starved the reader (15/30 abstentions).

    v1 NEGATIVE, STATED HONESTLY: all-PTR-index prompt (300-char front-loaded
    previews) + loose "reply-only-numbers" grammar + no retry produced a router
    that almost always returned block 0. v2 may still lose; pre-registered
    concession conditions are in experiments/MECH2_HANDOFF.md.

WHAT v2 CHANGES (at most two of the allowed three; uses (a) + (b)):
    (a) Keyword-enriched index lines: each registry line is
        `ID <i> | keywords: w1, w2, ... | preview...` where keywords are the
        block's top-TF terms after stopword/length filtering and a
        max-document-frequency cut (terms in >50% of blocks are dropped as
        non-discriminative). Capped at 8 short terms per block, so the token
        overhead is small and bounded. Goal: give the router a discriminative
        signal beyond the front-loaded preview, breaking the block-0 bias.
    (b) Constrained routing: the routing prompt numbers every block as a
        shortlist, demands the strict grammar `IDS: a, b` (or `IDS: NONE`),
        parses strict-first with the v1 lenient parser as fallback, and
        retries the route call ONCE with a repair prompt when both parses
        yield nothing usable. The retry is an extra LLM call and is counted
        honestly in `n_llm_calls` (2 normally, 3 on retry).
    NOT used: (c) route-then-rerank (resolve top-3 + single answer call).
    Rejected deliberately: it costs a third answer-side call and many more
    active tokens on every question, while the diagnosed failure is routing
    precision, not answer coverage.

CALL/TOKEN ACCOUNTING: base path is 2 LLM calls (route + answer), identical to
v1. A retry adds exactly 1 route call. active_tokens sums GenResult
prompt_tokens over ALL calls made (measured, not derived). Both/all prompts
are returned for provenance logging. Same System contract as v1:
ingest() chunks exactly like sempointer.pipeline.chunk_text.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    build_memory_prompt,
    ingest_blocks,
    require_ingested,
)
from .agentic_pointer import parse_routed_ids as _lenient_parse

ROUTING_INSTRUCTION_V2 = (
    "You are a memory router. Below is a numbered shortlist of memory blocks; "
    "each line gives the block ID, discriminative keywords, and a preview. "
    "Multi-hop questions need evidence from 2 or more blocks: select EVERY "
    "block needed to answer (up to {m}). Reply with ONLY one line in exactly "
    "this grammar: `IDS: 3, 17` (comma-separated block numbers, no brackets, "
    "no words) or `IDS: NONE` if no block is relevant. Do not explain."
)

REPAIR_INSTRUCTION = (
    "Your previous reply did not match the required grammar. Reply again with "
    "ONLY one line: `IDS: <comma-separated block numbers>` or `IDS: NONE`. "
    "No other text."
)

STRICT_RE = re.compile(r"^\s*IDS\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)

PREVIEW_CHARS = 200  # shorter than v1 (300): keywords carry the signal instead
MAX_KEYWORDS = 8
MAX_DF_FRAC = 0.5  # drop terms appearing in more than half the blocks

_STOPWORDS = frozenset(
    """a an the and or but if then else of at by for with about into through
    during before after above below to from up down in out on off over under
    again further once here there when where why how all any both each few more
    most other some such no nor not only own same so than too very can will
    just should now is are was were be been being have has had having do does
    did doing would could ought i you he she it we they them his her its our
    their this that these those as than what which who whom whose because until
    while per passage memory memories context given following states born""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']{2,}")


def block_keywords(blocks: List[str], max_kw: int = MAX_KEYWORDS) -> List[List[str]]:
    """Top-TF discriminative terms per block (pure Python, CPU-only).

    Tokenizes on [a-z0-9], drops stopwords/short tokens, drops terms with
    document frequency above MAX_DF_FRAC (non-discriminative corpus words),
    and keeps the top `max_kw` terms by within-block TF (ties broken
    alphabetically for determinism). Never raises; a block with no usable
    terms yields [].
    """
    try:
        tf_counters: List[Counter] = []
        for blk in blocks:
            toks = [t.strip("-'") for t in _TOKEN_RE.findall(blk.lower())]
            toks = [t for t in toks if t and t not in _STOPWORDS and len(t) >= 3]
            tf_counters.append(Counter(toks))
        n = max(1, len(blocks))
        df: Counter = Counter()
        for c in tf_counters:
            for term in c:
                df[term] += 1
        out: List[List[str]] = []
        for c in tf_counters:
            ranked = sorted(
                (t for t in c if df[t] / n <= MAX_DF_FRAC),
                key=lambda t: (-c[t], t),
            )
            out.append(ranked[:max(0, max_kw)])
        return out
    except Exception:
        return [[] for _ in blocks]


def parse_routed_ids_strict(text: str, n_blocks: int) -> Tuple[List[int], bool]:
    """Strict `IDS: ...` parse. Returns (ids, matched_strict_grammar)."""
    try:
        m = STRICT_RE.match(text or "")
        if not m:
            return [], False
        body = m.group(1).strip()
        if re.fullmatch(r"none", body, re.IGNORECASE):
            return [], True
        ids: List[int] = []
        for tok in re.findall(r"\d+", body):
            i = int(tok)
            if 0 <= i < n_blocks and i not in ids:
                ids.append(i)
        # Grammar matched but body had no usable ids -> strict match, empty set
        # only if the body contained no digits at all beyond NONE handling.
        if not ids and re.search(r"\d", body):
            return [], True  # digits present but all out-of-range: strict, unusable
        if not ids and not re.search(r"\d", body):
            return [], True  # garbage body under IDS: header: strict, unusable
        return ids, True
    except Exception:
        return [], False


def parse_routed_ids_v2(text: str, n_blocks: int) -> Tuple[List[int], bool]:
    """Strict-first, lenient-fallback parse. Returns (ids, strict_matched)."""
    ids, strict = parse_routed_ids_strict(text, n_blocks)
    if strict:
        return ids, True
    return _lenient_parse(text, n_blocks), False


def _index_lines(blocks: List[str], keywords: List[List[str]]) -> List[str]:
    lines = []
    for i, blk in enumerate(blocks):
        preview = " ".join(blk.split())[:PREVIEW_CHARS]
        kw = ", ".join(keywords[i]) if i < len(keywords) and keywords[i] else "-"
        lines.append(f"ID {i} | keywords: {kw} | {preview}")
    return lines


class AgenticPointerV2Baseline:
    """LLM-routed pointer selection with keyword index + constrained routing."""

    def __init__(
        self,
        tokenizer,
        m: int = 4,
        block_size: int = 512,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer = tokenizer
        self.m = m
        self.block_size = block_size
        self.system_prompt = system_prompt
        self.blocks: List[str] = []
        self.ingestion: Dict[str, Any] = {}
        self.keywords: List[List[str]] = []

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        self.blocks, self.ingestion = ingest_blocks(context, self.block_size, self.tokenizer)
        self.keywords = block_keywords(self.blocks)

    def build_routing_prompt(self, query: str) -> str:
        """Routing prompt (Call 1): numbered keyword-enriched shortlist + query."""
        require_ingested(self.blocks, type(self).__name__)
        sections: List[str] = []
        if self.system_prompt:
            sections.append(f"System: {self.system_prompt.strip()}")
        lines = _index_lines(self.blocks, self.keywords)
        sections.append("Memory Registry Shortlist (one block per line):\n" + "\n".join(lines))
        sections.append(
            f"Question: {query.strip()}\n"
            + ROUTING_INSTRUCTION_V2.format(m=max(0, self.m))
        )
        return "\n\n".join(sections)

    def build_repair_prompt(self, query: str, bad_reply: str) -> str:
        return "\n\n".join(
            [
                f"Question: {query.strip()}",
                f"Your previous reply was: {bad_reply.strip()[:500]}",
                REPAIR_INSTRUCTION,
            ]
        )

    def answer(self, query: str, llm) -> Dict[str, Any]:
        require_ingested(self.blocks, type(self).__name__)
        n_calls = 0
        prompt_tokens_total = 0
        completion_tokens_total = 0
        peak_vram = 0.0
        gen_latency_ms = 0.0
        prompts: List[str] = []

        # ---- Call 1: route ------------------------------------------------
        routing_prompt = self.build_routing_prompt(query)
        routing_core = len(self.tokenizer.encode(routing_prompt, add_special_tokens=False))
        t_route0 = time.perf_counter()
        route_gen = llm.generate(routing_prompt)
        routing_latency_ms = (time.perf_counter() - t_route0) * 1000.0
        n_calls += 1
        prompts.append(route_gen.prompt)
        prompt_tokens_total += route_gen.prompt_tokens
        completion_tokens_total += route_gen.completion_tokens
        peak_vram = max(peak_vram, route_gen.peak_vram_mb)
        gen_latency_ms += route_gen.latency_ms

        routed_ids, strict = parse_routed_ids_v2(route_gen.text, len(self.blocks))
        routing_raw = route_gen.text
        retried = False

        # ---- Call 1b (retry once): only when nothing usable was parsed ----
        if not routed_ids and not re.fullmatch(r"\s*IDS\s*:\s*NONE\s*", route_gen.text or "", re.IGNORECASE):
            repair_prompt = self.build_repair_prompt(query, route_gen.text or "")
            repair_gen = llm.generate(repair_prompt)
            n_calls += 1
            retried = True
            prompts.append(repair_gen.prompt)
            prompt_tokens_total += repair_gen.prompt_tokens
            completion_tokens_total += repair_gen.completion_tokens
            peak_vram = max(peak_vram, repair_gen.peak_vram_mb)
            gen_latency_ms += repair_gen.latency_ms
            routing_latency_ms += repair_gen.latency_ms
            r_ids, _ = parse_routed_ids_v2(repair_gen.text, len(self.blocks))
            if r_ids:
                routed_ids = r_ids
                routing_raw = route_gen.text + "\n[RETRY] " + repair_gen.text
        _ = strict  # kept for provenance below

        routed_ids = routed_ids[: max(0, self.m)]

        # ---- Final call: answer from resolved blocks ----------------------
        resolved = [self.blocks[i] for i in routed_ids]
        final_prompt = build_memory_prompt(query, resolved, self.system_prompt)
        final_core = len(self.tokenizer.encode(final_prompt, add_special_tokens=False))
        answer_gen = llm.generate(final_prompt)
        n_calls += 1
        prompts.append(answer_gen.prompt)
        prompt_tokens_total += answer_gen.prompt_tokens
        completion_tokens_total += answer_gen.completion_tokens
        peak_vram = max(peak_vram, answer_gen.peak_vram_mb)
        gen_latency_ms += answer_gen.latency_ms

        return {
            "answer": answer_gen.text,
            "prompt": answer_gen.prompt,
            "prompts": prompts,  # all calls logged
            "routing_prompt": route_gen.prompt,
            "routing_raw": routing_raw,
            "active_tokens": prompt_tokens_total + answer_gen.prompt_tokens,
            "active_tokens_core": routing_core + final_core,
            "routed_ids": routed_ids,
            "selected_ids": routed_ids,
            "selection_scores": None,  # LLM-routed, no numeric scores
            "selection_latency_ms": routing_latency_ms,
            "routing_latency_ms": routing_latency_ms,
            "generation_latency_ms": gen_latency_ms,
            "completion_tokens": completion_tokens_total,
            "peak_vram_mb": peak_vram,
            "n_blocks": len(self.blocks),
            "n_llm_calls": n_calls,
            "stop_reason": answer_gen.stop_reason,
            "extra": {
                "system": "agentic_pointer_v2",
                "retrieval": "llm-routed keyword-enriched pointer selection",
                "routing_instruction": ROUTING_INSTRUCTION_V2.format(m=max(0, self.m)),
                "strict_grammar_matched": strict,
                "routing_retried": retried,
                "ingestion": self.ingestion,
            },
        }

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "agentic_pointer_v2",
            "m": self.m,
            "block_size": self.block_size,
            "n_blocks": len(self.blocks),
            "n_llm_calls": "2 (3 on routing retry)",
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
