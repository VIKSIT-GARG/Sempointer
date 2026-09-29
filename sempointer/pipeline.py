"""End-to-end SemPointer pipeline (the `System` evaluated by run_benchmark.py).

Lifecycle, with a real causal LLM throughout:

    context -> blocks -> pointers P_i (k real vocabulary tokens per block,
            sempointer.pointers) -> pointer index in the model's sequence space
            -> query-conditioned selection: the LLM's OWN attention from the
               query to each pointer span (sempointer.attention_scorer), fused
               with BM25 + dense rankings via RRF (sempointer.selector)
            -> optional multi-hop re-selection (query + resolved evidence)
            -> resolution from a substrate:
                 M_text: selected block text re-encoded in the prompt
                 M_kv:   selected block KV precomputed once per context,
                         RoPE-re-positioned and spliced (sempointer.kv_substrate)
            -> active context = pointers of UNSELECTED blocks + resolved
               payloads of selected blocks + query -> greedy generation.

What distinguishes this from top-m RAG: every block stays addressable in the
active sequence through its pointer, selection is driven by the generating
model's attention over those pointers (not only an external embedder), and
resolved payloads can come from a precomputed KV substrate. Each of these is
switchable so ablations isolate their contribution:
    ptype    in {keyword, random_id, lead, none}
    selector in {dense, bm25, hybrid, attention, hybrid+attention}
    m        int, or "adaptive" (score-ratio cut-off, capped at m_max)
    hops     1 or 2
    substrate in {text, kv}
    show_pointers  whether unselected pointers stay in the active context
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .pointers import make_pointer_generator
from .selector import BlockSelector

DEFAULT_SYSTEM_PROMPT = (
    "You answer questions using only the provided memory context. "
    "If the context does not contain the answer, say you don't know. "
    "Answer concisely with only the answer itself."
)

INDEX_HEADER = "Memory index (one pointer per stored block):\n"
POINTER_HEADER = "Memory Pointers (unexpanded blocks):\n"
CONTEXT_HEADER = "Memory Context:\n"


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
    def __init__(
        self,
        tokenizer,
        k: int = 16,
        m: Union[int, str] = 2,
        block_size: int = 512,
        ptype: str = "keyword",
        selector: str = "hybrid+attention",
        m_max: Optional[int] = None,
        adaptive_ratio: float = 0.5,
        hops: int = 1,
        hop_evidence_tokens: int = 128,
        substrate: str = "text",
        kv_offload: str = "cpu",
        show_pointers: bool = True,
        attn_layers: str = "upper_half",
        attn_agg: str = "mean",
        rrf_k: int = 60,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        pointer_vocab_size: int = 256,
        seed: int = 42,
        device: str = "cuda",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        if substrate not in ("text", "kv"):
            raise ValueError("substrate must be 'text' or 'kv'")
        if hops not in (1, 2):
            raise ValueError("hops must be 1 or 2")
        self.tokenizer = tokenizer
        self.k = int(k)
        self.adaptive = isinstance(m, str) and m.lower() == "adaptive"
        if isinstance(m, str) and not self.adaptive:
            raise ValueError(f"m must be an int or 'adaptive', got {m!r}")
        self.m = 1 if self.adaptive else int(m)
        self.m_max = int(m_max) if m_max is not None else (4 if self.adaptive else self.m)
        self.adaptive_ratio = float(adaptive_ratio)
        self.block_size = block_size
        self.ptype = ptype
        self.hops = hops
        self.hop_evidence_tokens = hop_evidence_tokens
        self.substrate = substrate
        self.kv_offload = kv_offload
        self.show_pointers = show_pointers
        self.attn_layers = attn_layers
        self.attn_agg = attn_agg
        self.seed = seed
        self.device = device
        self.system_prompt = system_prompt

        gen_kw: Dict[str, Any] = {}
        if ptype.lower() in ("random_id", "random", "semantic_id"):
            gen_kw = {"embedder_id": embedder_id, "pointer_vocab_size": pointer_vocab_size, "device": device}
        self.generator = make_pointer_generator(ptype, k=self.k, seed=seed, **gen_kw)
        self.selector = BlockSelector(mode=selector, embedder_id=embedder_id, device=device, rrf_k=rrf_k)

        self.blocks: List[str] = []
        self.pointer_ids: List[List[int]] = []
        self._kv = None
        self._kv_context_id: Optional[int] = None
        self._kv_encode: Dict[str, Any] = {}
        self._context_id = 0

    # ------------------------------------------------------------------ #
    def _enc(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False) if text else []

    def ingest(self, context: str) -> None:
        t0 = time.perf_counter()
        self.blocks = chunk_text(context, self.block_size, self.tokenizer)
        if not self.blocks:
            raise ValueError("empty context")
        self.pointer_ids = self.generator.pointers(self.blocks, self.tokenizer)
        self.selector.fit(self.blocks)
        self.ingest_ms = (time.perf_counter() - t0) * 1000.0
        self._context_id += 1  # KV substrate is (re)built lazily for this context

    def _require_ingested(self) -> None:
        if not self.blocks:
            raise RuntimeError("SemPointerPipeline.ingest() must be called before use")

    # ------------------------------------------------------------------ #
    def _attention_scores(self, query: str, llm) -> Optional[List[float]]:
        if not self.selector.needs_attention:
            return None
        if llm is None or getattr(llm, "model", None) is None:
            raise ValueError(f"selector {self.selector.mode!r} needs the LLM (pass llm=)")
        from .attention_scorer import build_index_ids, pointer_attention_scores

        header = f"System: {self.system_prompt.strip()}\n\n{INDEX_HEADER}"
        entries = [(f"[P{i}]", ptr) for i, ptr in enumerate(self.pointer_ids)]
        ids, spans, q_span = build_index_ids(
            self.tokenizer, header, entries, f"\nQuestion: {query.strip()}", "\nRelevant pointers:",
            span="pointer",
        )
        dev = llm.model.get_input_embeddings().weight.device
        return pointer_attention_scores(
            llm.model, ids.to(dev), spans, q_span, layers=self.attn_layers, agg=self.attn_agg
        )

    def _select_once(self, query: str, llm, exclude: List[int], budget: int) -> Tuple[List[int], List[float]]:
        att = self._attention_scores(query, llm)
        ranked = [(i, s) for i, s in self.selector.rank(query, att) if i not in exclude]
        if not ranked or budget <= 0:
            return [], []
        if self.adaptive:
            top = ranked[0][1]
            chosen = [(i, s) for i, s in ranked[:budget] if s >= self.adaptive_ratio * top] or ranked[:1]
        else:
            chosen = ranked[:budget]
        return [i for i, _ in chosen], [s for _, s in chosen]

    def select(self, query: str, llm=None) -> Dict[str, Any]:
        """Query-conditioned selection (1 or 2 hops). Returns ids in selection order."""
        self._require_ingested()
        t0 = time.perf_counter()
        cap = self.m_max if self.adaptive else self.m
        cap = max(1, min(cap, len(self.blocks)))
        ids, scores = self._select_once(query, llm, [], cap)
        n_hops = 1
        if self.hops == 2 and len(ids) < len(self.blocks):
            # hop 2: re-query with the resolved evidence so bridge entities that
            # only appear in the first hop's payload can address the next block
            ev = self._enc(self.blocks[ids[0]])[: self.hop_evidence_tokens]
            q2 = query.strip() + " " + self.tokenizer.decode(ev, skip_special_tokens=True)
            extra = max(1, cap - len(ids)) if not self.adaptive else max(1, cap // 2)
            ids2, scores2 = self._select_once(q2, llm, ids, extra)
            ids, scores, n_hops = ids + ids2, scores + scores2, 2
        return {"ids": ids, "scores": scores, "n_hops": n_hops,
                "selection_latency_ms": (time.perf_counter() - t0) * 1000.0}

    # ------------------------------------------------------------------ #
    def _pointer_section(self, selected: List[int]) -> str:
        if not self.show_pointers:
            return ""
        entries = []
        for i, ptr in enumerate(self.pointer_ids):
            if i in selected:
                continue
            txt = self.tokenizer.decode(ptr, skip_special_tokens=True).strip() if ptr else ""
            entries.append(f"[P{i}] {txt}".rstrip())
        return (POINTER_HEADER + "\n".join(entries) + "\n\n") if entries else ""

    def _head_text(self, selected: List[int]) -> str:
        return f"System: {self.system_prompt.strip()}\n\n" + self._pointer_section(selected) + CONTEXT_HEADER

    def _tail_text(self, query: str) -> str:
        return f"Question: {query.strip()}\nAnswer:"

    def build_prompt(self, query: str, selected: List[int]) -> str:
        """M_text active prompt. Resolved blocks are shown in document order."""
        body = "\n\n".join(f"[MEMORY {i}]:\n{self.blocks[i].strip()}" for i in sorted(selected))
        return self._head_text(selected) + body + "\n\n" + self._tail_text(query)

    def active_prompt(self, query: str, llm=None) -> str:
        """Active prompt without generation (tests / dry-run)."""
        return self.build_prompt(query, self.select(query, llm)["ids"])

    # ------------------------------------------------------------------ #
    def _ensure_kv(self, llm) -> None:
        if self._kv is not None and self._kv_context_id == self._context_id:
            return
        from .kv_substrate import KVSubstrate

        if self._kv is None:
            self._kv = KVSubstrate(llm.model, llm.tokenizer, device=llm.device, offload=self.kv_offload)
        blocks_ids = [self._enc(b.strip()) for b in self.blocks]
        self._kv_encode = self._kv.encode_blocks(blocks_ids)
        self._kv_context_id = self._context_id

    def _answer_kv(self, query: str, selected: List[int], llm) -> Dict[str, Any]:
        self._ensure_kv(llm)
        segs: List[Tuple[str, Any]] = [("ids", self._enc(self._head_text(selected)))]
        for n, i in enumerate(sorted(selected)):
            segs.append(("ids", self._enc(("\n\n" if n else "") + f"[MEMORY {i}]:\n")))
            segs.append(("block", i))
        segs.append(("ids", self._enc("\n\n" + self._tail_text(query))))
        r = self._kv.generate(segs, max_new_tokens=llm.max_new_tokens)
        return {
            "answer": r["text"].strip(),
            "prompt": None,
            "active_tokens": r["total_positions"],
            "generation_latency_ms": r["latency_ms"],
            "ttft_ms": r["ttft_ms"],
            "completion_tokens": r["completion_tokens"],
            "peak_vram_mb": r["peak_vram_mb"],
            "stop_reason": r["stop_reason"],
            "kv_bytes": self._kv_encode.get("stored_bytes"),
            "kv_encode_ms": self._kv_encode.get("encode_ms"),
            "computed_tokens": r["computed_tokens"],
            "reused_tokens": r["reused_tokens"],
            "transfer_ms": r["transfer_ms"],
            "rope_mixed_frequencies": r.get("rope_mixed_frequencies", False),
        }

    def answer(self, query: str, llm) -> Dict[str, Any]:
        """Selection -> resolution -> active context -> REAL LLM generation."""
        self._require_ingested()
        sel = self.select(query, llm)
        selected = sel["ids"]
        if self.substrate == "kv":
            out = self._answer_kv(query, selected, llm)
        else:
            prompt = self.build_prompt(query, selected)
            gen = llm.generate(prompt)
            out = {
                "answer": gen.text,
                "prompt": gen.prompt,
                "active_tokens": gen.prompt_tokens,
                "generation_latency_ms": gen.latency_ms,
                "ttft_ms": gen.ttft_ms,
                "completion_tokens": gen.completion_tokens,
                "peak_vram_mb": gen.peak_vram_mb,
                "stop_reason": gen.stop_reason,
            }
        pointer_tokens = sum(len(p) for i, p in enumerate(self.pointer_ids) if i not in selected) \
            if self.show_pointers else 0
        out.update({
            "selected_ids": selected,
            "selection_scores": [float(s) for s in sel["scores"]],
            "selection_latency_ms": sel["selection_latency_ms"],
            "n_hops": sel["n_hops"],
            "expanded_ids": selected,
            "n_selected": len(selected),
            "n_blocks": len(self.blocks),
            "pointer_tokens": pointer_tokens,
        })
        return out

    def info(self) -> Dict[str, Any]:
        return {
            "system": "sempointer",
            "k": self.k,
            "m": "adaptive" if self.adaptive else self.m,
            "m_max": self.m_max,
            "adaptive_ratio": self.adaptive_ratio if self.adaptive else None,
            "block_size": self.block_size,
            "hops": self.hops,
            "substrate": "M_kv" if self.substrate == "kv" else "M_text",
            "show_pointers": self.show_pointers,
            "attn_layers": self.attn_layers,
            "attn_agg": self.attn_agg,
            "pointer_generator": self.generator.info(),
            "selector": self.selector.info(),
        }
