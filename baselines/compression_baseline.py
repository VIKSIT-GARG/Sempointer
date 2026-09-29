"""B4: prompt-compression baseline via LLMLingua-2 (uniform System contract).

ingest() chunks the context exactly like every other text system (ingestion
stats parity). answer() compresses the FULL joined context with LLMLingua-2
at `rate` — question-aware (the query is passed as the compression question,
the QA-relevant mode) — then generates through the shared finish_answer LLM
path over the single compressed block, so active_tokens is the MEASURED
compressed prompt length, directly comparable to other systems' tokens.

Scaffolding rule: `llmlingua` is LAZILY imported inside __init__. If it is
missing, construction raises ImportError with install instructions; the
module itself imports cleanly so one missing pilot dep never breaks the rest
of the suite (clean skip = omit the system from --systems, or probe with
is_available() before adding it to a run).

Tested against llmlingua>=0.2.4 (PromptCompressor.compress_prompt with
`question=` / `rate=` kwargs returning a dict with 'compressed_prompt').
Compressor errors at runtime propagate loudly — never an empty prompt, never
a silent fallback to uncompressed context.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ._text_common import (
    DEFAULT_SYSTEM_PROMPT,
    finish_answer,
    ingest_blocks,
    require_ingested,
)

DEFAULT_COMPRESSOR_ID = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

INSTALL_HINT = (
    "CompressionBaseline needs `llmlingua`. "
    "Install with: ./warden/bin/python -m pip install -U llmlingua"
)


def is_available() -> Tuple[bool, str]:
    """Probe the llmlingua dependency without raising. Returns (ok, detail)."""
    try:
        import llmlingua  # noqa: F401

        ver = getattr(llmlingua, "__version__", "unknown")
        return True, f"llmlingua {ver} importable"
    except ImportError as exc:
        return False, f"{type(exc).__name__}: {exc}. {INSTALL_HINT}"


class CompressionBaseline:
    """LLMLingua-2 full-context compression + real LLM answer."""

    def __init__(
        self,
        tokenizer,
        block_size: int = 512,
        compressor_id: str = DEFAULT_COMPRESSOR_ID,
        rate: float = 0.5,
        device: str = "cuda",
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    ):
        try:
            from llmlingua import PromptCompressor
        except ImportError as _exc:
            raise ImportError(INSTALL_HINT) from _exc
        if not 0.0 < rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")
        if not isinstance(compressor_id, str) or not compressor_id:
            raise ValueError("compressor_id must be a non-empty model id string")

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.compressor_id = compressor_id
        self.rate = rate
        self.device = device
        self.system_prompt = system_prompt
        self.blocks: List[str] = []
        self.ingestion: Dict[str, Any] = {}
        # Loud on failure: a compression baseline that silently skips
        # compression is just the full-context baseline mislabeled.
        try:
            self.compressor = PromptCompressor(model_name=compressor_id, device_map=device)
        except Exception as exc:  # noqa: BLE001 - re-raised loudly, never swallowed
            raise RuntimeError(
                f"CompressionBaseline: could not load compressor {compressor_id!r} "
                f"on device {device!r} ({type(exc).__name__}: {exc}). "
                "Check the model id, network access, and GPU memory."
            ) from exc

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        self.blocks, self.ingestion = ingest_blocks(context, self.block_size, self.tokenizer)

    def compress(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Compresses the full ingested context. Returns (text, stats)."""
        require_ingested(self.blocks, type(self).__name__)
        full_context = "\n\n".join(self.blocks)
        t0 = time.perf_counter()
        res = self.compressor.compress_prompt(
            full_context, instruction="", question=query, rate=self.rate
        )
        compression_ms = (time.perf_counter() - t0) * 1000.0
        try:
            compressed = res["compressed_prompt"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "CompressionBaseline: unexpected compress_prompt result "
                f"(keys: {list(res) if isinstance(res, dict) else type(res).__name__}). "
                "Expected a dict with 'compressed_prompt' (llmlingua>=0.2.4)."
            ) from exc
        if not isinstance(compressed, str) or not compressed.strip():
            raise RuntimeError(
                "CompressionBaseline: compressor returned an empty prompt — "
                "refusing to answer on nothing (never a silent fallback)."
            )
        stats = {
            "compression_latency_ms": compression_ms,
            "origin_tokens": res.get("origin_tokens"),
            "compressed_tokens": res.get("compressed_tokens"),
            "achieved_rate": res.get("rate", res.get("ratio")),
        }
        return compressed, stats

    def answer(self, query: str, llm) -> Dict[str, Any]:
        compressed, stats = self.compress(query)
        out = finish_answer(
            system="compression",
            query=query,
            blocks=[compressed],
            llm=llm,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
            # The compressor saw the whole context: every block was active
            # in compressed form.
            selected_ids=list(range(len(self.blocks))),
            selection_scores=None,
            selection_latency_ms=float(stats["compression_latency_ms"]),
            extra={
                "retrieval": f"llmlingua-2 full-context compression @rate={self.rate}",
                "compressor_id": self.compressor_id,
                "target_rate": self.rate,
                "compression": stats,
                "n_ingested_blocks": len(self.blocks),
                "ingestion": self.ingestion,
            },
        )
        return out

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        return {
            "system": "compression",
            "block_size": self.block_size,
            "compressor_id": self.compressor_id,
            "target_rate": self.rate,
            "compressor_device": self.device,
            "n_blocks": len(self.blocks),
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "ingestion_latency_ms": self.ingestion.get("ingestion_latency_ms"),
            "has_system_prompt": bool(self.system_prompt),
        }
