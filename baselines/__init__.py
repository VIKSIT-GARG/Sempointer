"""Baselines module for the SemPointer reconstruction.

All baselines implement the uniform `System` contract (see sempointer/pipeline.py):

    ingest(context: str) -> None
    answer(query: str, llm: LLMEngine) -> {"answer", "prompt", "active_tokens",
                                           "selected_ids", "extra", ...}
    info() -> dict

Exports are lazy so that importing `baselines` does not pull in heavy
dependencies (sentence-transformers, kvpress and its global attention patch).
Import submodules directly or via the attribute access below.
"""

from typing import Any

__all__ = [
    "FullContextBaseline",
    "BM25Baseline",
    "DenseRAGBaseline",
    "KVPressBaseline",
]

_LAZY = {
    "FullContextBaseline": ("baselines.full_context", "FullContextBaseline"),
    "BM25Baseline": ("baselines.bm25_baseline", "BM25Baseline"),
    "DenseRAGBaseline": ("baselines.dense_rag", "DenseRAGBaseline"),
    "KVPressBaseline": ("baselines.kvpress_baseline", "KVPressBaseline"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
