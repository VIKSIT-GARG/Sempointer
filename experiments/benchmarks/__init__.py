"""Public benchmark ingestion layer for the SemPointer reconstruction.

Phase 3 (owner: Agent A). Loaders return a uniform BenchExample format:

    from benchmarks import get_builder
    examples = get_builder("ruler")("niah_single_1", 4096, n_samples=2,
                                    tokenizer_fn=qwen_len_fn)
    examples = get_builder("longbench")("hotpotqa", n_samples=2)
    examples = get_builder("locomo")(conversation_idx=0, n_samples=5)

All builders record gold answers at construction time and tag every example
with its official metric (see metrics/official_metrics.py).
"""

from .common import BenchExample, SUPPORTED_METRICS
from .ruler import build_ruler, RULER_TASKS
from .longbench import build_longbench, SUPPORTED_TASKS as LONGBENCH_TASKS
from .locomo import build_locomo

BUILDERS = {
    "ruler": build_ruler,
    "longbench": build_longbench,
    "locomo": build_locomo,
}


def get_builder(name: str):
    """Return the builder callable for a benchmark by name.

    name in {"ruler", "longbench", "locomo"}; signatures differ per benchmark
    (see each module's build_* docstring).
    """
    if name not in BUILDERS:
        raise ValueError(f"Unknown benchmark {name!r}; expected one of "
                         f"{sorted(BUILDERS)}")
    return BUILDERS[name]


__all__ = [
    "BenchExample",
    "SUPPORTED_METRICS",
    "BUILDERS",
    "get_builder",
    "build_ruler",
    "build_longbench",
    "build_locomo",
    "RULER_TASKS",
    "LONGBENCH_TASKS",
]
