"""Common data structures shared by all benchmark loaders.

Every loader in this package returns a list of ``BenchExample`` objects so
that the benchmark execution layer (run_benchmark.py) can treat RULER,
LongBench and LoCoMo uniformly.

Part of the SemPointer reconstruction (Phase 3, owner: Agent A).
"""

from dataclasses import dataclass, field

SUPPORTED_METRICS = (
    "ruler_containment",  # NVIDIA/RULER string_match_all (case-insensitive containment)
    "longbench_f1",       # THUDM/LongBench official qa_f1_score
    "longbench_em",       # SQuAD-style exact match on LongBench normalization (supplementary)
    "locomo_f1",          # LoCoMo adaptation: token F1 (see benchmarks/locomo.py docstring)
)


@dataclass
class BenchExample:
    """One evaluation item with an explicitly recorded gold answer.

    Attributes:
        id: Unique identifier, stable across runs (used in predictions.jsonl).
        context: The full context text to ingest (haystack, documents,
            conversation, ...).
        question: The query/instruction to answer given the context.
        gold: ALL acceptable reference answers recorded at construction time.
            Scoring takes the maximum over these references.
        metric: One of SUPPORTED_METRICS; selects the official scoring fn.
        meta: Task name, source URLs and task-specific bookkeeping
            (e.g. needle depth fraction, LoCoMo QA category, official prompt
            segments for byte-exact prompt reconstruction).
    """

    id: str
    context: str
    question: str
    gold: list
    metric: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"metric {self.metric!r} not in {SUPPORTED_METRICS}"
            )
        self.gold = [str(g) for g in self.gold]

    def to_dict(self) -> dict:
        """JSON-serializable representation (for provenance / predictions)."""
        return {
            "id": self.id,
            "context": self.context,
            "question": self.question,
            "gold": list(self.gold),
            "metric": self.metric,
            "meta": dict(self.meta),
        }
