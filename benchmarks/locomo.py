"""LoCoMo loader (snap-research/locomo, ACL 2024).

Primary sources:
  - Repo:   https://github.com/snap-research/locomo
  - Paper:  arXiv:2402.17753 ("Evaluating Very Long-Term Conversational
            Memory of LLM Agents")
  - Data:   https://raw.githubusercontent.com/snap-research/locomo/main/data/
            locomo10.json (10 synthetic multi-session conversations; cached
            under benchmarks/data/locomo_cache/locomo10.json).

Data structure (verified against the actual file):
  root: list of 10 conversations; each has keys
    qa            -> list of {question, answer?, category, evidence,
                               adversarial_answer?}
    conversation  -> dict: speaker_a, speaker_b, session_N_date_time (str),
                     session_N (list of turns {speaker, dia_id, text, and for
                     image-sharing turns also img_url/blip_caption/query})
    sample_id, event_summary, observation, session_summary
  QA categories (LoCoMo paper Table 1): 1 = multi-hop, 2 = temporal
  reasoning, 3 = open-domain knowledge, 4 = single-hop, 5 = adversarial.

Gold handling (faithful to the file and to task_eval/evaluation.py of the
official repo):
  - categories 1-4: gold = str(answer); category 3 answers are truncated at
    the first ';' exactly like the official eval
    (evaluation.py: answer.split(';')[0].strip()).
  - category 5 (adversarial): items carry 'adversarial_answer' (a plausible
    but wrong statement) and no 'answer'. We record gold = ["not mentioned"]
    and keep the adversarial statement in meta["adversarial_answer"]. The
    official scoring for cat-5 is binary: prediction containing
    "no information available" or "not mentioned" scores 1, else 0
    (official repo evaluation.py, lines 217-221). Under our token-F1 metric a
    "not mentioned"-style prediction gets partial credit; results MUST be
    labeled with the adaptation below.

DOCUMENTED ADAPTATION (metric): the LoCoMo paper scores open-domain (cat 3)
and temporal (cat 2) answers with an LLM judge (GPT-4), and the official
repo's evaluation.py uses Porter-stemmed token F1 (with a normalization that
also removes 'and' and commas) for categories 1-4 plus the special binary
adversarial rule for category 5. No judge API is available in this
environment, so we use the LongBench-style token F1
(metrics/official_metrics.py::qa_f1_score, metric id "locomo_f1") for ALL
categories. This is a conservative adaptation and any results table built on
it must state "LoCoMo (token-F1 adaptation)".
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import BenchExample

LOCOMO_GITHUB = "https://github.com/snap-research/locomo"
LOCOMO_PAPER = "https://arxiv.org/abs/2402.17753"
LOCOMO_DATA_URL = ("https://raw.githubusercontent.com/snap-research/locomo/"
                   "main/data/locomo10.json")

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal reasoning",
    3: "open-domain knowledge",
    4: "single-hop",
    5: "adversarial",
}

NOT_MENTIONED_GOLD = "not mentioned"

_CACHE_DIR = Path(__file__).resolve().parent / "data" / "locomo_cache"
_DATA_FILE = _CACHE_DIR / "locomo10.json"


def _load_conversations() -> list:
    if not _DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {_DATA_FILE}. Download the official LoCoMo data once "
            f"with: curl -L {LOCOMO_DATA_URL} -o {_DATA_FILE}"
        )
    with open(_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def render_conversation(conversation: dict) -> str:
    """Render the full conversation as 'Speaker: text' turns.

    Sessions are ordered by their numeric suffix; within a session, turns are
    kept in file order. Image-sharing turns that also carry dialogue text are
    kept (they are real conversational turns); only turns with no usable text
    (image-only) are excluded. Session date/time lines are NOT added (kept
    minimal per the project's loader contract).
    """
    session_keys = sorted(
        (k for k in conversation
         if k.startswith("session_") and not k.endswith("_date_time")
         and isinstance(conversation[k], list)),
        key=lambda k: int(k.split("_")[1]),
    )
    lines = []
    for key in session_keys:
        for turn in conversation[key]:
            text = (turn.get("text") or "").strip()
            if not text:
                continue  # image-only turn
            lines.append(f"{turn.get('speaker', '?')}: {text}")
    return "\n".join(lines)


def build_locomo(conversation_idx: int = 0, categories: list | None = None,
                 n_samples: int | None = None) -> list:
    """Build one BenchExample per QA item of one LoCoMo conversation.

    Args:
        conversation_idx: index into the 10 official conversations (0-9).
        categories: optional filter of QA categories 1-5 (official ids).
        n_samples: optional cap applied AFTER the category filter (the QA
            list is kept in official file order; no shuffling).

    Returns:
        list[BenchExample] with metric="locomo_f1", category bookkeeping in
        meta. Gold values are read directly from the dataset file (never
        regenerated).
    """
    conversations = _load_conversations()
    if not 0 <= conversation_idx < len(conversations):
        raise ValueError(
            f"conversation_idx must be in [0, {len(conversations) - 1}], "
            f"got {conversation_idx}"
        )
    conv = conversations[conversation_idx]
    context = render_conversation(conv["conversation"])
    speakers = {k: conv["conversation"].get(k) for k in ("speaker_a", "speaker_b")}

    qa_items = conv["qa"]
    if categories is not None:
        allowed = set(int(c) for c in categories)
        unknown = allowed - set(CATEGORY_NAMES)
        if unknown:
            raise ValueError(f"Unknown LoCoMo categories: {sorted(unknown)}; "
                             f"valid: 1-5")
        qa_items = [q for q in qa_items if int(q["category"]) in allowed]
    if n_samples is not None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        qa_items = qa_items[:n_samples]

    examples = []
    for i, qa in enumerate(qa_items):
        category = int(qa["category"])
        if category == 5:
            gold = [NOT_MENTIONED_GOLD]
        else:
            answer = qa.get("answer")
            if answer is None:
                # Malformed item; fall back to the adversarial statement so
                # the gold list is never empty.
                gold = [str(qa.get("adversarial_answer", NOT_MENTIONED_GOLD))]
            elif isinstance(answer, list):
                gold = [str(a) for a in answer]
            else:
                gold = [str(answer)]
                if category == 3:  # official: truncate at first ';'
                    gold = [gold[0].split(";")[0].strip()]
        examples.append(BenchExample(
            id=f"locomo-{conv.get('sample_id', conversation_idx)}-qa{i}",
            context=context,
            question=str(qa["question"]),
            gold=gold,
            metric="locomo_f1",
            meta={
                "task": "locomo_qa",
                "benchmark": "locomo",
                "source_urls": [LOCOMO_GITHUB, LOCOMO_PAPER],
                "sample_id": conv.get("sample_id"),
                "conversation_idx": conversation_idx,
                "category": category,
                "category_name": CATEGORY_NAMES[category],
                "evidence": qa.get("evidence"),
                "adversarial_answer": qa.get("adversarial_answer"),
                "metric_adaptation": (
                    "token-F1 (LongBench normalization) instead of the "
                    "official LLM-judge / stemmed-F1 protocol; see "
                    "benchmarks/locomo.py docstring"
                ),
                "qa_index_in_file": i,
                "n_qa_in_conversation": len(conv["qa"]),
                "speakers": speakers,
                "context_turns": context.count("\n") + 1,
            },
        ))
    return examples
