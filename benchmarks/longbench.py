"""LongBench loader (THUDM/LongBench, ACL 2024).

Primary sources:
  - Dataset:  https://huggingface.co/datasets/THUDM/LongBench
  - Repo:     https://github.com/THUDM/LongBench
  - Paper:    arXiv:2308.14586 ("LongBench: A Bilingual, Multitask Benchmark
              for Long Context Understanding")
  - Official metric: LongBench/metrics.py::qa_f1_score (normalize_answer +
              Counter-based token F1), applied per ground truth and MAXed over
              ground truths by LongBench/eval.py::scorer. Re-implemented
              verbatim in metrics/official_metrics.py.

Data access deviation (documented): the official HF dataset is a loading
script (LongBench.py) that downloads
https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip and
reads per-task jsonl files. `datasets` 5.x no longer supports script datasets
("Dataset scripts are no longer supported"), and the repo has no
auto-converted parquet branch. We therefore replicate the official script
exactly: download the same data.zip once, cache it under
benchmarks/data/longbench_cache/, and parse the same per-task jsonl with the
same fields (input, context, answers, length, dataset, language,
all_classes, _id). The data is byte-identical to what load_dataset would
have returned.

Subset policy: when n_samples is given we take the FIRST n examples of the
official test split (no shuffling) -- LongBench test sets are small and
order is dataset-defined. This evaluates a prefix subset; documented in
meta["subset"] = "first_n" and results must state the effective count.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .common import BenchExample

LONGBENCH_HF = "https://huggingface.co/datasets/THUDM/LongBench"
LONGBENCH_GITHUB = "https://github.com/THUDM/LongBench"
LONGBENCH_PAPER = "https://arxiv.org/abs/2308.14586"
DATA_ZIP_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"

# Official LongBench.py task configs; identical to the script's task_list.
SUPPORTED_TASKS = [
    "hotpotqa", "2wikimqa", "musique", "multifieldqa_en", "narrativeqa", "qasper",
]

# Official LongBench/config/dataset2prompt.json entries for our tasks; kept in
# meta so runners can reproduce the official prompt verbatim.
OFFICIAL_PROMPT_TEMPLATES = {
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nThe following are given "
        "passages.\n{context}\n\nAnswer the question based on the given "
        "passages. Only give me the answer and do not output any other "
        "words.\n\nQuestion: {input}\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, "
        "answer the following question based on the above text, only give me "
        "the answer and do not output any other words.\n\nQuestion: {input}\n"
        "Answer:"
    ),
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie "
        "script, and a question. Answer the question asconcisely as you can, "
        "using a single phrase if possible. Do not provide any explanation."
        "\n\nStory: {context}\n\nNow, answer the question based on the story "
        "asconcisely as you can, using a single phrase if possible. Do not "
        "provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the "
        "question as concisely as you can, using a single phrase or sentence "
        "if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question "
        "is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\nArticle: {context}\n\n Answer the "
        "question based on the above article as concisely as you can, using a "
        "single phrase or sentence if possible. If the question cannot be "
        "answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
}

# Official LongBench/config/dataset2maxlen.json entries (generation budget).
OFFICIAL_MAX_GEN_LEN = {
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "multifieldqa_en": 64, "narrativeqa": 128, "qasper": 128,
}

_CACHE_DIR = Path(__file__).resolve().parent / "data" / "longbench_cache"
_DATA_ZIP = _CACHE_DIR / "data.zip"
_TASK_JSONL = "data/{task}.jsonl"


def _load_task_jsonl(task: str) -> list:
    """Replicates the official LongBench.py loader: read data/<task>.jsonl
    from the cached official data.zip."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unknown LongBench task {task!r}; expected one of {SUPPORTED_TASKS}")
    if not _DATA_ZIP.exists():
        raise FileNotFoundError(
            f"Missing {_DATA_ZIP}. Download the official LongBench data once "
            f"with: curl -L {DATA_ZIP_URL} -o {_DATA_ZIP}"
        )
    with zipfile.ZipFile(_DATA_ZIP) as z:
        with z.open(_TASK_JSONL.format(task=task)) as f:
            items = [json.loads(line) for line in f.read().decode("utf-8").splitlines()
                     if line.strip()]
    return items


def build_longbench(task: str, n_samples: int | None = None,
                    seed: int = 42) -> list:
    """Build LongBench examples for one task from the official data.

    Args:
        task: one of SUPPORTED_TASKS (exact official config names).
        n_samples: if given, take the FIRST n examples of the official test
            split (no shuffling; LongBench test sets are small and we document
            that a prefix subset is evaluated). None -> full split.
        seed: accepted for interface uniformity across builders; UNUSED
            because the subset policy is a deterministic prefix (documented).

    Returns:
        list[BenchExample]: context=item["context"], question=item["input"],
        gold=item["answers"] (list), metric="longbench_f1".
    """
    del seed  # deterministic prefix subset; see docstring
    items = _load_task_jsonl(task)
    total = len(items)
    if n_samples is not None:
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        items = items[:n_samples]

    prompt_template = OFFICIAL_PROMPT_TEMPLATES[task]
    examples = []
    for i, item in enumerate(items):
        answers = item["answers"]
        if isinstance(answers, str):
            answers = [answers]
        examples.append(BenchExample(
            id=f"longbench-{task}-{item.get('_id', i)}",
            context=item["context"],
            question=item["input"],
            gold=[str(a) for a in answers],
            metric="longbench_f1",
            meta={
                "task": task,
                "benchmark": "longbench",
                "source_urls": [LONGBENCH_HF, LONGBENCH_GITHUB, LONGBENCH_PAPER],
                "official_length": item.get("length"),
                "language": item.get("language"),
                "all_classes": item.get("all_classes"),
                "official_id": item.get("_id"),
                "official_prompt_template": prompt_template,
                "official_max_gen_len": OFFICIAL_MAX_GEN_LEN[task],
                "subset": "first_n" if n_samples is not None else "full",
                "n_available_total": total,
            },
        ))
    return examples
