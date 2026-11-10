"""Exact token counts and reproducible concurrent preparation; no model inference."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


def word_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "mars": 1, "rover": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return SimpleNamespace(backend_tokenizer=backend)


def test_token_cache_deduplicates_batches_and_survives_restart(tmp_path):
    from evaluation.preparation import ExactTokenCounter
    tokenizer = word_tokenizer()
    path = tmp_path / "counts.sqlite"
    with ExactTokenCounter(tokenizer, path, "tokenizer-A") as counter:
        assert counter.count_many(["mars rover", "mars", "mars rover"]) == [2, 1, 2]
        assert counter.stats()["encoded"] == 2
    with ExactTokenCounter(tokenizer, path, "tokenizer-A") as counter:
        assert counter.count_many(["mars", "mars rover"]) == [1, 2]
        assert counter.stats()["encoded"] == 0
    with ExactTokenCounter(tokenizer, path, "tokenizer-B") as counter:
        assert counter.count_many(["mars rover"]) == [2]
        assert counter.stats()["encoded"] == 1


def test_shared_counter_keeps_order_under_concurrent_batches(tmp_path):
    from evaluation.preparation import ExactTokenCounter
    with ExactTokenCounter(word_tokenizer(), tmp_path / "counts.sqlite", "fixture") as counter:
        batches = [["mars " * (i + 1), "rover"] for i in range(16)]
        with ThreadPoolExecutor(max_workers=4) as workers:
            actual = list(workers.map(counter.count_many, batches))
        assert actual == [[i + 1, 1] for i in range(16)]


def test_cache_invalidation_includes_effective_tokenizer_configuration(tmp_path):
    from evaluation.preparation import ExactTokenCounter
    from tokenizers import pre_tokenizers
    tokenizer = word_tokenizer()
    path = tmp_path / "counts.sqlite"
    with ExactTokenCounter(tokenizer, path, "same-source-file") as counter:
        assert counter.count_many(["mars-rover"]) == [3]
    tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    with ExactTokenCounter(tokenizer, path, "same-source-file") as counter:
        assert counter.count_many(["mars-rover"]) == [1]


def test_fast_counts_match_pinned_qwen_for_unicode_special_tokens_and_long_text(tmp_path):
    from evaluation.preparation import ExactTokenCounter
    from transformers import AutoTokenizer
    path = Path(__file__).resolve().parents[2] / "models/Qwen3.5-4B"
    if not (path / "tokenizer.json").exists():
        pytest.skip("Pinned local tokenizer is unavailable")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    texts = ["", "mars rover", "हिन्दी 中文 café e\u0301 👨‍👩‍👧‍👦", "<|im_start|>user\nhello<|im_end|>",
             "def hello():\n    return 42\n", "Long document. 中文 परीक्षण\n" * 12000]
    expected = [len(tok.encode(text, add_special_tokens=False)) for text in texts]
    with ExactTokenCounter(tok, tmp_path / "qwen.sqlite", "qwen-fixture") as counter:
        assert counter.count_many(texts) == expected


def test_batched_scan_preserves_exact_source_token_limit(tmp_path, monkeypatch):
    import prepare_modal as prep
    from evaluation.data import Example, Task
    from evaluation.preparation import ExactTokenCounter
    source = tmp_path / "source.json"; source.write_text("boundary fixture")
    task = Task("longbench", "hotpotqa", source)
    examples = [Example(str(n), "mars " * n, "where?", ["mars"], "qa_f1",
                        "longbench", "hotpotqa", "test", str(n)) for n in [262143, 262144, 262145]]
    monkeypatch.setattr(prep, "iter_examples", lambda *a, **kw: iter(examples))
    monkeypatch.setattr(prep, "partition_for", lambda ex: "test")
    with ExactTokenCounter(word_tokenizer(), tmp_path / "counts.sqlite", "fixture") as counter:
        rows, stats = prep.scan_family([task], None, counter, 19, 3, set(), tmp_path / "families", "code")
    assert {row["id"]: row["source_tokens"] for row in rows} == {"262143": 262143, "262144": 262144}
    assert stats["source_token_limit"] == 1


@pytest.mark.parametrize("family", ["longbench", "locomo"])
def test_batched_family_scan_matches_serial_selection_and_invalidates_checkpoints(tmp_path, monkeypatch, family):
    import prepare_modal as prep
    from evaluation.cloud_policy import CandidatePool, digest
    from evaluation.data import Example, Task
    from evaluation.preparation import ExactTokenCounter
    source = tmp_path / "source.json"; source.write_text("fixture version one")
    task = Task(family, "hotpotqa" if family == "longbench" else "qa", source)
    examples = [Example(str(i), "mars " * (i + 1), "where?", ["mars"], "qa_f1",
                        family, task.task, "test", str(i), meta={"category": i % 3}) for i in range(40)]
    examples += [examples[2], Example("too-long", "mars " * 262145, "where?", ["mars"],
                                    "qa_f1", family, task.task, "test", "over-limit", meta={"category": 0})]
    monkeypatch.setattr(prep, "partition_for", lambda ex: "test")
    reads = []
    def read(task, tokenizer=None):
        reads.append(task.path)
        yield from examples
    monkeypatch.setattr(prep, "iter_examples", read)
    baseline = CandidatePool(19, 10)
    for ex in examples:
        row = {"id": ex.id, "family": ex.family, "stratum": prep.stratum(task, ex),
               "source": digest(task.to_dict()), "semantic_id": digest([ex.family, ex.context, ex.question]),
               "output_cap": prep.output_cap(task)}
        if baseline.competitive(row) and len(ex.context.split()) <= 262144:
            baseline.add(row)
    with ExactTokenCounter(word_tokenizer(), tmp_path / "counts.sqlite", "fixture") as counter:
        arguments = dict(tasks=[task], tokenizer=None, counter=counter, seed=19, cap=10,
                         excluded=set(), cache_dir=tmp_path / "families", code_signature="code-A")
        rows, stats = prep.scan_family(**arguments)
        assert {r["id"] for r in rows} == {r["id"] for r in baseline.selected()}
        assert stats["scanned"] == len(examples)
        prep.scan_family(**arguments)
        assert len(reads) == 1  # a completed family is reused
        source.write_text("fixture version two is different")
        prep.scan_family(**arguments)
        assert len(reads) == 2
        prep.scan_family(**{**arguments, "seed": 20})
        assert len(reads) == 3
