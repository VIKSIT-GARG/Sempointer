"""CPU tests for the benchmark harness: system-spec parser, budget-mode
baselines, hybrid RRF ordering and metric aggregation (no LLM, offline)."""

import pytest
from types import SimpleNamespace

from provenance import init_run_dir
from system_specs import merge_kwargs, parse_system_specs, split_specs, validate_kwargs


def test_resume_keeps_original_config_and_rejects_mixed_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr("provenance.environment_record", lambda: {"git_commit": "abc"})
    run = tmp_path / "paper_ruler"
    cfg = {"run_name": "paper_ruler", "systems": "sempointer,full", "limit": 10,
           "resume": False, "dry_run": False}
    init_run_dir(str(run), cfg)
    (run / "predictions.jsonl").write_text('{"system":"full"}\n')

    with pytest.raises(ValueError, match="Pass --resume"):
        init_run_dir(str(run), cfg)
    with pytest.raises(ValueError, match="different settings: limit"):
        init_run_dir(str(run), {**cfg, "limit": 5, "resume": True}, resume=True)

    init_run_dir(str(run), {**cfg, "resume": True}, resume=True)
    assert __import__("json").loads((run / "config.json").read_text())["resume"] is False
    assert len((run / "resume_history.jsonl").read_text().splitlines()) == 1


def test_engine_info_records_loaded_model_revision_without_hub_lookup():
    from engine.llm import LLMEngine

    engine = object.__new__(LLMEngine)
    engine.model_id = "example/model"
    engine.dtype = "nf4"
    engine.device = "cuda"
    engine.max_new_tokens = 32
    engine.seed = 42
    engine.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="abc123"))
    engine.tokenizer = SimpleNamespace(init_kwargs={})
    assert engine.info()["model_config_revision"] == "abc123"


def test_nf4_compute_dtype_adapts_to_t4(monkeypatch):
    import torch
    from engine.llm import nf4_compute_dtype

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert nf4_compute_dtype("cuda") == torch.float16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert nf4_compute_dtype("cuda") == torch.bfloat16


def test_bundle_revision_used_without_git_directory(tmp_path):
    from provenance import git_commit

    (tmp_path / "BUNDLE_COMMIT").write_text("a" * 40 + "\n")
    assert git_commit(str(tmp_path)) == "a" * 40


def test_engine_records_first_generated_token_time_on_short_prompt():
    import torch
    from engine.llm import LLMEngine

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def __call__(self, text, return_tensors):
            return Batch(input_ids=torch.tensor([[10, 11]]))

        def decode(self, ids, skip_special_tokens):
            return "answer"

    class Model:
        device = torch.device("cpu")

        def generate(self, input_ids, streamer, **kwargs):
            streamer.put(input_ids)
            streamer.put(torch.tensor([12]))
            streamer.end()
            return torch.tensor([[10, 11, 12]])

    engine = object.__new__(LLMEngine)
    engine.model = Model()
    engine.tokenizer = Tokenizer()
    engine.device = "cpu"
    engine.max_new_tokens = 4
    result = engine.generate("prompt")
    assert result.text == "answer"
    assert result.ttft_ms is not None
    assert 0 <= result.ttft_ms <= result.latency_ms


@pytest.mark.parametrize("prompt_tokens,max_chunk", [(5000, 4096), (12001, 2048)])
def test_long_prefill_bounds_forward_chunk_size(prompt_tokens, max_chunk):
    import torch
    from engine.llm import LLMEngine

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text, return_tensors):
            return Batch(input_ids=torch.zeros((1, prompt_tokens), dtype=torch.long))

        def decode(self, ids, skip_special_tokens):
            return "done"

    class Model:
        device = torch.device("cpu")

        def __init__(self):
            self.chunks = []

        def __call__(self, input_ids, **kwargs):
            self.chunks.append(input_ids.shape[1])
            return SimpleNamespace(past_key_values=None,
                                   logits=torch.tensor([[[0.0, 1.0]]]))

    engine = object.__new__(LLMEngine)
    engine.model = Model()
    engine.tokenizer = Tokenizer()
    engine.device = "cpu"
    engine.max_new_tokens = 1
    result = engine.generate("prompt")
    assert result.prompt_tokens == prompt_tokens
    assert max(engine.model.chunks) <= max_chunk
    assert sum(engine.model.chunks) == prompt_tokens


def test_gold_string_block_hit_is_limited_to_identifiable_niah_evidence():
    from run_benchmark import gold_string_block_hit

    blocks = ["filler", "The special code is 7350753.", "More filler"]
    assert gold_string_block_hit("niah_single_1", ["7350753"], blocks, [1]) == 1.0
    assert gold_string_block_hit("niah_single_1", ["7350753"], blocks, [0]) == 0.0
    assert gold_string_block_hit("cwe", ["7350753"], blocks, [1]) is None
    assert gold_string_block_hit("niah_single_1", ["missing"], blocks, [1]) is None


# ------------------------------------------------------------------ specs
def test_split_specs_respects_brackets():
    s = "full,sempointer[selector=fused;m=adaptive],rag[m=4,rrf_k=30],bm25:2"
    assert split_specs(s) == ["full", "sempointer[selector=fused;m=adaptive]",
                              "rag[m=4,rrf_k=30]", "bm25:2"]
    with pytest.raises(ValueError):
        split_specs("sempointer[a=1")


def test_parse_retrieval_specs_and_names():
    specs = parse_system_specs("rag:4,bm25:2,hybrid:8,rag_budget:1500,full", rag_m=[1])
    by = {s.name: s for s in specs}
    assert set(by) == {"rag_m4", "bm25_m2", "hybrid_m8", "rag_b1500", "full"}
    assert by["rag_m4"].kwargs == {"m": 4, "token_budget": None}
    assert by["rag_b1500"].kwargs == {"m": None, "token_budget": 1500}


def test_plain_rag_backcompat_and_m_list_expansion():
    assert [s.name for s in parse_system_specs("rag,bm25", rag_m=[1])] == ["rag", "bm25"]
    names = [s.name for s in parse_system_specs("rag", rag_m=[1, 2, 4, 8])]
    assert names == ["rag_m1", "rag_m2", "rag_m4", "rag_m8"]
    names = [s.name for s in parse_system_specs("hybrid,bm25_budget", token_budget=900)]
    assert names == ["hybrid_b900", "bm25_b900"]
    with pytest.raises(ValueError):
        parse_system_specs("rag_budget")  # no budget anywhere


def test_sempointer_variant_specs():
    specs = parse_system_specs("sempointer,sempointer[selector=fused;ptype=salient;m=adaptive;hops=2]")
    assert specs[0].name == "sempointer" and specs[0].kwargs == {}
    v = specs[1]
    assert v.kwargs == {"selector": "fused", "ptype": "salient", "m": "adaptive", "hops": 2}
    assert v.name == "sempointer__hops-2__m-adaptive__ptype-salient__selector-fused"
    colon = parse_system_specs("sempointer:k=4;m=2")[0]
    assert colon.kwargs == {"k": 4, "m": 2}
    with pytest.raises(ValueError):
        parse_system_specs("sempointer,sempointer")  # duplicate names
    with pytest.raises(ValueError):
        parse_system_specs("nonsense")


def test_none_pointer_type_remains_a_named_mode():
    spec = parse_system_specs("sempointer[ptype=none]")[0]
    assert spec.name == "sempointer__ptype-none"
    assert spec.kwargs["ptype"] == "none"


def test_resume_counts_distinct_outstanding_errors():
    from bench_metrics import aggregate

    rows = [
        {"system": "full", "example_id": "a", "error": "OutOfMemoryError: first"},
        {"system": "full", "example_id": "a", "error": "OutOfMemoryError: retry"},
        {"system": "full", "example_id": "b", "error": "OutOfMemoryError: first"},
        {"system": "full", "example_id": "b", "score": 1.0, "metric": "exact_match"},
    ]
    result = aggregate(rows, ["full"])["systems"]["full"]
    assert result["n_scored"] == 1
    assert result["n_errors"] == 1


def test_unknown_kwargs_fail_loudly():
    class Fake:
        def __init__(self, tokenizer, k=8, m=1, device="cuda"):
            pass

    validate_kwargs(Fake, {"k": 4})
    with pytest.raises(ValueError, match="unknown Fake kwarg"):
        validate_kwargs(Fake, {"selectr": "fused"}, "sempointer[selectr=fused]")
    # run-level defaults not accepted by the constructor are dropped, spec wins
    kw = merge_kwargs(Fake, {"k": 8, "block_size": 512, "device": "cpu"}, {"k": 4})
    assert kw == {"k": 4, "device": "cpu"}


def test_real_pipeline_rejects_unknown_kwarg():
    from sempointer.pipeline import SemPointerPipeline

    with pytest.raises(ValueError):
        validate_kwargs(SemPointerPipeline, {"definitely_not_a_kwarg": 1})


# ------------------------------------------------------------------ selection
def test_select_ranked_budget_and_top_m():
    from baselines._text_common import select_ranked

    counts = [100, 200, 300, 50]
    ranked = [2, 0, 3, 1]
    assert select_ranked(ranked, counts, m=2) == [2, 0]
    assert select_ranked(ranked, counts, m=10) == ranked
    assert select_ranked(ranked, counts, token_budget=450) == [2, 0, 3]  # 300+100+50
    assert select_ranked(ranked, counts, token_budget=399) == [2]         # 300; +100 > 399
    assert select_ranked(ranked, counts, token_budget=10) == [2]          # at least one
    with pytest.raises(ValueError):
        select_ranked(ranked, counts, token_budget=0)


@pytest.fixture(scope="module")
def gpt2_tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


CONTEXT = " ".join(
    [f"Paragraph {i} discusses generic filler topic number {i} at length. " * 3 for i in range(6)]
    + ["The vault code of Captain Thistlewick is 7294031. " * 2]
)


def test_bm25_budget_mode_respects_budget(gpt2_tok):
    from baselines.bm25_baseline import BM25Baseline

    b = BM25Baseline(tokenizer=gpt2_tok, m=None, token_budget=80, block_size=32)
    b.ingest(CONTEXT)
    ids, _ = b.retrieve("What is the vault code of Captain Thistlewick?")
    counts = b.ingestion["block_token_counts"]
    assert len(ids) >= 1
    assert sum(counts[i] for i in ids) <= 80 or len(ids) == 1
    full_rank, _ = b.rank_all("What is the vault code of Captain Thistlewick?")
    assert ids == full_rank[: len(ids)]  # budget mode = rank prefix
    assert "Thistlewick" in b.blocks[ids[0]]
    top2 = BM25Baseline(tokenizer=gpt2_tok, m=2, block_size=32)
    top2.ingest(CONTEXT)
    assert top2.retrieve("vault code Thistlewick")[0] == full_rank[:2]
    with pytest.raises(ValueError):
        BM25Baseline(tokenizer=gpt2_tok, m=None)


# ------------------------------------------------------------------ hybrid RRF
def test_rrf_fuse_ordering():
    from baselines._text_common import rrf_fuse

    fused = rrf_fuse([[0, 1, 2], [1, 2, 0]], k=60)
    # 1: 1/62+1/61 ; 0: 1/61+1/63 ; 2: 1/63+1/62
    assert [d for d, _ in fused] == [1, 0, 2]
    # ties broken by lower index
    assert [d for d, _ in rrf_fuse([[0, 1], [1, 0]], k=60)] == [0, 1]
    # weights shift the order
    assert [d for d, _ in rrf_fuse([[0, 1], [1, 0]], k=1, weights=[1.0, 3.0])][0] == 1


class _FakeDense:
    """Dense index stub: fixed ranking (reverse block order)."""

    def fit(self, blocks):
        self.n = len(blocks)

    def rank_all(self, query):
        order = list(range(self.n))[::-1]
        return order, [float(-i) for i in range(self.n)]


def test_hybrid_baseline_matches_manual_rrf(gpt2_tok):
    from baselines._text_common import rrf_fuse
    from baselines.hybrid_rrf import HybridRRFBaseline

    h = HybridRRFBaseline(tokenizer=gpt2_tok, m=3, block_size=32, dense_index=_FakeDense())
    h.ingest(CONTEXT)
    q = "vault code Thistlewick"
    expected = [d for d, _ in rrf_fuse([h.bm25_rank(q), h.dense.rank_all(q)[0]], k=60)]
    ids, scores = h.rank_all(q)
    assert ids == expected and scores == sorted(scores, reverse=True)
    assert h.retrieve(q)[0] == expected[:3]


# ------------------------------------------------------------------ metrics
def _row(system, eid, score, tok, task="niah", clen=4096, **kw):
    return {"system": system, "example_id": eid, "score": score, "active_tokens": tok,
            "task": task, "context_length": clen, "metric": "ruler_containment",
            "latency_ms": 10.0, "peak_vram_mb": None, **kw}


def test_aggregate_paired_and_breakdown():
    from bench_metrics import aggregate

    rows = []
    for i in range(6):
        clen = 4096 if i < 3 else 8192
        rows.append(_row("full", f"e{i}", 1.0, 4000, clen=clen))
        rows.append(_row("rag_m1", f"e{i}", float(i % 2), 600, clen=clen))
        rows.append(_row("rag_m4", f"e{i}", 1.0 if i < 5 else 0.0, 2100, clen=clen))
        rows.append(_row("sempointer", f"e{i}", 1.0 if i != 2 else 0.0, 700, clen=clen,
                         selection_latency_ms=5.0, n_hops=2))
    # old-style row without optional fields/context_length + an error then retried
    rows.append({"system": "sempointer", "example_id": "e6", "error": "boom"})
    rows.append({"system": "sempointer", "example_id": "e6", "score": 1.0, "active_tokens": 700,
                 "task": "niah", "metric": "ruler_containment", "latency_ms": 9.0})
    rows.append({"system": "rag_m1", "example_id": "e7", "error": "never fixed"})

    agg = aggregate(rows, ["full", "rag_m1", "rag_m4", "sempointer"], seed=0)
    s = agg["systems"]["sempointer"]
    assert s["n_scored"] == 7 and s["n_errors"] == 0
    assert s["score_mean"] == pytest.approx(6 / 7)
    assert s["selection_latency_ms_mean"] == 5.0 and s["n_hops_mean"] == 2.0
    assert set(s["per_task_context_length"]) == {"niah@4096", "niah@8192"}
    assert s["per_task_context_length"]["niah@4096"]["score_mean"] == pytest.approx(2 / 3)
    assert agg["systems"]["rag_m1"]["n_errors"] == 1

    p = agg["paired"]["sempointer"]
    assert p["matched_retrieval"] == "rag_m1"  # 600 tokens closest to 700
    vs_full = p["vs_full"]
    assert vs_full["n_paired"] == 6  # e6 absent in full
    assert vs_full["mean_diff"] == pytest.approx(-1 / 6)
    assert vs_full["losses"] == 1 and vs_full["ties"] == 5
    lo, hi = vs_full["diff_ci95"]
    assert lo <= vs_full["mean_diff"] <= hi
    assert "vs_rag_m4" in p and "vs_sempointer" not in p


def test_aggregate_empty_system():
    from bench_metrics import aggregate

    agg = aggregate([{"system": "full", "example_id": "a", "error": "x"}], ["full"])
    assert agg["systems"]["full"]["n_scored"] == 0 and agg["paired"] == {}
