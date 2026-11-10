"""Prompt-budget contracts; CPU fixtures are not empirical evidence."""
from copy import deepcopy
from types import SimpleNamespace

from tests.test_all_datasets import Tokenizer


def fixture_memory():
    tok = Tokenizer()
    blocks = [f"evidence{i} fact " * 40 for i in range(30)]
    router = SimpleNamespace(blocks=blocks, pointer_ids=[[f"descriptor{i}"] for i in range(30)],
        system_prompt="Use the evidence", selector=SimpleNamespace(rank=lambda q: [(i,30-i) for i in range(30)]),
        select=lambda q: {"ids":[0,1], "selection_latency_ms":1})
    return tok, SimpleNamespace(router=router, blocks=blocks)


def test_bounded_variants_count_chat_and_keep_all_prompts_under_budget():
    from evaluation.bounded import budgeted_variants
    tok, prepared = fixture_memory()
    before = deepcopy(prepared.blocks)
    variants = budgeted_variants(prepared, "where is evidence?", tok, budget=100,
        pointer_caps=(16,32), prompt_formatter=lambda p:"CHAT START "+p+" CHAT END")
    for name, row in variants.items():
        assert row["prompt_tokens"] == len(tok.encode(row["prompt"]))
        assert row["prompt"].startswith("CHAT START")
        if name != "original_sempointer":
            assert row["prompt_tokens"] <= 100
    for cap in [16,32]:
        row = variants[f"bounded_{cap}"]
        assert 0 < row["pointer_tokens"] <= cap
        assert set(row["pointer_ids"]).isdisjoint(row["selected_ids"])
        assert set(row["pointer_ids"]) <= set(range(2,16))
        assert row["partial_block_id"] is not None
    assert prepared.blocks == before


def test_zero_pointer_allowance_matches_adaptive_payload_control():
    from evaluation.bounded import budgeted_variants
    tok, prepared = fixture_memory()
    rows = budgeted_variants(prepared, "where?", tok, budget=100, pointer_caps=(0,))
    assert rows["bounded_0"]["prompt"] == rows["adaptive_no_index"]["prompt"]
    assert rows["bounded_0"]["pointer_tokens"] == 0


def test_question_overflow_is_explicit_and_original_index_is_not_truncated():
    from evaluation.bounded import budgeted_variants
    tok, prepared = fixture_memory()
    rows = budgeted_variants(prepared, "question " * 40, tok, budget=20, pointer_caps=(8,))
    assert all(r["skip_reason"] == "question_and_framing_exceed_budget" for n,r in rows.items() if n != "original_sempointer")
    original = rows["original_sempointer"]
    assert "[P29]" in original["prompt"]
    assert original["skip_reason"] == "input_budget_exceeded"
