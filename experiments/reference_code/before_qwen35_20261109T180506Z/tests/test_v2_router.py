"""CPU-only checks for the experimental V2 router."""

from sempointer.v2_pipeline import SemPointerV2Pipeline
from system_specs import parse_system_specs
from run_benchmark import system_class
from types import SimpleNamespace


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def make_router(semantic_rescue=False, **kwargs):
    router = SemPointerV2Pipeline(
        tokenizer=Tokenizer(), ptype="none", device="cpu", m=2, m_max=4,
        semantic_rescue=semantic_rescue, **kwargs,
    )
    router.blocks = [
        "mars orbital launch 2020", "mars rover landed 2021",
        "venus atmospheric probe", "saturn rings telescope",
        "jupiter storm discovered",
    ]
    router.selector.fit(router.blocks)
    return router


def test_v2_spec_is_distinct_from_v1():
    specs = parse_system_specs("sempointer,sempointer_v2,sempointer_v2[semantic_rescue=false]")
    assert [s.name for s in specs] == [
        "sempointer", "sempointer_v2", "sempointer_v2__semantic_rescue-False"]
    assert system_class("sempointer_v2") is SemPointerV2Pipeline


def test_lexical_route_stays_small_when_query_is_covered():
    router = make_router()
    result = router.select("mars rover")
    assert result["router_route"] == "lexical"
    assert len(result["ids"]) == 2
    assert result["router_coverage"] == 1


def test_low_coverage_expands_and_protects_lexical_blocks():
    router = make_router()
    result = router.select("venus saturn jupiter")
    lexical = [i for i, _ in router.selector.rank("venus saturn jupiter")]
    assert result["router_route"] == "expanded_lexical"
    assert result["ids"][:2] == lexical[:2]
    assert result["ids"] == lexical[:4]


def test_semantic_rescue_adds_candidate_after_protected_blocks():
    router = make_router(semantic_rescue=True)
    router.dense_selector.fit = lambda blocks: None
    router.dense_selector.rank = lambda query: [(4, 1.0), (3, 0.5)]
    result = router.select("mars absent unknown")
    lexical = [i for i, _ in router.selector.rank("mars absent unknown")]
    assert result["ids"][:2] == lexical[:2]
    assert len(result["ids"]) == 3
    assert result["router_route"] == "semantic_rescue"


def test_dense_index_is_lazy_when_rescue_not_needed():
    router = make_router(semantic_rescue=True)
    router.dense_selector.fit = lambda blocks: (_ for _ in ()).throw(AssertionError("dense fit"))
    router.select("mars rover")


def test_repeated_query_terms_do_not_force_unnecessary_expansion():
    router = make_router()
    router.blocks = ["mars rover a", "mars rover b", "mars rover c", "venus", "saturn"]
    router.selector.fit(router.blocks)
    result = router.select("mars rover")
    assert result["router_coverage"] == 1
    assert result["router_route"] == "lexical"
    assert len(result["ids"]) == 2


def test_token_cap_keeps_first_block_and_skips_excess():
    router = make_router(max_resolved_tokens=1)
    result = router.select("mars rover missingword")
    assert len(result["ids"]) == 1
    assert result["router_resolved_tokens"] > 1


def test_answer_records_router_route_without_extra_selection():
    router = make_router()
    router.pointer_ids = [[] for _ in router.blocks]
    llm = SimpleNamespace(generate=lambda prompt: SimpleNamespace(
        text="2021", prompt=prompt, prompt_tokens=len(prompt.split()),
        latency_ms=1, ttft_ms=0.5, completion_tokens=1, peak_vram_mb=0,
        stop_reason="eos"))
    out = router.answer("mars rover", llm)
    assert out["answer"] == "2021"
    assert out["router_route"] == "lexical"
    assert out["n_selected"] == 2
