"""CPU tests for sempointer.pointers / lexical / selector / attention_scorer."""

import math

import numpy as np
import pytest
import torch

from sempointer.lexical import BM25Index, tokenize
from sempointer.pointers import make_pointer_generator
from sempointer.selector import BlockSelector

QWEN = "Qwen/Qwen2.5-1.5B-Instruct"

BLOCKS = [
    "The quarterly budget review for the marketing department was held on Monday. "
    "Spending on online advertising rose while print costs fell.",
    "Captain Velmora Thistlewick guarded the vault. The special magic number is 7294031. "
    "Remember it for the journey.",
    "Photosynthesis converts light energy into chemical energy inside chloroplasts of plant cells.",
    "The Eiffel Tower in Paris was completed in 1889 for the World's Fair.",
    "Volcanic eruptions release ash and sulfur dioxide into the stratosphere, cooling the climate.",
    "The chess tournament in Reykjavik ended with a dramatic endgame and a surprise champion.",
]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(QWEN)


# ------------------------------------------------------------------ pointers
def test_keyword_pointers_len_readable_distinctive(tok):
    k = 8
    gen = make_pointer_generator("keyword", k=k)
    ptrs = gen.pointers(BLOCKS, tok)
    assert len(ptrs) == len(BLOCKS)
    for p in ptrs:
        assert 0 < len(p) <= k
    text = tok.decode(ptrs[1])
    assert "7294031" in text  # the magic number is the top term (Qwen spends 8 tokens on it)
    text16 = tok.decode(make_pointer_generator("keyword", k=16).pointers(BLOCKS, tok)[1])
    assert "7294031" in text16
    assert "Velmora" in text16 and "Thistlewick" in text16
    # readable: decodes to real words, no replacement chars
    assert "�" not in text
    assert tok.decode(ptrs[3]).strip()
    assert len({tuple(p) for p in ptrs}) == len(ptrs)


def test_keyword_pointers_deterministic(tok):
    a = make_pointer_generator("keyword", k=12).pointers(BLOCKS, tok)
    b = make_pointer_generator("keyword", k=12).pointers(BLOCKS, tok)
    assert a == b


def test_keyword_prefers_block_distinctive_terms(tok):
    blocks = ["apple banana common", "apple cherry common", "apple durian common"]
    gen = make_pointer_generator("keyword", k=2)
    terms = gen.block_terms(blocks)
    assert terms[1][0][0] == "cherry"


def test_lead_and_none(tok):
    lead = make_pointer_generator("lead", k=5).pointers(BLOCKS, tok)
    for b, p in zip(BLOCKS, lead):
        assert p == tok.encode(b, add_special_tokens=False)[:5]
    none = make_pointer_generator("none", k=8)
    assert none.k == 0
    assert none.pointers(BLOCKS, tok) == [[] for _ in BLOCKS]
    assert make_pointer_generator("keyword", k=0).pointers(BLOCKS, tok) == [[] for _ in BLOCKS]


def test_random_id(tok):
    gen = make_pointer_generator("random_id", k=8, seed=42, device="cpu")
    p = gen.pointers(BLOCKS, tok)
    assert all(len(x) == 8 for x in p)
    assert all(1000 <= t < 1256 for x in p for t in x)
    assert p == make_pointer_generator("random_id", k=8, seed=42, device="cpu").pointers(BLOCKS, tok)
    assert gen.info()["pointer_kind"] == "random_id"


def test_unknown_kind():
    with pytest.raises(ValueError):
        make_pointer_generator("bogus", k=4)


# ------------------------------------------------------------------ bm25
def test_tokenize_drops_stopwords():
    assert tokenize("The cat is on the Mat") == ["cat", "mat"]


def test_bm25_rare_term_first():
    idx = BM25Index(BLOCKS)
    s = idx.scores("What is the magic number Thistlewick guarded?")
    assert s.shape == (len(BLOCKS),)
    assert int(np.argmax(s)) == 1
    assert idx.scores("zzzunknown").sum() == 0.0


def test_bm25_matches_rank_bm25_without_stopwords():
    rb = pytest.importorskip("rank_bm25")
    corpus = [tokenize(b) for b in BLOCKS]
    ref = rb.BM25Okapi(corpus).get_scores(tokenize("energy light Paris tower"))
    ours = BM25Index(BLOCKS).scores("energy light Paris tower")
    np.testing.assert_allclose(ours, ref, rtol=1e-9)


# ------------------------------------------------------------------ selector
@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_selector_top1(mode):
    sel = BlockSelector(mode=mode, device="cpu")
    sel.fit(BLOCKS)
    assert not sel.needs_attention
    assert sel.select("Which city hosted the chess tournament in Reykjavik?", m=1) == [5]
    assert sel.select("How do plants turn light into chemical energy (photosynthesis)?", m=1) == [2]
    ranked = sel.rank("volcanic ash stratosphere")
    assert sorted(i for i, _ in ranked) == list(range(len(BLOCKS)))
    assert ranked[0][0] == 4


def test_selector_attention_modes_and_embedder_reuse():
    sel = BlockSelector(mode="hybrid+attention", device="cpu")
    assert sel.needs_attention
    sel.fit(BLOCKS)
    emb = sel.embedder
    att = [0.0] * len(BLOCKS)
    att[1] = 1.0
    top = sel.select("magic number Thistlewick", m=1, attention_scores=att)
    assert top == [1]
    with pytest.raises(ValueError):
        sel.rank("magic number")  # attention required
    sel.fit(BLOCKS[:3])  # re-fit must not reload the embedder
    assert sel.embedder is emb
    assert BlockSelector(mode="dense", device="cpu").embedder is emb  # module-level cache

    only = BlockSelector(mode="attention", device="cpu")
    only.fit(BLOCKS)
    order = [i for i, _ in only.rank("q", attention_scores=[0.1, 0.5, 0.3, 0.9, 0.0, 0.2])]
    assert order == [3, 1, 2, 5, 0, 4]


def test_selector_adaptive_bounds():
    sel = BlockSelector(mode="bm25", device="cpu")
    sel.fit(BLOCKS)
    q = "magic number Thistlewick"
    for ratio in (0.0, 0.5, 0.99, 1.0):
        s = sel.select(q, m=3, adaptive=True, adaptive_ratio=ratio)
        assert 1 <= len(s) <= 3
        assert s[0] == 1
    assert len(sel.select(q, m=3, adaptive=True, adaptive_ratio=0.0)) == 3
    assert len(sel.select(q, m=2, adaptive=True, adaptive_ratio=0.0, m_max=5)) == 5
    assert sel.select(q, m=3, adaptive=True, adaptive_ratio=1.0) == [1]
    assert len(sel.select(q, m=100)) == len(BLOCKS)


# ------------------------------------------------------------------ attention scorer
def test_build_index_ids_spans(tok):
    from sempointer.attention_scorer import build_index_ids

    ptr = make_pointer_generator("keyword", k=6).pointers(BLOCKS, tok)
    entries = [(f"[P{i}]", p) for i, p in enumerate(ptr)]
    ids, spans, qspan = build_index_ids(tok, "<|im_start|>user\nIndex:\n", entries, "Question: x?", "<|im_end|>\n")
    flat = ids[0].tolist()
    assert ids.shape[0] == 1
    for (s, e), p in zip(spans, ptr):
        assert flat[s:e] == p
    assert qspan[0] >= spans[-1][1]
    assert tok.decode(flat[qspan[0] : qspan[1]]) == "Question: x?"
    assert flat[0] == tok.convert_tokens_to_ids("<|im_start|>")


@pytest.mark.slow
def test_pointer_attention_scores_qwen_cpu(tok):
    from transformers import AutoModelForCausalLM

    from sempointer.attention_scorer import build_index_ids, pointer_attention_scores

    model = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32)
    model.eval()
    orig = model.config._attn_implementation
    # k=16 so the query's key word (Thistlewick) is inside exactly one pointer;
    # at k=8 block 1's pointer is only the 8-token number.
    ptr = make_pointer_generator("keyword", k=16).pointers(BLOCKS, tok)
    assert sum("Thistlewick" in tok.decode(p) for p in ptr) == 1
    entries = [(f"[P{i}]", p) for i, p in enumerate(ptr)]
    ids, spans, qspan = build_index_ids(
        tok,
        "<|im_start|>user\nMemory index (one line per block):\n",
        entries,
        "Question: What is the magic number that Captain Thistlewick guarded?",
        "<|im_end|>\n<|im_start|>assistant\n",
    )
    scores = pointer_attention_scores(model, ids, spans, qspan)
    assert len(scores) == len(BLOCKS)
    assert all(isinstance(s, float) and math.isfinite(s) and s >= 0 for s in scores)
    assert model.config._attn_implementation == orig
    ranking = sorted(range(len(scores)), key=lambda i: -scores[i])
    print(f"\n[attention] attn impl restored={orig}; scores={[round(s, 4) for s in scores]}; ranking={ranking}; "
          f"matching pointer (1) rank={ranking.index(1) + 1}")
    assert 1 in ranking[:3]

    # also exercise alternative options and restoration on exception
    s2 = pointer_attention_scores(model, ids, spans, qspan, layers=[-1, -2], agg="max", length_norm=True)
    assert len(s2) == len(BLOCKS) and all(math.isfinite(s) for s in s2)
    with pytest.raises(ValueError):
        pointer_attention_scores(model, ids, [(qspan[1] - 1, qspan[1])], qspan)
    bad_ids = torch.full_like(ids, 10**9)  # out-of-vocab -> forward raises
    with pytest.raises(Exception):
        pointer_attention_scores(model, bad_ids, spans, qspan)
    assert model.config._attn_implementation == orig
