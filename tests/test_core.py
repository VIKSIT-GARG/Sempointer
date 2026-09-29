"""Unit tests for the repaired SemPointer core (CPU-only, no model download)."""

import pytest
import torch

from sempointer.pointer_generator import SemanticIDPointerGenerator


TEXT_A = "The Eiffel Tower is in Paris, France. It was built in 1889 for the World's Fair."
TEXT_B = "Photosynthesis converts light energy into chemical energy inside chloroplasts."


@pytest.fixture(scope="module")
def gen8():
    return SemanticIDPointerGenerator(k=8, device="cpu")


@pytest.fixture(scope="module")
def gen16():
    return SemanticIDPointerGenerator(k=16, device="cpu")


def test_pointer_length_matches_k(gen8, gen16):
    assert len(gen8.generate_token_ids(TEXT_A)) == 8
    assert len(gen16.generate_token_ids(TEXT_A)) == 16


def test_k_changes_the_address_but_prefix_is_stable(gen8, gen16):
    """k must genuinely matter: k=16 extends (not replaces) the k=8 address,
    because both come from the same seeded projections."""
    ids8 = gen8.generate_token_ids(TEXT_A)
    ids16 = gen16.generate_token_ids(TEXT_A)
    assert ids16[:8] == ids8


def test_seed_stability():
    a = SemanticIDPointerGenerator(k=8, seed=123, device="cpu").generate_token_ids(TEXT_A)
    b = SemanticIDPointerGenerator(k=8, seed=123, device="cpu").generate_token_ids(TEXT_A)
    c = SemanticIDPointerGenerator(k=8, seed=124, device="cpu").generate_token_ids(TEXT_A)
    assert a == b
    assert a != c


def test_different_texts_get_different_addresses(gen8):
    assert gen8.generate_token_ids(TEXT_A) != gen8.generate_token_ids(TEXT_B)


def test_token_ids_inside_reserved_range(gen8):
    for tid in gen8.generate_token_ids(TEXT_A):
        assert gen8.id_offset <= tid < gen8.id_offset + gen8.pointer_vocab_size


def test_dense_pointer_not_repeated(gen8):
    """Regression guard against the legacy expand(k, d) trick: the dense
    pointer must be a single (1, d) embedding, and k must not change it."""
    p = gen8.generate(TEXT_A)
    assert p.dim() == 2 and p.shape[0] == 1
    g2 = SemanticIDPointerGenerator(k=2, device="cpu")
    assert torch.allclose(gen8.generate(TEXT_A), g2.generate(TEXT_A))


def test_batch_consistency(gen8):
    ids_single = gen8.generate_token_ids(TEXT_A)
    ids_batch = gen8.generate_token_ids_batch([TEXT_A, TEXT_B])[0]
    assert ids_single == ids_batch


# ----------------------------------------------------------------- #
# chunking + active-token accounting
# ----------------------------------------------------------------- #
def test_chunk_text_respects_block_size():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    from sempointer.pipeline import chunk_text

    text = "word " * 1000
    chunks = chunk_text(text, block_size=128, tokenizer=tok)
    assert all(len(tok.encode(c, add_special_tokens=False)) <= 128 for c in chunks)
    assert len(chunks) >= 1000 * 1 // 128 - 2


def test_active_tokens_scale_with_theory():
    """Honest validation of L_active = (N-m)k + m*B_res + L_Q on the REAL
    blocks/pointers the pipeline produced (chunking is by tokens, so the
    number of blocks is read back, not assumed):
    - every pointer is at most k tokens
    - measured prompt = pointer tokens + resolved tokens + L_Q + framing,
      with framing non-negative and bounded per unselected pointer
    - active length grows with N at a slope bounded by k + label framing
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    from sempointer.pipeline import SemPointerPipeline

    k, q = 4, "What is in block 3?"
    enc = lambda t: tok.encode(t, add_special_tokens=False)

    def measure(n_sent: int):
        p = SemPointerPipeline(tokenizer=tok, k=k, m=1, block_size=32, device="cpu",
                               selector="hybrid", ptype="keyword")
        sents = [f"Record {i} describes the {w} harbour of Velmora{i} built in {1800 + i}."
                 for i, w in zip(range(n_sent), ["northern", "eastern", "old", "quiet",
                                                  "busy", "small", "grand", "hidden",
                                                  "stone", "wooden", "misty", "sunny"] * 4)]
        p.ingest(" ".join(sents))
        sel = p.select(q)["ids"]
        prompt = p.active_prompt(q)
        N = len(p.blocks)
        ptr_tok = sum(len(p.pointer_ids[i]) for i in range(N) if i not in sel)
        res_tok = sum(len(enc(p.blocks[i])) for i in sel)
        core = ptr_tok + res_tok + len(enc(q))
        return len(enc(prompt)), core, N, sel, p

    n_a, core_a, N_a, sel_a, _ = measure(8)
    n_b, core_b, N_b, sel_b, p_b = measure(24)
    assert N_b > N_a >= 2

    # pointers are compact: at most k tokens each, and some carry content
    assert all(len(ptr) <= k for ptr in p_b.pointer_ids)
    assert any(len(ptr) > 0 for ptr in p_b.pointer_ids)

    # framing (labels, headers, "Question:") is non-negative and bounded
    per_ptr_label = 8  # "[P12] " + newline
    for n, core, N, sel in [(n_a, core_a, N_a, sel_a), (n_b, core_b, N_b, sel_b)]:
        framing = n - core
        assert framing >= 0, f"measured {n} < core {core}"
        assert framing <= per_ptr_label * (N - len(sel)) + 80, f"framing {framing} too large"

    # growth per extra unselected block is bounded by k + label framing
    slope = (n_b - n_a) / (N_b - N_a)
    assert 0 < slope <= k + per_ptr_label + 32 / max(1, N_b - N_a), f"slope {slope}"


def test_registry_roundtrip(gen8):
    from sempointer.registry import PointerRegistry

    r = PointerRegistry(k=8, B=512, max_size=10)
    mid = r.register(TEXT_A, gen8, metadata={"x": 1})
    assert r.get_text(mid) == TEXT_A
    assert len(r.get_pointer_token_ids(mid)) == 8
    rec = r.get_record(mid)
    assert rec.metadata["x"] == 1
    with pytest.raises(RuntimeError):
        small = PointerRegistry(k=8, B=512, max_size=1)
        small.register(TEXT_A, gen8)
        small.register(TEXT_B, gen8)


def test_scorer_top_m_ordering(gen8):
    from sempointer.registry import PointerRegistry
    from sempointer.scorer import PointerScorer

    r = PointerRegistry(k=8, B=512, max_size=10)
    mids = [r.register(t, gen8) for t in [TEXT_A, TEXT_B]]
    s = PointerScorer(gen8, device="cpu")
    ids, scores = s.score("Eiffel Tower in Paris", r, m=1)
    assert ids[0] == mids[0]
    ids2, scores2 = s.score_all("chloroplasts light energy", r)
    assert ids2[0] == mids[1]
