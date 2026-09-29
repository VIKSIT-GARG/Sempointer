"""CPU-only tests for AddressScorer + AddressRAGBaseline (MECH-1).

Run:  experiments/warden/bin/python experiments/test_address_cpu.py
  (or from experiments/: warden/bin/python test_address_cpu.py)
  pytest: warden/bin/python -m pytest test_address_cpu.py -v  (from experiments/)

Constraints: MiniLM on CPU only, no LLM/engine import, no CUDA calls.
Failures are loud (assert + traceback); nothing is skipped silently.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from sempointer.address_scorer import AddressScorer, address_score
from sempointer.pointer_generator import SemanticIDPointerGenerator
from sempointer.registry import MemoryRecord, PointerRegistry
from sempointer.scorer import PointerScorer

DEVICE = "cpu"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

# Toy corpus: 8 topically spread blocks so address space is exercised.
CORPUS = [
    "The Eiffel Tower stands in Paris and was completed in 1889 for the World's Fair.",
    "Photosynthesis converts light energy into chemical energy inside chloroplasts.",
    "The Python programming language was created by Guido van Rossum in 1991.",
    "Black holes have an event horizon beyond which nothing can escape, not even light.",
    "The Great Wall of China stretches over thirteen thousand miles across northern China.",
    "Mitochondria generate most of the cell's supply of adenosine triphosphate.",
    "Julius Caesar crossed the Rubicon in 49 BC, starting a Roman civil war.",
    "Quantum entanglement links particles so measuring one instantly affects the other.",
]
QUERY = "What Paris landmark was built for the 1889 World's Fair?"


def make_registry(k: int, seed: int = 42) -> tuple:
    gen = SemanticIDPointerGenerator(
        embedder_id=EMBEDDER, k=k, seed=seed, device=DEVICE
    )
    reg = PointerRegistry(k=k, B=512, max_size=64)
    for blk in CORPUS:
        reg.register(blk, gen)
    return gen, reg


def test_k_sensitivity():
    """k MUST change rankings: k=2 vs k=8 must order the toy corpus differently.

    Structural reason: the score reads all k positions (prefix run can extend
    past position 2; Hamming denominator is k), so lengthening the address
    from V_p^2 to V_p^8 changes both addresses and score magnitudes.
    """
    _, reg2 = make_registry(k=2)
    _, reg8 = make_registry(k=8)
    gen2 = SemanticIDPointerGenerator(embedder_id=EMBEDDER, k=2, seed=42, device=DEVICE)
    gen8 = SemanticIDPointerGenerator(embedder_id=EMBEDDER, k=8, seed=42, device=DEVICE)
    ids2, _ = AddressScorer(gen2).score_all(QUERY, reg2)
    ids8, _ = AddressScorer(gen8).score_all(QUERY, reg8)
    assert ids2 != ids8, (
        f"k is structurally inert?! k=2 ranking {ids2} == k=8 ranking {ids8}. "
        "AddressScorer fails its core demand."
    )
    print(f"  k-sensitivity OK: k=2 {ids2} vs k=8 {ids8}")


def test_determinism_seed_stability():
    gen_a, reg_a = make_registry(k=8, seed=42)
    gen_b, reg_b = make_registry(k=8, seed=42)
    ids_a, scores_a = AddressScorer(gen_a).score_all(QUERY, reg_a)
    ids_b, scores_b = AddressScorer(gen_b).score_all(QUERY, reg_b)
    assert ids_a == ids_b, f"non-deterministic ranking: {ids_a} vs {ids_b}"
    assert torch.equal(scores_a, scores_b), "non-deterministic scores"
    # Same seed + same text => same address, across independent instances.
    assert gen_a.generate_token_ids(CORPUS[0]) == gen_b.generate_token_ids(CORPUS[0])
    print(f"  determinism OK: ranking {ids_a}")


def test_interface_parity_with_pointer_scorer():
    """AddressScorer.score mirrors PointerScorer.score: (List[int], Tensor),
    descending scores, exactly min(m, N) entries."""
    gen, reg = make_registry(k=8)
    addr = AddressScorer(gen)
    dense = PointerScorer(gen, device=DEVICE)
    for m in (1, 3, len(CORPUS), len(CORPUS) + 5):
        a_ids, a_scores = addr.score(QUERY, reg, m)
        d_ids, d_scores = dense.score(QUERY, reg, m)
        assert isinstance(a_ids, list) and isinstance(a_scores, torch.Tensor)
        assert isinstance(d_ids, list) and isinstance(d_scores, torch.Tensor)
        assert len(a_ids) == len(a_scores) == min(m, len(CORPUS))
        assert len(d_ids) == len(d_scores) == min(m, len(CORPUS))
        assert all(a_scores[i] >= a_scores[i + 1] for i in range(len(a_scores) - 1)), (
            f"address scores not descending for m={m}: {a_scores.tolist()}"
        )
    print("  interface parity OK (types, lengths, descending order for m=1,3,N,N+5)")


def test_nonempty_selection():
    gen, reg = make_registry(k=8)
    ids, scores = AddressScorer(gen).score(QUERY, reg, m=1)
    assert len(ids) == 1 and len(scores) == 1, "empty selection — retrieval returned nothing"
    assert ids[0] in reg.records, f"selected id {ids[0]} not in registry"
    print(f"  nonempty selection OK: id={ids[0]} score={float(scores[0]):.3f}")


def test_address_only_ignores_dense():
    """Proof of address-only selection: destroying every dense embedding must
    leave the ranking bit-identical (the scorer never reads them)."""
    gen, reg = make_registry(k=8)
    scorer = AddressScorer(gen)
    before_ids, before_scores = scorer.score_all(QUERY, reg)
    for rec in reg.records.values():
        rec.pointer_embedding = torch.zeros_like(rec.pointer_embedding)
    after_ids, after_scores = scorer.score_all(QUERY, reg)
    assert before_ids == after_ids, "ranking changed after zeroing dense embeddings — dense leak!"
    assert torch.equal(before_scores, after_scores), "scores changed after zeroing dense embeddings"
    print("  address-only proof OK (rankings identical with zeroed dense embeddings)")


class _StubGenerator:
    """Canned discrete addresses to pin ordering semantics deterministically."""

    def __init__(self, query_ids):
        self.query_ids = query_ids
        self.k = len(query_ids)

    def generate_token_ids(self, text, tokenizer=None):
        assert text == "__query__", f"stub saw unexpected text {text!r}"
        return list(self.query_ids)


def _stub_registry(block_addresses):
    reg = PointerRegistry(k=len(block_addresses[0]), B=512, max_size=16)
    for i, bids in enumerate(block_addresses):
        reg.records[i] = MemoryRecord(
            memory_id=i,
            text=f"block {i}",
            token_count=2,
            pointer_embedding=torch.zeros(1, 4),
            pointer_token_ids=list(bids),
            inserted_at=i,
        )
    reg._next_id = len(block_addresses)
    return reg


def test_prefix_beats_hamming_and_tiebreak_by_index():
    """Prefix run is the primary key: 2-prefix/2-match beats 1-prefix/3-match.
    Identical addresses tie-break by ascending block index."""
    gen = _StubGenerator([10, 11, 12, 13])
    reg = _stub_registry(
        [
            [10, 99, 12, 13],  # block 0: prefix 1, matches 3 -> 1.75
            [10, 11, 99, 99],  # block 1: prefix 2, matches 2 -> 2.50 (must win)
            [10, 11, 99, 99],  # block 2: identical to block 1 -> loses on index tie-break
            [99, 99, 99, 99],  # block 3: prefix 0, matches 0 -> 0.00
        ]
    )
    assert address_score([10, 11, 12, 13], [10, 11, 99, 99]) == 2.0 + 2.0 / 4.0
    ids, scores = AddressScorer(gen).score_all("__query__", reg)
    assert ids == [1, 2, 0, 3], f"ordering wrong: {ids} scores={scores.tolist()}"
    assert scores[0] > scores[2], "prefix-primary violated: 1-prefix/3-match beat 2-prefix/2-match"
    print(f"  ordering semantics OK: {ids} scores={scores.tolist()}")


def test_empty_registry_and_k_mismatch_loud():
    gen, _ = make_registry(k=8)
    scorer = AddressScorer(gen)
    ids, scores = scorer.score(QUERY, PointerRegistry(k=8, B=512), m=3)
    assert ids == [] and scores.numel() == 0, "empty registry must yield empty selection, not an error"
    # k skew between generator and stored addresses must raise, never silently truncate.
    bad = PointerRegistry(k=2, B=512, max_size=4)
    bad.records[0] = MemoryRecord(0, "x", 1, torch.zeros(1, 4), [1, 2], 0)
    bad._next_id = 1
    try:
        scorer.score_all(QUERY, bad)
    except RuntimeError as e:
        assert "k mismatch" in str(e), f"wrong error: {e}"
    else:
        raise AssertionError("k mismatch between generator and registry was SILENT — must raise")
    print("  empty-registry + k-mismatch-loud OK")


def test_address_rag_retrieve_path_cpu():
    """AddressRAGBaseline.retrieve runs on CPU without any LLM object."""
    from baselines.address_rag import AddressRAGBaseline

    class FakeTokenizer:
        def encode(self, t, add_special_tokens=False):
            return t.split()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids) if ids and isinstance(ids[0], str) else ""

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sempointer"))
    from sempointer.pipeline import chunk_text  # noqa: F401  (import-path sanity)

    tok = FakeTokenizer()
    rag = AddressRAGBaseline(tokenizer=tok, k=4, m=2, block_size=16, seed=42, device=DEVICE)
    rag.ingest(" ".join(CORPUS))
    assert len(rag.blocks) >= 2, "chunking produced fewer than 2 blocks"
    ids, scores = rag.retrieve(QUERY)
    assert len(ids) == 2 and len(scores) == 2, f"retrieve returned {len(ids)} ids, want 2"
    assert rag.info()["system"] == "address_rag" and rag.info()["k"] == 4
    print(f"  address_rag retrieve OK: ids={ids} scores={scores}")


TESTS = [
    test_k_sensitivity,
    test_determinism_seed_stability,
    test_interface_parity_with_pointer_scorer,
    test_nonempty_selection,
    test_address_only_ignores_dense,
    test_prefix_beats_hamming_and_tiebreak_by_index,
    test_empty_registry_and_k_mismatch_loud,
    test_address_rag_retrieve_path_cpu,
]


def main() -> int:
    assert DEVICE == "cpu", "these tests must run CPU-only"
    assert torch.cuda.is_available() is False or True  # informational; device pinned to CPU regardless
    failures = 0
    for fn in TESTS:
        try:
            print(f"[{fn.__name__}]")
            fn()
        except Exception as e:  # noqa: BLE001 — loud, then continue to show all failures
            failures += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed (device=cpu, no LLM).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
