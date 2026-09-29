"""Address-only selection over discrete pointer IDs (MECH-1).

Honest framing: this module tests whether the discrete pointer addresses
(V_p^k semantic IDs from SemanticIDPointerGenerator) carry retrieval signal
on their own. It is a direct attack on M1 — the finding that the current
pipeline's selection shortcut is dense cosine (PointerScorer never reads
`pointer_token_ids`). If AddressScorer retrieves competitively, pointer length
k is STRUCTURALLY load-bearing for retrieval; if it fails, we publish that
too.

Selection uses ONLY discrete token IDs, never dense embeddings:
  - The query is quantized through the generator's OWN seeded projections via
    `generate_token_ids` (i.e. the same `token_ids_from_embedding` path used
    at ingest time). No `generate`/`generate_batch` call, no embedding read.
  - Each block scores `prefix + matches / k`, where `prefix` is the exact
    leading-position match run and `matches` is the total agreement count
    over the k positions (Hamming similarity). Prefix is the primary key
    (longest shared address stem = nearest in the V_p^k trie), Hamming
    similarity breaks ties among blocks that diverge at the same depth.
  - Ties (identical composite) break deterministically by ascending block
    index (memory_id).

Why k MUST change rankings (structural, not parametric): the address of a
block is its full k-token string in V_p^k, and the composite score reads all
k positions (the prefix run can extend past any fixed bound, and the Hamming
term's denominator is k itself). Changing k therefore changes both the
addresses being compared and the score magnitudes — unlike the legacy
`embedding.expand(k, d)` trick where k was a no-op. The unit test
`test_k_sensitivity` pins this: k=2 vs k=8 must rank a toy corpus differently.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .registry import PointerRegistry


def _prefix_run(a: List[int], b: List[int]) -> int:
    """Length of the exact leading-position match run between two addresses."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def address_score(q_ids: List[int], b_ids: List[int]) -> float:
    """Composite address-overlap score: prefix run + Hamming similarity.

    Prefix dominates (Hamming term is strictly < 1 for k >= 1), so ranking is
    lexicographic on (prefix, matches) with deterministic index tie-break
    applied by the caller.
    """
    k = len(q_ids)
    if k == 0:
        return 0.0
    prefix = _prefix_run(q_ids, b_ids)
    matches = sum(1 for x, y in zip(q_ids, b_ids) if x == y)
    return float(prefix) + float(matches) / float(k)


class AddressScorer:
    """Ranks registry blocks by discrete address overlap with the query.

    Mirrors PointerScorer's interface (`score` / `score_all`) so the two are
    drop-in interchangeable in benchmarks. The generator is used ONLY for its
    discrete quantization path (`generate_token_ids`); dense pointer
    embeddings are never read — zeroing every record's `pointer_embedding`
    leaves rankings bit-identical (pinned by `test_address_only_ignores_dense`).
    """

    def __init__(self, generator):
        self.generator = generator

    def score_all(self, query: str, registry: PointerRegistry) -> Tuple[List[int], torch.Tensor]:
        if len(registry) == 0:
            return [], torch.empty(0)

        # Same quantization path as ingest: seeded projections + V_p buckets.
        q_ids = self.generator.generate_token_ids(query)
        k = len(q_ids)

        scored: List[Tuple[float, int]] = []
        for mid in sorted(registry.records.keys()):
            b_ids = registry.records[mid].pointer_token_ids
            if len(b_ids) != k:
                # Loud mismatch: silently truncating/padding would hide a
                # generator/registry k skew and make k-sensitivity claims void.
                raise RuntimeError(
                    f"AddressScorer k mismatch: query address has k={k} but "
                    f"block {mid} stored {len(b_ids)} token IDs. Re-ingest with "
                    f"the same generator k."
                )
            scored.append((address_score(q_ids, b_ids), mid))

        # Descending composite; deterministic tie-break by block index.
        scored.sort(key=lambda t: (-t[0], t[1]))
        ranked_ids = [mid for _, mid in scored]
        scores = torch.tensor([s for s, _ in scored], dtype=torch.float32)
        return ranked_ids, scores

    def score(self, query: str, registry: PointerRegistry, m: int) -> Tuple[List[int], torch.Tensor]:
        memory_ids, scores = self.score_all(query, registry)
        if m >= len(memory_ids):
            return memory_ids, scores
        return memory_ids[:m], scores[:m]

    def info(self) -> Dict:
        gen = self.generator
        return {
            "scorer": "AddressScorer",
            "retrieval": "discrete address overlap (prefix + Hamming/k), no dense embeddings",
            "k": getattr(gen, "k", None),
            "pointer_vocab_size": getattr(gen, "pointer_vocab_size", None),
            "seed": getattr(gen, "seed", None),
        }
