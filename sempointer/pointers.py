"""Pointer generators: the compact k-token pointer each block keeps in the
LLM's active sequence.

All generators emit, per block, a list of <= k token ids in the TARGET LLM's
vocabulary (the LLM tokenizer is passed to `pointers()`):

- "keyword"   (main): salient terms of the block, TF-IDF-scored with IDF over
               the blocks of THIS context (terms distinctive to a block win),
               boosted for numbers / capitalized entity-like tokens / long rare
               words; stopwords dropped; deduplicated; rendered as real,
               readable vocabulary tokens (" term" with leading space) in score
               order until k tokens (last term truncated if needed).
- "random_id" (ablation): the legacy semantic-ID pointer (seeded random
               projection of a MiniLM embedding bucketed into token ids
               id_offset + bucket). Tokens are arbitrary to the LLM.
- "lead"      (ablation): the first k LLM tokens of the block.
- "none"      (ablation): k = 0, empty pointers.

"keyword"/"lead"/"none" are pure python (no model calls) and deterministic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .lexical import STOPWORDS

# A "term" = alphanumeric run, allowing internal - . ' (e.g. "COVID-19",
# "3.14", "O'Neil"); trailing punctuation is stripped by the regex shape.
_TERM_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|[-.'](?=[A-Za-z0-9]))*")
_SENT_END_RE = re.compile(r"[.!?:\n]\s*$")


class PointerGenerator:
    """Base class. Subclasses implement `pointers`."""

    name: str = "base"
    k: int = 0

    def pointers(self, blocks: List[str], tokenizer) -> List[List[int]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def info(self) -> dict:
        return {"pointer_kind": self.name, "k": self.k}


# --------------------------------------------------------------------------- #
# keyword pointers
# --------------------------------------------------------------------------- #
def extract_terms(text: str) -> List[Tuple[str, str, int, bool]]:
    """Returns (lower, surface, char_pos, capitalized_mid_sentence) per term occurrence."""
    out = []
    for mt in _TERM_RE.finditer(text):
        surf = mt.group(0)
        low = surf.lower()
        prefix = text[max(0, mt.start() - 3) : mt.start()]
        sentence_start = mt.start() == 0 or bool(_SENT_END_RE.search(prefix))
        cap = surf[0].isupper() and not sentence_start
        out.append((low, surf, mt.start(), cap))
    return out


class KeywordPointerGenerator(PointerGenerator):
    """Context-relative TF-IDF keyword pointers (see module docstring).

    score(t, block) = (1 + ln tf) * idf(t) * boost(t)
      idf(t)   = ln((N + 1) / (df + 1)) + 1          (df over this context's blocks)
      boost(t) = number_boost        if t contains a digit
               * entity_boost        if t appears capitalized mid-sentence
                                     (or is ALLCAPS / MixedCase)
               * long_boost          if len(t) >= long_len
    Ties: higher score, then earlier first occurrence in the block.
    """

    name = "keyword"

    def __init__(
        self,
        k: int,
        number_boost: float = 2.0,
        entity_boost: float = 1.5,
        long_boost: float = 1.2,
        long_len: int = 8,
        min_len: int = 2,
        max_term_tokens: Optional[int] = None,
    ):
        if k < 0:
            raise ValueError("k must be >= 0")
        self.k = k
        self.number_boost = number_boost
        self.entity_boost = entity_boost
        self.long_boost = long_boost
        self.long_len = long_len
        self.min_len = min_len
        self.max_term_tokens = max_term_tokens

    def _keep(self, low: str) -> bool:
        if low in STOPWORDS:
            return False
        has_digit = any(c.isdigit() for c in low)
        return has_digit or len(low) >= self.min_len

    def block_terms(self, blocks: List[str]) -> List[List[Tuple[str, float]]]:
        """Per block: [(surface_term, score)] sorted best-first (deduped, case-insensitive)."""
        per_block = []
        df: Counter = Counter()
        for blk in blocks:
            occ = [o for o in extract_terms(blk) if self._keep(o[0])]
            tf: Counter = Counter()
            first_pos: Dict[str, int] = {}
            surface: Dict[str, str] = {}
            entity: Dict[str, bool] = {}
            for low, surf, pos, cap in occ:
                tf[low] += 1
                if low not in first_pos:
                    first_pos[low] = pos
                    surface[low] = surf
                is_ent = cap or (len(surf) > 1 and surf.isupper() and surf.isalpha()) or (
                    any(c.isupper() for c in surf[1:]) and any(c.islower() for c in surf)
                )
                if is_ent:
                    entity[low] = True
                    if not surface[low][0].isupper():
                        surface[low] = surf  # prefer the entity-cased surface form
            df.update(tf.keys())
            per_block.append((tf, first_pos, surface, entity))

        n = len(blocks)
        ranked = []
        for tf, first_pos, surface, entity in per_block:
            scored = []
            for low, f in tf.items():
                idf = math.log((n + 1) / (df[low] + 1)) + 1.0
                s = (1.0 + math.log(f)) * idf
                if any(c.isdigit() for c in low):
                    s *= self.number_boost
                if entity.get(low):
                    s *= self.entity_boost
                if len(low) >= self.long_len:
                    s *= self.long_boost
                scored.append((-s, first_pos[low], surface[low], s))
            scored.sort()
            ranked.append([(surf, s) for _, _, surf, s in scored])
        return ranked

    def pointers(self, blocks: List[str], tokenizer) -> List[List[int]]:
        if self.k == 0:
            return [[] for _ in blocks]
        out = []
        for terms in self.block_terms(blocks):
            ids: List[int] = []
            first_overflow: Optional[List[int]] = None
            for surf, _ in terms:
                if len(ids) >= self.k:
                    break
                t_ids = tokenizer.encode(" " + surf, add_special_tokens=False)
                if self.max_term_tokens is not None and len(t_ids) > self.max_term_tokens:
                    continue
                if len(ids) + len(t_ids) <= self.k:
                    ids.extend(t_ids)
                elif first_overflow is None:
                    # doesn't fit whole: keep looking for lower-ranked terms that
                    # fit whole (a truncated number/name is misleading)
                    first_overflow = t_ids
            if len(ids) < self.k and first_overflow is not None:
                # nothing else filled the budget: truncate the best overflowing term
                ids.extend(first_overflow[: self.k - len(ids)])
            out.append(ids)
        return out

    def info(self) -> dict:
        return {
            "pointer_kind": self.name,
            "k": self.k,
            "idf_scope": "blocks of the current context",
            "number_boost": self.number_boost,
            "entity_boost": self.entity_boost,
            "long_boost": self.long_boost,
            "long_len": self.long_len,
            "min_len": self.min_len,
            "max_term_tokens": self.max_term_tokens,
        }


# --------------------------------------------------------------------------- #
# ablations
# --------------------------------------------------------------------------- #
class RandomIDPointerGenerator(PointerGenerator):
    """Legacy semantic-ID pointers (seeded random projection buckets)."""

    name = "random_id"

    def __init__(
        self,
        k: int,
        seed: int = 42,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        pointer_vocab_size: int = 256,
        id_offset: int = 1000,
        device: str = "cuda",
    ):
        from .pointer_generator import SemanticIDPointerGenerator

        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        self.k = k
        self.seed = seed
        self._gen = SemanticIDPointerGenerator(
            embedder_id=embedder_id,
            k=k,
            pointer_vocab_size=pointer_vocab_size,
            id_offset=id_offset,
            seed=seed,
            device=device,
        )

    def pointers(self, blocks: List[str], tokenizer) -> List[List[int]]:
        if self.k == 0 or not blocks:
            return [[] for _ in blocks]
        return self._gen.generate_token_ids_batch(list(blocks))

    def info(self) -> dict:
        return {"pointer_kind": self.name, **self._gen.info()}


class LeadPointerGenerator(PointerGenerator):
    """First k LLM tokens of the block."""

    name = "lead"

    def __init__(self, k: int):
        if k < 0:
            raise ValueError("k must be >= 0")
        self.k = k

    def pointers(self, blocks: List[str], tokenizer) -> List[List[int]]:
        if self.k == 0:
            return [[] for _ in blocks]
        return [tokenizer.encode(b, add_special_tokens=False)[: self.k] for b in blocks]


class NonePointerGenerator(PointerGenerator):
    """No pointer tokens at all (k forced to 0)."""

    name = "none"

    def __init__(self, k: int = 0):
        self.k = 0

    def pointers(self, blocks: List[str], tokenizer) -> List[List[int]]:
        return [[] for _ in blocks]


POINTER_KINDS = ("keyword", "random_id", "lead", "none")


def make_pointer_generator(kind: str, k: int, seed: int = 42, **kw) -> PointerGenerator:
    """Factory. Extra kwargs go to the generator (e.g. device/embedder_id for
    random_id, number_boost etc. for keyword)."""
    kind = kind.lower()
    if kind == "keyword":
        return KeywordPointerGenerator(k=k, **kw)
    if kind in ("random_id", "random", "semantic_id"):
        return RandomIDPointerGenerator(k=k, seed=seed, **kw)
    if kind == "lead":
        return LeadPointerGenerator(k=k)
    if kind == "none":
        return NonePointerGenerator()
    raise ValueError(f"unknown pointer kind {kind!r}; expected one of {POINTER_KINDS}")
