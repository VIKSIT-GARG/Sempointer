"""Pointer generators.

Two pointer representations exist and both are real:

1. Dense pointer embedding (scoring representation) — the block's sentence
   embedding; used by PointerScorer for query-conditioned selection. This is
   the paper's pooling-class generator G_phi (Section "Concrete Instantiation").

2. Semantic-ID pointer tokens (addressing representation) — k discrete token
   IDs produced by quantizing the block embedding. Each pointer token j is the
   bucket index of a seeded random projection <r_j, e> of the block embedding,
   rendered as a single vocabulary token ID. The address space is therefore
   V_p^k: the pointer length k controls both the number of active tokens the
   block contributes and the discrete addressing capacity. Unlike the legacy
   `embedding.expand(k, d)` trick, changing k changes the address itself.

`SemanticIDPointerGenerator` provides both representations through one object
and is the generator used by the current pipeline.
"""

from __future__ import annotations

from typing import List

import torch
from sentence_transformers import SentenceTransformer


def embed_long(embedder, texts):
    """Embeds texts longer than the embedder's max sequence length.

    SentenceTransformer.encode silently truncates at max_seq_length (e.g. 256
    for MiniLM), so a 512-token block would be represented by only its first
    half. We split the text into max_seq_length token windows (embedder
    tokenizer) and mean-pool the window embeddings, so the block embedding
    represents the WHOLE block. Used by SemPointer and the dense-RAG baseline
    so both see identical representations.
    """
    single = isinstance(texts, str)
    items = [texts] if single else list(texts)
    max_len = int(getattr(embedder, "max_seq_length", 0) or 256)
    tok = embedder.tokenizer
    out = []
    for text in items:
        ids = tok.encode(text, add_special_tokens=False, verbose=False)
        if len(ids) <= max_len:
            out.append(embedder.encode(text, convert_to_tensor=True))
            continue
        window_texts = [tok.decode(ids[i : i + max_len]) for i in range(0, len(ids), max_len)]
        embs = embedder.encode(window_texts, convert_to_tensor=True)
        out.append(embs.mean(dim=0))
    res = torch.stack(out)
    return res[0] if single else res


class PoolingPointerGenerator:
    """Legacy generator kept for the archived E01-E17 runners.

    NOTE: generate() returns the same embedding repeated k times, so k has no
    effect on retrieval or on the pointer content. Do not use for new
    experiments; use SemanticIDPointerGenerator.
    """

    def __init__(self, model_name: str, k: int, device: str = "cuda"):
        self.k = k
        self.device = device
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            if device != "cpu":
                import warnings

                warnings.warn(f"Failed to initialize SentenceTransformer on {device} ({e}). Falling back to CPU.")
                self.device = "cpu"
                self.model = SentenceTransformer(model_name, device="cpu")
            else:
                raise
        self.tokenizer = self.model.tokenizer

    def generate(self, text: str) -> torch.Tensor:
        embedding = self._embed(text)
        d = embedding.shape[-1]
        return embedding.unsqueeze(0).expand(self.k, d)

    def generate_batch(self, texts: List[str]) -> torch.Tensor:
        embeddings = self._embed_many(texts)
        d = embeddings.shape[-1]
        return embeddings.unsqueeze(1).expand(len(texts), self.k, d)

    def generate_token_ids(self, text: str, tokenizer=None) -> List[int]:
        import math
        from collections import Counter

        tok = tokenizer or self.tokenizer
        tokens = tok.tokenize(text)
        token_ids = tok.convert_tokens_to_ids(tokens)
        special_ids = set(tok.all_special_ids)
        counts = {tid: c for tid, c in Counter(token_ids).items() if tid not in special_ids}
        top = [tid for tid, _ in sorted(counts.items(), key=lambda x: -x[1])[: self.k]]
        while len(top) < self.k:
            top.append(tok.eos_token_id or 0)
        return top[: self.k]

    def _embed(self, text: str) -> torch.Tensor:
        try:
            return self.model.encode(text, convert_to_tensor=True, device=self.device)
        except Exception:
            return self.model.encode(text, convert_to_tensor=True, device="cpu")

    def _embed_many(self, texts: List[str]) -> torch.Tensor:
        try:
            return self.model.encode(texts, convert_to_tensor=True, device=self.device)
        except Exception:
            return self.model.encode(texts, convert_to_tensor=True, device="cpu")


class SemanticIDPointerGenerator:
    """Pooling-class generator whose pointer tokens are semantic quantizer outputs.

    Construction (documented so the address semantics are auditable):
      - The block embedding e is L2-normalized.
      - k seeded random unit vectors r_1..r_k in R^d are drawn once (part of
        the generator; identical for every block and every run with the same
        seed).
      - Token j of the pointer = bucket index of clip(<r_j, e>, -1, 1) mapped
        to [-1, 1] and discretized into V_p equal-width buckets; bucket b_j is
        rendered as vocabulary token ID (id_offset + b_j).
      - Address space = V_p^k; k therefore controls addressing capacity and
        the number of active tokens, as the paper's L_active assumes.

    Caveat recorded for the collision analysis: equal-width binning makes the
    bucket distribution non-uniform, so the uniform birthday bound is a
    reference model, not an exact predictor.
    """

    def __init__(
        self,
        embedder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 8,
        pointer_vocab_size: int = 256,
        id_offset: int = 1000,
        seed: int = 42,
        device: str = "cuda",
    ):
        if k < 0:
            raise ValueError("k must be >= 0")
        self.k = k
        self.pointer_vocab_size = pointer_vocab_size
        self.id_offset = id_offset
        self.seed = seed
        self.device = device
        try:
            self.embedder = SentenceTransformer(embedder_id, device=device)
        except Exception as e:
            if device != "cpu":
                import warnings

                warnings.warn(f"Embedder init failed on {device} ({e}); falling back to CPU.")
                self.device = "cpu"
                self.embedder = SentenceTransformer(embedder_id, device="cpu")
            else:
                raise
        g = torch.Generator().manual_seed(seed)
        self._projections: torch.Tensor | None = None  # built lazily once d is known

        def _gen(d: int) -> torch.Tensor:
            torch.manual_seed(seed)
            p = torch.randn(k, d, generator=g)
            return p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)

        self._build_projections = _gen

    @property
    def tokenizer(self):
        return self.embedder.tokenizer

    def _embed(self, text: str) -> torch.Tensor:
        try:
            return embed_long(self.embedder, text).to(self.device)
        except Exception:
            return embed_long(self.embedder, text).to("cpu")

    def _embed_batch(self, texts: List[str]) -> torch.Tensor:
        try:
            return embed_long(self.embedder, texts).to(self.device)
        except Exception:
            return embed_long(self.embedder, texts).to("cpu")

    # -- dense representation ------------------------------------------ #
    def generate(self, text: str) -> torch.Tensor:
        e = self._embed(text)
        return e.unsqueeze(0)  # (1, d): the dense pointer, not repeated

    def generate_batch(self, texts: List[str]) -> torch.Tensor:
        return self._embed_batch(texts).unsqueeze(1)  # (n, 1, d)

    # -- discrete address ------------------------------------------------#
    def _project(self, emb: torch.Tensor) -> torch.Tensor:
        d = emb.shape[-1]
        if self._projections is None or self._projections.shape[-1] != d:
            self._projections = self._build_projections(int(d))
        proj = self._projections.to(emb.device).to(emb.dtype)
        return proj @ emb.unsqueeze(-1)  # (k, 1) or (n, k, 1)

    def token_ids_from_embedding(self, emb: torch.Tensor) -> List[int]:
        """emb: (d,) normalized block embedding -> k pointer token IDs."""
        proj = self._project(emb).squeeze(-1)  # (k,)
        clipped = proj.clamp(-1.0, 1.0 - 1e-6)
        buckets = ((clipped + 1.0) / 2.0 * self.pointer_vocab_size).long()
        return [self.id_offset + int(b) for b in buckets.tolist()]

    def generate_token_ids(self, text: str, tokenizer=None) -> List[int]:
        emb = self._embed(text)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return self.token_ids_from_embedding(emb)

    def generate_token_ids_batch(self, texts: List[str]) -> List[List[int]]:
        embs = self._embed_batch(texts)
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        proj = self._project(embs).squeeze(-1)  # (n, k)
        clipped = proj.clamp(-1.0, 1.0 - 1e-6)
        buckets = ((clipped + 1.0) / 2.0 * self.pointer_vocab_size).long()
        return [[self.id_offset + int(b) for b in row.tolist()] for row in buckets]

    def info(self) -> dict:
        return {
            "generator": "SemanticIDPointerGenerator",
            "embedder": getattr(self.embedder, "model_name_or_path", None) or str(self.embedder),
            "k": self.k,
            "pointer_vocab_size": self.pointer_vocab_size,
            "id_offset": self.id_offset,
            "seed": self.seed,
        }
