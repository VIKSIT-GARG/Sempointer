"""M_kv substrate: a MEASURED per-block KV-cache store with RoPE re-positioning.

Each context block is prefilled ONCE, alone, and its per-layer keys/values are
stored (pinned CPU memory when ``offload="cpu"``, device memory when
``offload="gpu"``). At query time the selected blocks' KV is spliced into one
assembled cache instead of re-encoding their text (M_text).

Mechanics
---------
* A block is encoded at positions ``base_pos .. base_pos+L-1`` where
  ``base_pos = 0`` (or ``len(shared_prefix_ids)``, see ``encode_blocks``).
* Splicing it at absolute position ``p`` rotates every layer's cached keys by
  ``delta = p - base_pos`` with the model's own RoPE frequencies (values are
  unchanged). RoPE is a per-pair complex rotation, so R(delta)R(pos) =
  R(pos+delta) and the attention scores inside the block depend only on
  relative positions: a block's internal hidden states are offset-invariant.
* ``("ids", ...)`` segments are prefilled into the growing cache with correct
  ``cache_position``/``position_ids`` and attend to EVERYTHING before them
  (prefix, spliced blocks, earlier ids). Decoding is greedy with the cache.

Approximation (block-parallel encoding, as in TurboRAG / APE)
-------------------------------------------------------------
Spliced blocks do NOT attend to each other or to the tokens preceding them in
the assembled sequence (they only saw themselves, plus ``shared_prefix_ids``
if given). Later fresh segments (the query) attend to everything. Hence:
  - a single block at position 0 (or after exactly the shared prefix it was
    encoded with) is EXACT w.r.t. full prefill;
  - multiple blocks / a different preceding prefix is an approximation.
Blocks encoded alone also make their first token an attention sink; passing
the prompt's system/user header as ``shared_prefix_ids`` avoids that.

If the last segment is a block, its final token is prefilled fresh (it needs
next-token logits), so that token counts as computed, not reused.

Limitations
-----------
* LongRoPE (Phi-3.5): HF switches from short to long rescale factors when a
  forward's max position exceeds ``original_max_position_embeddings`` (4096).
  Stored keys are rotated with the frequencies they were encoded with
  (snapshotted per block), so assembled sequences <= 4096 positions are
  consistent; beyond that the fresh tokens use long factors while spliced keys
  use short ones (HF's own chunked prefill has the same mixing). The output
  flags this as ``rope_mixed_frequencies``.
* Batch size 1, greedy decoding only. DynamicCache concatenation copies KV.
"""

from __future__ import annotations

import gc
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import DynamicCache

PREFILL_CHUNK = 4096  # same cap as engine/llm.py


# --------------------------------------------------------------------------- #
# RoPE helpers
# --------------------------------------------------------------------------- #
def _find_rotary(model) -> torch.nn.Module:
    base = getattr(model, "model", None)
    rot = getattr(base, "rotary_emb", None) if base is not None else None
    if rot is not None and hasattr(rot, "inv_freq"):
        return rot
    for name, mod in model.named_modules():
        if name.endswith("rotary_emb") and hasattr(mod, "inv_freq"):
            return mod
    raise ValueError("model has no rotary embedding module (rotary_emb with inv_freq)")


def _base_inv_freq(model) -> torch.Tensor:
    """Frequencies used for sequences within the pretraining length (for
    longrope: the short factors; for default RoPE: the only frequencies)."""
    rot = _find_rotary(model)
    f = getattr(rot, "original_inv_freq", None)
    if f is None:
        f = rot.inv_freq
    return f.detach().float().cpu()


def rotate_cached_keys(
    keys: torch.Tensor,
    delta: int,
    model,
    inv_freq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Re-positions RoPE'd keys by ``+delta`` positions.

    ``keys[..., head_dim]`` were produced by the model's
    ``k*cos(pos*f) + rotate_half(k)*sin(pos*f)`` (times attention_scaling).
    Multiplying again by the pure rotation R(delta*f) yields exactly the key at
    ``pos + delta``; attention_scaling is already inside ``keys`` and is NOT
    re-applied. Partial rotary (rotary_dim < head_dim) is honoured: dims past
    ``2*len(inv_freq)`` pass through. ``inv_freq`` defaults to the model's own
    rotary module frequencies (``original_inv_freq``); pass the frequencies the
    keys were encoded with when they may differ (longrope beyond 4096).
    Angles are computed in float64 and the rotation in float32, then cast back
    to ``keys.dtype``."""
    if delta == 0:
        return keys
    if inv_freq is None:
        inv_freq = _base_inv_freq(model)
    ang = float(delta) * inv_freq.detach().to(device="cpu", dtype=torch.float64)
    emb = torch.cat([ang, ang])
    rot_dim = int(emb.shape[0])
    if rot_dim > keys.shape[-1]:
        raise ValueError(f"rotary dim {rot_dim} > head_dim {keys.shape[-1]}")
    cos = emb.cos().to(device=keys.device, dtype=torch.float32)
    sin = emb.sin().to(device=keys.device, dtype=torch.float32)
    k = keys.float()
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    half = rot_dim // 2
    rh = torch.cat((-k_rot[..., half:], k_rot[..., :half]), dim=-1)
    out = k_rot * cos + rh * sin
    if k_pass.shape[-1]:
        out = torch.cat([out, k_pass], dim=-1)
    return out.to(keys.dtype)


def _cache_kv(cache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))  # legacy API


# --------------------------------------------------------------------------- #
class KVSubstrate:
    """Per-block KV store + splice-and-generate. See module docstring."""

    def __init__(self, model, tokenizer, device: str = "cuda", offload: str = "cpu"):
        if offload not in ("cpu", "gpu"):
            raise ValueError(f"offload must be 'cpu' or 'gpu', got {offload!r}")
        self.model = model
        self.tokenizer = tokenizer
        self.offload = offload
        # compute device = where the model's input embeddings live (works for
        # device_map-placed NF4 models); `device` is the caller's intent
        self.compute_device = model.get_input_embeddings().weight.device
        self.device = torch.device(device) if device else self.compute_device
        self._cuda = self.compute_device.type == "cuda"
        self.store_device = torch.device("cpu") if offload == "cpu" else self.compute_device
        self._pin = offload == "cpu" and self._cuda
        self._rotary = _find_rotary(model)
        self.rope_type = getattr(self._rotary, "rope_type", "default")
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        rp = getattr(cfg, "rope_parameters", None)
        orig = rp.get("original_max_position_embeddings") if isinstance(rp, dict) else None
        self.original_max_positions = orig or getattr(cfg, "original_max_position_embeddings", None)
        self._blocks: List[Dict[str, Any]] = []
        self.shared_prefix_ids: List[int] = []

    # ------------------------------------------------------------------ #
    def _sync(self) -> None:
        if self._cuda:
            torch.cuda.synchronize(self.compute_device)

    def _new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.model.config)

    def _prefill(self, ids: Sequence[int], cache, start: int, pos_offset: int = 0):
        """Prefills `ids` into `cache` at cache slots start.. (RoPE positions
        start+pos_offset..). Returns (last-position logits, cache)."""
        t = torch.tensor([list(ids)], dtype=torch.long, device=self.compute_device)
        logits, i, n = None, 0, t.shape[1]
        while i < n:
            chunk = t[:, i : i + PREFILL_CHUNK]
            cp = torch.arange(start + i, start + i + chunk.shape[1], device=self.compute_device)
            out = self.model(
                chunk,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
                cache_position=cp,
                position_ids=(cp + pos_offset).unsqueeze(0),
                logits_to_keep=1,
            )
            cache = out.past_key_values
            logits = out.logits[:, -1, :].float()
            del out
            i += chunk.shape[1]
        return logits, cache

    @torch.no_grad()
    def encode_kv(
        self,
        ids: Sequence[int],
        position_offset: int = 0,
        prefix_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes `ids` alone (optionally after a precomputed prefix KV of
        shape [n_layers, H, P, D]) at RoPE positions P+position_offset...
        Returns stacked (K, V) of shape [n_layers, kv_heads, len(ids), head_dim]
        on the compute device (prefix part excluded)."""
        cache = self._new_cache()
        P = 0
        if prefix_kv is not None:
            pk, pv = prefix_kv
            P = int(pk.shape[2])
            for li in range(pk.shape[0]):
                cache.update(pk[li].unsqueeze(0), pv[li].unsqueeze(0), li)
        _, cache = self._prefill(ids, cache, start=P, pos_offset=position_offset)
        kv = _cache_kv(cache)
        shapes = {tuple(k.shape) for k, _ in kv}
        if len(shapes) != 1:
            raise ValueError(f"heterogeneous per-layer KV shapes {shapes}; not supported")
        K = torch.stack([k[0, :, P:, :] for k, _ in kv]).contiguous()
        V = torch.stack([v[0, :, P:, :] for _, v in kv]).contiguous()
        del cache, kv
        return K, V

    def _to_store(self, t: torch.Tensor) -> torch.Tensor:
        if self.store_device.type == "cpu":
            if self._pin:
                buf = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
                buf.copy_(t)
                return buf
            return t.to("cpu").contiguous()
        return t.to(self.store_device).contiguous()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_blocks(
        self,
        blocks_ids: List[List[int]],
        shared_prefix_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Replaces the store with the KV of `blocks_ids` (block i = index i).

        Each block is encoded independently at positions 0..len-1, or, if
        `shared_prefix_ids` is given, after that prefix (positions P..P+len-1;
        the prefix KV itself is not stored). Splicing such a block right after
        the same prefix ("ids" segment at position 0) is then exact.
        Returns MEASURED encode_ms and stored_bytes."""
        self.clear()
        prefix = list(shared_prefix_ids or [])
        self._sync()
        t0 = time.perf_counter()
        prefix_kv = self.encode_kv(prefix) if prefix else None
        n_tokens = 0
        for bi, ids in enumerate(blocks_ids):
            ids = [int(x) for x in ids]
            if not ids:
                raise ValueError(f"block {bi} is empty")
            K, V = self.encode_kv(ids, prefix_kv=prefix_kv)
            # frequencies actually used by this forward (longrope may switch)
            inv = self._rotary.inv_freq.detach().float().cpu().clone()
            self._blocks.append(
                {"k": self._to_store(K), "v": self._to_store(V), "ids": ids,
                 "base_pos": len(prefix), "inv_freq": inv}
            )
            n_tokens += len(ids)
            del K, V
            if self._cuda and bi % 16 == 15:
                torch.cuda.empty_cache()
        del prefix_kv
        self._sync()
        encode_ms = (time.perf_counter() - t0) * 1000.0
        self.shared_prefix_ids = prefix
        if self._cuda:
            torch.cuda.empty_cache()
        return {
            "encode_ms": encode_ms,
            "stored_bytes": float(self.stored_bytes()),
            "n_blocks": float(len(self._blocks)),
            "n_tokens": float(n_tokens),
            "shared_prefix_tokens": float(len(prefix)),
        }

    # ------------------------------------------------------------------ #
    def _eos_set(self, eos_token_id) -> set:
        if eos_token_id is None:
            ids = set()
            gc_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
            for e in (gc_eos, getattr(self.tokenizer, "eos_token_id", None)):
                if e is None:
                    continue
                ids.update(e if isinstance(e, (list, tuple)) else [e])
            return {int(x) for x in ids}
        if isinstance(eos_token_id, (list, tuple, set)):
            return {int(x) for x in eos_token_id}
        return {int(eos_token_id)}

    @staticmethod
    def _pick(logits: torch.Tensor, seen: set, rp: float) -> int:
        if rp != 1.0 and seen:
            idx = torch.tensor(sorted(seen), device=logits.device).unsqueeze(0)
            sc = logits.gather(-1, idx)
            sc = torch.where(sc < 0, sc * rp, sc / rp)  # HF RepetitionPenalty
            logits = logits.scatter(-1, idx, sc)
        return int(logits.argmax(dim=-1).item())

    @torch.no_grad()
    def generate(
        self,
        segments: List[Tuple[str, Any]],
        max_new_tokens: int,
        eos_token_id=None,
        repetition_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        """Assembles one cache from `segments` (in order) and greedy-decodes.

        segments: ("ids", List[int]) = prefill fresh; ("block", i) = splice
        stored block i (keys rotated to its absolute position). eos_token_id:
        int/list; default = model.generation_config eos + tokenizer eos.
        repetition_penalty: 1.0 = pure greedy (HF semantics over all prompt
        tokens incl. spliced block tokens, plus generated ones)."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        plan: List[Tuple[str, Any]] = []
        for kind, val in segments:
            if kind == "ids":
                ids = [int(x) for x in val]
                if ids:
                    plan.append(("ids", ids))
            elif kind == "block":
                bi = int(val)
                if not 0 <= bi < len(self._blocks):
                    raise IndexError(f"block {bi} not in store ({len(self._blocks)} blocks)")
                plan.append(("block", bi))
            else:
                raise ValueError(f"unknown segment kind {kind!r}")
        if not plan:
            raise ValueError("no non-empty segments")
        eos = self._eos_set(eos_token_id)
        dev = self.compute_device

        if self._cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        self._sync()
        t0 = time.perf_counter()

        cache = self._new_cache()
        pos = computed = reused = 0
        transfer_ms = 0.0
        logits = None
        seen: set = set()
        i = 0
        while i < len(plan):
            kind, val = plan[i]
            if kind == "ids":
                logits, cache = self._prefill(val, cache, start=pos)
                pos += len(val)
                computed += len(val)
                seen.update(val)
                i += 1
                continue
            # run of consecutive blocks -> one concatenation per layer
            run_k: List[torch.Tensor] = []
            run_v: List[torch.Tensor] = []
            tail_token: Optional[int] = None
            while i < len(plan) and plan[i][0] == "block":
                blk = self._blocks[plan[i][1]]
                is_last = i == len(plan) - 1
                L = len(blk["ids"]) - (1 if is_last else 0)
                if is_last:
                    tail_token = blk["ids"][-1]
                if L > 0:
                    self._sync()
                    tt = time.perf_counter()
                    k = blk["k"][:, :, :L].to(dev, non_blocking=True)
                    v = blk["v"][:, :, :L].to(dev, non_blocking=True)
                    self._sync()
                    transfer_ms += (time.perf_counter() - tt) * 1000.0
                    splice_at = pos + sum(x.shape[2] for x in run_k)
                    k = rotate_cached_keys(k, splice_at - blk["base_pos"], self.model, blk["inv_freq"])
                    run_k.append(k)
                    run_v.append(v)
                    reused += L
                seen.update(blk["ids"])
                i += 1
            if run_k:
                n_layers = run_k[0].shape[0]
                for li in range(n_layers):
                    kl = torch.cat([k[li] for k in run_k], dim=1).unsqueeze(0)
                    vl = torch.cat([v[li] for v in run_v], dim=1).unsqueeze(0)
                    cache.update(kl, vl, li)
                pos += sum(k.shape[2] for k in run_k)
                del run_k, run_v, kl, vl
            if tail_token is not None:
                logits, cache = self._prefill([tail_token], cache, start=pos)
                pos += 1
                computed += 1

        total_positions = pos
        nxt = self._pick(logits, seen, repetition_penalty)
        self._sync()
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        generated: List[int] = []
        stop_reason = "max_new_tokens"
        for step in range(max_new_tokens):
            generated.append(nxt)
            seen.add(nxt)
            if nxt in eos:
                stop_reason = "eos"
                break
            if step == max_new_tokens - 1:
                break
            out = self.model(
                torch.tensor([[nxt]], device=dev),
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([total_positions + step], device=dev),
                logits_to_keep=1,
            )
            cache = out.past_key_values
            nxt = self._pick(out.logits[:, -1, :].float(), seen, repetition_penalty)
            del out
        self._sync()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2) if self._cuda else 0.0

        mixed = bool(
            self.rope_type == "longrope"
            and self.original_max_positions
            and total_positions + len(generated) > self.original_max_positions
            and reused > 0
        )
        if mixed:
            warnings.warn(
                "longrope: assembled sequence exceeds original_max_position_embeddings; "
                "spliced keys use short-factor RoPE while new tokens use long factors",
                RuntimeWarning,
            )
        del cache
        if self._cuda:
            torch.cuda.empty_cache()
        return {
            "output_ids": generated,
            "text": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "peak_vram_mb": peak_vram_mb,
            "computed_tokens": computed,
            "reused_tokens": reused,
            "transfer_ms": transfer_ms,
            "total_positions": total_positions,
            "completion_tokens": len(generated),
            "stop_reason": stop_reason,
            "rope_mixed_frequencies": mixed,
        }

    # ------------------------------------------------------------------ #
    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    def block_len(self, i: int) -> int:
        return len(self._blocks[i]["ids"])

    def stored_bytes(self, where: str = "all") -> int:
        """MEASURED bytes of stored K/V tensors. where: 'all' | 'cpu' | 'gpu'."""
        if where not in ("all", "cpu", "gpu"):
            raise ValueError("where must be 'all', 'cpu' or 'gpu'")
        total = 0
        for blk in self._blocks:
            for t in (blk["k"], blk["v"]):
                on_gpu = t.device.type == "cuda"
                if where == "all" or (where == "gpu") == on_gpu:
                    total += t.numel() * t.element_size()
        return total

    def clear(self) -> None:
        self._blocks = []
        self.shared_prefix_ids = []
        gc.collect()
        if self._cuda:
            torch.cuda.empty_cache()
