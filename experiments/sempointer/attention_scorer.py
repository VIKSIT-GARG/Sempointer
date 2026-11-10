"""LLM-attention pointer scoring.

One forward pass of the target LLM over an "index" prompt (header, one entry
per block = label + pointer tokens, the query, footer). For each selected
layer we take the attention from the query tokens to each pointer span, sum
over the span's keys, and average over heads and query tokens; layers are then
aggregated (mean or max) -> one score per pointer.

Implementation notes
- Attention weights are only materialized by the eager implementation, so the
  model is switched to eager for the call and the ORIGINAL implementation is
  restored in a `finally` block (also on exception).
- Instead of `output_attentions=True` (which would keep all layers' (H, L, L)
  tensors alive until the end of the forward), we register forward hooks on
  the selected layers' self-attention modules. Eager attention always returns
  its weights as output[1]; the hook reduces them to an (L,) vector
  immediately, so at most one layer's weights are alive at a time.
- `logits_to_keep=1`, `use_cache=False`: no full-vocab logits, no KV cache.
- The model is never moved (.to()); inputs go to the embedding's device, so
  bitsandbytes NF4 models work unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch


def _num_layers(model) -> int:
    cfg = model.config
    n = getattr(cfg, "num_hidden_layers", None)
    if n is None and hasattr(cfg, "get_text_config"):
        n = getattr(cfg.get_text_config(), "num_hidden_layers", None)
    if n is None:
        raise ValueError("cannot determine num_hidden_layers from model.config")
    return int(n)


def resolve_layers(model, layers: Union[str, Sequence[int]]) -> List[int]:
    n = _num_layers(model)
    if isinstance(layers, str):
        if layers == "upper_half":
            return list(range(n // 2, n))
        if layers == "lower_half":
            return list(range(0, n // 2))
        if layers == "all":
            return list(range(n))
        if layers == "last":
            return [n - 1]
        raise ValueError(f"unknown layers spec {layers!r}")
    out = sorted({(int(i) + n) % n if int(i) < 0 else int(i) for i in layers})
    if not out or out[-1] >= n:
        raise ValueError(f"layer indices {list(layers)} out of range for {n} layers")
    return out


def _attention_modules(model) -> Dict[int, torch.nn.Module]:
    """layer_idx -> self-attention module (Qwen2Attention, Phi3Attention, ...)."""
    mods: Dict[int, torch.nn.Module] = {}
    for name, mod in model.named_modules():
        if name.endswith("self_attn") and hasattr(mod, "layer_idx") and mod.layer_idx is not None:
            mods.setdefault(int(mod.layer_idx), mod)
    if not mods:
        raise RuntimeError("no self_attn modules with layer_idx found in model")
    return mods


def _set_attn(model, impl: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(impl)
    else:  # very old transformers
        model.config._attn_implementation = impl


@torch.no_grad()
def pointer_attention_scores(
    model,
    input_ids: torch.LongTensor,
    pointer_spans: List[Tuple[int, int]],
    query_span: Tuple[int, int],
    layers: Union[str, List[int]] = "upper_half",
    agg: str = "mean",
    length_norm: bool = False,
) -> List[float]:
    """Query->pointer attention mass per pointer span (see module docstring).

    input_ids: (1, L). pointer_spans / query_span: [start, end) positions.
    Empty pointer spans score 0.0. Returns python floats, one per span.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must be (1, L), got {tuple(input_ids.shape)}")
    if agg not in ("mean", "max"):
        raise ValueError("agg must be 'mean' or 'max'")
    L = int(input_ids.shape[1])
    qs, qe = int(query_span[0]), int(query_span[1])
    if not (0 <= qs < qe <= L):
        raise ValueError(f"query_span {query_span} invalid for L={L}")
    for s, e in pointer_spans:
        if not (0 <= s <= e <= L):
            raise ValueError(f"pointer span {(s, e)} invalid for L={L}")
        if e > qs:
            # causal model: query tokens can only attend to earlier positions
            raise ValueError(f"pointer span {(s, e)} must precede query_span {query_span}")

    layer_ids = resolve_layers(model, layers)
    attn_mods = _attention_modules(model)
    missing = [i for i in layer_ids if i not in attn_mods]
    if missing:
        raise RuntimeError(f"attention modules missing for layers {missing}")

    per_layer: Dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_mod, _inp, output):
            w = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None
            if w is None:
                raise RuntimeError("attention module returned no weights; eager attention not active")
            # (1, H, L, L) -> query rows -> mean over heads & query tokens -> (L,)
            per_layer[idx] = w[0, :, qs:qe, :].float().mean(dim=(0, 1)).cpu()
            # return a copy of output without the weights so nothing downstream keeps them
            return (output[0], None) + tuple(output[2:])

        return hook

    device = model.get_input_embeddings().weight.device
    ids = input_ids.to(device)
    orig_impl = model.config._attn_implementation
    handles = []
    try:
        if orig_impl != "eager":
            _set_attn(model, "eager")
        for i in layer_ids:
            handles.append(attn_mods[i].register_forward_hook(make_hook(i)))
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
        if model.config._attn_implementation != orig_impl:
            _set_attn(model, orig_impl)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(per_layer) != len(layer_ids):
        raise RuntimeError(f"captured {len(per_layer)} of {len(layer_ids)} layers")

    stacked = torch.stack([per_layer[i] for i in layer_ids])  # (n_layers, L)
    csum = torch.nn.functional.pad(stacked.cumsum(dim=1), (1, 0))  # (n_layers, L+1)
    scores: List[float] = []
    for s, e in pointer_spans:
        if e <= s:
            scores.append(0.0)
            continue
        mass = csum[:, e] - csum[:, s]  # (n_layers,)
        if length_norm:
            mass = mass / float(e - s)
        v = mass.mean() if agg == "mean" else mass.max()
        scores.append(float(v))
    return scores


def build_index_ids(
    tokenizer,
    header: str,
    entries: List[Tuple[str, List[int]]],
    query_text: str,
    footer: str,
    sep: str = "\n",
    span: str = "pointer",
) -> Tuple[torch.LongTensor, List[Tuple[int, int]], Tuple[int, int]]:
    """Builds the index prompt token-by-token so spans are exact.

    Layout: header | label_0 ptr_0 sep | label_1 ptr_1 sep | ... | query_text | footer
    Each text piece is encoded separately with add_special_tokens=False (chat
    special tokens such as "<|im_start|>" inside header/footer are still
    recognized by the Qwen/Phi tokenizers); pointer ids are inserted verbatim.

    span="pointer": each span covers the pointer tokens only (falls back to
      the label tokens when the pointer is empty, e.g. kind "none").
    span="entry":   each span covers label + pointer tokens.

    Returns (input_ids (1, L) LongTensor, pointer_spans, query_span).
    IMPORTANT: feed exactly these ids to the model; re-tokenizing the decoded
    string can merge tokens across piece boundaries and shift the spans.
    """
    if span not in ("pointer", "entry"):
        raise ValueError("span must be 'pointer' or 'entry'")

    def enc(t: str) -> List[int]:
        return tokenizer.encode(t, add_special_tokens=False) if t else []

    ids: List[int] = list(enc(header))
    sep_ids = enc(sep)
    spans: List[Tuple[int, int]] = []
    for label, ptr in entries:
        lab_ids = enc(label)
        lab_start = len(ids)
        ids.extend(lab_ids)
        p_start = len(ids)
        ids.extend(int(t) for t in ptr)
        p_end = len(ids)
        if span == "entry" or p_end == p_start:
            spans.append((lab_start, p_end))
        else:
            spans.append((p_start, p_end))
        ids.extend(sep_ids)
    q_ids = enc(query_text)
    if not q_ids:
        raise ValueError("query_text must be non-empty")
    q_start = len(ids)
    ids.extend(q_ids)
    q_span = (q_start, len(ids))
    ids.extend(enc(footer))
    return torch.tensor([ids], dtype=torch.long), spans, q_span
