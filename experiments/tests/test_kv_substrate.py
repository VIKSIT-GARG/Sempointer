"""Tests for the M_kv substrate (sempointer/kv_substrate.py).

CPU tests load Qwen/Qwen2.5-1.5B-Instruct in float32 from the local HF cache
(set HF_HUB_OFFLINE=1). The `gpu` variant uses cuda + bf16 when available.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 warden/bin/python -m pytest \
        tests/test_kv_substrate.py -q -m "not gpu"
"""

import os

import pytest

torch = pytest.importorskip("torch")

from sempointer.kv_substrate import KVSubstrate, rotate_cached_keys  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

BLOCK_TEXT = "The secret code of the blue door is 48213."
QUERY_TEXT = "Question: What is the secret code of the blue door? Answer:"
SYSTEM = "You answer questions using only the provided context. Answer concisely."


def _load(device: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype)
    model = model.to(device).eval()
    return model, tok


def _chat_parts(tok):
    """Splits the chat-templated prompt into (header, tail) around the context
    so the block text sits inside the user turn."""
    marker = "<<CTX>>"
    full = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": marker + "\n" + QUERY_TEXT}],
        tokenize=False, add_generation_prompt=True,
    )
    head, tail = full.split(marker)
    return (tok.encode(head, add_special_tokens=False),
            tok.encode(tail, add_special_tokens=False))


def _greedy_reference(model, ids, max_new_tokens):
    """Plain full-prefill greedy decoding (no splicing)."""
    inp = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
    out = model.generate(
        inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new_tokens,
        do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None,
        top_k=None, pad_token_id=151643,
    )
    return out[0, len(ids):].tolist()


def _truncate_at_eos(ids, eos):
    out = []
    for t in ids:
        out.append(t)
        if t in eos:
            break
    return out


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cpu_model():
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        pytest.skip("set HF_HUB_OFFLINE=1 (network blocked)")
    return _load("cpu", torch.float32)


@pytest.mark.slow
def test_rotation_exactness(cpu_model):
    model, tok = cpu_model
    sub = KVSubstrate(model, tok, device="cpu", offload="cpu")
    ids = tok.encode("Alice keeps her bicycle in the garage behind the old bakery.",
                     add_special_tokens=False)
    k0, v0 = sub.encode_kv(ids, position_offset=0)
    k37, v37 = sub.encode_kv(ids, position_offset=37)
    rot = rotate_cached_keys(k0, 37, model)
    for li in range(k0.shape[0]):
        scale = k37[li].abs().max().item()
        err_k = (rot[li] - k37[li]).abs().max().item()
        err_v = (v0[li] - v37[li]).abs().max().item()
        # fp32 round-off grows with activation magnitude (Qwen key biases are
        # large), so compare relative to the layer's key scale
        assert err_k <= 1e-4 * max(1.0, scale) + 1e-4, (li, err_k, scale)
        assert err_v <= 1e-4 * max(1.0, v37[li].abs().max().item()) + 1e-4, (li, err_v)
    # rotation is a no-op for delta 0 and composes: R(a)R(b) = R(a+b)
    assert torch.equal(rotate_cached_keys(k0, 0, model), k0)
    two = rotate_cached_keys(rotate_cached_keys(k0, 20, model), 17, model)
    assert torch.allclose(two, rot, atol=1e-3, rtol=1e-5)


@pytest.mark.slow
def test_splice_single_block_matches_plain_greedy(cpu_model):
    model, tok = cpu_model
    sub = KVSubstrate(model, tok, device="cpu", offload="cpu")
    head, tail = _chat_parts(tok)
    block = tok.encode(BLOCK_TEXT + "\n", add_special_tokens=False)
    eos = sub._eos_set(None)
    n_new = 12

    # (a) empty prefix, the block is the whole prompt -> exact
    prompt = head + block + tail
    sub.encode_blocks([prompt])
    res = sub.generate([("ids", []), ("block", 0)], max_new_tokens=n_new)
    ref = _truncate_at_eos(_greedy_reference(model, prompt, n_new), eos)
    assert res["output_ids"] == ref
    assert res["reused_tokens"] == len(prompt) - 1
    assert res["computed_tokens"] == 1
    assert res["total_positions"] == len(prompt)

    # (b) block encoded independently, spliced after a prefix -> approximate
    sub.encode_blocks([block])
    res2 = sub.generate([("ids", head), ("block", 0), ("ids", tail)], max_new_tokens=n_new)
    ref2 = _truncate_at_eos(_greedy_reference(model, head + block + tail, n_new), eos)
    assert len(res2["output_ids"]) > 0
    assert res2["reused_tokens"] == len(block)
    assert res2["computed_tokens"] == len(head) + len(tail)
    print(f"\n[prefix splice] matches full prefill: {res2['output_ids'] == ref2} "
          f"| splice={tok.decode(res2['output_ids'])!r} full={tok.decode(ref2)!r}")

    # (c) same block encoded WITH the shared prefix -> exact again
    sub.encode_blocks([block], shared_prefix_ids=head)
    res3 = sub.generate([("ids", head), ("block", 0), ("ids", tail)], max_new_tokens=n_new)
    assert res3["output_ids"] == ref2


@pytest.mark.slow
def test_factual_recall_from_spliced_block(cpu_model):
    model, tok = cpu_model
    sub = KVSubstrate(model, tok, device="cpu", offload="cpu")
    head, tail = _chat_parts(tok)
    distractor = tok.encode("The red door has no code; it is always open.\n",
                            add_special_tokens=False)
    block = tok.encode(BLOCK_TEXT + "\n", add_special_tokens=False)
    sub.encode_blocks([distractor, block])
    res = sub.generate([("ids", head), ("block", 1), ("ids", tail)], max_new_tokens=16)
    assert "48213" in res["text"], res["text"]
    # two spliced blocks (block-parallel) + fresh query
    res2 = sub.generate([("ids", head), ("block", 0), ("block", 1), ("ids", tail)],
                        max_new_tokens=16)
    assert "48213" in res2["text"], res2["text"]
    assert res2["reused_tokens"] == len(distractor) + len(block)
    for k in ("latency_ms", "ttft_ms", "peak_vram_mb", "transfer_ms"):
        assert res2[k] >= 0.0
    assert res2["ttft_ms"] <= res2["latency_ms"]


@pytest.mark.slow
def test_stored_bytes_and_offload(cpu_model):
    model, tok = cpu_model
    sub = KVSubstrate(model, tok, device="cpu", offload="cpu")
    blocks = [tok.encode(t, add_special_tokens=False)
              for t in ("First block of text.", "A second, somewhat longer block of text here.")]
    stats = sub.encode_blocks(blocks)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    elt = torch.tensor([], dtype=model.dtype).element_size()
    expected = sum(2 * cfg.num_hidden_layers * cfg.num_key_value_heads * len(b) * head_dim * elt
                   for b in blocks)
    actual = sum(b[x].numel() * b[x].element_size() for b in sub._blocks for x in ("k", "v"))
    assert sub.stored_bytes() == actual == expected
    assert stats["stored_bytes"] == expected
    assert stats["encode_ms"] > 0
    assert sub.stored_bytes("cpu") == expected and sub.stored_bytes("gpu") == 0
    assert all(b["k"].device.type == "cpu" and b["v"].device.type == "cpu" for b in sub._blocks)
    sub.clear()
    assert sub.stored_bytes() == 0 and sub.num_blocks == 0


# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_bf16_rotation_and_recall():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, tok = _load("cuda", torch.bfloat16)
    try:
        for offload in ("cpu", "gpu"):
            sub = KVSubstrate(model, tok, device="cuda", offload=offload)
            ids = tok.encode("Alice keeps her bicycle in the garage.", add_special_tokens=False)
            k0, _ = sub.encode_kv(ids)
            k37, _ = sub.encode_kv(ids, position_offset=37)
            rot = rotate_cached_keys(k0, 37, model)
            rel = ((rot.float() - k37.float()).norm() / k37.float().norm()).item()
            assert rel < 5e-2, rel  # bf16 activations drift with position
            head, tail = _chat_parts(tok)
            block = tok.encode(BLOCK_TEXT + "\n", add_special_tokens=False)
            sub.encode_blocks([block])
            dev_type = "cpu" if offload == "cpu" else "cuda"
            assert sub._blocks[0]["k"].device.type == dev_type
            if offload == "cpu":
                assert sub._blocks[0]["k"].is_pinned()
            res = sub.generate([("ids", head), ("block", 0), ("ids", tail)], max_new_tokens=16)
            assert "48213" in res["text"], res["text"]
            assert res["peak_vram_mb"] > 0
    finally:
        del model
        torch.cuda.empty_cache()
