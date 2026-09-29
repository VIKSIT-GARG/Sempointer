# SYS-2 Handoff — engine TTFT logging + encoder/rerank/compression pilots

All commands run from `experiments/` (so `warden/bin/python` resolves).
SYS-2 wrote no GPU code paths that change numerics: the `llm.py` patch is
timestamps + one `cuda.synchronize()` per path; greedy outputs are untouched.
CPU verification (py_compile + import + mock-LLM contract + loud-fail paths)
passes; `tests/test_core.py` 11/11 green. You run everything below.

## (a) Engine-logging smoke (GPU, small + long prompt)

Short prompt (exercises the streamer-tap TTFT path):

```bash
cd experiments
./warden/bin/python -c "
from engine.llm import LLMEngine
llm = LLMEngine(model_id='Qwen/Qwen2.5-1.5B-Instruct', dtype='bf16', device='cuda', max_new_tokens=16)
r = llm.generate('Say hello in one short sentence.')
print('latency_ms:      ', round(r.latency_ms, 1))
print('ttft_ms:         ', None if r.ttft_ms is None else round(r.ttft_ms, 1))
print('decode_tok/s:    ', r.decode_tokens_per_sec)
print('prompt/compl:    ', r.prompt_tokens, r.completion_tokens, '| stop:', r.stop_reason)
print('text:            ', r.text)
"
```

Long prompt (forces the chunked-prefill TTFT path, prompt > 4096 tokens):

```bash
cd experiments
./warden/bin/python -c "
from engine.llm import LLMEngine
llm = LLMEngine(model_id='Qwen/Qwen2.5-1.5B-Instruct', dtype='bf16', device='cuda', max_new_tokens=16)
long_ctx = ('The quick brown fox jumps over the lazy dog. ' * 1200).strip()
r = llm.generate(long_ctx + ' What animal jumped?')
print('prompt_tokens:   ', r.prompt_tokens)
print('latency_ms:      ', round(r.latency_ms, 1))
print('ttft_ms:         ', None if r.ttft_ms is None else round(r.ttft_ms, 1))
print('decode_tok/s:    ', r.decode_tokens_per_sec)
print('text:            ', r.text)
"
```

Expected: `ttft_ms` populated on both paths, `ttft_ms < latency_ms`,
`decode_tok/s` populated when completions ≥ 2. Short-path `ttft_ms=None`
means the streamer tap is unavailable (old transformers) — report it, do
not work around it.

## (b) Encoder-swap pilot (BGE default; E5 variant included)

```bash
cd experiments
./warden/bin/python -c "
from transformers import AutoTokenizer
from engine.llm import LLMEngine
from baselines.encoder_swap import EncoderSwapBaseline
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
llm = LLMEngine(model_id='Qwen/Qwen2.5-1.5B-Instruct', dtype='bf16', device='cuda', max_new_tokens=32)
ctx = open('README.md').read()
for enc in ['BAAI/bge-small-en-v1.5', 'sentence-transformers/all-MiniLM-L6-v2']:
    s = EncoderSwapBaseline(tokenizer=tok, m=1, embedder_id=enc, device='cuda')
    s.ingest(ctx)
    out = s.answer('What is this repo about?', llm)
    print('SYSTEM:', out['extra']['system'])
    print('ANSWER:', out['answer'][:200])
    print('TOK:', out['active_tokens'], '| SEL:', out['selected_ids'])
"
```

E5 pilot: same snippet with `embedder_id='intfloat/e5-small-v2'`.
Caveat (by design): NO `query:`/`passage:` prefixes are applied — protocol
is byte-identical to the MiniLM dense baseline so the pilot isolates encoder
weights apples-to-apples. A prefix study is a separate follow-up.

NOTE for whoever owns `run_benchmark.py` (not a SYS-2 path): `build_systems`
has no `encoder_swap:` / `rerank` / `compression` specs yet. Suggested spec
lines (4-line pattern each, mirroring the `rag` branch):

```python
elif spec.startswith("encoder_swap:"):
    from baselines.encoder_swap import EncoderSwapBaseline
    enc = spec.split(":", 1)[1]
    systems[f"encoder_swap:{enc}"] = EncoderSwapBaseline(
        tokenizer=tokenizer, m=args.rag_m, block_size=args.block_size,
        embedder_id=enc, device=args.device)
elif spec == "rerank":
    from baselines.rerank import RerankBaseline
    systems["rerank"] = RerankBaseline(
        tokenizer=tokenizer, m=args.rag_m, block_size=args.block_size,
        device=args.device)
elif spec.startswith("compression:"):
    from baselines.compression_baseline import CompressionBaseline
    rate = float(spec.split(":")[1]) if ":" in spec else 0.5
    systems[f"compression:{rate}"] = CompressionBaseline(
        tokenizer=tokenizer, block_size=args.block_size,
        rate=rate, device=args.device)
```

Same for TTFT in `predictions.jsonl`: `GenResult.ttft_ms` /
`.decode_tokens_per_sec` exist but `finish_answer`/`run_benchmark.py` do not
forward them yet (not SYS-2 paths) — one-line passthrough each if wanted.

## (c) Installs (reranker dep already present; llmlingua missing)

```bash
cd experiments
uv pip install --python warden/bin/python -U "sentence-transformers>=2.2"
uv pip install --python warden/bin/python -U "llmlingua>=0.2.4"
```

Notes: `sentence-transformers 6.1.0` is already installed (CrossEncoder
imports fine — no new dep needed for rerank, the line above just pins the
floor). `llmlingua` is confirmed absent; after install, probe with
`./warden/bin/python -c "from baselines.compression_baseline import
is_available; print(is_available())"` → expect `(True, ...)`.

## (d) What to paste back to SYS-2

1. Full stdout of both (a) smoke commands (numbers + any traceback verbatim).
2. Full stdout of the (b) pilot loop.
3. Output of both (c) install commands + the `is_available()` probe.
4. If anything raises: the complete traceback, transformers/sentence-transformers versions, and whether CUDA was visible (`nvidia-smi -L` one-liner).

Do NOT paste weights, dataset contents, or full benchmark logs — the four
items above only.
