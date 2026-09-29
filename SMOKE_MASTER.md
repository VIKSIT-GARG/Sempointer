# SMOKE_MASTER.md — single copy-paste queue for the GPU host operator

Run top to bottom from `experiments/` (`cd /home/viksit/Projects/token-opencode/experiments`).
Detail + expected signatures live in the source docs — this file is only the queue:
- new-baseline smokes → [`SMOKE_CHECKLIST.md`](SMOKE_CHECKLIST.md) (SYS-1)
- engine/encoder smokes → [`SYS2_HANDOFF.md`](SYS2_HANDOFF.md) (SYS-2)

All `--dry-run` variants of the commands below were parse-verified CPU-only on
2026-09-29 (exit 0). Remove `--dry-run` to execute for real. No step here was
GPU-executed during verification.

## Queue

### 0. Pre-flight (CPU only, no model load)

```bash
./warden/bin/python -m py_compile baselines/agentic_pointer.py baselines/iterative_rag.py baselines/hybrid.py baselines/session_summary.py run_benchmark.py && echo COMPILE_OK
PYTHONPATH=. ./warden/bin/python -c "import baselines.agentic_pointer, baselines.iterative_rag, baselines.hybrid, baselines.session_summary; print('IMPORT_OK')"
```

### A. SYS-1 new-system smokes (GPU, ~3 min each, 1.5B bf16; NF4 fallback noted in source)

```bash
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems agentic_pointer --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems iterative_rag --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems hybrid --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems session_summary --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems sempointer,full,rag,bm25,agentic_pointer,iterative_rag,hybrid,session_summary --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
```

Pass = per-example `score=0.xxx` lines, zero `ERROR:` lines,
`metrics.json` shows `n_scored: 3, n_errors: 0` per system (see SMOKE_CHECKLIST.md).

### B. SYS-2 engine/encoder smokes (GPU, minutes)

```bash
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

```bash
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

```bash
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

```bash
uv pip install --python warden/bin/python -U "sentence-transformers>=2.2"
uv pip install --python warden/bin/python -U "llmlingua>=0.2.4"
./warden/bin/python -c "from baselines.compression_baseline import is_available; print(is_available())"
```

Pass = `ttft_ms` populated with `ttft_ms < latency_ms` on both engine paths;
encoder loop prints two systems; `is_available()` → `(True, ...)`
(full caveats in SYS2_HANDOFF.md — incl. the no-prefix E5 protocol note and the
`run_benchmark.py` wiring suggestions, which are follow-ups, not smokes).

### C. Pilot multi-hop run (~1 h GPU: MuSiQue 2–4-hop, 30 ex × 4 systems, 3B NF4)

```bash
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks musique --limit 30 --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 --systems sempointer,full,rag,bm25 --max-new-tokens 64 --run-name pilot_multihop
cat results/pilot_multihop/metrics.json
```

## D. Paste back exactly this (nothing else)

1. Full stdout of §0 (two lines: `COMPILE_OK`, `IMPORT_OK`).
2. For each §A command: `n_scored` / `n_errors` per system from its `metrics.json`
   (one line per system), plus any `ERROR:` lines verbatim.
3. Full stdout of all four §B commands (numbers + complete tracebacks if any).
4. `cat results/pilot_multihop/metrics.json` (§C) in full.
5. If anything raised: complete traceback + `nvidia-smi -L` one-liner.

Do NOT paste weights, dataset contents, or full benchmark logs.
