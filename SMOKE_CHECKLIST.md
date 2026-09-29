# Smoke checklist — new baselines (agentic_pointer, iterative_rag, hybrid, session_summary)

Run from `experiments/` (imports assume `experiments/` is CWD / on `sys.path`).
Python: `./warden/bin/python` (aka `experiments/warden/bin/python` from repo root).
NEVER run GPU inference here — the user runs these commands. All use `--limit 3`
(3 examples) and greedy decoding. Pick the model line that is cached locally;
default `--dtype bf16`. If only 3B is cached, swap in the NF4 line.

## 0. Pre-flight (CPU only, no model load)

```bash
./warden/bin/python -m py_compile baselines/agentic_pointer.py baselines/iterative_rag.py baselines/hybrid.py baselines/session_summary.py run_benchmark.py && echo COMPILE_OK
PYTHONPATH=. ./warden/bin/python -c "import baselines.agentic_pointer, baselines.iterative_rag, baselines.hybrid, baselines.session_summary; print('IMPORT_OK')"
```

Expected: `COMPILE_OK`, then `IMPORT_OK` (no model download, no CUDA needed).

## 1. agentic_pointer (2 LLM calls: route + answer)

```bash
# tiny model if cached:
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems agentic_pointer --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
# else 3B NF4 fallback:
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems agentic_pointer --model Qwen/Qwen2.5-3B-Instruct --dtype nf4
```

## 2. iterative_rag (2 LLM calls: draft + final, dense reuse)

```bash
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems iterative_rag --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
# NF4 fallback: same command with `--model Qwen/Qwen2.5-3B-Instruct --dtype nf4`
```

## 3. hybrid (RRF of BM25 + dense, 1 LLM call)

```bash
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems hybrid --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
# NF4 fallback: same command with `--model Qwen/Qwen2.5-3B-Instruct --dtype nf4`
```

## 4. session_summary (MemGPT-lite: N summary calls + 1 answer call)

```bash
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems session_summary --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
# NF4 fallback: same command with `--model Qwen/Qwen2.5-3B-Instruct --dtype nf4`
```

## 5. All four together (+ regression: existing systems untouched)

```bash
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --limit 3 --systems sempointer,full,rag,bm25,agentic_pointer,iterative_rag,hybrid,session_summary --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16
```

## Expected success signatures (every command)

- Per-example lines: `agentic_pointer|iterative_rag|hybrid|session_summary  <ex_id> score=0.xxx tok=<int> lat=<int>ms` — NO `ERROR:` lines.
- `results/<run_name>/predictions.jsonl` has 3 `score`-bearing rows per new system.
- `results/<run_name>/metrics.json` contains keys for each requested system with `n_scored: 3, n_errors: 0`.
- Failures (if any) appear as `{"system": ..., "example_id": ..., "error": ...}` rows — recorded, never hidden; retry with `--resume` after a fix.
