# N500_QUEUE.md — n≈500 LongBench multihop queue (prepared, NOT run; CPU-only verify 2026-09-30)

## Decision (one-line justification)
USE FULL 600 (hotpotqa 200 + 2wikimqa 200 + musique 200, `--limit` omitted) — `benchmarks/longbench.py:build_longbench` takes a deterministic FIRST_n prefix with `del seed` (seed UNUSED), so any trim to 500 would be prefix-biased, not a seed-42 random subsample; full 600 keeps official splits unbiased at ~same cost as 500.

## Pilot basis
- `results/pilot_multihop/metrics.json`: n=30 musique × 4 sys (sempointer,full,rag,bm25), Qwen2.5-3B-NF4, max-new 64, seed 42.
- Rate: 30 ex × 4 sys ≈ 7.6 min → 0.253 min/ex (4-sys) ≈ 3.8 s/gen avg; ≈1.9 min/system/30ex.
- Cross-check: `results/final_longbench` 1150 ex × 4 sys = 258.7 min → 6.75 min/30ex-4sys (consistent).
- Per-system note: `full` mean latency 9629 ms vs ~1300 ms (sempointer/rag/bm25); `full` dominates wall-time.

## Runtime estimate (3B-NF4, 4 systems, max-new 64)
- 600 ex × 4 sys = 2400 gens × ~3.8 s ≈ 152 min ≈ **2.5 GPU-h** (+10–20% ingest/encode overhead → budget **2.5–3.0 h**).
- 500-ex equivalent would be ≈127 min ≈ 2.1 h — saving only ~25 min at cost of prefix bias; rejected.
- Per-system ≈ 38 min (sempointer/rag/bm25 faster, full slower; sum ≈ 152 min).

## Exact commands (run from `experiments/`, top-to-bottom)
```bash
# 0. Parse-only verify (CPU-safe, no model load) — verified PARSE_OK 2026-09-30, exit 0
./warden/bin/python run_benchmark.py --help | grep -E "seed|run-name|resume|longbench-tasks|limit"

# 1. PRIMARY: full-600 multihop run (no --limit = full splits), resumable, stable name
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa,2wikimqa,musique --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 --systems sempointer,full,rag,bm25 --max-new-tokens 64 --seed 42 --run-name longbench_multihop600_s42 --resume

# 1b. Resume after crash/OOM: re-run the SAME line verbatim (completed (system,example) pairs skipped; ERRORED pairs retried).
# 1c. Optional sharded alternative (same total; isolates failures per task):
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 --systems sempointer,full,rag,bm25 --max-new-tokens 64 --seed 42 --run-name longbench_multihop600_hotpotqa_s42 --resume
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks 2wikimqa --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 --systems sempointer,full,rag,bm25 --max-new-tokens 64 --seed 42 --run-name longbench_multihop600_2wikimqa_s42 --resume
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks musique --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 --systems sempointer,full,rag,bm25 --max-new-tokens 64 --seed 42 --run-name longbench_multihop600_musique_s42 --resume
```

## Flags / resume contract
- All flags verified against `run_benchmark.py --help` + `parse_args()` (CPU-only): `--benchmark longbench`, `--longbench-tasks`, `--model`, `--dtype nf4`, `--systems`, `--max-new-tokens 64`, `--seed 42`, `--run-name`, `--resume`; no `--limit` (full splits).
- `--resume` skips scored (system,example) pairs in `predictions.jsonl`, retries ERRORED pairs; keep `--run-name` + `--seed` fixed across retries, never mix seeds in one dir.
- Data prereq (once): `benchmarks/data/longbench_cache/data.zip` present (verified: hotpotqa 200 / 2wikimqa 200 / musique 200 lines).

## Output checklist (PRIMARY run)
- [ ] `results/longbench_multihop600_s42/config.json` (seed 42, no limit, 4 systems, 3B-NF4)
- [ ] `results/longbench_multihop600_s42/environment.json`
- [ ] `results/longbench_multihop600_s42/predictions.jsonl` — 2400 scored rows (600 × 4), zero `ERROR:` lines (or triaged)
- [ ] `results/longbench_multihop600_s42/metrics.json` — per system `n_scored: 600, n_errors: 0` + per_task hotpotqa/2wikimqa/musique n=200 each
- [ ] Paste back: full `metrics.json` + any `ERROR:` lines verbatim + `nvidia-smi -L`
- [ ] NOT run on this CPU host; GPU host executes §commands only.
