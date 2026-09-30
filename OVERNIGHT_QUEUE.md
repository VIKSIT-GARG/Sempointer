# OVERNIGHT_QUEUE.md — copy-paste queue for GPU host (CPU-verified parse-only, no GPU execution here)

Run top-to-bottom from `experiments/` (`cd /home/viksit/Projects/token-opencode/experiments`).
All `run_benchmark.py` flags below verified against `--help` + `parse_args()` CPU-only 2026-09-30 (`PARSE_OK`).
`--seed` IS supported (`run_benchmark.py:54`, default 42) — use distinct `--run-name` per seed so `--resume` never mixes seeds.
`compression:` is NOT wired in `build_systems` (only `kvpress:*` matched; `grep compression` hits docstring+kvpress only) — apply §2 diff first.

## (0) Pre-flight (CPU-safe, no model load)

```bash
./warden/bin/python run_benchmark.py --help | grep -E "seed|run-name|resume|longbench-tasks"
grep -n "compression" run_benchmark.py || echo "NO_COMPRESSION_WIRING_CONFIRMED"
```

## (1) 2-seed RULER rerun — seeds 43, 44 (mirrors `run_final_suite.sh` ruler stage + `--seed`)

```bash
./warden/bin/python run_benchmark.py --benchmark ruler --ruler-tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,vt,cwe --context-lengths 4096,8192,16384 --limit 10 --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 --systems sempointer,full,rag,bm25 --max-new-tokens 32 --run-name final_ruler_s43 --resume --seed 43
./warden/bin/python run_benchmark.py --benchmark ruler --ruler-tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,vt,cwe --context-lengths 4096,8192,16384 --limit 10 --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 --systems sempointer,full,rag,bm25 --max-new-tokens 32 --run-name final_ruler_s44 --resume --seed 44
```

- Expected runtime: ~3–5 h GPU per seed on 3B NF4 (180 ex × 4 sys = 720 gens, max-new 32; ~4× SMOKE_MASTER §C 1 h pilot). Resumable: re-run same line after crash/OOM.
- Outputs: `results/final_ruler_s43/{config.json,environment.json,predictions.jsonl,metrics.json}`, same for `s44`.
- If seed were unsupported (it is supported — fallback only if needed): drop `--seed`, use `--run-name final_ruler_resume_a/b` with `--resume`.

## (2) LLMLingua musique-30 pilot (needs wiring)

Minimal wiring diff in `run_benchmark.py`, inside `build_systems`, before the `elif spec.startswith("kvpress:"):` branch (SYS2_HANDOFF.md §b pattern, adapted to actual ctor in `baselines/compression_baseline.py:56`):

```diff
+        elif spec.startswith("compression:"):
+            from baselines.compression_baseline import CompressionBaseline
+            rate = float(spec.split(":")[1]) if ":" in spec else 0.5
+            systems[f"compression:{rate}"] = CompressionBaseline(
+                tokenizer=tokenizer, block_size=args.block_size,
+                rate=rate, device=args.device)
         elif spec.startswith("kvpress:"):
```

Pre-step (once):

```bash
uv pip install --python warden/bin/python -U "llmlingua>=0.2.4"
./warden/bin/python -c "from baselines.compression_baseline import is_available; print(is_available())"
```

Pilot (parse-verified; `compression:0.5` requires the diff above):

```bash
./warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks musique --limit 30 --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 --systems sempointer,full,rag,bm25,compression:0.5 --max-new-tokens 64 --run-name pilot_compression_musique30 --resume --seed 42
```

- Expected runtime: ~1–1.5 h GPU (30 ex × 5 sys = 150 gens, max-new 64; cf. SMOKE_MASTER §C ~1 h for 4 sys). Expect `(True, ...)` from probe; compressor `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` downloads once.
- Outputs: `results/pilot_compression_musique30/{config.json,predictions.jsonl,metrics.json}`.

## (3) Paste back exactly this (nothing else)

1. `cat results/final_ruler_s43/metrics.json` + `cat results/final_ruler_s44/metrics.json` (full).
2. `cat results/pilot_compression_musique30/metrics.json` (full).
3. Any `ERROR:` lines verbatim + complete traceback if a stage raised; plus `nvidia-smi -L` one-liner.
4. `git diff --stat` + `git diff run_benchmark.py` (to confirm only §2 wiring changed).

Do NOT paste weights, dataset contents, or full benchmark logs.
