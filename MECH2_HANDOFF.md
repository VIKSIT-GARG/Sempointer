# MECH-2 handoff — agentic_pointer_v2 head-to-head (user execution)

v2 hypothesis: keyword-enriched index + constrained routing fixes v1's
block-0/single-block router collapse (77% `[0]`, 0/30 multi-block, F1 0.030).
Full diagnosis: `experiments/MECH2_DIAGNOSIS.md`. CPU contract already green
(py_compile + mock-LLM test, no weights touched).

## 1. Smoke test (CPU-light GPU check, ~minutes)

Copy-paste from `experiments/`:

```bash
PYTHONPATH=. ./warden/bin/python run_benchmark.py --benchmark longbench \
  --longbench-tasks hotpotqa --limit 3 --systems agentic_pointer_v2 \
  --model Qwen/Qwen2.5-3B-Instruct --dtype nf4
```

Expected: 3 prediction lines, no `ERROR:` lines, `n_llm_calls` 2 (3 only if a
routing retry fired — check `extra.routing_retried`), both prompts logged
(`prompts` length == `n_llm_calls`).

## 2. Head-to-head pilot (the actual test, n=30 musique, 3B NF4)

```bash
PYTHONPATH=. ./warden/bin/python run_benchmark.py --benchmark longbench \
  --longbench-tasks musique --limit 30 \
  --systems agentic_pointer_v2,agentic_pointer,sempointer,rag \
  --model Qwen/Qwen2.5-3B-Instruct --dtype nf4
```

## 3. Expected signatures (what "routing won" looks like)

- `agentic_pointer_v2`: multi-block selections on a clear majority of the 30
  (v1: 0/30); `[0]`-only routes well below v1's 23/30; strict-grammar match
  (`extra.strict_grammar_matched`) true on most first attempts.
- F1 above v1's 0.030 with active_tokens at or below v1's ~3.3k mean
  (previews cut 300→200 chars; keywords add bounded overhead).
- `sempointer`/`rag` lines are the control arms — v2 must beat v1 AND narrow
  the gap to dense (0.054 in v2 pilot), not just move tokens around.

## 4. Paste back to MECH-2

Paste the raw terminal output plus the new `predictions.jsonl` path (or file).
Minimum sufficient: per-system mean F1, mean active_tokens, and for
`agentic_pointer_v2` the `selected_ids` histogram + retry rate.

## 5. Pre-registered concession (routing loses if ANY hold on n=30 musique)

1. v2 mean F1 ≤ v1 mean F1 (no gain from fixing the top failure mode); or
2. v2 still routes single-block on ≥ 80% of questions (constraint failed to
   change router behavior — the collapse is model-capacity, not prompt
   design); or
3. v2 routes diverse multi-blocks but F1 stays flat vs v1 (router fixed,
   reader still starved → the bottleneck was never routable signal at 3B,
   and dense retrieval wins on cost-adjusted merit).

If (2) or (3): concede — LLM routing at 3B/NF4 loses; do not build v3 without
new evidence.
