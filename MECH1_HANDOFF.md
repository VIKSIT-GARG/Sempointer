# MECH-1 HANDOFF — address-only retrieval (k STRUCTURALLY matters?)

MECH-1 built: `AddressScorer` (selection from discrete pointer IDs only, zero
dense reads) + `AddressRAGBaseline` (`address_rag`: same chunking/prompt/System
contract as `rag`, retrieval swapped to address overlap). CPU tests green 8/8.
Unmodified: `scorer.py`, `dense_rag.py`, everything else (only additive `elif`
in `run_benchmark.py`).

## (a) CPU test — copy-paste (no GPU, no LLM)

```bash
cd /home/viksit/Projects/token-opencode/experiments
warden/bin/python -m py_compile sempointer/address_scorer.py baselines/address_rag.py test_address_cpu.py
warden/bin/python test_address_cpu.py
```

Expected: `8/8 passed (device=cpu, no LLM)`, incl.
`k-sensitivity OK: k=2 [7, 0, 1, 2, 3, 4, 5, 6] vs k=8 [7, 2, 0, 4, 1, 3, 5, 6]`.
Paste back: the full 8-line output (or any FAIL traceback — failures are loud).

## (b) GPU smoke — address_rag runs end-to-end (1.5B, hotpotqa, limit 3)

```bash
cd /home/viksit/Projects/token-opencode/experiments
warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa \
  --systems address_rag --model Qwen/Qwen2.5-1.5B-Instruct --dtype bf16 \
  --device cuda --limit 3 --k 8 --rag-m 1 --run-name mech1_smoke_address
```

Expected signature: 3 scored rows for `address_rag`, 0 errors; `selected_ids`
length 1 each; `selection_scores` look like `prefix + matches/8` (e.g. 0.125
steps, values ≥ 0, top score often > 1.0 when a prefix run hits).
Paste back: the 3 `score=` lines + any ERROR lines.

## (c) Head-to-head pilot — the k-matters test (3B NF4, musique, limit 30)

```bash
cd /home/viksit/Projects/token-opencode/experiments
warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks musique \
  --systems address_rag,rag,sempointer --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 \
  --device cuda --limit 30 --k 8 --rag-m 1 --run-name mech1_pilot_kmatters
# then the k-ablation on the address system only:
warden/bin/python run_benchmark.py --benchmark longbench --longbench-tasks musique \
  --systems address_rag --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 \
  --device cuda --limit 30 --k 2 --rag-m 1 --run-name mech1_pilot_k2
```

Expected signatures:
- `metrics.json` has all systems with `n_scored=30`, CIs present.
- Decision rule (pre-registered): if `address_rag(k=8)` F1 is within noise of
  `rag` AND `address_rag(k=8)` beats `address_rag(k=2)` outside the 95% CI,
  k is load-bearing for retrieval → teardown demand "sempointer is actually
  built" is met structurally. If `address_rag` collapses to near-zero while
  `rag` scores normally, discrete addresses carry no signal → publish that.
- Sanity: `address_rag` active_tokens ≈ `rag` active_tokens (same prompt
  format; only retrieved block ids differ).

## What to paste back

1. CPU test output (a).
2. Smoke `score=` lines (b).
3. `metrics.json` → `systems` block for both pilot runs (c), i.e.
   `cat results/mech1_pilot_kmatters/metrics.json` and
   `cat results/mech1_pilot_k2/metrics.json`.
4. Any ERROR rows from `predictions.jsonl` (`grep -h error results/<run>/predictions.jsonl | head`).

## Files (owned paths only)

- `experiments/sempointer/address_scorer.py` — AddressScorer + address_score.
- `experiments/baselines/address_rag.py` — AddressRAGBaseline (`address_rag`).
- `experiments/test_address_cpu.py` — 8 CPU tests.
- `experiments/run_benchmark.py` — additive `elif spec == "address_rag"` only.
- `experiments/MECH1_HANDOFF.md` — this file.
