# K-ablation sweep (pointer length k ∈ {0, 8, 32})

One command runs the same examples at three pointer lengths for
`address_rag` / `sempointer`. `k=0` = no pointer strings (pure RAG control).

## What `--k` already did (verified, no change needed)

- `--k` plumbs end-to-end: `run_benchmark.py → SemPointerPipeline(k) /
  AddressRAGBaseline(k) → SemanticIDPointerGenerator(k) → registry
  `pointer_token_ids` (length k) → prompt + `AddressScorer` (reads all k
  positions; `test_k_sensitivity` pins k=2 vs k=8 ranking change).
- Caveat found: `SemPointerPipeline` retrieves by dense cosine
  (`PointerScorer` never reads `pointer_token_ids`), so for `sempointer`
  `--k` varies prompt pointer size, not retrieval. `address_rag` retrieval
  IS k-sensitive end-to-end.

## Additive fixes in this change (existing flags untouched)

1. `sempointer/prompt_builder.py`: k=0 (empty `pointer_token_ids`) now omits
   the `Memory Registry Pointers:` section instead of emitting empty
   `[PTR_i: ]` placeholders → sempointer k=0 prompt is a pure-RAG control.
2. `run_benchmark.py`: new `--k-sweep` flag only; loops the requested k values
   sequentially on identical examples, one provenance dir per value.

## k=0 semantics per system

| system | k=0 behavior |
|---|---|
| `sempointer` | prompt = resolved blocks + query only (no pointer strings); retrieval unchanged (dense cosine) |
| `address_rag` | all address scores 0.0 → deterministic index tie-break (no-signal control); prompt unchanged (retrieved blocks only, never had pointers) |

## Copy-paste commands (run from `experiments/`, CPU-only box: use `--device cpu`)

```bash
# 1. One-command sweep: k ∈ {0, 8, 32}, same examples, suffixed run dirs
python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa \
  --limit 3 --systems address_rag,sempointer --device cpu \
  --run-name kabl_hotpot --k-sweep 0,8,32

# 2. Manual equivalent (identical effect, no new flag)
for k in 0 8 32; do
  python run_benchmark.py --benchmark longbench --longbench-tasks hotpotqa \
    --limit 3 --systems address_rag,sempointer --device cpu \
    --k "$k" --run-name "kabl_hotpot_k$k"
done

# 3. RULER variant
python run_benchmark.py --benchmark ruler --ruler-tasks niah_single_1 \
  --context-lengths 4096 --limit 5 --systems address_rag,sempointer \
  --device cpu --run-name kabl_ruler --k-sweep 0,8,32
```

## Expected outputs

- Dirs: `results/kabl_hotpot_k0/`, `results/kabl_hotpot_k8/`,
  `results/kabl_hotpot_k32/`, each with `config.json` (`"k": 0/8/32`,
  `"k_sweep": "0,8,32"`), `environment.json`, `predictions.jsonl`,
  `metrics.json`.
- `config.json` differs across the three dirs ONLY in `k` + `run_name`.
- `predictions.jsonl` row counts per system identical across k dirs (same
  examples); `address_rag` rankings shift with k; `address_rag` k=0 selects
  lowest block indices (all-zero scores).
- `metrics.json` has all three systems scored; compare
  `systems.address_rag.score_mean` across k dirs for the ablation curve.
- Checks: `python -m py_compile run_benchmark.py sempointer/prompt_builder.py`
  (no output = OK); `python run_benchmark.py --help` shows `--k-sweep`.
