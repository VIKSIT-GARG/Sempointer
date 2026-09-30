# SemPointer: Empirical Validation Suite & Results

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0%2Bcu130-orange.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-13%20runtime-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Status: Results Recorded](https://img.shields.io/badge/Results-Final%20%2B%20pilot%20suites%20recorded-success.svg)](#2-empirical-results--validation-status)

Official reference implementation, benchmark suite, and empirical results for **SemPointer** (Random-Access Semantic Memory for Large Language Models).

---

## 1. Technical Framework & Mechanics

SemPointer formalizes sequence-level indirection within the transformer sequence space, separating context retention from context reference:

```
[ Context Blocks X_1 ... X_N ] ──► Generator G_phi ──► [ Compact Pointers P_1 ... P_N ] (k tokens each)
                                                          │
Query Q_t ──► Pointer Scorer ──► Select top-m Pointers ───┴──► Dereference Payloads from Substrate
                                                          │
                                                          ▼ Active Inference Sequence:
                       L_active = (N - m)*k + m*B_res + L_Q  <<  N*B + L_Q
```

### Core Equations

- **Active Sequence Length:**
  $$L_{\mathrm{active}} = (N-m)k + m \cdot B_{\mathrm{res}} + L_Q$$
  where $N$ is the registry size, $B$ is the block size, $k$ is pointer length ($k \ll B$), and $m$ is the number of resolved blocks ($m \ll N$).

- **Crossover Threshold $K^*$ (Stateless Regime):**
  $$K^* = \frac{\mathcal{C}_{\mathrm{init}}}{\Delta\mathcal{C}_{\mathrm{step}}}$$
  Single-query viability ($K^* \le 1$) is achieved for registry sizes $N \ge 2$ under pooling-class generators ($K^*(1)=1.015, K^*(2)=0.958, K^*(4)=0.861, K^*(64)=0.213$).

- **Persistent KV Cache Footprint (Regime 2):**
  Caching only $N \cdot k$ pointer tokens instead of $N \cdot B$ history tokens gives the
  *derived* footprint ratio $B/k$; the old headline "**$59.9\times$ at $N=1024$**" is an
  analytical formula, not a measurement. The recorded runs measure **active-token**
  reduction instead: on LongBench, 963 vs 13,066 tokens/example ($\approx 13.6\times$) — see §2.

---

## 2. Empirical Results & Validation Status

The suite is `run_benchmark.py` (`--benchmark ruler|longbench|locomo|kv_compare|all`), driven
by `run_final_suite.sh`. The legacy 17 standalone E01–E17 runners are retired
(`archive/legacy_runners/`) and are the source of **no** current number. Recorded results:
`results/final_*/` (P1–P4, Qwen2.5-3B-Instruct, NF4, seed 42) and `results/pilot_*/`
(30-example LongBench `musique` pilots). Selectable systems: `sempointer`, `full`, `rag`,
`bm25`, `address_rag`, `agentic_pointer`, `agentic_pointer_v2`, `iterative_rag`, `hybrid`,
plus `kvpress:<press>:<ratio>`. Environment: RTX 5060 Laptop `sm_120a`, torch 2.14.0+cu130,
CUDA runtime 13.0.

### Final runs (`results/final_*/`)

| Suite | Benchmark | n | sempointer | full | rag | bm25 | tokens (sempointer vs full) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `final_ruler` | RULER (6 tasks, 4k/8k/16k) | 180 | 0.330 | 0.700 | 0.329 | 0.578 | 1,270 vs 13,089 |
| `final_longbench` | LongBench (6 tasks) | 1150 | 0.077 | 0.111 | 0.078 | 0.081 | 963 vs 13,066 |
| `final_locomo` | LoCoMo (token-F1 adaptation) | 199 | 0.052 | 0.064 | 0.050 | 0.063 | 987 vs 14,525 |
| `final_kvcompare` | RULER (kvpress @0.3) | 10 | 0.800 | 1.000 | — | — | 726 vs 3,806 |

`final_kvcompare` baselines: `kvpress_snapkv_0.3` 0.50, `kvpress_tova_0.3` 0.80,
`kvpress_knorm_0.3` 0.40. **Honest reading:** SemPointer does not beat `full` context on
these suites; it is competitive with RAG/BM25 at a fraction of the active-token cost.
The only `full` errors are 14 recorded LongBench rows (`n_errors=14`), never hidden.

### Pilot runs (`results/pilot_*/`, 30 examples)

| Suite | Systems (score) |
| :--- | :--- |
| `pilot_multihop` | sempointer 0.054 · full 0.047 · rag 0.031 · bm25 0.025 |
| `pilot_multihop_v2` | agentic_pointer 0.030 · iterative_rag 0.028 · hybrid 0.022 |
| `pilot_address` | sempointer 0.054 · rag 0.031 · address_rag 0.027 |
| `pilot_routing_v2` | sempointer 0.054 · rag 0.031 · agentic_pointer 0.030 · agentic_pointer_v2 0.023 |

Derived statistics (`results/stats.json`, paired bootstrap CIs / McNemar / Wilcoxon,
`SEED=42`, `N_BOOT=10_000`), retrieval-vs-answer breakdown (`results/error_breakdown.json`),
and the True-B table (`results/config_table.json`) are all CPU-only and regenerable from
`final_*/`. `results/envcheck/` and `results/smoke_*` hold 1-example environment checks and
single-task smoke runs; they are not benchmark evidence.

---

## 3. Repository Architecture

```
.
├── sempointer/                     # Core library
│   ├── cost_model.py               # Analytical FLOP and crossover cost formulas
│   ├── pointer_generator.py        # Semantic-ID pointer generator (k token addresses)
│   ├── prompt_builder.py           # Active sequence assembler & validator
│   ├── registry.py                 # Pointer registry
│   ├── resolver.py                 # M_text, M_kv, M_latent substrates
│   └── scorer.py                   # Query-conditioned selector & cosine scorer
├── baselines/                      # full, dense RAG, BM25, address_rag, agentic_pointer[_v2],
│                                   # iterative_rag, hybrid, kvpress, session_summary, no_context
├── benchmarks/                     # RULER / LongBench / LoCoMo loaders + cached data
├── engine/                         # llm.py — model load, generation, VRAM/token instrumentation
├── metrics/                        # official_metrics, retrieval, QA, efficiency
├── stats/                          # Bootstrap CIs, Wilcoxon, McNemar primitives
├── tests/                          # test_core.py (CPU) + test_integration_gpu.py (GPU)
├── results/                        # final_*/ and pilot_*/ recorded runs (+ envcheck, smoke_*)
├── archive/legacy_*                # retired E01–E17 runners/configs/data (not evidence)
├── paper/                          # manuscript sources, claim-evidence matrix, provenance
├── run_benchmark.py                # the benchmark entry point
├── run_final_suite.sh              # P1–P4 sequential orchestrator
├── requirements_locked.txt        # exact pinned stack (uv)
└── README.md
```

---

## 4. Quickstart & Reproduction

### Installation

All commands run from `experiments/` (relative paths are resolved from there).
This rig uses `uv`, not conda:

```bash
cd experiments
uv venv warden --python 3.11
uv pip install --python warden/bin/python -r requirements_locked.txt
```

The pinned stack (`requirements_locked.txt`, 104 packages) resolves torch
`2.14.0+cu130` from plain PyPI. **Do not use the legacy `.../whl/cu124` index:**
cu124 wheels predate Blackwell and cannot drive the RTX 5060's `sm_120a`.

### Running the Suite

```bash
cd experiments

# P1–P4 sequentially (ruler → longbench → locomo → kv_compare), resumable:
./run_final_suite.sh

# A single stage:
./run_final_suite.sh ruler
./run_final_suite.sh kvcompare      # stage alias for --benchmark kv_compare

# Dry run first (no inference):
warden/bin/python run_benchmark.py \
    --benchmark all --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 --dry-run
```

### Running Individual Benchmarks

```bash
cd experiments

# RULER, 10 examples/task, 4k–16k contexts:
warden/bin/python run_benchmark.py \
    --benchmark ruler \
    --ruler-tasks niah_single_1,niah_multikey_1,vt,cwe \
    --context-lengths 4096,8192,16384 --limit 10 \
    --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 \
    --systems sempointer,full,rag,bm25 --max-new-tokens 32 \
    --run-name final_ruler --resume

# KV-compression comparison at equal budget:
warden/bin/python run_benchmark.py \
    --benchmark kv_compare \
    --ruler-tasks niah_single_1 --context-lengths 4096 --limit 5 \
    --model Qwen/Qwen2.5-3B-Instruct --dtype nf4 \
    --systems sempointer,full,kvpress:snapkv:0.3,kvpress:tova:0.3 \
    --run-name final_kvcompare --resume
```

---

## 5. Result Logs

Each run directory holds `config.json`, `environment.json`, `predictions.jsonl`
(one raw record per `(system, example)`, failures included as error rows), and
`metrics.json` (per-system aggregates + bootstrap CIs):

```
results/
├── final_ruler/       # RULER: 180 examples, 6 tasks
├── final_longbench/   # LongBench: 1150 examples, 6 tasks
├── final_locomo/      # LoCoMo: 199 examples (token-F1 adaptation)
├── final_kvcompare/   # RULER + kvpress SnapKV/TOVA/Knorm @0.3
├── pilot_multihop/    # 30-example musique pilots (v1/v2)
├── pilot_address/     # address_rag vs rag vs sempointer
├── pilot_routing_v2/  # agentic_pointer[_v2] routing
├── envcheck/          # 1-example environment verification
├── smoke_*/           # single-task smoke runs (ruler/longbench/locomo/kvpress)
├── stats.json         # paired bootstrap CIs, McNemar, Wilcoxon
├── error_breakdown.json  # retrieval-miss vs block-hit-wrong-answer join
└── config_table.json  # True-B table (n_blocks, tokens/block, k, m, kappa)
```

---

## 6. License

This repository is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
