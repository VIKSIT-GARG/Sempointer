# SemPointer: Empirical Validation Suite & Results

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-orange.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-12%2B%20%7C%2013%2B-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Status: Experiments Confirmed](https://img.shields.io/badge/Experiments-17%2F17%20Passed-success.svg)](#2-empirical-results--validation-status)

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
  Caching only $N \cdot k$ pointer tokens instead of $N \cdot B$ history tokens yields up to **$59.9\times$ VRAM savings** at $N=1024$.

---

## 2. Empirical Results & Validation Status

All 17 experiments executed on an NVIDIA GeForce RTX 5060 Laptop GPU under CUDA 13.4 and PyTorch 2.13. Results are saved as structured JSON in [`results/`](results/):

| Experiment | Focus Area | Empirical Outcome | Status |
| :--- | :--- | :--- | :---: |
| **`E01_retrieval`** | Pointer Addressing Accuracy | Recall@1 = 1.000, MRR = 1.000, nDCG@10 = 1.000 at $N \le 16$; Recall@1 = 0.900 at $N=256$ | **PASS** |
| **`E02_qa`** | Long-Context Memory QA | Token F1 gap = 0.0pp vs Full Context; active token savings: $56.0\%$--$68.6\%$ | **PASS** |
| **`E03_compression`** | Active Sequence Formula | Exact zero token residual ($L_{\text{meas}} \equiv L_{\text{pred}}$) across 1,200 parameter cells | **PASS** |
| **`E04_flops`** | Stateless FLOP Validation | $48.4\times$ compute reduction at $N=64, K=16$ (176.54 $\to$ 3.65 TFLOPs) | **PASS** |
| **`E05_kstar`** | Empirical $K^*$ Crossover | $0.00\%$ theoretical error ($K^*(1)=1.0150, K^*(2)=0.9578, K^*(4)=0.8609$) | **PASS** |
| **`E06_scaling`** | Latency Scaling with $N$ | Selector latency scales strictly linearly ($R^2 = 0.962$); Recall degrades logarithmically | **PASS** |
| **`E07_pointer_length`** | Pointer Length $k$ Tradeoff | Strict monotonicity across $k \in [2..64]$, Pareto saturation at $k \approx 8$ | **PASS** |
| **`E08_payload_size`** | Block Size $B$ Ablation | Verified compression scaling up to $27.8\times$ across $B \in [128..2048]$ | **PASS** |
| **`E09_selected_pointers`** | Selected Pointers $m$ | Accuracy plateaus at $m \ge m_{\text{true}}$, confirming access sparsity | **PASS** |
| **`E10_query_length`** | Query Length $L_Q$ Ablation | $L_Q$ impact on selector latency is asymptotically negligible ($b \approx 0$) | **PASS** |
| **`E11_compositionality`** | 2-Hop Compositionality | Joint pointer resolution achieves 100% 2-hop retrieval precision | **PASS** |
| **`E12_collision`** | Birthday Collision Bounds | Empirical collision rates strictly bounded by theoretical birthday model | **PASS** |
| **`E13_drift`** | Temporal Semantic Drift | Exponential decay $\lambda = 0.0028$/turn; Recall@1 = 0.969 across 100 turns | **PASS** |
| **`E14_fidelity`** | Substrate Resolution Fidelity | $\mathcal{M}_{\text{text}}$ and $\mathcal{M}_{\text{kv}}$ exhibit 0.000 distortion; $\mathcal{M}_{\text{latent}}$ lossy ($\epsilon=0.980$) | **PASS** |
| **`E15_kv_cache`** | Persistent KV Cache Scaling | $59.9\times$ VRAM savings ($1.09\text{ GB}$ vs $65.5\text{ GB}$ at $N=1024$), breaks 8GB OOM | **PASS** |
| **`E16_paged`** | Paged Host-to-Device Offload | $64\times$--$2048\times$ transfer reduction; latency drops $318.09\text{ ms} \to 4.97\text{ ms}$ | **PASS** |
| **`E17_cross_model`** | Cross-Model Generalization | Validated across Qwen-2.5-7B, Llama-3.1-8B, Gemma-2-9B, Phi-3.5-mini | **PASS** |

---

## 3. Repository Architecture

```
.
├── sempointer/                     # Core library
│   ├── cost_model.py               # Analytical FLOP and crossover cost formulas
│   ├── pointer_generator.py        # Pooling-class & attention-class generators
│   ├── prompt_builder.py           # Active sequence assembler & validator
│   ├── registry.py                 # Thread-safe in-memory pointer registry
│   ├── resolver.py                 # M_text, M_kv, and M_latent substrates
│   └── scorer.py                   # Query-conditioned selector & cosine scorer
├── baselines/                      # Full Context (B1), Dense RAG (B2), BM25 (B3), No Context (B5)
├── runners/                        # 17 standalone experiment runner scripts (e01 to e17)
├── config/                         # YAML configs for every experiment
├── data/                           # SPBench dataset builder & loaders (FEVER, NarrativeQA, etc.)
├── metrics/                        # Retrieval, QA, and Efficiency profiling
├── stats/                          # Bootstrap CIs, Wilcoxon, McNemar tests
├── results/                        # Production JSON result files (e01 through e17)
├── analysis/                       # Table and figure generator scripts
├── run_all.sh                      # Master orchestrator script
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 4. Quickstart & Reproduction

### Installation

```bash
# Clone the repository
git clone https://github.com/VIKSIT-GARG/Sempointer.git
cd Sempointer

# Recommended: create a conda environment
conda create -n sempointer python=3.11 -y
conda activate sempointer

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install requirements
pip install -r requirements.txt
```

### Running Experiments

Execute the master orchestrator script:

```bash
chmod +x run_all.sh

# Run Tier 1 experiments (E01-E07, E14) on CUDA
./run_all.sh --tier 1 --device cuda

# Run Tier 2 experiments (E08-E13, E15, E17) on CUDA
./run_all.sh --tier 2 --device cuda

# Run Tier 3 paged offload experiment (E16)
./run_all.sh --tier 3 --device cuda

# Or run the entire suite end-to-end:
./run_all.sh --device cuda
```

### Running Individual Experiments

```bash
# E03: Active Sequence Compression Formula Verification
python -m runners.e03_compression --config config/e03_compression.yaml

# E05: Empirical K* Crossover Threshold Validation
python -m runners.e05_kstar --config config/e05_kstar.yaml

# E01: Semantic Pointer Retrieval Accuracy
python -m runners.e01_retrieval --config config/e01_retrieval.yaml

# E15: Persistent KV Cache Memory Scaling
python -m runners.e15_kv_cache --config config/e15_kv_cache.yaml --device cuda
```

---

## 5. Result Logs

All raw experiment measurements are stored in [`results/`](results/) as structured JSON files organized by experiment ID:
```
results/
├── e01/    # Retrieval accuracy, MRR, nDCG, latencies
├── e02/    # Downstream QA EM/F1, token savings
├── e03/    # Active sequence length grid (1,200 cells)
├── e04/    # Profiler & analytical FLOP counts
├── e05/    # K* empirical vs theoretical crossover
├── e06/    # Latency decomposition & scaling
├── ...
└── e17/    # Cross-model generalization results
```

---

## 6. License

This repository is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
