# SemPointer: An Indirection Layer over Dense Retrieval — Measurements and Negative Results

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Paper PDF](https://img.shields.io/badge/Paper-IEEE_Conference-red.svg)](sempointer_ieee.pdf)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-orange.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-12%2B%20%7C%2013%2B-green.svg)](https://developer.nvidia.com/cuda-zone)
[![Status: Test Suite](https://img.shields.io/badge/Tests-pytest_suite-green.svg)](#2-empirical-evaluation--evidence)

This repository contains the official reference implementation, empirical benchmark suite, and LaTeX manuscript for the paper:

> **SemPointer: Toward Random-Access Semantic Memory for Large Language Models**  
> **Author:** Viksit Garg (*Department of AI & ML, Dr. Akhilesh Das Gupta Institute of Professional Studies, New Delhi, India*)  
> **Camera-Ready Manuscript:** [`sempointer_ieee.pdf`](sempointer_ieee.pdf)  
> **Source Repository:** [https://github.com/VIKSIT-GARG/Sempointer](https://github.com/VIKSIT-GARG/Sempointer)

---

## 1. Overview and Core Theoretical Foundations

Standard autoregressive transformer language models process contextual history as an unbroken linear sequence. In multi-turn dialogue, long-horizon agents, or iterative synthesis, accumulating context triggers quadratic self-attention FLOP costs ($\mathcal{O}(L^2 d)$) in stateless or un-cached regimes, and rapid GPU out-of-memory (OOM) failures in cached regimes.

Existing prompt compression techniques (hard token pruning, soft prompt tokens, dynamic KV eviction) reduce sequence footprints in place, but permanently couple representation with addressing: **once context is compressed or pruned, individual source facts cannot be selectively dereferenced on demand.**

**SemPointer** formalizes sequence-level indirection within the transformer sequence space, separating context retention from context reference:

```
[ Historical Context Blocks X_1 ... X_N ]
                  │
                  ▼ Pointer Generator G_phi
[ Compact Token Pointers P_1 ... P_N ] (k tokens each, k << B)
                  │
                  ├──► Retained in Active Sequence Space
                  └──► Payload X_i decoupled to Independent Substrate (M_text, M_kv, M_latent)
                  │
[ Query Q_t ] ────┴──► Pointer Scorer ──► Select top-m Pointers ──► Resolve Payloads
                  │
                  ▼ Active Sequence for Transformer Inference:
     L_active = (N - m)*k + m*B_res + L_Q  <<  N*B + L_Q
```

### Key Theoretical Guarantees

1. **Active Sequence Compression (Proposition 1):**  
   For $N$ registered memory blocks of length $B$, selecting $m$ relevant blocks per query ($m \ll N$) with pointer length $k$ ($k \ll B$) compresses active sequence length to:
   $$L_{\mathrm{active}} = (N-m)k + m \cdot B_{\mathrm{res}} + L_Q$$
   achieving up to $128\times$ sequence compression relative to linear concatenation ($N \cdot B + L_Q$).

2. **The Indirection Crossover Threshold $K^*$ (Proposition 2):**  
   Pointer setup overhead ($\mathcal{C}_{\mathrm{init}}$) amortizes over repeated queries. Under stateless inference, indirection yields strictly lower cumulative FLOPs than linear re-ingestion whenever query count $K > K^*$:
   $$K^* = \frac{\mathcal{C}_{\mathrm{init}}}{\Delta\mathcal{C}_{\mathrm{step}}}$$
   For pooling-class pointer generators at $d=4096, B=512, k=8$, single-query viability ($K^* \le 1$) is achieved at registry size $N \ge 2$ ($K^*(1)=1.015, K^*(2)=0.958, K^*(4)=0.861, K^*(64)=0.213$).

3. **Persistent KV Memory Footprint Scaling (Regime 2):**  
   In persistent KV cache regimes, caching only $N \cdot k$ pointer tokens instead of $N \cdot B$ history tokens yields up to **$59.9\times$ VRAM savings**, scaling beyond $N=1024$ without encountering 8GB GPU memory limits.

---

## 2. Empirical Evaluation & Evidence

Empirical evidence lives in executed artifacts, not in this file. The runnable
contract is the pytest suite in `experiments/tests/` (11 CPU unit tests in
`test_core.py` + 6 GPU integration tests in `test_integration_gpu.py`; collect
with `warden/bin/python -m pytest tests/ --collect-only` from
`experiments/`), plus the four executed final benchmark runs under
`experiments/results/final_{ruler,longbench,locomo,kvcompare}/`
(`config.json`, `environment.json`, `predictions.jsonl`, `metrics.json` per
run). Claim-to-evidence mapping is maintained in
[`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md); dataset provenance
(including documented deviations) in
[`BENCHMARK_PROVENANCE.md`](BENCHMARK_PROVENANCE.md); run instructions in
[`RUNNING_EXPERIMENTS.md`](RUNNING_EXPERIMENTS.md).

> **Not in the paper:** legacy experiment IDs E04 (stateless FLOP validation),
> E08 (payload-size ablation), E10 (query-length ablation), and E12 (birthday
> collision bounds) were retired during reconstruction (tautological or
> category-error analyses — see `CLAIM_EVIDENCE_MATRIX.md` rows 9–16 and
> `INDEPENDENT_RECONSTRUCTION.md` §7). The hand-typed 17-experiment results
> table previously shown here has been removed for the same reason; consult the
> artifacts above instead.

---

## 3. Repository Structure

```
.
├── sempointer_ieee.tex             # Main IEEE conference LaTeX manuscript
├── sempointer_ieee.pdf             # Compiled camera-ready PDF (9 pages)
├── references.bib                  # Complete, verified primary-source BibTeX bibliography
├── IEEEtran.cls                    # Official IEEE conference document class
├── figure_architecture.tex         # TikZ vector source for system architecture
├── figure_architecture.pdf         # Compiled vector architecture diagram
├── figures/                        # 10 publication vector figures (PDF)
│   ├── F1_kstar_crossover.pdf
│   ├── F2_kstar_theory_vs_empirical.pdf
│   ├── F3_recall_vs_N.pdf
│   ├── F4_active_tokens_vs_N.pdf
│   ├── F5_latency_decomposition.pdf
│   ├── F6_pointer_length_tradeoff.pdf
│   ├── F7_qa_accuracy_vs_N.pdf
│   ├── F8_kv_memory_capacity.pdf
│   ├── F9_collision_rate.pdf
│   └── F10_resolution_fidelity.pdf
├── tables/                         # 9 publication LaTeX tables
│   ├── T1_model_configurations.tex
│   ├── T2_dataset_summary.tex
│   ├── T3_main_retrieval_results.tex
│   ├── T4_qa_accuracy.tex
│   ├── T5_complexity_validation.tex
│   ├── T6_memory_footprint.tex
│   ├── T7_pointer_length_ablation.tex
│   ├── T8_cross_model_comparison.tex
│   └── T9_falsification_summary.tex
└── experiments/                    # Core library and experiment execution suite
    ├── run_all.sh                  # Master orchestrator script (Tiers 1, 2, 3)
    ├── requirements.txt            # Python dependencies
    ├── sempointer/                 # Core SemPointer implementation
    │   ├── cost_model.py           # Analytical FLOP and crossover equations
    │   ├── pointer_generator.py    # Pooling-class & attention-class generators
    │   ├── registry.py             # In-memory thread-safe pointer registry
    │   ├── scorer.py               # Query-conditioned selector & scorer
    │   ├── resolver.py             # M_text, M_kv, and M_latent substrates
    │   └── prompt_builder.py       # Active sequence assembler & validator
    ├── baselines/                  # Full Context, Dense RAG, BM25, No Context
    ├── data/                       # SPBench synthetic benchmark & standard loaders
    ├── metrics/                    # Retrieval, QA, and Efficiency profiling
    ├── stats/                      # Bootstrap CIs, Wilcoxon, McNemar tests
    ├── runners/                    # Individual runners e01_*.py through e17_*.py
    ├── config/                     # YAML configuration files for all runs
    ├── analysis/                   # Figure and table generation scripts
    ├── results/                    # Structured JSON result outputs
    └── logs/                       # Full execution logs from production runs
```

---

## 4. Quickstart & Installation

### Environment Setup

Clone the repository and install dependencies in a clean virtual environment:

```bash
git clone https://github.com/VIKSIT-GARG/Sempointer.git
cd Sempointer/experiments

# Create conda environment (recommended)
conda create -n sempointer python=3.11 -y
conda activate sempointer

# Install PyTorch with CUDA support (adjust for your CUDA driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install experiment requirements
pip install -r requirements.txt
```

---

## 5. Reproducing Experiments

The master experiment orchestrator [`experiments/run_all.sh`](experiments/run_all.sh) executes the test suite sequentially with logging and rich CLI progress reporting:

```bash
cd experiments

# Make runner executable
chmod +x run_all.sh

# Run Tier 1 experiments (E01-E07, E14) on CUDA
./run_all.sh --tier 1 --device cuda

# Run Tier 2 experiments (E08-E13, E15, E17) on CUDA
./run_all.sh --tier 2 --device cuda

# Run Tier 3 paged offload experiment (E16)
./run_all.sh --tier 3 --device cuda

# Or run the entire suite in one pass:
./run_all.sh --device cuda
```

### Running Individual Experiments

Individual experiments can also be executed directly via Python module invocations:

```bash
# E03: Active Sequence Compression Verification (Proposition 1)
python -m runners.e03_compression --config config/e03_compression.yaml

# E05: Empirical K* Crossover Threshold Validation (Proposition 2)
python -m runners.e05_kstar --config config/e05_kstar.yaml

# E01: Semantic Pointer Retrieval Accuracy
python -m runners.e01_retrieval --config config/e01_retrieval.yaml

# E15: Persistent KV Cache Memory Scaling
python -m runners.e15_kv_cache --config config/e15_kv_cache.yaml --device cuda
```

---

## 6. Regenerating Figures and LaTeX Tables

After executing the experiment runners, publication figures and LaTeX tables can be regenerated directly from the empirical JSON logs in `experiments/results/`:

```bash
cd experiments

# Regenerate all 10 publication figures (PDF and PNG)
python -m analysis.generate_figures

# Regenerate all 9 LaTeX tabular files
python -m analysis.generate_tables
```

---

## 7. Compiling the Paper Manuscript

The paper is typeset using the IEEEtran conference format. To compile the camera-ready PDF using [`tectonic`](https://tectonic-typesetting.github.io/):

```bash
# From repository root:
tectonic sempointer_ieee.tex
```

This compiles `sempointer_ieee.tex`, resolves all citations from `references.bib`, embeds vector figures and tables, and produces `sempointer_ieee.pdf` (9 pages, exactly balanced).

---

## 8. Citation

If you find this theoretical framework or empirical benchmark useful in your research, please cite:

```bibtex
@inproceedings{garg2026sempointer,
  author    = {Viksit Garg},
  title     = {SemPointer: Toward Random-Access Semantic Memory for Large Language Models},
  booktitle = {IEEE Conference Proceedings},
  year      = {2026},
  url       = {https://github.com/VIKSIT-GARG/Sempointer}
}
```

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
