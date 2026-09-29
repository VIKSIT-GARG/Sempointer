# INDEPENDENT_RECONSTRUCTION.md

**Status:** Mandatory first deliverable of the zero-trust investigation.
**Author of record:** Independent re-investigation performed 2026-09-28.
**Method:** Every statement below is based on direct inspection of source code, execution of the code by this investigator (fresh runs producing new timestamped artifacts in `experiments/results/*/`), inspection of stored artifacts and logs, hardware/dependency checks, and primary-source verification of external references. No prior agent conclusion, README claim, paper claim, or result file was trusted without independent verification.

---

## 0. What was done during this investigation

1. Mapped the repository: `~/Projects/tokens2/` (not a git repo) containing `sempointer_ieee.tex/pdf` (paper), `experiments/` (git repo, **one** squashed commit `3f314f8` "feat: SemPointer experiment codebase and empirical evaluation results", remote `github.com/VIKSIT-GARG/Sempointer`, verified public and reachable).
2. Read every core module (`sempointer/*.py`), all 17 runners (`runners/e01`–`e17`), the 4 untracked scripts added 2026-09-28 21:43 (`e18_ruler.py`, `e19_longbench.py`, `e20_kv_compare.py`, `e21_contrastive_baselines.py`), data builders, baselines, metrics, table/figure generators, configs, README, EMPIRICAL_PLAN.md, PROMPT_FOR_CODE_GENERATION.md, GEMINI.md.
3. Grepped the entire codebase for LLM inference: **no runner ever instantiates a causal LM or calls `generate()`.**
4. Executed the code myself with the project conda env (`/home/viksit/miniconda3/envs/warden/bin/python`, torch 2.13.0+cu130, CUDA available): fresh runs of E01, E02, E03, E05, E08, E12, E14, E15. Fresh outputs reproduce the stored artifacts (same point estimates).
5. Compared every headline number in `sempointer_ieee.tex` against the stored result JSONs.
6. Verified external sources for the planned public benchmarks: RULER (arXiv:2404.06654, NVIDIA/RULER), LoCoMo (arXiv:2402.17753, snap-research/locomo), LongBench (arXiv:2308.14586, THUDM/LongBench), SnapKV (arXiv:2404.14469), H2O (arXiv:2306.14048), PyramidKV (arXiv:2406.02069), ScissorHands (arXiv:2305.17118).
7. Hardware verified: NVIDIA GeForce RTX 5060 Laptop 8 GB, driver CUDA 13.4, GPU functional today. The 2026-09-24 master run attempted CUDA and logged allocator OOM warnings; `run_all.sh` auto-falls back to CPU on CUDA failure. Device provenance of stored runs is mixed (CPU/CUDA), and is not recorded per-result.

---

## 1. What does the paper claim?

From `sempointer_ieee.tex` (abstract, Sections I, VII–VIII; all verified quotes):

- **C1.** SemPointer is a framework for *semantic addressing and selective context access*: compact token-level pointers serve as addresses into an independent memory substrate; five-stage lifecycle (encounter, storage, pointer generation, query-conditioned selection, resolution).
- **C2.** Active sequence compression: `L_active = (N−m)·k + m·B_res + L_Q` replacing `N·B + L_Q`.
- **C3.** Algebraic crossover threshold `K* = C_init / ΔC_step`; single-query viability for `N ≥ 2` under pooling-class generators.
- **C4.** In cached (KV) regimes, "up to 59.9× KV memory savings without downstream task degradation".
- **C5.** "Empirical evaluation across four model families (Qwen-2.5, Llama-3.1, Gemma-2, Phi-3.5) and six diagnostic benchmarks."
- **C6.** "Verifying the active-sequence formula with exact zero token residual across 1,200 parameter cells."
- **C7.** "Confirming the theoretical crossover K* with 0.00% relative error" — including "Profiler FLOPs match algebraic model".
- **C8.** E01 retrieval: Recall@1 = 1.000 at N=8; 0.969±0.031 at N=32 "outperforming BM25 (0.940±0.040) while tracking Dense RAG (0.980±0.020)"; 0.900±0.045 at N=256 "significantly surpassing BM25 (0.780±0.060)"; "across 5 random seeds"; selector latency "5.50±0.21 ms on CUDA".
- **C9.** E02 QA: "across NarrativeQA and SPBench", F1 = 1.000 for SemPointer/Full-Context/Dense RAG/BM25, 0.0pp gap, McNemar p=1.000, token savings 56.0–68.6%.
- **C10.** Evaluation setup (Sec. VII-A, line 512): "four established benchmarks … NarrativeQA, FEVER, HotpotQA, and MuSiQue" plus self-constructed SPBench and "VI-NIAH" (needles up to 128k tokens).
- **C11.** Falsification table (`tab:falsification`): H1 "0.969±0.031 (SPBench) / 0.940±0.028 (VI-NIAH), MRR=0.984"; H5 "k=2: 0.840±0.042, k=4: 0.920±0.038, k≥8: 0.969±0.031"; H6 "0.900±0.045 at N=256 (β=0.042)".
- **C12.** E07 ablation: "k=2 suffers combinatorial collisions (10.96% collision rate, dropping recall to 0.840±0.042), confirming Proposition 3".
- **C13.** E14 substrate fidelity: M_text and M_kv "lossless (ε=0.000, EM=1.000)"; M_latent "ε=0.980, cosine fidelity 0.020, confirming that lossy latent compression induces hallucinations".
- **C14.** E11: "On 2-hop HotpotQA reasoning, joint multi-pointer dereferencing achieved Token F1 1.000 … compared to 0.720±0.040 for individual resolution, verifying that joint dereferencing prevents attention interference."
- **C15.** E16: "decreasing paging latency from 318.09 ms to 4.97 ms at N=64."
- **C16.** E17: retrieval and compression "generalized across all models" with K* scaling by d.
- **C17.** All experiments "executed on an NVIDIA GeForce RTX 5060 Laptop GPU … under CUDA 13.4 and PyTorch 2.13"; results reproducible from the public repo.

---

## 2. What does the repository actually contain?

`experiments/` (~3,200 LOC of first-party Python):

- **Core package** (`sempointer/`): pointer generation, registry, scorer, resolvers (M_text/M_kv/M_latent), prompt builder, analytic cost model. All modules load and run.
- **Runners** (`runners/e01`–`e17`) with YAML configs; results for all 17 in `results/eNN/` (multiple timestamped runs each); 13 master-run logs in `logs/` (last: 2026-09-24, "Passed: 15/17" — e08 crashed on a `torch` NameError, e15 rejected `--device`; both were re-run individually and succeeded, so README's "17/17" is defensible only post-hoc).
- **Baselines** (`baselines/`): Full Context, Dense RAG (MiniLM + cosine/FAISS), BM25, No-Context. These classes *optionally* wrap a causal LM, but no runner enables it.
- **Data** (`data/`): `spbench_builder.py` (synthetic Wikipedia-chunk benchmark) and `benchmark_loaders.py` (real loaders for FEVER/NarrativeQA/HotpotQA/MuSiQue/LongBench; a **mock RULER** explicitly labeled "Simplified mock implementation").
- **Analysis** (`analysis/`): table/figure generators. Table generator T9 (falsification matrix) is fully hardcoded; others load stored results with fabricated fallback branches.
- **Untracked stubs** (`e18`–`e21`, added 2026-09-28, one hour before this session): import pip packages that do not exist (`snapkv`, `h2o`, `pyramidkv`, `scissorhands`, `longbench`, `mamba`, `cache_augmented` as named APIs), reference data files that do not exist (`data/ruler/ruler_dataset.jsonl`, `data/kv_compare/*.txt`), and e19 sets `pred = "placeholder"` as the SemPointer prediction. **None of them has ever run; no results exist for them.** They cannot execute as written (imports fail; `experiments.` package prefix unresolved).

---

## 3. What actually executes? (verified by fresh runs)

| Experiment | What it actually does | LLM? | Real data? | Verdict |
|---|---|---|---|---|
| E01 | Dense cosine retrieval (MiniLM/mpnet) over ≤1024 template passages; query contains the target's unique `ENTITY_...` string verbatim; BM25 comparison | No | No — 8-domain synthetic templates | Executes; task is near-trivial string matching (BM25 = 1.0 at every N) |
| E02 | Builds prompts for 4 systems; "answer" = `ground_truth in prompt` substring check; EM/F1 on that | No | No — 8 hardcoded entity templates | Executes; **not QA** — retrieval-containment proxy. All four systems score 1.0 by construction |
| E03 | Tokenizes synthesized prompts, compares to `L_active` formula | No | Synthetic | Executes; genuine but a **software unit test** of the prompt builder |
| E04 | Evaluates analytic FLOP formulas; matmul timing | No | n/a | Executes; **derived numbers, not measurements** |
| E05 | `kstar_emp = kstar_theory` (line ~134, comment: "algebraic identity"); separate wall-clock matmul "latency crossover" | No | n/a | Executes; **0.00% error is tautological** |
| E06 | Latency/recall vs N | No | Synthetic | Executes |
| E07 | k ∈ [2..64] ablation | No | Synthetic | Executes but **structurally vacuous**: pointer embedding is one vector repeated k×, so retrieval and collision are *invariant* in k (stored: R@1=1.0, collision=0.1096 at every k) |
| E08 | Payload size B ablation (token counting, K* at B) | No | Synthetic | Executes (current code runs; earlier logged crash was a `torch` NameError, since fixed) |
| E09 | m ablation, retrieval-only | No | Synthetic | Executes |
| E10 | Query-length ablation | No | Synthetic | Executes |
| E11 | 2-hop retrieval: joint pointer set vs dense RAG | No | Synthetic | Executes; stored: joint F1=1.0 **and dense RAG F1=1.0, improvement = 0.0pp** |
| E12 | "Collision" = fraction of pairs with cosine > 0.95, compared to discrete birthday bound V^k | No | Synthetic | Executes; **category error** (continuous similarity vs discrete address space). "Hard" config uses near-duplicate template passages → 100% collision |
| E13 | Cosine similarity decay across Δt; recall constant 0.96875 (=31/32) at every horizon | No | Synthetic | Executes; "λ=0.0028" is a fit to embedding decay, no addressing mechanism involved |
| E14 | M_text: text vs itself → 1.0. **M_kv: hardcoded `cos=1.0, epsilon=0.0`** (runner lines 57–60). M_latent: **untrained** 2-layer MLP → ε≈0.98–1.0, **fails its own ≤0.20 gate** | No | Synthetic | Executes; two of three substrate rows are assumption or degenerate |
| E15 | Closed-form `2·n_layers·seq·n_kv_heads·d_head·2` for two sequence lengths | No | n/a | Executes; **pure arithmetic**; the "59.9×" headline is the ratio of two formula evaluations |
| E16 | `bytes = N·B·2` and `latency = bytes / measured PCIe bandwidth` | No | n/a | Executes; **derived latency**, not measured paging |
| E17 | Four "model families" are hardcoded `{name, hf_id, d}` dicts; **no model is ever loaded**; one MiniLM embedder does all retrieval; K* re-evaluated at different d | No | Synthetic | Executes; **cross-model claim unsupported by any model run** |

Fresh runs by this investigator reproduced the stored outputs (E01/E02/E03/E05/E08/E12/E14/E15), so the stored artifacts are genuine outputs of this code. The problems are in what the code measures, not in whether it was run.

---

## 4. What does the end-to-end data flow actually look like?

As implemented (M_text path, the only complete one):

```
synthetic template passage
  → PoolingPointerGenerator: SentenceTransformer embeds block → 1 dense vector, .expand(k, d)   [pointer "tokens" are one vector repeated k times]
  → PointerRegistry: {text, embedding, top-k most-frequent token IDs}
  → PointerScorer: cosine(query embedding, mean-pooled block embeddings) → top-m
  → TextResolver: dict lookup of raw text for selected ids
  → PromptBuilder: prompt = [decoded top-k frequent tokens of unselected blocks] + [full text of selected blocks] + query
  → (no LLM; either substring-containment scoring or token counting)
```

Two architectures actually exist in the code and are never connected:

1. **Retrieval architecture** (scorer): dense embedding similarity, k-agnostic.
2. **Prompt architecture** (prompt builder): k *frequent-token* strings, which influence only token counts, never retrieval.

The M_kv substrate (`KVCacheResolver.precompute_kv`) exists but is invoked by **no experiment**: no KV page is ever dereferenced into a real `generate()` call, so the paper's central mechanism — pointers addressing cached KV states during actual LLM inference — is **never executed anywhere**.

---

## 5. Which parts of SemPointer genuinely exist / are partial / are absent?

**Genuinely exist (and execute):**
- Block registry with stable IDs (`registry.py`).
- Query-conditioned dense selection, top-m (`scorer.py`).
- M_text resolution and active-prompt assembly; exact token accounting; verification of `L_active` against measured token counts (E03: 1,200/1,200 cells — REPRODUCED BY US from stored full runs).
- Analytic cost model (correct algebra; K* values 1.0150/0.9578/0.8609/0.7159/0.2132 at N=1/2/4/8/64 reproduce exactly).
- M_kv precompute path (`KVCacheResolver`) as a standalone utility; M_latent as an untrained module.
- Honest baselines: BM25, Dense RAG (retrieval parts).

**Partial:**
- Pointer generation: two incompatible notions (embedding-repeat vs top-k frequent token IDs); neither is the paper's "token-level semantic pointer generated by G_φ" as a unified mechanism; the attention-class generator exists but is unused by runners.
- E16/E15: measured PCIe bandwidth and formula constants are real; the *conclusions* are derived, not measured end-to-end.
- Statistical machinery (bootstrap CI, McNemar) exists and is invoked, but on containment-proxy outcomes.

**Absent:**
- Any LLM inference in the entire experimental path (no `generate()` call is ever made in a stored run).
- Any execution on FEVER, NarrativeQA, HotpotQA, MuSiQue, RULER, or LongBench (loaders exist; zero call sites; zero results).
- "VI-NIAH" benchmark: no implementation beyond the fake filler in `load_ruler_niah` and no results.
- KV-substrate end-to-end dereferencing (pointers → KV pages → real inference).
- Trained M_latent (the module is random-init; nothing is ever trained).
- Cross-model execution (models never loaded).
- Per-run environment/device metadata in result JSONs (only `timestamp`).
- The four e18–e21 public-benchmark pipelines (stubs only).

---

## 6. Which datasets are actually used?

| Dataset | Status |
|---|---|
| Synthetic template passages (8 domains, unique `ENTITY_n` IDs) — `runners/e01_retrieval.py:33-70` | Only "dataset" used by E01/E06/E07/E12/E13/E14/E17 |
| Synthetic entity QA (8 hardcoded entities) — `runners/e02_qa.py:46-105` | Only "dataset" used by E02/E09/E10/E11 |
| "SPBench" (Wikipedia chunks; **query = first 5 words of target passage, answer = words 5–15 of the same passage**) — `data/spbench_builder.py:148-170` | Builder exists; no stored result demonstrates a completed build (no `spbench.jsonl` present); degenerate task design regardless |
| FEVER / NarrativeQA / HotpotQA / MuSiQue | Real loaders; **never called**; paper claims they were evaluated (C10) |
| RULER | Only a mock ("This is a long context." × n, needle/hardcoded answer); never called |
| LongBench (THUDM/LongBench) | Real loader; never called (the e19 stub uses a nonexistent `longbench` pip API instead) |
| LoCoMo | Not present anywhere |

**No public benchmark dataset has ever been used in any experiment in this repository.**

---

## 7. Classification of every load-bearing number

Categories: **REPRODUCED BY US** (fresh run matches) / **DERIVED FROM SOURCE** (closed-form from stated formula; correct algebra, but not a measurement) / **SOURCE-DERIVED** (independently documented external fact) / **UNVERIFIED** (no traceable source in repo) / **CONTRADICTED** (repo's own data says otherwise) / **INVALID** (tautological, fabricated, or category error).

| Claim (paper/README) | Stored evidence | Classification |
|---|---|---|
| `L_active` formula, zero residual, 1,200 cells (C6) | `results/e03/run_1790187434.json` etc.: 1200/1200; my fresh run matches | REPRODUCED (but it is a unit test of the prompt builder, as the paper itself concedes) |
| K*(N)=1.0150/0.9578/0.8609/0.7159/0.2132 (C3) | Cost model algebra; my fresh E05 run matches | DERIVED FROM SOURCE |
| "K* confirmed with 0.00% relative error" / "Profiler FLOPs match" (C7) | `kstar_emp = kstar_theory` (e05_kstar.py ~line 134); `torch.profiler` has zero call sites | **INVALID** (tautology; profiler claim unsupported by code) |
| 59.9× KV savings at N=1024; 79/520, 103/2056, 199/8200 MB (C4) | E15 formula; my fresh run reproduces exactly | DERIVED FROM SOURCE (not measured; "empirical validation" framing in Fig. 4 caption is misleading) |
| E01 Recall@1=1.000 @N=8 | Stored + fresh: 1.0 | REPRODUCED (degenerate task) |
| E01 "0.969±0.031 @N=32" | Stored: 1.000 | **CONTRADICTED** |
| E01 "outperforming BM25 (0.940@N=32 / 0.780±0.060@N=256)" | Stored: BM25 = 1.000 at every N (beats SemPointer's 0.92/0.90 at N≥128) | **CONTRADICTED** |
| E01 Dense RAG 0.980@N=32 / 0.890@N=256 | E01 never runs Dense RAG | **UNVERIFIED → INVALID** (no source) |
| "across 5 random seeds" | Runner uses fixed seed=42; no seed sweep code or results | **UNVERIFIED** |
| "±0.031/±0.045/±0.060/±0.021" intervals | Not derivable from stored artifacts (bootstrap CIs exist but differ) | **UNVERIFIED** |
| "5.50±0.21 ms on CUDA @N=32" | Stored E06: 5.43 ms (CPU/CUDA mixed) | UNVERIFIED (point estimate plausible; interval and CUDA attribution unsupported) |
| "R² = 0.962" linear latency scaling | No stored computation of R² found | **UNVERIFIED** |
| E02 F1=1.0, gap 0.0pp, savings 56.0–68.6% | Stored + fresh match | REPRODUCED (but it is substring-containment, **not QA**; "NarrativeQA" never used) |
| E11 "0.720±0.040 for individual resolution", "2-hop HotpotQA" | Stored: joint F1=1.0, dense RAG F1=1.0, improvement 0.0pp; no HotpotQA; no 0.720 anywhere | **CONTRADICTED / INVALID** |
| H1 "0.940±0.028 (VI-NIAH); MRR=0.984" | VI-NIAH does not exist; stored MRR=1.000 | **INVALID / CONTRADICTED** |
| H5 "k=2: 0.840, k=4: 0.920, k≥8: 0.969" | Stored E07: R@1 = 1.0 at every k | **CONTRADICTED** |
| E07 "k=2 collisions 10.96% → recall 0.840" | Collision 0.1096 **identical at every k** (embedding invariant in k); recall 1.0 at k=2 | **CONTRADICTED** (causal story fabricated; k-ablation is vacuous) |
| E12 "empirical collision rates strictly bounded by theoretical birthday model" (README) | Stored/fresh: empirical 6.7–100% vs bound ~1e-8; "hard" config = near-duplicate synthetic passages | **CONTRADICTED / INVALID** (incommensurable quantities) |
| E13 λ=0.0028; Recall@1=0.969 at turn 100 | Stored: λ=0.002804; recall constant 0.96875 at all horizons | REPRODUCED point estimates (±0.025 invented; decay of *similarity*, no addressing mechanism tested) |
| E14 M_kv lossless ε=0.000 | Hardcoded constants in runner; KVCacheResolver never invoked | **INVALID** (assumption presented as measurement) |
| E14 M_latent ε=0.980 "confirms latent compression induces hallucinations" | Untrained random MLP; my fresh run ε=1.003; no text reconstruction, no hallucination metric | REPRODUCED number, **INVALID interpretation** |
| E16 "318.09 ms → 4.97 ms" | Stored = bytes ÷ measured PCIe bandwidth (derived) | DERIVED FROM SOURCE (not measured paging) |
| E17 "generalized across all models" | No model loaded; same embedder 4×; theory re-evaluated at different d | **INVALID as cross-model evidence** |
| T1 model metadata (Qwen2.5-7B 28L/4KV heads d=3584; Llama-3.1-8B 32L/8KV d=4096; Gemma-2-9B 42L/8KV d=3584; Phi-3.5-mini 32L/32KV d=3072) | Matches published model cards | SOURCE-DERIVED |
| "17/17 passed" (README) | Last master log: 15/17; e08/e15 passed individually afterwards | MISLEADING as stated |
| "six diagnostic benchmarks" (abstract) vs "4 established + 2 self-constructed" (Sec. VII-A) | Only synthetic suites ran | **CONTRADICTED** internally and empirically |

---

## 8. Answers to the mandate's questions, in one place

- **Which experiments genuinely execute?** All 17 runners execute (16 cleanly; e15's CLI rejects `--device` in the orchestrator context). All 17 produce genuine artifacts of *what they actually compute* — which for E02/E05/E14/E15/E16/E17 is not what the paper says it computes.
- **Which datasets are actually used?** Only synthetic template corpora generated inside the runners. Zero public datasets in any stored result.
- **Which numbers can actually be reproduced?** Every stored point estimate I tested (E01, E02, E03, E05, E08, E12, E14, E15) reproduces from current code. Reproducibility of the *artifacts* is not in doubt; their *meaning* is.
- **Which claims are unsupported?** All ± intervals; "5 seeds"; Dense RAG E01 numbers; R²=0.962; VI-NIAH; NarrativeQA/FEVER/HotpotQA/MuSiQue evaluations; "Profiler FLOP consistency"; "cross-model generalization"; "hallucination" claims; e18–e21 capabilities.
- **Which claims are contradicted?** BM25 comparisons in E01/H1; E01 R@1=0.969@N=32; H5 k-sweep values; E07 k=2 story; E11 "0.720 individual resolution"; E12 "bounded by birthday model"; "six diagnostic benchmarks"; README 17/17 (as of the last master log).
- **Which claims are invalid as framed?** "0.00% K* error" (tautology); E02 "QA accuracy" (containment proxy); E14 M_kv row (hardcoded); E15/E16 "empirical validation" (formula evaluation); "59.9×" as a measured saving.

---

## 9. What must be rebuilt (in priority order)

1. **A real inference path.** The single most important gap: SemPointer must be evaluated with an actual causal LM (e.g., Qwen2.5-7B-Instruct, fits 8 GB at 4-bit; document quantization), where the active prompt is genuinely generated from. Without this, no downstream-task claim is possible.
2. **M_kv end-to-end.** Wire `KVCacheResolver` into actual generation (resolve selected blocks' KV pages into a real forward pass), so the cached-regime claims become measurements rather than formulas. E15-style formulas may remain as theory, clearly labeled derived.
3. **Delete or rebuild e18–e21 stubs.** They import hallucinated APIs and cannot run. Replace with real harnesses against NVIDIA/RULER, THUDM/LongBench, snap-research/locomo, and kvpress/H2O/PyramidKV-class implementations that actually execute.
4. **Real benchmark data.** P1 RULER, P2 LongBench, P3 LoCoMo, P4 KV-baseline comparison — with documented provenance (dataset, revision, split, metric) and honest metrics computed from model outputs.
5. **Retire or rework degenerate internals:** the `expand(k,d)` pointer (makes k meaningless), the query=prefix SPBench design, the substring "QA" scorer, the cosine-vs-birthday collision comparison, the hardcoded T9 rows, and the fabricated table fallbacks.
6. **Paper corrections (minimum set):** E01 baseline numbers, H1/H5 rows, E11 claim, E14 M_kv row framing, "Profiler" claim, "six diagnostic benchmarks", four-model claim, dataset claims (C10), E02 "QA" framing, and any language implying measurements where formulas were evaluated.
7. **Provenance plumbing:** record device/dtype/env per run; single runner with config; separate `results/` (current) from `results_archive/` (historical, unverified-for-paper-purposes).

---

## 10. Investigator's fresh-run artifacts created during this audit

New timestamped files (no stored artifact was modified or deleted):
`experiments/results/e01/run_1790614808.json`, `e02/run_1790614832.json`, `e03/run_1790614774.json`, `e05/run_1790614862.json`, `e08/run_1790614908.json`, `e12/run_1790615088.json`, `e14/run_1790614859.json`, `e15/run_1790614884.json`.

---

## 11. Bottom line

The repository contains a real, executing, reproducible **simulation harness for the paper's algebra**, plus a genuine embedding-retrieval stack with real baselines — and nothing beyond that. Not a single experiment involved a language model, a public dataset, or a measured KV-cache mechanism. The paper's evaluation section (C8–C17) is supported only where the claim is algebraic (C2, C3, the E03 unit test, E15/E16 formulas) and is contradicted or unsupported in every claim that involves baselines, seeds, established benchmarks, model families, or measured system behavior. The architecture gap is specific and buildable: retrieval selection exists; pointer-mediated *inference* (text or KV dereferencing into a real LLM) does not. Per the mandate, no implementation changes have been made; this document establishes reality first.
