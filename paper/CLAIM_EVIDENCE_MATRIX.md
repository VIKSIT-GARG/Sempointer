# CLAIM_EVIDENCE_MATRIX.md

Every major claim of the current paper draft (`sempointer_ieee.tex`) mapped to evidence. Status values: **SUPPORTED** (measured, traceable) / **DERIVED** (correct algebra, labeled as theory) / **UNIT** (software-level test only) / **NOT_TESTED** (no execution exists yet) / **CONTRADICTED** (repo data says otherwise — see INDEPENDENT_RECONSTRUCTION.md §7).

**Rule:** the paper may be updated only when NOT_TESTED rows become SUPPORTED by real artifacts in `experiments/results/`.

| # | Claim | Code path | Dataset | Raw artifact | Status |
|---|---|---|---|---|---|
| 1 | Active sequence compresses from N·B to (N−m)k + m·B_res + L_Q | `sempointer/prompt_builder.py`, `sempointer/pipeline.py` | any | `results/*/predictions.jsonl` (active_tokens) + `tests/test_core.py` | UNIT (formula verified on constructed prompts); SUPPORTED-on-benchmarks pending user run |
| 2 | Pointer generation produces compact token-level addresses | `sempointer/pointer_generator.py::SemanticIDPointerGenerator` | — | `tests/test_core.py` (k-dependence, V^k space) | UNIT (legacy expand(k,d) bug fixed; k now changes the address) |
| 3 | Query-conditioned selection retrieves the correct memory | `sempointer/scorer.py` + `pipeline.py` | RULER haystacks | `results/ruler_*` | NOT_TESTED (awaiting user run; legacy E01 evidence retired as degenerate) |
| 4 | Downstream task fidelity ≈ full context | `run_benchmark.py` systems | RULER/LongBench/LoCoMo | `results/{ruler,longbench,locomo}_*/` | NOT_TESTED (legacy E02 was substring containment — retired) |
| 5 | Active-token savings vs full context | `run_benchmark.py` | same | same | NOT_TESTED |
| 6 | Latency / peak-VRAM reduction vs full context | `engine/llm.py` instrumentation | same | same | NOT_TESTED (legacy E15/E16 were formulas — now labeled DERIVED) |
| 7 | SemPointer competitive with established KV compression at equal budget | `baselines/kvpress_baseline.py` (SnapKV/PyramidKV/Knorm) | RULER | `results/kv_compare_*` | NOT_TESTED |
| 8 | Cross-model generalization (4 model families) | `run_benchmark.py --model` | — | — | NOT_TESTED; only models that actually run will be claimed (8GB constraint documented) |
| 9 | K* crossover algebra; single-query viability N≥2 | `sempointer/cost_model.py` | — | analytical | DERIVED (correct algebra; "0.00% empirical error" claim retired as tautology) |
| 10 | KV-memory footprint ratio B/k (59.9× at N=1024) | `sempointer/cost_model.py` + `engine/llm.py::prefill_footprint` | — | `results/kv_compare_*` (measured) + formula (derived) | DERIVED as theory; MEASURED version pending |
| 11 | Collision behavior bounded by birthday model | `sempointer/pointer_generator.py` (V^k address space) | — | — | NOT_TESTED (legacy E12 comparison was a category error — retired) |
| 12 | M_latent substrate | `sempointer/resolver.py::LatentResolver` (untrained) | — | — | NOT_TESTED; no claim may be made unless the module is trained; legacy "confirms hallucinations" claim retired |
| 13 | Robustness to registry size N (graceful degradation) | `run_benchmark.py` (context-length × N sweep) | RULER | pending | NOT_TESTED |
| 14 | "Evaluated on NarrativeQA/FEVER/HotpotQA/MuSiQue" | — | — | — | **CONTRADICTED** (legacy: never executed). Replaced by P1–P3 claims after runs |
| 15 | "0.00% relative error, Profiler FLOPs match" | — | — | — | **RETIRED** (tautology in legacy code; `torch.profiler` never invoked) |
| 16 | "6.6×/19.9×/41.2×/59.9× KV savings" as empirical results | — | — | — | **RECLASSIFIED** as DERIVED (formula); measured version pending in P4 |

## Evidence that already exists (from the implementation phase, MEASURED)

- Core end-to-end path executes: query → semantic pointer selection → M_text resolution → real Qwen2.5-1.5B generation → correct answer; retrieval selected the block containing the answer (2026-09-28 implementation smoke test, console-verified; will be reproduced in `tests/` marked gpu).
