# Local preparation optimization verification — 2026-11-10

This change accelerates local dataset preparation. No model inference, cloud build,
cloud GPU run, or new empirical benchmark score was produced.

## Hardware and implementation

The inspected laptop has an AMD Ryzen 7 250 (8 cores / 16 threads), 30 GiB RAM,
and an RTX 5060 Laptop GPU with 8 GiB VRAM. Preparation defaults to eight
concurrent family workers sharing a 16-thread Rust tokenization pool. It uses
bounded batches, a durable content-keyed SQLite count cache, and per-family
checkpoints. Main example materialization is also concurrent by family. The
installed tokenizer has no CUDA execution path; no GPU replacement was introduced.

The original per-family source ordering, hash-priority heap, eligibility guards,
deduplication and balanced quotas are retained. Exact source counts use the pinned
Qwen tokenizer with truncation and padding disabled. Cache keys include the effective
tokenizer pipeline, and family checkpoints include source metadata, code and settings.
RULER generation stays in one family worker to preserve shared RNG behavior.

## Measured performance

A cold scan of 600 real LongBench questions across 2WikiMQA, HotpotQA and QASPER
selected the same 300 rows, with identical IDs, token counts and context hashes:

| Preparation scan | Seconds |
| --- | ---: |
| Previous serial path | 11.077 |
| Batched path, cold cache | 2.067 |

The measured improvement for that scan was 5.36x. This is not an estimate of the
entire ten-family scan or Modal inference. A subsequent completed-family cache
lookup took under 1 ms; it did not reread the source. The shared process peak RSS
for this single-family comparison was 1,333 MiB, not a measurement of the full
8-worker peak. Raw timings are in `modal_runs/preparation_validation/speed_report.json`.
A smaller 48-context tokenization-only probe measured 1.675 s versus 0.143 s.

## Verification

The focused command passed **66 tests**, with one pre-existing dependency warning:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 warden/bin/python -m pytest tests/test_preparation.py tests/test_modal_suite.py tests/test_qwen35.py tests/test_all_datasets.py tests/test_reviewer_suite.py tests/test_v2_router.py -q --tb=short --disable-warnings
```

Checks cover persistent cache reuse, concurrent batch ordering, configuration
invalidation, actual pinned Qwen Unicode/special-token/long-text counts, both
sides of the 262,144-token cutoff, serial/batched selection parity, multi-category
balance, completed-family restart, full frozen-bundle preparation and reuse.
`bash -n run_modal_suite.sh`, isolated Modal SDK import without RPC, and
`git diff --check` also passed. All 90 inventoried historical result files kept
their recorded sizes and modification timestamps.

The full offline non-GPU suite completed with **99 passed, 7 failed, 21 errors,
7 deselected**. Most failures/errors require uncached older tokenizers/models
(Qwen2.5, GPT-2, MiniLM or other legacy fixtures). The separate
`test_partial_stage_preserves_scored_rows_and_marks_missing_examples` assertion
expects a numeric partial-score marker from the existing paper-table builder.
That builder and its tests were not modified. These unrelated failures were not
fixed as part of preparation optimization. Full tracebacks remain in
`modal_runs/preparation_full_test.log`.

Every failing/error test from that run:

- `tests/test_core.py::test_seed_stability`
- `tests/test_core.py::test_chunk_text_respects_block_size`
- `tests/test_core.py::test_active_tokens_scale_with_theory`
- `tests/test_paper_assets.py::test_partial_stage_preserves_scored_rows_and_marks_missing_examples`
- `tests/test_pointers_selector.py::test_selector_top1[dense]`
- `tests/test_pointers_selector.py::test_selector_top1[hybrid]`
- `tests/test_pointers_selector.py::test_selector_attention_modes_and_embedder_reuse`
- `tests/test_benchmark_harness.py::test_bm25_budget_mode_respects_budget`
- `tests/test_benchmark_harness.py::test_hybrid_baseline_matches_manual_rrf`
- `tests/test_core.py::test_pointer_length_matches_k`
- `tests/test_core.py::test_k_changes_the_address_but_prefix_is_stable`
- `tests/test_core.py::test_different_texts_get_different_addresses`
- `tests/test_core.py::test_token_ids_inside_reserved_range`
- `tests/test_core.py::test_dense_pointer_not_repeated`
- `tests/test_core.py::test_batch_consistency`
- `tests/test_core.py::test_registry_roundtrip`
- `tests/test_core.py::test_scorer_top_m_ordering`
- `tests/test_kv_substrate.py::test_rotation_exactness`
- `tests/test_kv_substrate.py::test_splice_single_block_matches_plain_greedy`
- `tests/test_kv_substrate.py::test_factual_recall_from_spliced_block`
- `tests/test_kv_substrate.py::test_stored_bytes_and_offload`
- `tests/test_pointers_selector.py::test_keyword_pointers_len_readable_distinctive`
- `tests/test_pointers_selector.py::test_keyword_pointers_deterministic`
- `tests/test_pointers_selector.py::test_keyword_prefers_block_distinctive_terms`
- `tests/test_pointers_selector.py::test_lead_and_none`
- `tests/test_pointers_selector.py::test_random_id`
- `tests/test_pointers_selector.py::test_build_index_ids_spans`
- `tests/test_pointers_selector.py::test_pointer_attention_scores_qwen_cpu`

## Preservation and operation

Original preparation, launcher and guide copies with SHA256 checksums are in
`reference_code/preparation_before_parallel_20261110/`. Existing benchmark results
and sealed bundles were not rewritten. No preparation process or cloud run was
started on the user's behalf. The normal launcher automatically uses the new
CPU settings; see `MODAL_TEN_BENCHMARKS.md` for overrides and budget prerequisites.
