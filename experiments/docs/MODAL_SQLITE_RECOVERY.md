# Modal SQLite recovery — 2026-11-10

The original BF16 run `d13722f18a626529a306` completed all 216 development evaluations, then its fourth worker failed before the cost pilot. `reviewer_suite.records()` used SQLite's transaction context manager, which does not close the connection. A reused Modal container refused `volume.reload()` because `dev_cautious/results.sqlite` remained open. This happened before the worker error handler, leaving its displayed status stale.

The runtime fix wraps the read-only SQLite connection in `contextlib.closing`. The new regression test failed before this change and passes after it. No inference settings or selected examples changed.

The new frozen run is `8c2749692d31cff34a48`, in `modal_runs/ten_qwen35_recovered_v1/bundle`. Only `reviewer_suite.py` differs between runtime snapshots. All 3,000 main samples, 36 independent development samples, model revision, BF16/batch-two setting and precision evidence are identical. The original frozen bundles remain intact.

The one-time `modal_runs/ten_qwen35_recovered_v1/recover_modal.py` operation confirmed the source worker was terminal before claiming the global lease, reserved $0.15 against the original shared $25 resource allowance (`8ef26bfd68509bae3b6b`), and copied the checkpoint on Modal's CPU. The new results' configs record the new bundle/code hashes, while raw SQLite databases, events and environment records were verified byte-for-byte against the original. Only copied threshold file paths were remapped. The selected thresholds and original raw evidence remain unchanged. Provenance is stored remotely in `results/checkpoint_recovery.json`.

The CPU recovery exercised the fixed reader on every copied development database, committed, and successfully performed a real Modal volume refresh before spawning GPU work. The existing cached model is reused; the 216 completed development evaluations are not rerun. The launcher selector now points to the recovery bundle. The previous stale state was marked `paused_error`, with its original state backed up as `state_before_recovery.json`.

Detached app: https://modal.com/apps/viksitg427/main/ap-CW3ngCs8ieilDcygHkwh2h

## Verification

- The new resource-lifecycle regression failed before the fix and passed after it.
- All 25 focused reviewer/orchestration tests passed.
- The broader relevant evaluation test selection passed: 75 tests, one pre-existing deprecation warning.
- Actual Modal recovery returned `restarted`, `retained_evaluations: 216`, and `volume_refresh_verified: true`.
- The complete offline non-GPU suite returned 108 passed, 7 failed, 21 errors, and 7 deselected. Legacy model/tokenizer caches are missing; the previously observed paper-assets assertion also remains. Full traces: `modal_runs/ten_qwen35_recovered_v1/full_tests.log`.

The complete suite's failing/error test names are retained below, rather than implying the entire repository is green:

- `FAILED tests/test_core.py::test_seed_stability - OSError: We couldn't connect...`
- `FAILED tests/test_core.py::test_chunk_text_respects_block_size - OSError: We ...`
- `FAILED tests/test_core.py::test_active_tokens_scale_with_theory - OSError: We...`
- `FAILED tests/test_paper_assets.py::test_partial_stage_preserves_scored_rows_and_marks_missing_examples`
- `FAILED tests/test_pointers_selector.py::test_selector_top1[dense] - OSError: ...`
- `FAILED tests/test_pointers_selector.py::test_selector_top1[hybrid] - OSError:...`
- `FAILED tests/test_pointers_selector.py::test_selector_attention_modes_and_embedder_reuse`
- `ERROR tests/test_benchmark_harness.py::test_bm25_budget_mode_respects_budget`
- `ERROR tests/test_benchmark_harness.py::test_hybrid_baseline_matches_manual_rrf`
- `ERROR tests/test_core.py::test_pointer_length_matches_k - OSError: We couldn'...`
- `ERROR tests/test_core.py::test_k_changes_the_address_but_prefix_is_stable - O...`
- `ERROR tests/test_core.py::test_different_texts_get_different_addresses - OSEr...`
- `ERROR tests/test_core.py::test_token_ids_inside_reserved_range - OSError: We ...`
- `ERROR tests/test_core.py::test_dense_pointer_not_repeated - OSError: We could...`
- `ERROR tests/test_core.py::test_batch_consistency - OSError: We couldn't conne...`
- `ERROR tests/test_core.py::test_registry_roundtrip - OSError: We couldn't conn...`
- `ERROR tests/test_core.py::test_scorer_top_m_ordering - OSError: We couldn't c...`
- `ERROR tests/test_kv_substrate.py::test_rotation_exactness - OSError: We could...`
- `ERROR tests/test_kv_substrate.py::test_splice_single_block_matches_plain_greedy`
- `ERROR tests/test_kv_substrate.py::test_factual_recall_from_spliced_block - OS...`
- `ERROR tests/test_kv_substrate.py::test_stored_bytes_and_offload - OSError: We...`
- `ERROR tests/test_pointers_selector.py::test_keyword_pointers_len_readable_distinctive`
- `ERROR tests/test_pointers_selector.py::test_keyword_pointers_deterministic - ...`
- `ERROR tests/test_pointers_selector.py::test_keyword_prefers_block_distinctive_terms`
- `ERROR tests/test_pointers_selector.py::test_lead_and_none - OSError: We could...`
- `ERROR tests/test_pointers_selector.py::test_random_id - OSError: We couldn't ...`
- `ERROR tests/test_pointers_selector.py::test_build_index_ids_spans - OSError: ...`
- `ERROR tests/test_pointers_selector.py::test_pointer_attention_scores_qwen_cpu`
