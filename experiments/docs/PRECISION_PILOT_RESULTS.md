# Measured precision and batching selection — 2026-11-10

Selected configuration: **Qwen3.5-4B BF16, native inference batches of two**.
The complete 3,000-question sample and 36-question development set are unchanged.
The original NF4 bundle and all 90 inventoried historical result files are preserved.

## Real measurements

Both precisions used the same NVIDIA L40S, pinned checkpoint revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, greedy non-thinking prompts and
Transformers 5.2.0. Actual optimized kernels were installed using `fla-core==0.6.0`
and `causal-conv1d==1.7.0`. The full FLA extension was replaced after an actual
image-build dependency conflict; it requires Transformers >=5.6.2.

Precision timing forced exactly 32 decoded tokens for six prompts spanning
prompt lengths, warmed each shape, and used NF4/BF16/BF16/NF4 order. BF16 had a
median paired speedup of **1.292x**, approximately **22.6% less time**, with all
six prompts meeting the >=10% threshold. Load and compilation time were excluded
from those equal-work timings. Raw normal answers were also retained: 8 of 22
texts differed between NF4 and BF16. This is not a claim of improved benchmark
quality, and no correctness score selected the precision.

Initial cold batching timings included compilation and are retained as reference.
A separate audit warmed each serial/batch shape and ran two measured passes over
all 22 prompts, preserving original output caps and grouping equal caps together.

| Mode | Median time for 22 prompts | Compared with serial | Answer agreement |
| --- | ---: | ---: | --- |
| BF16 serial | 19.438 s | baseline | reference |
| BF16 batch 2 | 18.339 s | 1.060x throughput; 5.7% less time | all 22 match in both passes |
| BF16 batch 4 | 16.715 s | 1.163x throughput | one prompt differs; rejected |

Because the user requested any advantage, the final batch decision accepts a
>=5% advantage in **each** measured pass with matching outputs, rather than the
original >=10% target. Batch-2 passes were 18.423 s and 18.255 s. The original
audit's stricter decision remains intact; `comparison_warmed.json` explicitly
records the applied rule, original cold results, warmed audit and final decision.
The modest observed batching benefit is not a full-run speedup guarantee.
Batched agreement on 22 prompts does not guarantee full-test bitwise parity.

## Configuration and evidence

- Original immutable bundle: `modal_runs/ten_qwen35_v1/bundle/`, run `8ef26bfd68509bae3b6b`.
- Selected immutable bundle: `modal_runs/ten_qwen35_measured_v1/bundle/`, run `d13722f18a626529a306`.
- Selected configuration evidence: `precision_comparison.json` inside that bundle.
- Prompt-only pilot input: `modal_runs/precision_pilot/input_v2/pilot.json`.
- Original paired precision result: `modal_runs/precision_pilot/results/comparison.json`.
- Warmed batching result: `modal_runs/precision_pilot/results/batch_audit.json`.
- Applied combined result: `modal_runs/precision_pilot/results/comparison_warmed.json`.
- Machine-readable summary: `modal_runs/precision_pilot/results/summary.json`.
- Before-change source snapshot: `reference_code/before_precision_pilot_20261110/`.

The launcher selects the derived bundle via `modal_runs/selected_inference_bundle.json`.
All six test systems share BF16 and a requested batch size of two. Development
runs remain serial. A generation OOM retries a smaller batch; no source/history,
output cap, sample count or reference is reduced. Per-request latency, TTFT and
peak memory are omitted for batched rows. Shared timing, effective batch size,
padding, memory and completed-generation throughput are recorded separately.

## Cost discipline

The shared resource allowance records **$1.577369**: $1 reserved for CPU setup,
image building and reporting, plus $0.265820 and $0.311549 for the two settled GPU
jobs including startup/idle headroom. **$23.422631 remains** in the $25 allowance.
These are conservative allowance values, not a verified provider invoice. The
new run inherits the same account, so precision spending is not reset. Workspace
usage <=$25 and net spend $0 remain the provider-side guards as confirmed by the
user's original launch flag. The server settings were not read or changed by code.

The pending original NF4 upload was stopped before inference. After selection,
the existing start command was resumed using the derived bundle. Its detached
submission is recorded in `modal_runs/ten_qwen35_measured_v1/launch.log`. The transfer of thousands of individual files was replaced with one 212.3 MiB
compressed archive of the same 576 MiB bundle. A CPU helper verified every example,
runtime file and evidence checksum before publishing the input directory, and
reserved a further $0.05 in the same resource account. The main cost pilot still
determines whether all remaining questions fit the allowance.

## Verification

Focused offline checks: **74 passed**, one dependency warning. They include tiny
real hybrid-cache models, padded native batch parsing, exact prompt accounting,
OOM downshift, shared allowance inheritance, unchanged samples/configuration
checksums, checkpoint recovery and raw-measurement decision rules.
The real hardware pilot independently exercised actual full-size BF16/NF4 models.
Shell syntax, SDK import and whitespace checks passed. Historical result sizes
and modification timestamps remained unchanged for all 90 inventoried files.

The broader offline non-GPU suite had **107 passed, 7 failed, 21 errors,
7 deselected**. Missing cached older model/tokenizer fixtures account for the
model-loading failures. The unrelated paper-table assertion
`test_partial_stage_preserves_scored_rows_and_marks_missing_examples` also fails;
its implementation and tests were not changed. Full tracebacks are retained in
`modal_runs/precision_pilot/full_tests.log`.

Affected test names from that run:

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
