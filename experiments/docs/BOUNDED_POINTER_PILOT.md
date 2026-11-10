# Bounded pointer local pilot

Run `7947e6bdc044900a222d` tests a bounded, query-relevant pointer index on the local RTX 5060 using pinned Qwen3.5-4B NF4, batch one. It is a small exploratory experiment, not a completed claim of improvement. The completed 1,000-question run remains unchanged.

## Method and comparisons

The registry remains outside the prompt. The bounded variant includes descriptors only from the top 16 BM25-ranked candidates outside the route's selected payloads. A development-selected allowance of at most 128 or 256 tokens bounds descriptor overhead. Question text, system instruction, labels, pointer section, payload and native non-thinking chat framing all count toward the whole-input cap.

The independent whole-input caps are 1,024, 2,048 and 4,096. Resolved evidence fits the remaining budget; a partially included final block is recorded as `partial_block_id`. Original source files are unchanged. Pointer allowance zero reproduces the adaptive no-index prompt. Original SemPointer is rendered exactly as before and skipped if its complete prompt exceeds the current cap. A question plus framing that is itself too long is explicitly skipped, never shortened.

Development arms: `bounded_128`, `bounded_256`, `adaptive_no_index`, `bm25_budget`, `original_sempointer`.

Test arms: `bounded_pointer` using the frozen choice for that budget, `adaptive_no_index`, `bm25_budget`, `original_sempointer`.

`bm25_budget` fills the external budget with ranked evidence, including a possible final partial block. It is a stronger variable-passage budget control than a fixed four-block baseline. All arms have the same upper input cap; their actual token consumption can differ. The no-index arm shares the adaptive candidate route, but can allocate to payload space that the bounded arm spends on descriptors, so the comparison measures that allocation trade-off rather than holding payload text exactly constant.

## Splits and selection

There are 30 development questions (3 per family) and 100 test questions (10 per family). All 130 question IDs exclude the completed local evaluation and the prior Modal cost-pilot IDs. The new development/test group IDs are disjoint. Strata are hash-ordered to avoid alphabetical preference when a small pilot cannot represent all subcategories. No scores enter sampling.

New questions can share source documents with older experiments. The frozen manifest reports 6 development questions and 25 test questions with group IDs already seen in the previous local run. This especially affects LoCoMo; none of its test source groups is wholly new. The pilot cannot establish generalization to unseen conversations solely from question freshness.

For each input budget, the pointer allowance is selected only from development results: maximize question-wise wins minus losses against budgeted BM25, penalizing a skipped candidate when BM25 scores. Ties prefer fewer candidate skips, fewer actual prompt tokens, then the smaller allowance. This utility is a configuration rule, not an average accuracy over heterogeneous metrics. Choices are frozen in `results/pointer_selection.json` before any fresh test stage. The previous independently chosen routing thresholds remain fixed.

There are 450 scheduled development system–question–budget evaluations and 1,200 test evaluations, including explicit skips. No extra model calls select pointers. No model training or cloud execution is performed.

## Commands and saved evidence

From `experiments/`:

```bash
bash run_bounded_suite.sh status
tail -f results/bounded_pointer_pilot_v1/worker.log
bash run_bounded_suite.sh stop
bash run_bounded_suite.sh start
```

The local process runs in the background. Keep the laptop plugged in and awake. `start` resumes the frozen input/code snapshot and skips finished evaluations. `stop` waits for a safe checkpoint. One file lock prevents duplicate workers for this output directory. This launcher does not touch the older local or Modal run selectors.

Results live in `results/bounded_pointer_pilot_v1`. Each of six stages retains SQLite records, raw events, exported predictions and frozen config. Progress checkpoints are written after each question. The completed or paused run automatically produces `results/REPORT.md`, `summary.json`, per-task metrics and paired confidence intervals. An inference error pauses the experiment rather than silently dropping failed calls. Scores of unrelated metrics are not averaged into a single score.

Generation timing includes prefill and answer decoding. Per-example shared ingestion and complete prompt-suite construction are recorded separately. Selection timing does not establish a complete serving-speed comparison; do not attribute shared setup to individual arms or call generation-only savings end-to-end savings.

## Validation and review

- Six new tests cover whole prompt/pointer caps, zero-pointer ablation, overflow without question truncation, fresh and group-disjoint splits, development-only choice, the real runner hook, custom-reference paired summaries, and resume without repeated calls.
- 84 relevant evaluation tests passed (one existing deprecation warning).
- Actual pinned Qwen tokenizer checks passed on RULER, LongMemEval and LongBench v2 at all three budgets. Rendered original prompts matched the existing implementation byte-for-byte. LongMemEval's checked legacy prompt had 6,696 tokens; bounded prompts stayed at or below the configured total cap with descriptor overhead below 128/256.
- All 65 hashed prior result/provenance files remained unchanged. An original runner snapshot is retained in `reference_code/before_bounded_pointer_20261110`.
- A real CUDA worker produced and saved bounded development scores; process submission alone was not treated as execution proof.
- The full offline suite returned 117 passed, 7 failed, 21 errors and 7 deselected. The same missing legacy model/tokenizer caches and paper-assets assertion remain; full traces and failing names are in `results/bounded_pointer_pilot_v1/full_tests.log`.
- Final review was an author self-review because sub-agents are prohibited in this side conversation. It is not an independent review.

The causal mechanism for any accuracy change remains a hypothesis. Preserve negative/null results and do not tune against the fresh test results after seeing them. Adaptive retrieval already has relevant prior work; a bounded-index implementation by itself does not establish publication novelty.
