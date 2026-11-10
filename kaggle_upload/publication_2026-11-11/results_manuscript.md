# Resource-Efficient Evidence Access: Kaggle Evaluation

> Draft evidence status: complete combined-report members were read from an incomplete archive download. All displayed counts reconcile with the saved metrics, but full raw-database validation remains pending. This T4/FP16 run is separate from the bounded-interface study in the existing manuscript.

## Supported empirical claim

On 622 questions scored by both SemPointer and full context, SemPointer used 76.7% fewer active input tokens and 61.3% lower reported per-query latency. Across the same 2,400 questions as matched-budget BM25, it used essentially the same mean active input budget and recorded 8.9% lower reported latency. These measurements characterize an efficient evidence-access operating point for the evaluated checkpoint and hardware.

## Evaluation setting

This supplement reports frozen run 8ef26bfd68509bae3b6b using Qwen3.5-4B (revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a), NF4 weights, FP16 compute, batch size one, and one independent worker on each of two NVIDIA T4 GPUs. Eight 300-question dataset subsets are complete: BABILong, InfiniteBench, L-Eval, LoCoMo, LongBench, LongBench v2, LongMemEval, and LooGLE. Their first five systems score the same 2,400 selected questions. The archived snapshot records 13,055 scored outcomes and 1,798 skips across all ten selected families; RULER and SCROLLS have partial coverage. The planned 3,000 questions produce up to 18,000 system-question outcomes, rather than 18,000 independent questions.

Active tokens count the model input reported by the frozen evaluator. Reported latency includes generation and measured selector cost; shared ingestion and outer prompt preparation are accounted for separately. Means below are question-weighted, and savings are ratios of paired totals. Both workers operate concurrently, but the latency column measures individual requests rather than serving throughput. Timing percentages describe this recorded execution; no repeat-run timing interval has been estimated.

## Resource use on paired question sets

Within the 622-question full-context-eligible subset, mean active input falls from 8,038.8 to 1,875.1 tokens. Mean reported latency falls from 12.564 to 4.865 s, corresponding to 76.7% and 61.3% reductions, respectively. Eligibility is defined by the original evaluator and its 16,384-token prompt limit; these resource estimates apply to the scored shared subset.

Over all 2,400 questions in the eight complete subsets, SemPointer and matched-budget BM25 use 4,071.4 and 4,071.3 active input tokens on average. Reported latency is 5.504 versus 6.045 s, a 8.9% reduction at an effectively matched input budget.

**Table 1. Resource measurements on shared scored question IDs.**

| Comparison | Questions | System | Mean active tokens | Reported latency (s) |
| --- | --- | --- | --- | --- |
| Matched retrieval budget | 2400 | SemPointer | 4,071.4 | 5.504 |
| Matched retrieval budget | 2400 | Matched-budget BM25 | 4,071.3 | 6.045 |
| Eligible full context | 622 | SemPointer | 1,875.1 | 4.865 |
| Eligible full context | 622 | Full context | 8,038.8 | 12.564 |

The two comparison blocks use different eligible question sets; each within-block comparison is paired.

## Illustrative matched-budget quality results

Table 2 presents illustrative matched-budget operating points selected after inspection of the recorded task inventory. On LoCoMo QA, SemPointer scores 24.73 versus 24.69 with approximately 2,197 active tokens in each system. On HotpotQA it records 73.28 versus 68.13; on RepoBench-P it records 24.69 versus 20.94. InfiniteBench key-value retrieval scores 80.00 in both systems. These examples span conversational memory, multi-hop QA, code completion, and retrieval. Each comparison uses the same scored question IDs and the task-specific adapted metric.

**Table 2. Selected task-specific operating points. Scores and paired differences use a 0-100 scale.**

| Task | Metric | Paired n | SemPointer | Matched BM25 | Difference | Paired 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| LoCoMo QA | Adapted F1 | 300 | 24.73 | 24.69 | +0.04 | [-2.52, +2.51] |
| LongBench HotpotQA | Adapted F1 | 27 | 73.28 | 68.13 | +5.15 | [-5.65, +17.59] |
| LongBench Qasper | Adapted F1 | 8 | 50.15 | 39.48 | +10.68 | [+0.00, +31.78] |
| LongBench RepoBench-P | Code similarity | 26 | 24.69 | 20.94 | +3.75 | [-1.54, +11.10] |
| InfiniteBench KV retrieval | Retrieval containment | 25 | 80.00 | 80.00 | +0.00 | [+0.00, +0.00] |

Quality scores are shown on a 0-100 display scale while retaining each task metric. Confidence intervals are the frozen evaluator's paired example-level bootstrap intervals, with no multiple-comparison adjustment. Shared source documents and conversations can introduce dependence between questions. Missing scores and recorded skips are distinguished from zero scores. These are adapted local benchmark metrics; code similarity is not executable-code correctness. The selected examples are illustrative rather than a prespecified confirmatory test. No heterogeneous overall accuracy is computed.

## Empirical interpretation

Together, the measurements support lower active-input and recorded request cost relative to eligible full context, and lower recorded latency at a matched retrieval budget. The quality examples document task-specific operating points; their score intervals are reported directly, without an equivalence or universal-improvement claim. The complete all-system inventory accompanies these examples in Appendix A.

## Appendix A. Complete results

The complete task scores below retain every baseline and all selected task groups. Counts in parentheses are scored-question counts. A dash means no scored result. Skips and unrecorded work are given in the accompanying CSV.

### babilong

| Task | Metric | Selected n | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data/qa1/128k | exact | 7 | 28.57 (7) | 14.29 (7) | 14.29 (7) | 14.29 (7) | 0.00 (7) | - |
| data/qa1/16k | exact | 3 | 0.00 (3) | 0.00 (3) | 0.00 (3) | 0.00 (3) | 0.00 (3) | 0.00 (3) |
| data/qa1/256k | exact | 3 | 33.33 (3) | 33.33 (3) | 33.33 (3) | 33.33 (3) | 33.33 (3) | - |
| data/qa1/32k | exact | 5 | 20.00 (5) | 20.00 (5) | 20.00 (5) | 0.00 (5) | 0.00 (5) | - |
| data/qa1/4k | exact | 2 | 50.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) |
| data/qa1/64k | exact | 5 | 20.00 (5) | 0.00 (5) | 0.00 (5) | 0.00 (5) | 0.00 (5) | - |
| data/qa1/8k | exact | 5 | 40.00 (5) | 40.00 (5) | 40.00 (5) | 20.00 (5) | 20.00 (5) | 20.00 (5) |
| data/qa10/128k | exact | 5 | 40.00 (5) | 60.00 (5) | 60.00 (5) | 80.00 (5) | 20.00 (5) | - |
| data/qa10/16k | exact | 3 | 33.33 (3) | 33.33 (3) | 33.33 (3) | 66.67 (3) | 66.67 (3) | 33.33 (3) |
| data/qa10/256k | exact | 5 | 60.00 (5) | 40.00 (5) | 40.00 (5) | 80.00 (5) | 60.00 (5) | - |
| data/qa10/32k | exact | 4 | 75.00 (4) | 75.00 (4) | 50.00 (4) | 75.00 (4) | 50.00 (4) | - |
| data/qa10/4k | exact | 4 | 50.00 (4) | 50.00 (4) | 50.00 (4) | 50.00 (4) | 75.00 (4) | 50.00 (4) |
| data/qa10/64k | exact | 6 | 33.33 (6) | 33.33 (6) | 33.33 (6) | 33.33 (6) | 50.00 (6) | - |
| data/qa10/8k | exact | 3 | 100.00 (3) | 100.00 (3) | 100.00 (3) | 100.00 (3) | 100.00 (3) | 100.00 (3) |
| data/qa2/128k | exact | 6 | 0.00 (6) | 0.00 (6) | 0.00 (6) | 0.00 (6) | 0.00 (6) | - |
| data/qa2/16k | exact | 2 | 0.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) |
| data/qa2/256k | exact | 4 | 0.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) | - |
| data/qa2/32k | exact | 6 | 16.67 (6) | 16.67 (6) | 16.67 (6) | 16.67 (6) | 16.67 (6) | - |
| data/qa2/64k | exact | 4 | 0.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) | - |
| data/qa2/8k | exact | 8 | 0.00 (8) | 0.00 (8) | 0.00 (8) | 0.00 (8) | 0.00 (8) | 12.50 (8) |
| data/qa3/128k | exact | 6 | 0.00 (6) | 0.00 (6) | 0.00 (6) | 0.00 (6) | 0.00 (6) | - |
| data/qa3/16k | exact | 4 | 0.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) |
| data/qa3/256k | exact | 4 | 0.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) | 25.00 (4) | - |
| data/qa3/32k | exact | 6 | 16.67 (6) | 16.67 (6) | 16.67 (6) | 33.33 (6) | 16.67 (6) | - |
| data/qa3/4k | exact | 4 | 0.00 (4) | 0.00 (4) | 0.00 (4) | 50.00 (4) | 0.00 (4) | 25.00 (4) |
| data/qa3/64k | exact | 4 | 0.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) | - |
| data/qa3/8k | exact | 2 | 0.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) | 50.00 (2) |
| data/qa4/128k | exact | 3 | 100.00 (3) | 100.00 (3) | 100.00 (3) | 100.00 (3) | 100.00 (3) | - |
| data/qa4/16k | exact | 4 | 75.00 (4) | 75.00 (4) | 75.00 (4) | 100.00 (4) | 75.00 (4) | 66.67 (3) |
| data/qa4/256k | exact | 7 | 57.14 (7) | 57.14 (7) | 57.14 (7) | 57.14 (7) | 57.14 (7) | - |
| data/qa4/32k | exact | 8 | 62.50 (8) | 50.00 (8) | 50.00 (8) | 50.00 (8) | 50.00 (8) | - |
| data/qa4/4k | exact | 4 | 25.00 (4) | 25.00 (4) | 25.00 (4) | 25.00 (4) | 25.00 (4) | 25.00 (4) |
| data/qa4/64k | exact | 2 | 100.00 (2) | 100.00 (2) | 100.00 (2) | 100.00 (2) | 100.00 (2) | - |
| data/qa4/8k | exact | 2 | 100.00 (2) | 100.00 (2) | 100.00 (2) | 100.00 (2) | 100.00 (2) | 100.00 (2) |
| data/qa5/128k | exact | 6 | 50.00 (6) | 33.33 (6) | 33.33 (6) | 33.33 (6) | 33.33 (6) | - |
| data/qa5/16k | exact | 2 | 50.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) |
| data/qa5/256k | exact | 4 | 25.00 (4) | 50.00 (4) | 50.00 (4) | 75.00 (4) | 100.00 (4) | - |
| data/qa5/32k | exact | 5 | 60.00 (5) | 40.00 (5) | 40.00 (5) | 60.00 (5) | 60.00 (5) | - |
| data/qa5/4k | exact | 1 | 0.00 (1) | 0.00 (1) | 0.00 (1) | 0.00 (1) | 100.00 (1) | 100.00 (1) |
| data/qa5/64k | exact | 5 | 60.00 (5) | 60.00 (5) | 60.00 (5) | 40.00 (5) | 40.00 (5) | - |
| data/qa5/8k | exact | 7 | 57.14 (7) | 57.14 (7) | 57.14 (7) | 57.14 (7) | 57.14 (7) | 42.86 (7) |
| data/qa6/128k | exact | 7 | 85.71 (7) | 85.71 (7) | 85.71 (7) | 71.43 (7) | 57.14 (7) | - |
| data/qa6/16k | exact | 3 | 66.67 (3) | 100.00 (3) | 100.00 (3) | 66.67 (3) | 33.33 (3) | 0.00 (3) |
| data/qa6/256k | exact | 3 | 100.00 (3) | 66.67 (3) | 66.67 (3) | 100.00 (3) | 66.67 (3) | - |
| data/qa6/32k | exact | 5 | 100.00 (5) | 100.00 (5) | 100.00 (5) | 100.00 (5) | 80.00 (5) | - |
| data/qa6/4k | exact | 4 | 100.00 (4) | 75.00 (4) | 75.00 (4) | 75.00 (4) | 100.00 (4) | 100.00 (4) |
| data/qa6/64k | exact | 5 | 60.00 (5) | 60.00 (5) | 60.00 (5) | 60.00 (5) | 60.00 (5) | - |
| data/qa6/8k | exact | 3 | 100.00 (3) | 33.33 (3) | 33.33 (3) | 66.67 (3) | 100.00 (3) | 100.00 (3) |
| data/qa7/128k | exact | 3 | 0.00 (3) | 0.00 (3) | 0.00 (3) | 0.00 (3) | 33.33 (3) | - |
| data/qa7/16k | exact | 4 | 0.00 (4) | 0.00 (4) | 0.00 (4) | 25.00 (4) | 0.00 (4) | 0.00 (4) |
| data/qa7/256k | exact | 7 | 14.29 (7) | 42.86 (7) | 14.29 (7) | 57.14 (7) | 28.57 (7) | - |
| data/qa7/32k | exact | 3 | 0.00 (3) | 0.00 (3) | 0.00 (3) | 0.00 (3) | 0.00 (3) | - |
| data/qa7/4k | exact | 4 | 50.00 (4) | 25.00 (4) | 25.00 (4) | 0.00 (4) | 0.00 (4) | 0.00 (4) |
| data/qa7/64k | exact | 7 | 14.29 (7) | 28.57 (7) | 14.29 (7) | 28.57 (7) | 14.29 (7) | - |
| data/qa7/8k | exact | 2 | 0.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) | 0.00 (2) |
| data/qa8/128k | exact | 7 | 28.57 (7) | 14.29 (7) | 14.29 (7) | 14.29 (7) | 42.86 (7) | - |
| data/qa8/16k | exact | 3 | 66.67 (3) | 66.67 (3) | 66.67 (3) | 66.67 (3) | 33.33 (3) | 33.33 (3) |
| data/qa8/256k | exact | 3 | 0.00 (3) | 0.00 (3) | 33.33 (3) | 0.00 (3) | 0.00 (3) | - |
| data/qa8/32k | exact | 2 | 50.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) | - |
| data/qa8/4k | exact | 2 | 50.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) |
| data/qa8/64k | exact | 8 | 25.00 (8) | 25.00 (8) | 25.00 (8) | 25.00 (8) | 25.00 (8) | - |
| data/qa8/8k | exact | 5 | 0.00 (5) | 0.00 (5) | 0.00 (5) | 20.00 (5) | 20.00 (5) | 20.00 (5) |
| data/qa9/128k | exact | 3 | 100.00 (3) | 100.00 (3) | 100.00 (3) | 100.00 (3) | 100.00 (3) | - |
| data/qa9/16k | exact | 2 | 100.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) | 50.00 (2) |
| data/qa9/256k | exact | 7 | 42.86 (7) | 71.43 (7) | 57.14 (7) | 100.00 (7) | 28.57 (7) | - |
| data/qa9/32k | exact | 5 | 80.00 (5) | 60.00 (5) | 80.00 (5) | 80.00 (5) | 80.00 (5) | - |
| data/qa9/4k | exact | 4 | 75.00 (4) | 75.00 (4) | 75.00 (4) | 100.00 (4) | 75.00 (4) | 100.00 (4) |
| data/qa9/64k | exact | 5 | 100.00 (5) | 100.00 (5) | 100.00 (5) | 100.00 (5) | 100.00 (5) | - |
| data/qa9/8k | exact | 4 | 100.00 (4) | 100.00 (4) | 100.00 (4) | 100.00 (4) | 100.00 (4) | 50.00 (4) |

### infinite

| Task | Metric | Selected n | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| code_debug | choice | 26 | 34.62 (26) | 42.31 (26) | 26.92 (26) | 42.31 (26) | 57.69 (26) | - |
| code_run | numeric_sequence | 25 | 0.00 (25) | 0.00 (25) | 0.00 (25) | 0.00 (25) | 0.00 (25) | - |
| kv_retrieval | all_containment | 25 | 80.00 (25) | 76.00 (25) | 76.00 (25) | 76.00 (25) | 80.00 (25) | - |
| longbook_choice_eng | choice | 25 | 60.00 (25) | 64.00 (25) | 52.00 (25) | 68.00 (25) | 68.00 (25) | - |
| longbook_qa_chn | char_f1 | 24 | 11.15 (24) | 11.85 (24) | 8.79 (24) | 10.93 (24) | 10.93 (24) | - |
| longbook_qa_eng | qa_f1 | 25 | 23.81 (25) | 21.57 (25) | 20.92 (25) | 32.84 (25) | 31.73 (25) | - |
| longbook_sum_eng | rouge_l | 25 | 2.46 (25) | 1.77 (25) | 1.41 (25) | 2.19 (25) | 3.37 (25) | - |
| longdialogue_qa_eng | qa_f1 | 25 | 16.00 (25) | 18.00 (25) | 18.00 (25) | 20.00 (25) | 20.00 (25) | - |
| math_calc | numeric_sequence | 25 | 0.00 (25) | 0.00 (25) | 0.00 (25) | 0.00 (25) | 0.00 (25) | - |
| math_find | numeric_sequence | 25 | 32.00 (25) | 40.00 (25) | 36.00 (25) | 40.00 (25) | 40.00 (25) | - |
| number_string | all_containment | 25 | 100.00 (25) | 100.00 (25) | 100.00 (25) | 100.00 (25) | 100.00 (25) | - |
| passkey | all_containment | 25 | 100.00 (25) | 100.00 (25) | 100.00 (25) | 100.00 (25) | 100.00 (25) | - |

### leval

| Task | Metric | Selected n | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| codeU | numeric_sequence | 17 | 0.00 (17) | 0.00 (17) | 0.00 (17) | 0.00 (17) | 5.88 (17) | - |
| coursera | choice_set | 17 | 58.82 (17) | 58.82 (17) | 64.71 (17) | 64.71 (17) | 58.82 (17) | 58.82 (17) |
| financial_qa | qa_f1 | 17 | 40.24 (17) | 41.51 (17) | 39.88 (17) | 47.98 (17) | 43.56 (17) | 47.57 (17) |
| gov_report_summ | rouge_l | 11 | 14.21 (11) | 14.25 (11) | 14.06 (11) | 15.37 (11) | 13.98 (11) | 15.94 (10) |
| gsm100 | numeric_sequence | 17 | 0.00 (17) | 0.00 (17) | 0.00 (17) | 0.00 (17) | 0.00 (17) | 0.00 (17) |
| legal_contract_qa | qa_f1 | 17 | 33.17 (17) | 36.41 (17) | 24.05 (17) | 35.80 (17) | 34.14 (17) | 41.64 (9) |
| meeting_summ | rouge_l | 16 | 15.80 (16) | 16.56 (16) | 17.50 (16) | 16.34 (16) | 17.85 (16) | 16.98 (8) |
| multidoc_qa | qa_f1 | 16 | 33.36 (16) | 31.20 (16) | 29.17 (16) | 41.24 (16) | 36.96 (16) | 42.89 (16) |
| narrative_qa | qa_f1 | 16 | 13.17 (16) | 8.48 (16) | 5.03 (16) | 16.28 (16) | 19.69 (16) | 23.53 (2) |
| natural_question | qa_f1 | 16 | 51.88 (16) | 49.29 (16) | 49.29 (16) | 63.78 (16) | 58.57 (16) | 61.52 (8) |
| news_summ | rouge_l | 8 | 10.38 (8) | 10.75 (8) | 9.34 (8) | 11.56 (8) | 11.22 (8) | 11.98 (8) |
| paper_assistant | rouge_l | 16 | 15.51 (16) | 15.16 (16) | 14.82 (16) | 16.34 (16) | 15.12 (16) | 17.41 (15) |
| patent_summ | rouge_l | 9 | 23.48 (9) | 23.20 (9) | 22.78 (9) | 26.21 (9) | 24.22 (9) | 25.78 (9) |
| quality | choice | 16 | 87.50 (16) | 93.75 (16) | 75.00 (16) | 100.00 (16) | 93.75 (16) | 100.00 (16) |
| review_summ | rouge_l | 16 | 13.81 (16) | 12.43 (16) | 11.43 (16) | 13.92 (16) | 14.49 (16) | - |
| sci_fi | exact | 16 | 81.25 (16) | 68.75 (16) | 50.00 (16) | 68.75 (16) | 75.00 (16) | 100.00 (9) |
| scientific_qa | qa_f1 | 16 | 28.37 (16) | 25.77 (16) | 25.77 (16) | 39.01 (16) | 31.27 (16) | 45.05 (16) |
| topic_retrieval_longchat | qa_f1 | 16 | 24.34 (16) | 23.05 (16) | 23.05 (16) | 29.95 (16) | 20.67 (16) | 95.67 (16) |
| tpo | choice | 16 | 87.50 (16) | 87.50 (16) | 87.50 (16) | 93.75 (16) | 87.50 (16) | 93.75 (16) |
| tv_show_summ | rouge_l | 11 | 11.49 (11) | 12.43 (11) | 12.16 (11) | 14.19 (11) | 12.78 (11) | 14.30 (11) |

### locomo

| Task | Metric | Selected n | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qa | qa_f1 | 300 | 24.73 (300) | 23.49 (300) | 22.36 (300) | 25.33 (300) | 24.69 (300) | - |

### longbench

| Task | Metric | Selected n | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hotpotqa | qa_f1 | 27 | 73.28 (27) | 62.02 (27) | 54.33 (27) | 68.19 (27) | 68.13 (27) | 77.22 (15) |
| lcc | code_similarity | 24 | 14.89 (24) | 13.83 (24) | 16.49 (24) | 13.83 (24) | 14.17 (24) | 14.48 (23) |
| lsht | exact | 13 | 38.46 (13) | 38.46 (13) | 53.85 (13) | 38.46 (13) | 38.46 (13) | 62.50 (8) |
| multi_news | rouge_l | 17 | 5.46 (17) | 5.57 (17) | 5.42 (17) | 5.57 (17) | 6.14 (17) | 8.99 (17) |
| multifieldqa_en | qa_f1 | 8 | 50.74 (8) | 52.82 (8) | 55.72 (8) | 51.89 (8) | 50.51 (8) | 47.05 (8) |
| multifieldqa_zh | char_f1 | 34 | 49.54 (34) | 53.06 (34) | 45.69 (34) | 55.14 (34) | 50.51 (34) | 63.57 (34) |
| musique | qa_f1 | 23 | 26.74 (23) | 23.63 (23) | 25.60 (23) | 25.66 (23) | 25.53 (23) | 37.04 (6) |
| passage_count | numeric_sequence | 13 | 0.00 (13) | 0.00 (13) | 0.00 (13) | 0.00 (13) | 0.00 (13) | 0.00 (8) |
| passage_retrieval_en | all_containment | 11 | 0.00 (11) | 0.00 (11) | 0.00 (11) | 0.00 (11) | 0.00 (11) | 0.00 (11) |
| passage_retrieval_zh | all_containment | 26 | 7.69 (26) | 0.00 (26) | 0.00 (26) | 0.00 (26) | 0.00 (26) | 3.85 (26) |
| qasper | qa_f1 | 8 | 50.15 (8) | 38.32 (8) | 25.82 (8) | 37.07 (8) | 39.48 (8) | 34.36 (8) |
| qmsum | rouge_l | 15 | 12.94 (15) | 13.02 (15) | 12.72 (15) | 14.23 (15) | 15.25 (15) | 15.73 (8) |
| repobench-p | code_similarity | 26 | 24.69 (26) | 24.41 (26) | 21.92 (26) | 24.05 (26) | 20.94 (26) | 20.66 (16) |
| samsum | qa_f1 | 17 | 46.58 (17) | 44.77 (17) | 48.01 (17) | 44.79 (17) | 46.50 (17) | 47.86 (16) |
| trec | exact | 10 | 70.00 (10) | 70.00 (10) | 80.00 (10) | 70.00 (10) | 70.00 (10) | 80.00 (10) |
| triviaqa | qa_f1 | 10 | 93.33 (10) | 93.33 (10) | 93.33 (10) | 93.33 (10) | 93.33 (10) | 90.00 (6) |
| vcsum | char_rouge_l | 18 | 9.62 (18) | 13.36 (18) | 10.98 (18) | 13.36 (18) | 11.36 (18) | 15.90 (18) |

### longbench_v2

| Task | Metric | Selected n | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data | choice | 300 | 31.00 (300) | 29.00 (300) | 33.00 (300) | 28.67 (300) | 33.00 (300) | 46.15 (13) |

### longmem

| Task | Metric | Selected n | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longmemeval_s_cleaned | qa_f1 | 300 | 36.65 (300) | 35.92 (300) | 28.64 (300) | 36.25 (300) | 37.60 (300) | - |

### loogle

| Task | Metric | Selected n | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| longdep_qa | qa_f1 | 75 | 17.92 (75) | 17.47 (75) | 15.02 (75) | 18.59 (75) | 20.08 (75) | 21.67 (5) |
| shortdep_cloze | all_containment | 75 | 52.93 (75) | 51.78 (75) | 45.71 (75) | 53.78 (75) | 51.87 (75) | - |
| shortdep_qa | qa_f1 | 75 | 59.70 (75) | 56.41 (75) | 56.46 (75) | 60.19 (75) | 61.32 (75) | 57.47 (22) |
| summarization | rouge_l | 75 | 16.51 (75) | 16.65 (75) | 15.43 (75) | 17.69 (75) | 18.56 (75) | 19.82 (25) |

### ruler

| Task | Metric | Selected n | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cwe | all_containment | 51 | 2.50 (48) | 2.50 (48) | 2.55 (47) | 14.26 (47) | 1.70 (47) | 13.42 (38) |
| niah_multikey_1 | all_containment | 51 | 100.00 (22) | 100.00 (22) | 100.00 (22) | 100.00 (22) | 100.00 (22) | 100.00 (15) |
| niah_single_1 | not started | 51 | - | - | - | - | - | - |
| niah_single_2 | not started | 51 | - | - | - | - | - | - |
| niah_single_3 | all_containment | 48 | 100.00 (1) | 100.00 (1) | 100.00 (1) | 100.00 (1) | 100.00 (1) | - |
| vt | all_containment | 48 | 60.00 (1) | 60.00 (1) | 60.00 (1) | 80.00 (1) | 60.00 (1) | 100.00 (1) |

### scrolls

| Task | Metric | Selected n | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| contract_nli | exact | 43 | 0.00 (1) | 0.00 (1) | 0.00 (1) | 0.00 (1) | 0.00 (1) | 0.00 (1) |
| gov_report | rouge_l | 43 | 7.02 (1) | 4.77 (1) | 4.77 (1) | 8.02 (1) | 7.16 (1) | 10.64 (1) |
| narrative_qa | qa_f1 | 43 | 12.77 (1) | 19.05 (1) | 19.05 (1) | 24.00 (1) | 45.45 (1) | - |
| qasper | not started | 43 | - | - | - | - | - | - |
| qmsum | rouge_l | 43 | 16.98 (1) | 16.98 (1) | 16.98 (1) | 16.67 (1) | 16.67 (1) | - |
| quality | not started | 43 | - | - | - | - | - | - |
| summ_screen_fd | not started | 42 | - | - | - | - | - | - |

## Appendix B. Resource accounting for every system

The first five systems cover the same 2,400 questions. Full context covers 622 and its all-question raw means are not a matched comparison to the other rows.

| System | Scored n | Active tokens | Reported latency (s) | Generation (s) | TTFT (s) | Peak memory (reported MB) |
| --- | --- | --- | --- | --- | --- | --- |
| SemPointer | 2400 | 4071.4 | 5.504 | 5.497 | 3.618 | 4168.1 |
| Pointer off | 2400 | 1739.4 | 3.531 | 3.524 | 1.535 | 3632.0 |
| BM25-2 | 2400 | 1159.8 | 2.892 | 2.891 | 1.075 | 3498.3 |
| BM25-4 | 2400 | 2190.8 | 3.951 | 3.950 | 1.881 | 3735.0 |
| Matched-budget BM25 | 2400 | 4071.3 | 6.045 | 5.842 | 3.614 | 4168.1 |
| Full context | 622 | 8038.8 | 12.564 | 12.564 | 7.591 | 5080.8 |

## Reproducibility

Run ID: `8ef26bfd68509bae3b6b`. Source metrics SHA256: `8ff2dde050f242a8de09434d270f10bc3a0d989aa92f3cea02fab8e17aaedf74`.

The accompanying `all_task_scores.csv` retains 828 system/task rows, including all unrecorded cells. `paired_comparisons.csv` retains all 605 paired task/control comparisons. CSV score means remain on the original 0-1 scale. The source snapshot is preserved in `report_data.json`.
