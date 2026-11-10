# SemPointer: complete score and resource comparison

Frozen run `8ef26bfd68509bae3b6b`; Qwen3.5-4B, NF4/FP16, two T4 GPUs.

**Snapshot:** 13,055 scored outcomes, 1,798 skips, zero recorded errors. Eight datasets complete; RULER and SCROLLS partial. Development: 216 scored outcomes across three candidates. CWE cap audit is unfinished.

**Source limitation:** the latest download is an incomplete `.part` archive. Its complete combined report members were readable, but the full archive and raw prediction databases have not been audited.

## Reading the comparisons

- All quality scores below are multiplied by 100. They are the reported adapted task metrics, not a universal accuracy measure.
- Compare systems within the same task. Do not average different metrics into an overall quality score.
- Every score includes its scored-question count. Full context often covers a smaller subset; compare its paired results rather than unrelated raw means.
- A missing score is not zero. Recorded skips and unrecorded work are separate.
- Task higher/lower/tie counts weight every task group equally; they are descriptive and not question-weighted benchmark wins. There are 69 BABILong task/length groups.
- Confidence intervals are the evaluator’s existing example-level intervals, without multiple-comparison correction. Source/document clustering and tiny sample counts limit inference.

## Dataset coverage

| Dataset | Selected questions | Scored | Skipped | Unrecorded outcomes | Status |
|---|---:|---:|---:|---:|---|
| babilong | 300 | 1599 | 201 | 0 | Complete |
| infinite | 300 | 1500 | 300 | 0 | Complete |
| leval | 300 | 1720 | 80 | 0 | Complete |
| locomo | 300 | 1500 | 300 | 0 | Complete |
| longbench | 300 | 1738 | 62 | 0 | Complete |
| longbench_v2 | 300 | 1513 | 287 | 0 | Complete |
| longmem | 300 | 1500 | 300 | 0 | Complete |
| loogle | 300 | 1552 | 248 | 0 | Complete |
| ruler | 300 | 411 | 18 | 1371 | Partial |
| scrolls | 300 | 22 | 2 | 1776 | Partial |

## Task comparisons on the eight complete datasets

| Baseline | SemPointer higher | Lower | Tied | Paired CI above zero | Below zero | Includes zero |
|---|---:|---:|---:|---:|---:|---:|
| Pointer off | 39 | 24 | 62 | 4 | 1 | 120 |
| BM25-2 | 48 | 17 | 60 | 11 | 0 | 114 |
| BM25-4 | 27 | 45 | 53 | 2 | 6 | 117 |
| Matched-budget BM25 | 33 | 38 | 54 | 0 | 4 | 121 |
| Full context | 14 | 34 | 20 | 0 | 8 | 60 |

## Resource use on the eight complete datasets

Means are weighted by scored-question count. The first five systems cover the same 2,400 selected questions; full context covers only 622. Reported latency is generation plus measured selector cost. Shared ingestion and outer preparation are separate; this is not end-to-end wall time.

| System | Scored questions | Mean active tokens | Mean latency (s) | Generation (s) | TTFT (s) | Mean per-question peak GPU memory (reported MB) |
|---|---:|---:|---:|---:|---:|---:|
| SemPointer | 2400 | 4071.4 | 5.504 | 5.497 | 3.618 | 4168.1 |
| Pointer off | 2400 | 1739.4 | 3.531 | 3.524 | 1.535 | 3632.0 |
| BM25-2 | 2400 | 1159.8 | 2.892 | 2.891 | 1.075 | 3498.3 |
| BM25-4 | 2400 | 2190.8 | 3.951 | 3.950 | 1.881 | 3735.0 |
| Matched-budget BM25 | 2400 | 4071.3 | 6.045 | 5.842 | 3.614 | 4168.1 |
| Full context | 622 | 8038.8 | 12.564 | 12.564 | 7.591 | 5080.8 |

On the same 622 full-context-eligible questions, SemPointer averaged 1875.1 active tokens and 4.865 s reported latency; full context averaged 8038.8 tokens and 12.564 s. This demonstrates resource savings on that subset, but does not establish preserved quality: the task comparisons favor full context more often.

## Every selected task and every system

Scores ×100; n is the scored-question count for that cell. All five unstarted task groups are included.

### babilong

| Task | Metric | Selected questions | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| data/qa1/128k | exact | 7 | 28.57 (n=7) | 14.29 (n=7) | 14.29 (n=7) | 14.29 (n=7) | 0.00 (n=7) | —; skipped=7 |
| data/qa1/16k | exact | 3 | 0.00 (n=3) | 0.00 (n=3) | 0.00 (n=3) | 0.00 (n=3) | 0.00 (n=3) | 0.00 (n=3) |
| data/qa1/256k | exact | 3 | 33.33 (n=3) | 33.33 (n=3) | 33.33 (n=3) | 33.33 (n=3) | 33.33 (n=3) | —; skipped=3 |
| data/qa1/32k | exact | 5 | 20.00 (n=5) | 20.00 (n=5) | 20.00 (n=5) | 0.00 (n=5) | 0.00 (n=5) | —; skipped=5 |
| data/qa1/4k | exact | 2 | 50.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) |
| data/qa1/64k | exact | 5 | 20.00 (n=5) | 0.00 (n=5) | 0.00 (n=5) | 0.00 (n=5) | 0.00 (n=5) | —; skipped=5 |
| data/qa1/8k | exact | 5 | 40.00 (n=5) | 40.00 (n=5) | 40.00 (n=5) | 20.00 (n=5) | 20.00 (n=5) | 20.00 (n=5) |
| data/qa10/128k | exact | 5 | 40.00 (n=5) | 60.00 (n=5) | 60.00 (n=5) | 80.00 (n=5) | 20.00 (n=5) | —; skipped=5 |
| data/qa10/16k | exact | 3 | 33.33 (n=3) | 33.33 (n=3) | 33.33 (n=3) | 66.67 (n=3) | 66.67 (n=3) | 33.33 (n=3) |
| data/qa10/256k | exact | 5 | 60.00 (n=5) | 40.00 (n=5) | 40.00 (n=5) | 80.00 (n=5) | 60.00 (n=5) | —; skipped=5 |
| data/qa10/32k | exact | 4 | 75.00 (n=4) | 75.00 (n=4) | 50.00 (n=4) | 75.00 (n=4) | 50.00 (n=4) | —; skipped=4 |
| data/qa10/4k | exact | 4 | 50.00 (n=4) | 50.00 (n=4) | 50.00 (n=4) | 50.00 (n=4) | 75.00 (n=4) | 50.00 (n=4) |
| data/qa10/64k | exact | 6 | 33.33 (n=6) | 33.33 (n=6) | 33.33 (n=6) | 33.33 (n=6) | 50.00 (n=6) | —; skipped=6 |
| data/qa10/8k | exact | 3 | 100.00 (n=3) | 100.00 (n=3) | 100.00 (n=3) | 100.00 (n=3) | 100.00 (n=3) | 100.00 (n=3) |
| data/qa2/128k | exact | 6 | 0.00 (n=6) | 0.00 (n=6) | 0.00 (n=6) | 0.00 (n=6) | 0.00 (n=6) | —; skipped=6 |
| data/qa2/16k | exact | 2 | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) |
| data/qa2/256k | exact | 4 | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | —; skipped=4 |
| data/qa2/32k | exact | 6 | 16.67 (n=6) | 16.67 (n=6) | 16.67 (n=6) | 16.67 (n=6) | 16.67 (n=6) | —; skipped=6 |
| data/qa2/64k | exact | 4 | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | —; skipped=4 |
| data/qa2/8k | exact | 8 | 0.00 (n=8) | 0.00 (n=8) | 0.00 (n=8) | 0.00 (n=8) | 0.00 (n=8) | 12.50 (n=8) |
| data/qa3/128k | exact | 6 | 0.00 (n=6) | 0.00 (n=6) | 0.00 (n=6) | 0.00 (n=6) | 0.00 (n=6) | —; skipped=6 |
| data/qa3/16k | exact | 4 | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) |
| data/qa3/256k | exact | 4 | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 25.00 (n=4) | —; skipped=4 |
| data/qa3/32k | exact | 6 | 16.67 (n=6) | 16.67 (n=6) | 16.67 (n=6) | 33.33 (n=6) | 16.67 (n=6) | —; skipped=6 |
| data/qa3/4k | exact | 4 | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 50.00 (n=4) | 0.00 (n=4) | 25.00 (n=4) |
| data/qa3/64k | exact | 4 | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | —; skipped=4 |
| data/qa3/8k | exact | 2 | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 50.00 (n=2) |
| data/qa4/128k | exact | 3 | 100.00 (n=3) | 100.00 (n=3) | 100.00 (n=3) | 100.00 (n=3) | 100.00 (n=3) | —; skipped=3 |
| data/qa4/16k | exact | 4 | 75.00 (n=4) | 75.00 (n=4) | 75.00 (n=4) | 100.00 (n=4) | 75.00 (n=4) | 66.67 (n=3); skipped=1 |
| data/qa4/256k | exact | 7 | 57.14 (n=7) | 57.14 (n=7) | 57.14 (n=7) | 57.14 (n=7) | 57.14 (n=7) | —; skipped=7 |
| data/qa4/32k | exact | 8 | 62.50 (n=8) | 50.00 (n=8) | 50.00 (n=8) | 50.00 (n=8) | 50.00 (n=8) | —; skipped=8 |
| data/qa4/4k | exact | 4 | 25.00 (n=4) | 25.00 (n=4) | 25.00 (n=4) | 25.00 (n=4) | 25.00 (n=4) | 25.00 (n=4) |
| data/qa4/64k | exact | 2 | 100.00 (n=2) | 100.00 (n=2) | 100.00 (n=2) | 100.00 (n=2) | 100.00 (n=2) | —; skipped=2 |
| data/qa4/8k | exact | 2 | 100.00 (n=2) | 100.00 (n=2) | 100.00 (n=2) | 100.00 (n=2) | 100.00 (n=2) | 100.00 (n=2) |
| data/qa5/128k | exact | 6 | 50.00 (n=6) | 33.33 (n=6) | 33.33 (n=6) | 33.33 (n=6) | 33.33 (n=6) | —; skipped=6 |
| data/qa5/16k | exact | 2 | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) |
| data/qa5/256k | exact | 4 | 25.00 (n=4) | 50.00 (n=4) | 50.00 (n=4) | 75.00 (n=4) | 100.00 (n=4) | —; skipped=4 |
| data/qa5/32k | exact | 5 | 60.00 (n=5) | 40.00 (n=5) | 40.00 (n=5) | 60.00 (n=5) | 60.00 (n=5) | —; skipped=5 |
| data/qa5/4k | exact | 1 | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) | 100.00 (n=1) | 100.00 (n=1) |
| data/qa5/64k | exact | 5 | 60.00 (n=5) | 60.00 (n=5) | 60.00 (n=5) | 40.00 (n=5) | 40.00 (n=5) | —; skipped=5 |
| data/qa5/8k | exact | 7 | 57.14 (n=7) | 57.14 (n=7) | 57.14 (n=7) | 57.14 (n=7) | 57.14 (n=7) | 42.86 (n=7) |
| data/qa6/128k | exact | 7 | 85.71 (n=7) | 85.71 (n=7) | 85.71 (n=7) | 71.43 (n=7) | 57.14 (n=7) | —; skipped=7 |
| data/qa6/16k | exact | 3 | 66.67 (n=3) | 100.00 (n=3) | 100.00 (n=3) | 66.67 (n=3) | 33.33 (n=3) | 0.00 (n=3) |
| data/qa6/256k | exact | 3 | 100.00 (n=3) | 66.67 (n=3) | 66.67 (n=3) | 100.00 (n=3) | 66.67 (n=3) | —; skipped=3 |
| data/qa6/32k | exact | 5 | 100.00 (n=5) | 100.00 (n=5) | 100.00 (n=5) | 100.00 (n=5) | 80.00 (n=5) | —; skipped=5 |
| data/qa6/4k | exact | 4 | 100.00 (n=4) | 75.00 (n=4) | 75.00 (n=4) | 75.00 (n=4) | 100.00 (n=4) | 100.00 (n=4) |
| data/qa6/64k | exact | 5 | 60.00 (n=5) | 60.00 (n=5) | 60.00 (n=5) | 60.00 (n=5) | 60.00 (n=5) | —; skipped=5 |
| data/qa6/8k | exact | 3 | 100.00 (n=3) | 33.33 (n=3) | 33.33 (n=3) | 66.67 (n=3) | 100.00 (n=3) | 100.00 (n=3) |
| data/qa7/128k | exact | 3 | 0.00 (n=3) | 0.00 (n=3) | 0.00 (n=3) | 0.00 (n=3) | 33.33 (n=3) | —; skipped=3 |
| data/qa7/16k | exact | 4 | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 25.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) |
| data/qa7/256k | exact | 7 | 14.29 (n=7) | 42.86 (n=7) | 14.29 (n=7) | 57.14 (n=7) | 28.57 (n=7) | —; skipped=7 |
| data/qa7/32k | exact | 3 | 0.00 (n=3) | 0.00 (n=3) | 0.00 (n=3) | 0.00 (n=3) | 0.00 (n=3) | —; skipped=3 |
| data/qa7/4k | exact | 4 | 50.00 (n=4) | 25.00 (n=4) | 25.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) | 0.00 (n=4) |
| data/qa7/64k | exact | 7 | 14.29 (n=7) | 28.57 (n=7) | 14.29 (n=7) | 28.57 (n=7) | 14.29 (n=7) | —; skipped=7 |
| data/qa7/8k | exact | 2 | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) | 0.00 (n=2) |
| data/qa8/128k | exact | 7 | 28.57 (n=7) | 14.29 (n=7) | 14.29 (n=7) | 14.29 (n=7) | 42.86 (n=7) | —; skipped=7 |
| data/qa8/16k | exact | 3 | 66.67 (n=3) | 66.67 (n=3) | 66.67 (n=3) | 66.67 (n=3) | 33.33 (n=3) | 33.33 (n=3) |
| data/qa8/256k | exact | 3 | 0.00 (n=3) | 0.00 (n=3) | 33.33 (n=3) | 0.00 (n=3) | 0.00 (n=3) | —; skipped=3 |
| data/qa8/32k | exact | 2 | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | —; skipped=2 |
| data/qa8/4k | exact | 2 | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) |
| data/qa8/64k | exact | 8 | 25.00 (n=8) | 25.00 (n=8) | 25.00 (n=8) | 25.00 (n=8) | 25.00 (n=8) | —; skipped=8 |
| data/qa8/8k | exact | 5 | 0.00 (n=5) | 0.00 (n=5) | 0.00 (n=5) | 20.00 (n=5) | 20.00 (n=5) | 20.00 (n=5) |
| data/qa9/128k | exact | 3 | 100.00 (n=3) | 100.00 (n=3) | 100.00 (n=3) | 100.00 (n=3) | 100.00 (n=3) | —; skipped=3 |
| data/qa9/16k | exact | 2 | 100.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) | 50.00 (n=2) |
| data/qa9/256k | exact | 7 | 42.86 (n=7) | 71.43 (n=7) | 57.14 (n=7) | 100.00 (n=7) | 28.57 (n=7) | —; skipped=7 |
| data/qa9/32k | exact | 5 | 80.00 (n=5) | 60.00 (n=5) | 80.00 (n=5) | 80.00 (n=5) | 80.00 (n=5) | —; skipped=5 |
| data/qa9/4k | exact | 4 | 75.00 (n=4) | 75.00 (n=4) | 75.00 (n=4) | 100.00 (n=4) | 75.00 (n=4) | 100.00 (n=4) |
| data/qa9/64k | exact | 5 | 100.00 (n=5) | 100.00 (n=5) | 100.00 (n=5) | 100.00 (n=5) | 100.00 (n=5) | —; skipped=5 |
| data/qa9/8k | exact | 4 | 100.00 (n=4) | 100.00 (n=4) | 100.00 (n=4) | 100.00 (n=4) | 100.00 (n=4) | 50.00 (n=4) |

### infinite

| Task | Metric | Selected questions | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| code_debug | choice | 26 | 34.62 (n=26) | 42.31 (n=26) | 26.92 (n=26) | 42.31 (n=26) | 57.69 (n=26) | —; skipped=26 |
| code_run | numeric_sequence | 25 | 0.00 (n=25) | 0.00 (n=25) | 0.00 (n=25) | 0.00 (n=25) | 0.00 (n=25) | —; skipped=25 |
| kv_retrieval | all_containment | 25 | 80.00 (n=25) | 76.00 (n=25) | 76.00 (n=25) | 76.00 (n=25) | 80.00 (n=25) | —; skipped=25 |
| longbook_choice_eng | choice | 25 | 60.00 (n=25) | 64.00 (n=25) | 52.00 (n=25) | 68.00 (n=25) | 68.00 (n=25) | —; skipped=25 |
| longbook_qa_chn | char_f1 | 24 | 11.15 (n=24) | 11.85 (n=24) | 8.79 (n=24) | 10.93 (n=24) | 10.93 (n=24) | —; skipped=24 |
| longbook_qa_eng | qa_f1 | 25 | 23.81 (n=25) | 21.57 (n=25) | 20.92 (n=25) | 32.84 (n=25) | 31.73 (n=25) | —; skipped=25 |
| longbook_sum_eng | rouge_l | 25 | 2.46 (n=25) | 1.77 (n=25) | 1.41 (n=25) | 2.19 (n=25) | 3.37 (n=25) | —; skipped=25 |
| longdialogue_qa_eng | qa_f1 | 25 | 16.00 (n=25) | 18.00 (n=25) | 18.00 (n=25) | 20.00 (n=25) | 20.00 (n=25) | —; skipped=25 |
| math_calc | numeric_sequence | 25 | 0.00 (n=25) | 0.00 (n=25) | 0.00 (n=25) | 0.00 (n=25) | 0.00 (n=25) | —; skipped=25 |
| math_find | numeric_sequence | 25 | 32.00 (n=25) | 40.00 (n=25) | 36.00 (n=25) | 40.00 (n=25) | 40.00 (n=25) | —; skipped=25 |
| number_string | all_containment | 25 | 100.00 (n=25) | 100.00 (n=25) | 100.00 (n=25) | 100.00 (n=25) | 100.00 (n=25) | —; skipped=25 |
| passkey | all_containment | 25 | 100.00 (n=25) | 100.00 (n=25) | 100.00 (n=25) | 100.00 (n=25) | 100.00 (n=25) | —; skipped=25 |

### leval

| Task | Metric | Selected questions | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| codeU | numeric_sequence | 17 | 0.00 (n=17) | 0.00 (n=17) | 0.00 (n=17) | 0.00 (n=17) | 5.88 (n=17) | —; skipped=17 |
| coursera | choice_set | 17 | 58.82 (n=17) | 58.82 (n=17) | 64.71 (n=17) | 64.71 (n=17) | 58.82 (n=17) | 58.82 (n=17) |
| financial_qa | qa_f1 | 17 | 40.24 (n=17) | 41.51 (n=17) | 39.88 (n=17) | 47.98 (n=17) | 43.56 (n=17) | 47.57 (n=17) |
| gov_report_summ | rouge_l | 11 | 14.21 (n=11) | 14.25 (n=11) | 14.06 (n=11) | 15.37 (n=11) | 13.98 (n=11) | 15.94 (n=10); skipped=1 |
| gsm100 | numeric_sequence | 17 | 0.00 (n=17) | 0.00 (n=17) | 0.00 (n=17) | 0.00 (n=17) | 0.00 (n=17) | 0.00 (n=17) |
| legal_contract_qa | qa_f1 | 17 | 33.17 (n=17) | 36.41 (n=17) | 24.05 (n=17) | 35.80 (n=17) | 34.14 (n=17) | 41.64 (n=9); skipped=8 |
| meeting_summ | rouge_l | 16 | 15.80 (n=16) | 16.56 (n=16) | 17.50 (n=16) | 16.34 (n=16) | 17.85 (n=16) | 16.98 (n=8); skipped=8 |
| multidoc_qa | qa_f1 | 16 | 33.36 (n=16) | 31.20 (n=16) | 29.17 (n=16) | 41.24 (n=16) | 36.96 (n=16) | 42.89 (n=16) |
| narrative_qa | qa_f1 | 16 | 13.17 (n=16) | 8.48 (n=16) | 5.03 (n=16) | 16.28 (n=16) | 19.69 (n=16) | 23.53 (n=2); skipped=14 |
| natural_question | qa_f1 | 16 | 51.88 (n=16) | 49.29 (n=16) | 49.29 (n=16) | 63.78 (n=16) | 58.57 (n=16) | 61.52 (n=8); skipped=8 |
| news_summ | rouge_l | 8 | 10.38 (n=8) | 10.75 (n=8) | 9.34 (n=8) | 11.56 (n=8) | 11.22 (n=8) | 11.98 (n=8) |
| paper_assistant | rouge_l | 16 | 15.51 (n=16) | 15.16 (n=16) | 14.82 (n=16) | 16.34 (n=16) | 15.12 (n=16) | 17.41 (n=15); skipped=1 |
| patent_summ | rouge_l | 9 | 23.48 (n=9) | 23.20 (n=9) | 22.78 (n=9) | 26.21 (n=9) | 24.22 (n=9) | 25.78 (n=9) |
| quality | choice | 16 | 87.50 (n=16) | 93.75 (n=16) | 75.00 (n=16) | 100.00 (n=16) | 93.75 (n=16) | 100.00 (n=16) |
| review_summ | rouge_l | 16 | 13.81 (n=16) | 12.43 (n=16) | 11.43 (n=16) | 13.92 (n=16) | 14.49 (n=16) | —; skipped=16 |
| sci_fi | exact | 16 | 81.25 (n=16) | 68.75 (n=16) | 50.00 (n=16) | 68.75 (n=16) | 75.00 (n=16) | 100.00 (n=9); skipped=7 |
| scientific_qa | qa_f1 | 16 | 28.37 (n=16) | 25.77 (n=16) | 25.77 (n=16) | 39.01 (n=16) | 31.27 (n=16) | 45.05 (n=16) |
| topic_retrieval_longchat | qa_f1 | 16 | 24.34 (n=16) | 23.05 (n=16) | 23.05 (n=16) | 29.95 (n=16) | 20.67 (n=16) | 95.67 (n=16) |
| tpo | choice | 16 | 87.50 (n=16) | 87.50 (n=16) | 87.50 (n=16) | 93.75 (n=16) | 87.50 (n=16) | 93.75 (n=16) |
| tv_show_summ | rouge_l | 11 | 11.49 (n=11) | 12.43 (n=11) | 12.16 (n=11) | 14.19 (n=11) | 12.78 (n=11) | 14.30 (n=11) |

### locomo

| Task | Metric | Selected questions | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qa | qa_f1 | 300 | 24.73 (n=300) | 23.49 (n=300) | 22.36 (n=300) | 25.33 (n=300) | 24.69 (n=300) | —; skipped=300 |

### longbench

| Task | Metric | Selected questions | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hotpotqa | qa_f1 | 27 | 73.28 (n=27) | 62.02 (n=27) | 54.33 (n=27) | 68.19 (n=27) | 68.13 (n=27) | 77.22 (n=15); skipped=12 |
| lcc | code_similarity | 24 | 14.89 (n=24) | 13.83 (n=24) | 16.49 (n=24) | 13.83 (n=24) | 14.17 (n=24) | 14.48 (n=23); skipped=1 |
| lsht | exact | 13 | 38.46 (n=13) | 38.46 (n=13) | 53.85 (n=13) | 38.46 (n=13) | 38.46 (n=13) | 62.50 (n=8); skipped=5 |
| multi_news | rouge_l | 17 | 5.46 (n=17) | 5.57 (n=17) | 5.42 (n=17) | 5.57 (n=17) | 6.14 (n=17) | 8.99 (n=17) |
| multifieldqa_en | qa_f1 | 8 | 50.74 (n=8) | 52.82 (n=8) | 55.72 (n=8) | 51.89 (n=8) | 50.51 (n=8) | 47.05 (n=8) |
| multifieldqa_zh | char_f1 | 34 | 49.54 (n=34) | 53.06 (n=34) | 45.69 (n=34) | 55.14 (n=34) | 50.51 (n=34) | 63.57 (n=34) |
| musique | qa_f1 | 23 | 26.74 (n=23) | 23.63 (n=23) | 25.60 (n=23) | 25.66 (n=23) | 25.53 (n=23) | 37.04 (n=6); skipped=17 |
| passage_count | numeric_sequence | 13 | 0.00 (n=13) | 0.00 (n=13) | 0.00 (n=13) | 0.00 (n=13) | 0.00 (n=13) | 0.00 (n=8); skipped=5 |
| passage_retrieval_en | all_containment | 11 | 0.00 (n=11) | 0.00 (n=11) | 0.00 (n=11) | 0.00 (n=11) | 0.00 (n=11) | 0.00 (n=11) |
| passage_retrieval_zh | all_containment | 26 | 7.69 (n=26) | 0.00 (n=26) | 0.00 (n=26) | 0.00 (n=26) | 0.00 (n=26) | 3.85 (n=26) |
| qasper | qa_f1 | 8 | 50.15 (n=8) | 38.32 (n=8) | 25.82 (n=8) | 37.07 (n=8) | 39.48 (n=8) | 34.36 (n=8) |
| qmsum | rouge_l | 15 | 12.94 (n=15) | 13.02 (n=15) | 12.72 (n=15) | 14.23 (n=15) | 15.25 (n=15) | 15.73 (n=8); skipped=7 |
| repobench-p | code_similarity | 26 | 24.69 (n=26) | 24.41 (n=26) | 21.92 (n=26) | 24.05 (n=26) | 20.94 (n=26) | 20.66 (n=16); skipped=10 |
| samsum | qa_f1 | 17 | 46.58 (n=17) | 44.77 (n=17) | 48.01 (n=17) | 44.79 (n=17) | 46.50 (n=17) | 47.86 (n=16); skipped=1 |
| trec | exact | 10 | 70.00 (n=10) | 70.00 (n=10) | 80.00 (n=10) | 70.00 (n=10) | 70.00 (n=10) | 80.00 (n=10) |
| triviaqa | qa_f1 | 10 | 93.33 (n=10) | 93.33 (n=10) | 93.33 (n=10) | 93.33 (n=10) | 93.33 (n=10) | 90.00 (n=6); skipped=4 |
| vcsum | char_rouge_l | 18 | 9.62 (n=18) | 13.36 (n=18) | 10.98 (n=18) | 13.36 (n=18) | 11.36 (n=18) | 15.90 (n=18) |

### longbench_v2

| Task | Metric | Selected questions | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| data | choice | 300 | 31.00 (n=300) | 29.00 (n=300) | 33.00 (n=300) | 28.67 (n=300) | 33.00 (n=300) | 46.15 (n=13); skipped=287 |

### longmem

| Task | Metric | Selected questions | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| longmemeval_s_cleaned | qa_f1 | 300 | 36.65 (n=300) | 35.92 (n=300) | 28.64 (n=300) | 36.25 (n=300) | 37.60 (n=300) | —; skipped=300 |

### loogle

| Task | Metric | Selected questions | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| longdep_qa | qa_f1 | 75 | 17.92 (n=75) | 17.47 (n=75) | 15.02 (n=75) | 18.59 (n=75) | 20.08 (n=75) | 21.67 (n=5); skipped=70 |
| shortdep_cloze | all_containment | 75 | 52.93 (n=75) | 51.78 (n=75) | 45.71 (n=75) | 53.78 (n=75) | 51.87 (n=75) | —; skipped=75 |
| shortdep_qa | qa_f1 | 75 | 59.70 (n=75) | 56.41 (n=75) | 56.46 (n=75) | 60.19 (n=75) | 61.32 (n=75) | 57.47 (n=22); skipped=53 |
| summarization | rouge_l | 75 | 16.51 (n=75) | 16.65 (n=75) | 15.43 (n=75) | 17.69 (n=75) | 18.56 (n=75) | 19.82 (n=25); skipped=50 |

### ruler

| Task | Metric | Selected questions | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cwe | all_containment | 51 | 2.50 (n=48) | 2.50 (n=48) | 2.55 (n=47) | 14.26 (n=47) | 1.70 (n=47) | 13.42 (n=38); skipped=10 |
| niah_multikey_1 | all_containment | 51 | 100.00 (n=22) | 100.00 (n=22) | 100.00 (n=22) | 100.00 (n=22) | 100.00 (n=22) | 100.00 (n=15); skipped=7 |
| niah_single_1 | not started | 51 | — | — | — | — | — | — |
| niah_single_2 | not started | 51 | — | — | — | — | — | — |
| niah_single_3 | all_containment | 48 | 100.00 (n=1) | 100.00 (n=1) | 100.00 (n=1) | 100.00 (n=1) | 100.00 (n=1) | —; skipped=1 |
| vt | all_containment | 48 | 60.00 (n=1) | 60.00 (n=1) | 60.00 (n=1) | 80.00 (n=1) | 60.00 (n=1) | 100.00 (n=1) |

### scrolls

| Task | Metric | Selected questions | SemPointer | Pointer off | BM25-2 | BM25-4 | Matched-budget BM25 | Full context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| contract_nli | exact | 43 | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) | 0.00 (n=1) |
| gov_report | rouge_l | 43 | 7.02 (n=1) | 4.77 (n=1) | 4.77 (n=1) | 8.02 (n=1) | 7.16 (n=1) | 10.64 (n=1) |
| narrative_qa | qa_f1 | 43 | 12.77 (n=1) | 19.05 (n=1) | 19.05 (n=1) | 24.00 (n=1) | 45.45 (n=1) | —; skipped=1 |
| qasper | not started | 43 | — | — | — | — | — | — |
| qmsum | rouge_l | 43 | 16.98 (n=1) | 16.98 (n=1) | 16.98 (n=1) | 16.67 (n=1) | 16.67 (n=1) | —; skipped=1 |
| quality | not started | 43 | — | — | — | — | — | — |
| summ_screen_fd | not started | 42 | — | — | — | — | — | — |

## Every paired comparison

Positive differences favor SemPointer. Score differences and their 95% intervals are on the 0–100 score scale. Negative token or latency differences mean SemPointer used less. Intervals are uncorrected, and all incomplete-dataset comparisons remain preliminary.

| Dataset | Task | Comparison | Paired n | Score difference | 95% interval | Active token difference | Latency difference (s) |
|---|---|---|---:|---:|---|---:|---:|
| babilong | data/qa1/128k | sempointer_vs_bm25_2 | 7 | +14.29 | [+0.00, +42.86] | +5233.71 | +4.097 |
| babilong | data/qa1/128k | sempointer_vs_bm25_4 | 7 | +14.29 | [+0.00, +42.86] | +4194.00 | +3.163 |
| babilong | data/qa1/128k | sempointer_vs_bm25_matched_budget | 7 | +28.57 | [+0.00, +57.14] | +0.14 | -0.379 |
| babilong | data/qa1/128k | sempointer_vs_pointer_off | 7 | +14.29 | [+0.00, +42.86] | +5233.71 | +4.107 |
| babilong | data/qa1/16k | sempointer_vs_bm25_2 | 3 | +0.00 | [+0.00, +0.00] | +588.33 | +0.427 |
| babilong | data/qa1/16k | sempointer_vs_bm25_4 | 3 | +0.00 | [+0.00, +0.00] | -450.33 | -0.185 |
| babilong | data/qa1/16k | sempointer_vs_bm25_matched_budget | 3 | +0.00 | [+0.00, +0.00] | +0.33 | -0.063 |
| babilong | data/qa1/16k | sempointer_vs_full_context | 3 | +0.00 | [+0.00, +0.00] | -13691.33 | -13.263 |
| babilong | data/qa1/16k | sempointer_vs_pointer_off | 3 | +0.00 | [+0.00, +0.00] | +588.33 | +0.402 |
| babilong | data/qa1/256k | sempointer_vs_bm25_2 | 3 | +0.00 | [+0.00, +0.00] | +10717.00 | +10.803 |
| babilong | data/qa1/256k | sempointer_vs_bm25_4 | 3 | +0.00 | [+0.00, +0.00] | +9677.00 | +9.981 |
| babilong | data/qa1/256k | sempointer_vs_bm25_matched_budget | 3 | +0.00 | [+0.00, +0.00] | +0.00 | -0.466 |
| babilong | data/qa1/256k | sempointer_vs_pointer_off | 3 | +0.00 | [+0.00, +0.00] | +10717.00 | +10.799 |
| babilong | data/qa1/32k | sempointer_vs_bm25_2 | 5 | +0.00 | [+0.00, +0.00] | +1242.60 | +0.922 |
| babilong | data/qa1/32k | sempointer_vs_bm25_4 | 5 | +20.00 | [+0.00, +60.00] | +203.80 | +0.091 |
| babilong | data/qa1/32k | sempointer_vs_bm25_matched_budget | 5 | +20.00 | [+0.00, +60.00] | +0.00 | -0.125 |
| babilong | data/qa1/32k | sempointer_vs_pointer_off | 5 | +0.00 | [+0.00, +0.00] | +1242.60 | +0.918 |
| babilong | data/qa1/4k | sempointer_vs_bm25_2 | 2 | +50.00 | [+0.00, +100.00] | +129.00 | -0.430 |
| babilong | data/qa1/4k | sempointer_vs_bm25_4 | 2 | +50.00 | [+0.00, +100.00] | -909.00 | -0.766 |
| babilong | data/qa1/4k | sempointer_vs_bm25_matched_budget | 2 | +50.00 | [+0.00, +100.00] | +0.00 | -0.097 |
| babilong | data/qa1/4k | sempointer_vs_full_context | 2 | +50.00 | [+0.00, +100.00] | -2575.00 | -2.157 |
| babilong | data/qa1/4k | sempointer_vs_pointer_off | 2 | +50.00 | [+0.00, +100.00] | +129.00 | -0.452 |
| babilong | data/qa1/64k | sempointer_vs_bm25_2 | 5 | +20.00 | [+0.00, +60.00] | +2571.40 | +1.941 |
| babilong | data/qa1/64k | sempointer_vs_bm25_4 | 5 | +20.00 | [+0.00, +60.00] | +1532.60 | +1.168 |
| babilong | data/qa1/64k | sempointer_vs_bm25_matched_budget | 5 | +20.00 | [+0.00, +60.00] | +0.00 | -0.348 |
| babilong | data/qa1/64k | sempointer_vs_pointer_off | 5 | +20.00 | [+0.00, +60.00] | +2571.40 | +1.940 |
| babilong | data/qa1/8k | sempointer_vs_bm25_2 | 5 | +0.00 | [+0.00, +0.00] | +260.60 | +0.158 |
| babilong | data/qa1/8k | sempointer_vs_bm25_4 | 5 | +20.00 | [+0.00, +60.00] | -778.00 | -0.552 |
| babilong | data/qa1/8k | sempointer_vs_bm25_matched_budget | 5 | +20.00 | [+0.00, +60.00] | +0.00 | -0.027 |
| babilong | data/qa1/8k | sempointer_vs_full_context | 5 | +20.00 | [+0.00, +60.00] | -6044.60 | -5.578 |
| babilong | data/qa1/8k | sempointer_vs_pointer_off | 5 | +0.00 | [+0.00, +0.00] | +260.60 | +0.156 |
| babilong | data/qa10/128k | sempointer_vs_bm25_2 | 5 | -20.00 | [-60.00, +0.00] | +5136.20 | +4.324 |
| babilong | data/qa10/128k | sempointer_vs_bm25_4 | 5 | -40.00 | [-80.00, +0.00] | +4095.00 | +3.657 |
| babilong | data/qa10/128k | sempointer_vs_bm25_matched_budget | 5 | +20.00 | [-40.00, +80.00] | +0.60 | -0.524 |
| babilong | data/qa10/128k | sempointer_vs_pointer_off | 5 | -20.00 | [-60.00, +0.00] | +5136.20 | +4.316 |
| babilong | data/qa10/16k | sempointer_vs_bm25_2 | 3 | +0.00 | [+0.00, +0.00] | +563.33 | +0.428 |
| babilong | data/qa10/16k | sempointer_vs_bm25_4 | 3 | -33.33 | [-100.00, +0.00] | -475.67 | -0.363 |
| babilong | data/qa10/16k | sempointer_vs_bm25_matched_budget | 3 | -33.33 | [-100.00, +0.00] | +2.00 | -0.070 |
| babilong | data/qa10/16k | sempointer_vs_full_context | 3 | +0.00 | [+0.00, +0.00] | -13177.33 | -13.457 |
| babilong | data/qa10/16k | sempointer_vs_pointer_off | 3 | +0.00 | [+0.00, +0.00] | +563.33 | +0.432 |
| babilong | data/qa10/256k | sempointer_vs_bm25_2 | 5 | +20.00 | [+0.00, +60.00] | +10587.00 | +10.193 |
| babilong | data/qa10/256k | sempointer_vs_bm25_4 | 5 | -20.00 | [-60.00, +0.00] | +9546.20 | +9.513 |
| babilong | data/qa10/256k | sempointer_vs_bm25_matched_budget | 5 | +0.00 | [-60.00, +60.00] | +0.00 | -0.615 |
| babilong | data/qa10/256k | sempointer_vs_pointer_off | 5 | +20.00 | [+0.00, +60.00] | +10587.00 | +10.173 |
| babilong | data/qa10/32k | sempointer_vs_bm25_2 | 4 | +25.00 | [-50.00, +100.00] | +1472.00 | +0.993 |
| babilong | data/qa10/32k | sempointer_vs_bm25_4 | 4 | +0.00 | [+0.00, +0.00] | +453.50 | +0.352 |
| babilong | data/qa10/32k | sempointer_vs_bm25_matched_budget | 4 | +25.00 | [+0.00, +75.00] | +0.00 | -0.097 |
| babilong | data/qa10/32k | sempointer_vs_pointer_off | 4 | +0.00 | [-75.00, +75.00] | +1212.00 | +0.942 |
| babilong | data/qa10/4k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +124.00 | -0.103 |
| babilong | data/qa10/4k | sempointer_vs_bm25_4 | 4 | +0.00 | [+0.00, +0.00] | -893.00 | -0.744 |
| babilong | data/qa10/4k | sempointer_vs_bm25_matched_budget | 4 | -25.00 | [-75.00, +0.00] | +0.25 | -0.047 |
| babilong | data/qa10/4k | sempointer_vs_full_context | 4 | +0.00 | [+0.00, +0.00] | -2530.75 | -2.150 |
| babilong | data/qa10/4k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +124.00 | -0.121 |
| babilong | data/qa10/64k | sempointer_vs_bm25_2 | 6 | +0.00 | [+0.00, +0.00] | +2567.50 | +2.130 |
| babilong | data/qa10/64k | sempointer_vs_bm25_4 | 6 | +0.00 | [+0.00, +0.00] | +1527.17 | +1.325 |
| babilong | data/qa10/64k | sempointer_vs_bm25_matched_budget | 6 | -16.67 | [-50.00, +0.00] | +0.00 | -0.131 |
| babilong | data/qa10/64k | sempointer_vs_pointer_off | 6 | +0.00 | [+0.00, +0.00] | +2567.50 | +2.120 |
| babilong | data/qa10/8k | sempointer_vs_bm25_2 | 3 | +0.00 | [+0.00, +0.00] | +280.00 | +0.174 |
| babilong | data/qa10/8k | sempointer_vs_bm25_4 | 3 | +0.00 | [+0.00, +0.00] | -758.67 | -0.634 |
| babilong | data/qa10/8k | sempointer_vs_bm25_matched_budget | 3 | +0.00 | [+0.00, +0.00] | +0.00 | -0.050 |
| babilong | data/qa10/8k | sempointer_vs_full_context | 3 | +0.00 | [+0.00, +0.00] | -6483.67 | -6.023 |
| babilong | data/qa10/8k | sempointer_vs_pointer_off | 3 | +0.00 | [+0.00, +0.00] | +280.00 | +0.169 |
| babilong | data/qa2/128k | sempointer_vs_bm25_2 | 6 | +0.00 | [+0.00, +0.00] | +5271.17 | +4.562 |
| babilong | data/qa2/128k | sempointer_vs_bm25_4 | 6 | +0.00 | [+0.00, +0.00] | +4230.50 | +3.669 |
| babilong | data/qa2/128k | sempointer_vs_bm25_matched_budget | 6 | +0.00 | [+0.00, +0.00] | +0.00 | -0.713 |
| babilong | data/qa2/128k | sempointer_vs_pointer_off | 6 | +0.00 | [+0.00, +0.00] | +5271.17 | +4.575 |
| babilong | data/qa2/16k | sempointer_vs_bm25_2 | 2 | +0.00 | [+0.00, +0.00] | +607.00 | +0.313 |
| babilong | data/qa2/16k | sempointer_vs_bm25_4 | 2 | +0.00 | [+0.00, +0.00] | -433.00 | -1.084 |
| babilong | data/qa2/16k | sempointer_vs_bm25_matched_budget | 2 | +0.00 | [+0.00, +0.00] | +0.00 | -0.792 |
| babilong | data/qa2/16k | sempointer_vs_full_context | 2 | +0.00 | [+0.00, +0.00] | -14127.00 | -17.682 |
| babilong | data/qa2/16k | sempointer_vs_pointer_off | 2 | +0.00 | [+0.00, +0.00] | +607.00 | +0.255 |
| babilong | data/qa2/256k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +10657.00 | +10.159 |
| babilong | data/qa2/256k | sempointer_vs_bm25_4 | 4 | +0.00 | [+0.00, +0.00] | +9617.00 | +9.200 |
| babilong | data/qa2/256k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [+0.00, +0.00] | +0.00 | -1.113 |
| babilong | data/qa2/256k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +10657.00 | +10.108 |
| babilong | data/qa2/32k | sempointer_vs_bm25_2 | 6 | +0.00 | [+0.00, +0.00] | +1231.17 | +1.214 |
| babilong | data/qa2/32k | sempointer_vs_bm25_4 | 6 | +0.00 | [+0.00, +0.00] | +191.17 | +0.359 |
| babilong | data/qa2/32k | sempointer_vs_bm25_matched_budget | 6 | +0.00 | [+0.00, +0.00] | +0.00 | +0.124 |
| babilong | data/qa2/32k | sempointer_vs_pointer_off | 6 | +0.00 | [+0.00, +0.00] | +1231.17 | +1.224 |
| babilong | data/qa2/64k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +2440.00 | +1.999 |
| babilong | data/qa2/64k | sempointer_vs_bm25_4 | 4 | +0.00 | [+0.00, +0.00] | +1400.75 | +1.232 |
| babilong | data/qa2/64k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [+0.00, +0.00] | +1.25 | -0.366 |
| babilong | data/qa2/64k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +2440.00 | +2.014 |
| babilong | data/qa2/8k | sempointer_vs_bm25_2 | 8 | +0.00 | [+0.00, +0.00] | +268.12 | +0.156 |
| babilong | data/qa2/8k | sempointer_vs_bm25_4 | 8 | +0.00 | [+0.00, +0.00] | -769.88 | -1.040 |
| babilong | data/qa2/8k | sempointer_vs_bm25_matched_budget | 8 | +0.00 | [+0.00, +0.00] | +0.00 | -0.197 |
| babilong | data/qa2/8k | sempointer_vs_full_context | 8 | -12.50 | [-37.50, +0.00] | -6187.75 | -5.995 |
| babilong | data/qa2/8k | sempointer_vs_pointer_off | 8 | +0.00 | [+0.00, +0.00] | +268.12 | +0.142 |
| babilong | data/qa3/128k | sempointer_vs_bm25_2 | 6 | +0.00 | [+0.00, +0.00] | +5559.83 | +4.730 |
| babilong | data/qa3/128k | sempointer_vs_bm25_4 | 6 | +0.00 | [+0.00, +0.00] | +4560.33 | +3.926 |
| babilong | data/qa3/128k | sempointer_vs_bm25_matched_budget | 6 | +0.00 | [+0.00, +0.00] | +0.67 | -0.377 |
| babilong | data/qa3/128k | sempointer_vs_pointer_off | 6 | +0.00 | [+0.00, +0.00] | +5213.33 | +4.737 |
| babilong | data/qa3/16k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +847.75 | +0.588 |
| babilong | data/qa3/16k | sempointer_vs_bm25_4 | 4 | +0.00 | [+0.00, +0.00] | -192.00 | -0.123 |
| babilong | data/qa3/16k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [+0.00, +0.00] | +0.25 | +0.209 |
| babilong | data/qa3/16k | sempointer_vs_full_context | 4 | +0.00 | [+0.00, +0.00] | -13685.25 | -13.842 |
| babilong | data/qa3/16k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +587.75 | +0.128 |
| babilong | data/qa3/256k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +10804.50 | +10.145 |
| babilong | data/qa3/256k | sempointer_vs_bm25_4 | 4 | +0.00 | [+0.00, +0.00] | +9763.25 | +9.641 |
| babilong | data/qa3/256k | sempointer_vs_bm25_matched_budget | 4 | -25.00 | [-75.00, +0.00] | +0.00 | -1.282 |
| babilong | data/qa3/256k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +10544.25 | +9.926 |
| babilong | data/qa3/32k | sempointer_vs_bm25_2 | 6 | +0.00 | [+0.00, +0.00] | +1719.33 | +1.225 |
| babilong | data/qa3/32k | sempointer_vs_bm25_4 | 6 | -16.67 | [-50.00, +0.00] | +679.83 | +0.063 |
| babilong | data/qa3/32k | sempointer_vs_bm25_matched_budget | 6 | +0.00 | [-50.00, +50.00] | +0.33 | -0.032 |
| babilong | data/qa3/32k | sempointer_vs_pointer_off | 6 | +0.00 | [+0.00, +0.00] | +1199.67 | +0.433 |
| babilong | data/qa3/4k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +378.50 | +0.046 |
| babilong | data/qa3/4k | sempointer_vs_bm25_4 | 4 | -50.00 | [-100.00, +0.00] | -562.25 | -0.469 |
| babilong | data/qa3/4k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [+0.00, +0.00] | +0.00 | -0.373 |
| babilong | data/qa3/4k | sempointer_vs_full_context | 4 | -25.00 | [-75.00, +0.00] | -2436.50 | -2.090 |
| babilong | data/qa3/4k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +119.00 | -0.055 |
| babilong | data/qa3/64k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +3269.00 | +2.812 |
| babilong | data/qa3/64k | sempointer_vs_bm25_4 | 4 | +0.00 | [+0.00, +0.00] | +2229.25 | +2.544 |
| babilong | data/qa3/64k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [+0.00, +0.00] | +0.00 | +0.306 |
| babilong | data/qa3/64k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +2489.25 | +2.622 |
| babilong | data/qa3/8k | sempointer_vs_bm25_2 | 2 | +0.00 | [+0.00, +0.00] | +252.00 | -0.272 |
| babilong | data/qa3/8k | sempointer_vs_bm25_4 | 2 | +0.00 | [+0.00, +0.00] | -702.50 | -0.425 |
| babilong | data/qa3/8k | sempointer_vs_bm25_matched_budget | 2 | +0.00 | [+0.00, +0.00] | +0.00 | -0.009 |
| babilong | data/qa3/8k | sempointer_vs_full_context | 2 | -50.00 | [-100.00, +0.00] | -5689.50 | -4.335 |
| babilong | data/qa3/8k | sempointer_vs_pointer_off | 2 | +0.00 | [+0.00, +0.00] | +252.00 | -0.248 |
| babilong | data/qa4/128k | sempointer_vs_bm25_2 | 3 | +0.00 | [+0.00, +0.00] | +5251.00 | +4.638 |
| babilong | data/qa4/128k | sempointer_vs_bm25_4 | 3 | +0.00 | [+0.00, +0.00] | +4209.67 | +3.830 |
| babilong | data/qa4/128k | sempointer_vs_bm25_matched_budget | 3 | +0.00 | [+0.00, +0.00] | +0.00 | -0.322 |
| babilong | data/qa4/128k | sempointer_vs_pointer_off | 3 | +0.00 | [+0.00, +0.00] | +5251.00 | +4.640 |
| babilong | data/qa4/16k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +592.75 | +0.460 |
| babilong | data/qa4/16k | sempointer_vs_bm25_4 | 4 | -25.00 | [-75.00, +0.00] | -445.50 | -0.341 |
| babilong | data/qa4/16k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [-75.00, +75.00] | +0.00 | -0.049 |
| babilong | data/qa4/16k | sempointer_vs_full_context | 3 | +33.33 | [+0.00, +100.00] | -13620.33 | -14.156 |
| babilong | data/qa4/16k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +592.75 | +0.446 |
| babilong | data/qa4/256k | sempointer_vs_bm25_2 | 7 | +0.00 | [+0.00, +0.00] | +10822.14 | +10.243 |
| babilong | data/qa4/256k | sempointer_vs_bm25_4 | 7 | +0.00 | [+0.00, +0.00] | +9781.29 | +9.715 |
| babilong | data/qa4/256k | sempointer_vs_bm25_matched_budget | 7 | +0.00 | [+0.00, +0.00] | +0.00 | -0.729 |
| babilong | data/qa4/256k | sempointer_vs_pointer_off | 7 | +0.00 | [+0.00, +0.00] | +10673.57 | +10.213 |
| babilong | data/qa4/32k | sempointer_vs_bm25_2 | 8 | +12.50 | [+0.00, +37.50] | +1238.25 | +0.927 |
| babilong | data/qa4/32k | sempointer_vs_bm25_4 | 8 | +12.50 | [+0.00, +37.50] | +199.12 | +0.125 |
| babilong | data/qa4/32k | sempointer_vs_bm25_matched_budget | 8 | +12.50 | [+0.00, +37.50] | +0.00 | -0.219 |
| babilong | data/qa4/32k | sempointer_vs_pointer_off | 8 | +12.50 | [+0.00, +37.50] | +1238.25 | +0.926 |
| babilong | data/qa4/4k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +129.00 | +0.053 |
| babilong | data/qa4/4k | sempointer_vs_bm25_4 | 4 | +0.00 | [+0.00, +0.00] | -908.75 | -0.747 |
| babilong | data/qa4/4k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [+0.00, +0.00] | +0.00 | -0.040 |
| babilong | data/qa4/4k | sempointer_vs_full_context | 4 | +0.00 | [+0.00, +0.00] | -2549.25 | -2.168 |
| babilong | data/qa4/4k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +129.00 | +0.059 |
| babilong | data/qa4/64k | sempointer_vs_bm25_2 | 2 | +0.00 | [+0.00, +0.00] | +2431.00 | +1.965 |
| babilong | data/qa4/64k | sempointer_vs_bm25_4 | 2 | +0.00 | [+0.00, +0.00] | +1390.50 | +1.142 |
| babilong | data/qa4/64k | sempointer_vs_bm25_matched_budget | 2 | +0.00 | [+0.00, +0.00] | +0.00 | -0.143 |
| babilong | data/qa4/64k | sempointer_vs_pointer_off | 2 | +0.00 | [+0.00, +0.00] | +2431.00 | +1.965 |
| babilong | data/qa4/8k | sempointer_vs_bm25_2 | 2 | +0.00 | [+0.00, +0.00] | +263.00 | +0.157 |
| babilong | data/qa4/8k | sempointer_vs_bm25_4 | 2 | +0.00 | [+0.00, +0.00] | -775.00 | -0.657 |
| babilong | data/qa4/8k | sempointer_vs_bm25_matched_budget | 2 | +0.00 | [+0.00, +0.00] | +0.50 | -0.045 |
| babilong | data/qa4/8k | sempointer_vs_full_context | 2 | +0.00 | [+0.00, +0.00] | -5941.00 | -5.505 |
| babilong | data/qa4/8k | sempointer_vs_pointer_off | 2 | +0.00 | [+0.00, +0.00] | +263.00 | +0.153 |
| babilong | data/qa5/128k | sempointer_vs_bm25_2 | 6 | +16.67 | [+0.00, +50.00] | +5543.00 | +4.841 |
| babilong | data/qa5/128k | sempointer_vs_bm25_4 | 6 | +16.67 | [+0.00, +50.00] | +4502.83 | +4.162 |
| babilong | data/qa5/128k | sempointer_vs_bm25_matched_budget | 6 | +16.67 | [+0.00, +50.00] | +0.00 | -0.294 |
| babilong | data/qa5/128k | sempointer_vs_pointer_off | 6 | +16.67 | [+0.00, +50.00] | +5196.50 | +4.617 |
| babilong | data/qa5/16k | sempointer_vs_bm25_2 | 2 | +0.00 | [+0.00, +0.00] | +555.50 | +0.415 |
| babilong | data/qa5/16k | sempointer_vs_bm25_4 | 2 | +0.00 | [+0.00, +0.00] | -484.50 | -0.391 |
| babilong | data/qa5/16k | sempointer_vs_bm25_matched_budget | 2 | +0.00 | [+0.00, +0.00] | +0.00 | -0.069 |
| babilong | data/qa5/16k | sempointer_vs_full_context | 2 | +0.00 | [+0.00, +0.00] | -13136.50 | -13.542 |
| babilong | data/qa5/16k | sempointer_vs_pointer_off | 2 | +0.00 | [+0.00, +0.00] | +555.50 | +0.396 |
| babilong | data/qa5/256k | sempointer_vs_bm25_2 | 4 | -25.00 | [-100.00, +50.00] | +11028.75 | +11.067 |
| babilong | data/qa5/256k | sempointer_vs_bm25_4 | 4 | -50.00 | [-100.00, +0.00] | +9987.75 | +10.442 |
| babilong | data/qa5/256k | sempointer_vs_bm25_matched_budget | 4 | -75.00 | [-100.00, -25.00] | +0.25 | -0.289 |
| babilong | data/qa5/256k | sempointer_vs_pointer_off | 4 | -25.00 | [-100.00, +50.00] | +10508.50 | +10.677 |
| babilong | data/qa5/32k | sempointer_vs_bm25_2 | 5 | +20.00 | [+0.00, +60.00] | +1209.00 | +0.822 |
| babilong | data/qa5/32k | sempointer_vs_bm25_4 | 5 | +0.00 | [+0.00, +0.00] | +169.00 | +0.113 |
| babilong | data/qa5/32k | sempointer_vs_bm25_matched_budget | 5 | +0.00 | [+0.00, +0.00] | +0.00 | -0.088 |
| babilong | data/qa5/32k | sempointer_vs_pointer_off | 5 | +20.00 | [+0.00, +60.00] | +1209.00 | +0.801 |
| babilong | data/qa5/4k | sempointer_vs_bm25_2 | 1 | +0.00 | [+0.00, +0.00] | +89.00 | +0.070 |
| babilong | data/qa5/4k | sempointer_vs_bm25_4 | 1 | +0.00 | [+0.00, +0.00] | -947.00 | -0.618 |
| babilong | data/qa5/4k | sempointer_vs_bm25_matched_budget | 1 | -100.00 | [-100.00, -100.00] | +2.00 | -0.009 |
| babilong | data/qa5/4k | sempointer_vs_full_context | 1 | -100.00 | [-100.00, -100.00] | -1983.00 | -1.342 |
| babilong | data/qa5/4k | sempointer_vs_pointer_off | 1 | +0.00 | [+0.00, +0.00] | +89.00 | +0.015 |
| babilong | data/qa5/64k | sempointer_vs_bm25_2 | 5 | +0.00 | [+0.00, +0.00] | +2750.80 | +2.198 |
| babilong | data/qa5/64k | sempointer_vs_bm25_4 | 5 | +20.00 | [+0.00, +60.00] | +1710.40 | +1.339 |
| babilong | data/qa5/64k | sempointer_vs_bm25_matched_budget | 5 | +20.00 | [+0.00, +60.00] | +0.00 | -0.266 |
| babilong | data/qa5/64k | sempointer_vs_pointer_off | 5 | +0.00 | [+0.00, +0.00] | +2542.80 | +2.106 |
| babilong | data/qa5/8k | sempointer_vs_bm25_2 | 7 | +0.00 | [-42.86, +42.86] | +270.71 | +0.157 |
| babilong | data/qa5/8k | sempointer_vs_bm25_4 | 7 | +0.00 | [-57.14, +57.14] | -757.14 | -0.644 |
| babilong | data/qa5/8k | sempointer_vs_bm25_matched_budget | 7 | +0.00 | [-42.86, +42.86] | +0.00 | -0.055 |
| babilong | data/qa5/8k | sempointer_vs_full_context | 7 | +14.29 | [-28.57, +57.14] | -6278.43 | -5.813 |
| babilong | data/qa5/8k | sempointer_vs_pointer_off | 7 | +0.00 | [-42.86, +42.86] | +270.71 | +0.156 |
| babilong | data/qa6/128k | sempointer_vs_bm25_2 | 7 | +0.00 | [-42.86, +42.86] | +5322.43 | +4.618 |
| babilong | data/qa6/128k | sempointer_vs_bm25_4 | 7 | +14.29 | [-28.57, +57.14] | +4282.14 | +3.898 |
| babilong | data/qa6/128k | sempointer_vs_bm25_matched_budget | 7 | +28.57 | [+0.00, +57.14] | +0.29 | -0.456 |
| babilong | data/qa6/128k | sempointer_vs_pointer_off | 7 | +0.00 | [-42.86, +42.86] | +5173.71 | +4.493 |
| babilong | data/qa6/16k | sempointer_vs_bm25_2 | 3 | -33.33 | [-100.00, +0.00] | +609.67 | +0.469 |
| babilong | data/qa6/16k | sempointer_vs_bm25_4 | 3 | +0.00 | [+0.00, +0.00] | -428.33 | -0.334 |
| babilong | data/qa6/16k | sempointer_vs_bm25_matched_budget | 3 | +33.33 | [+0.00, +100.00] | +0.00 | -0.254 |
| babilong | data/qa6/16k | sempointer_vs_full_context | 3 | +66.67 | [+0.00, +100.00] | -14158.67 | -15.802 |
| babilong | data/qa6/16k | sempointer_vs_pointer_off | 3 | -33.33 | [-100.00, +0.00] | +609.67 | +0.459 |
| babilong | data/qa6/256k | sempointer_vs_bm25_2 | 3 | +33.33 | [+0.00, +100.00] | +10584.00 | +10.155 |
| babilong | data/qa6/256k | sempointer_vs_bm25_4 | 3 | +0.00 | [+0.00, +0.00] | +9542.00 | +9.548 |
| babilong | data/qa6/256k | sempointer_vs_bm25_matched_budget | 3 | +33.33 | [+0.00, +100.00] | +0.00 | -0.921 |
| babilong | data/qa6/256k | sempointer_vs_pointer_off | 3 | +33.33 | [+0.00, +100.00] | +10584.00 | +10.152 |
| babilong | data/qa6/32k | sempointer_vs_bm25_2 | 5 | +0.00 | [+0.00, +0.00] | +1234.60 | +0.922 |
| babilong | data/qa6/32k | sempointer_vs_bm25_4 | 5 | +0.00 | [+0.00, +0.00] | +196.60 | +0.121 |
| babilong | data/qa6/32k | sempointer_vs_bm25_matched_budget | 5 | +20.00 | [+0.00, +60.00] | +0.00 | -0.092 |
| babilong | data/qa6/32k | sempointer_vs_pointer_off | 5 | +0.00 | [+0.00, +0.00] | +1234.60 | +0.917 |
| babilong | data/qa6/4k | sempointer_vs_bm25_2 | 4 | +25.00 | [+0.00, +75.00] | +124.00 | -0.080 |
| babilong | data/qa6/4k | sempointer_vs_bm25_4 | 4 | +25.00 | [+0.00, +75.00] | -914.25 | -0.907 |
| babilong | data/qa6/4k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [+0.00, +0.00] | +0.25 | -0.042 |
| babilong | data/qa6/4k | sempointer_vs_full_context | 4 | +0.00 | [+0.00, +0.00] | -2551.25 | -2.140 |
| babilong | data/qa6/4k | sempointer_vs_pointer_off | 4 | +25.00 | [+0.00, +75.00] | +124.00 | -0.080 |
| babilong | data/qa6/64k | sempointer_vs_bm25_2 | 5 | +0.00 | [+0.00, +0.00] | +2581.40 | +2.121 |
| babilong | data/qa6/64k | sempointer_vs_bm25_4 | 5 | +0.00 | [+0.00, +0.00] | +1542.40 | +1.316 |
| babilong | data/qa6/64k | sempointer_vs_bm25_matched_budget | 5 | +0.00 | [+0.00, +0.00] | +0.00 | -0.154 |
| babilong | data/qa6/64k | sempointer_vs_pointer_off | 5 | +0.00 | [+0.00, +0.00] | +2581.40 | +2.120 |
| babilong | data/qa6/8k | sempointer_vs_bm25_2 | 3 | +66.67 | [+0.00, +100.00] | +273.33 | -0.489 |
| babilong | data/qa6/8k | sempointer_vs_bm25_4 | 3 | +33.33 | [+0.00, +100.00] | -763.67 | -1.097 |
| babilong | data/qa6/8k | sempointer_vs_bm25_matched_budget | 3 | +0.00 | [+0.00, +0.00] | +0.00 | -0.030 |
| babilong | data/qa6/8k | sempointer_vs_full_context | 3 | +0.00 | [+0.00, +0.00] | -6139.00 | -5.594 |
| babilong | data/qa6/8k | sempointer_vs_pointer_off | 3 | +66.67 | [+0.00, +100.00] | +273.33 | -0.498 |
| babilong | data/qa7/128k | sempointer_vs_bm25_2 | 3 | +0.00 | [+0.00, +0.00] | +5615.33 | +4.957 |
| babilong | data/qa7/128k | sempointer_vs_bm25_4 | 3 | +0.00 | [+0.00, +0.00] | +4573.67 | +4.369 |
| babilong | data/qa7/128k | sempointer_vs_bm25_matched_budget | 3 | -33.33 | [-100.00, +0.00] | +0.00 | -0.182 |
| babilong | data/qa7/128k | sempointer_vs_pointer_off | 3 | +0.00 | [+0.00, +0.00] | +5268.00 | +4.677 |
| babilong | data/qa7/16k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +598.50 | +0.377 |
| babilong | data/qa7/16k | sempointer_vs_bm25_4 | 4 | -25.00 | [-75.00, +0.00] | -440.50 | -0.344 |
| babilong | data/qa7/16k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [+0.00, +0.00] | +0.00 | -0.061 |
| babilong | data/qa7/16k | sempointer_vs_full_context | 4 | +0.00 | [+0.00, +0.00] | -13915.00 | -14.374 |
| babilong | data/qa7/16k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +598.50 | +0.378 |
| babilong | data/qa7/256k | sempointer_vs_bm25_2 | 7 | +0.00 | [-42.86, +42.86] | +11385.14 | +10.965 |
| babilong | data/qa7/256k | sempointer_vs_bm25_4 | 7 | -42.86 | [-85.71, -14.29] | +10343.29 | +10.267 |
| babilong | data/qa7/256k | sempointer_vs_bm25_matched_budget | 7 | -14.29 | [-42.86, +0.00] | +0.00 | -0.851 |
| babilong | data/qa7/256k | sempointer_vs_pointer_off | 7 | -28.57 | [-57.14, +0.00] | +10641.14 | +10.487 |
| babilong | data/qa7/32k | sempointer_vs_bm25_2 | 3 | +0.00 | [+0.00, +0.00] | +1585.67 | +1.133 |
| babilong | data/qa7/32k | sempointer_vs_bm25_4 | 3 | +0.00 | [+0.00, +0.00] | +545.67 | +0.439 |
| babilong | data/qa7/32k | sempointer_vs_bm25_matched_budget | 3 | +0.00 | [+0.00, +0.00] | +0.00 | -0.088 |
| babilong | data/qa7/32k | sempointer_vs_pointer_off | 3 | +0.00 | [+0.00, +0.00] | +1239.33 | +0.979 |
| babilong | data/qa7/4k | sempointer_vs_bm25_2 | 4 | +25.00 | [+0.00, +75.00] | +122.00 | +0.086 |
| babilong | data/qa7/4k | sempointer_vs_bm25_4 | 4 | +50.00 | [+0.00, +100.00] | -916.00 | -0.723 |
| babilong | data/qa7/4k | sempointer_vs_bm25_matched_budget | 4 | +50.00 | [+0.00, +100.00] | +0.00 | -0.060 |
| babilong | data/qa7/4k | sempointer_vs_full_context | 4 | +50.00 | [+0.00, +100.00] | -2551.25 | -2.096 |
| babilong | data/qa7/4k | sempointer_vs_pointer_off | 4 | +25.00 | [+0.00, +75.00] | +122.00 | +0.078 |
| babilong | data/qa7/64k | sempointer_vs_bm25_2 | 7 | +0.00 | [+0.00, +0.00] | +2939.00 | +2.184 |
| babilong | data/qa7/64k | sempointer_vs_bm25_4 | 7 | -14.29 | [-42.86, +0.00] | +1898.71 | +1.638 |
| babilong | data/qa7/64k | sempointer_vs_bm25_matched_budget | 7 | +0.00 | [+0.00, +0.00] | +0.00 | -0.162 |
| babilong | data/qa7/64k | sempointer_vs_pointer_off | 7 | -14.29 | [-42.86, +0.00] | +2493.00 | +1.925 |
| babilong | data/qa7/8k | sempointer_vs_bm25_2 | 2 | +0.00 | [+0.00, +0.00] | +284.00 | +0.171 |
| babilong | data/qa7/8k | sempointer_vs_bm25_4 | 2 | +0.00 | [+0.00, +0.00] | -738.50 | -0.607 |
| babilong | data/qa7/8k | sempointer_vs_bm25_matched_budget | 2 | +0.00 | [+0.00, +0.00] | +0.00 | -0.044 |
| babilong | data/qa7/8k | sempointer_vs_full_context | 2 | +0.00 | [+0.00, +0.00] | -6512.00 | -6.074 |
| babilong | data/qa7/8k | sempointer_vs_pointer_off | 2 | +0.00 | [+0.00, +0.00] | +284.00 | +0.168 |
| babilong | data/qa8/128k | sempointer_vs_bm25_2 | 7 | +14.29 | [+0.00, +42.86] | +6015.57 | +5.373 |
| babilong | data/qa8/128k | sempointer_vs_bm25_4 | 7 | +14.29 | [+0.00, +42.86] | +4974.43 | +4.588 |
| babilong | data/qa8/128k | sempointer_vs_bm25_matched_budget | 7 | -14.29 | [-42.86, +0.00] | +0.00 | -0.261 |
| babilong | data/qa8/128k | sempointer_vs_pointer_off | 7 | +14.29 | [+0.00, +42.86] | +5123.29 | +4.701 |
| babilong | data/qa8/16k | sempointer_vs_bm25_2 | 3 | +0.00 | [+0.00, +0.00] | +601.67 | +0.457 |
| babilong | data/qa8/16k | sempointer_vs_bm25_4 | 3 | +0.00 | [-100.00, +100.00] | -438.33 | -0.333 |
| babilong | data/qa8/16k | sempointer_vs_bm25_matched_budget | 3 | +33.33 | [+0.00, +100.00] | +0.00 | -0.083 |
| babilong | data/qa8/16k | sempointer_vs_full_context | 3 | +33.33 | [+0.00, +100.00] | -13979.33 | -14.371 |
| babilong | data/qa8/16k | sempointer_vs_pointer_off | 3 | +0.00 | [+0.00, +0.00] | +601.67 | +0.463 |
| babilong | data/qa8/256k | sempointer_vs_bm25_2 | 3 | -33.33 | [-100.00, +0.00] | +10850.00 | +10.974 |
| babilong | data/qa8/256k | sempointer_vs_bm25_4 | 3 | +0.00 | [+0.00, +0.00] | +9809.33 | +9.890 |
| babilong | data/qa8/256k | sempointer_vs_bm25_matched_budget | 3 | +0.00 | [+0.00, +0.00] | +0.00 | -0.411 |
| babilong | data/qa8/256k | sempointer_vs_pointer_off | 3 | +0.00 | [+0.00, +0.00] | +10503.00 | +10.665 |
| babilong | data/qa8/32k | sempointer_vs_bm25_2 | 2 | +0.00 | [+0.00, +0.00] | +1747.50 | +1.536 |
| babilong | data/qa8/32k | sempointer_vs_bm25_4 | 2 | +0.00 | [+0.00, +0.00] | +707.00 | -0.275 |
| babilong | data/qa8/32k | sempointer_vs_bm25_matched_budget | 2 | +0.00 | [+0.00, +0.00] | +0.00 | -0.909 |
| babilong | data/qa8/32k | sempointer_vs_pointer_off | 2 | +0.00 | [+0.00, +0.00] | +1227.50 | +1.121 |
| babilong | data/qa8/4k | sempointer_vs_bm25_2 | 2 | +0.00 | [+0.00, +0.00] | +618.00 | +0.452 |
| babilong | data/qa8/4k | sempointer_vs_bm25_4 | 2 | +0.00 | [+0.00, +0.00] | -420.00 | -0.361 |
| babilong | data/qa8/4k | sempointer_vs_bm25_matched_budget | 2 | +0.00 | [+0.00, +0.00] | +0.00 | -0.062 |
| babilong | data/qa8/4k | sempointer_vs_full_context | 2 | +0.00 | [+0.00, +0.00] | -1914.00 | -1.602 |
| babilong | data/qa8/4k | sempointer_vs_pointer_off | 2 | +0.00 | [+0.00, +0.00] | +99.00 | +0.052 |
| babilong | data/qa8/64k | sempointer_vs_bm25_2 | 8 | +0.00 | [+0.00, +0.00] | +2904.62 | +2.083 |
| babilong | data/qa8/64k | sempointer_vs_bm25_4 | 8 | +0.00 | [+0.00, +0.00] | +1864.62 | +1.595 |
| babilong | data/qa8/64k | sempointer_vs_bm25_matched_budget | 8 | +0.00 | [+0.00, +0.00] | +0.00 | -0.178 |
| babilong | data/qa8/64k | sempointer_vs_pointer_off | 8 | +0.00 | [+0.00, +0.00] | +2514.25 | +1.764 |
| babilong | data/qa8/8k | sempointer_vs_bm25_2 | 5 | +0.00 | [+0.00, +0.00] | +272.80 | +0.152 |
| babilong | data/qa8/8k | sempointer_vs_bm25_4 | 5 | -20.00 | [-60.00, +0.00] | -765.00 | -0.692 |
| babilong | data/qa8/8k | sempointer_vs_bm25_matched_budget | 5 | -20.00 | [-60.00, +0.00] | +0.00 | -0.029 |
| babilong | data/qa8/8k | sempointer_vs_full_context | 5 | -20.00 | [-60.00, +0.00] | -6312.20 | -5.875 |
| babilong | data/qa8/8k | sempointer_vs_pointer_off | 5 | +0.00 | [+0.00, +0.00] | +272.80 | +0.147 |
| babilong | data/qa9/128k | sempointer_vs_bm25_2 | 3 | +0.00 | [+0.00, +0.00] | +5217.33 | +4.531 |
| babilong | data/qa9/128k | sempointer_vs_bm25_4 | 3 | +0.00 | [+0.00, +0.00] | +4176.00 | +3.733 |
| babilong | data/qa9/128k | sempointer_vs_bm25_matched_budget | 3 | +0.00 | [+0.00, +0.00] | +0.00 | -0.285 |
| babilong | data/qa9/128k | sempointer_vs_pointer_off | 3 | +0.00 | [+0.00, +0.00] | +5217.33 | +4.532 |
| babilong | data/qa9/16k | sempointer_vs_bm25_2 | 2 | +50.00 | [+0.00, +100.00] | +587.50 | +0.146 |
| babilong | data/qa9/16k | sempointer_vs_bm25_4 | 2 | +50.00 | [+0.00, +100.00] | -450.50 | -0.641 |
| babilong | data/qa9/16k | sempointer_vs_bm25_matched_budget | 2 | +50.00 | [+0.00, +100.00] | +0.00 | -0.361 |
| babilong | data/qa9/16k | sempointer_vs_full_context | 2 | +50.00 | [+0.00, +100.00] | -13827.50 | -14.882 |
| babilong | data/qa9/16k | sempointer_vs_pointer_off | 2 | +50.00 | [+0.00, +100.00] | +587.50 | +0.144 |
| babilong | data/qa9/256k | sempointer_vs_bm25_2 | 7 | -14.29 | [-42.86, +0.00] | +10694.00 | +10.660 |
| babilong | data/qa9/256k | sempointer_vs_bm25_4 | 7 | -57.14 | [-85.71, -14.29] | +9652.57 | +9.907 |
| babilong | data/qa9/256k | sempointer_vs_bm25_matched_budget | 7 | +14.29 | [-28.57, +57.14] | +0.00 | -0.887 |
| babilong | data/qa9/256k | sempointer_vs_pointer_off | 7 | -28.57 | [-57.14, +0.00] | +10545.14 | +10.579 |
| babilong | data/qa9/32k | sempointer_vs_bm25_2 | 5 | +0.00 | [-60.00, +60.00] | +1383.20 | +0.926 |
| babilong | data/qa9/32k | sempointer_vs_bm25_4 | 5 | +0.00 | [+0.00, +0.00] | +344.80 | +0.273 |
| babilong | data/qa9/32k | sempointer_vs_bm25_matched_budget | 5 | +0.00 | [+0.00, +0.00] | +0.40 | -0.109 |
| babilong | data/qa9/32k | sempointer_vs_pointer_off | 5 | +20.00 | [+0.00, +60.00] | +1175.20 | +0.765 |
| babilong | data/qa9/4k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +119.00 | +0.030 |
| babilong | data/qa9/4k | sempointer_vs_bm25_4 | 4 | -25.00 | [-75.00, +0.00] | -918.75 | -0.576 |
| babilong | data/qa9/4k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [+0.00, +0.00] | +0.00 | -0.051 |
| babilong | data/qa9/4k | sempointer_vs_full_context | 4 | -25.00 | [-75.00, +0.00] | -2438.00 | -1.860 |
| babilong | data/qa9/4k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +119.00 | +0.031 |
| babilong | data/qa9/64k | sempointer_vs_bm25_2 | 5 | +0.00 | [+0.00, +0.00] | +2558.00 | +2.088 |
| babilong | data/qa9/64k | sempointer_vs_bm25_4 | 5 | +0.00 | [+0.00, +0.00] | +1517.60 | +1.293 |
| babilong | data/qa9/64k | sempointer_vs_bm25_matched_budget | 5 | +0.00 | [+0.00, +0.00] | +0.00 | -0.140 |
| babilong | data/qa9/64k | sempointer_vs_pointer_off | 5 | +0.00 | [+0.00, +0.00] | +2558.00 | +2.088 |
| babilong | data/qa9/8k | sempointer_vs_bm25_2 | 4 | +0.00 | [+0.00, +0.00] | +251.50 | +0.157 |
| babilong | data/qa9/8k | sempointer_vs_bm25_4 | 4 | +0.00 | [+0.00, +0.00] | -786.25 | -0.672 |
| babilong | data/qa9/8k | sempointer_vs_bm25_matched_budget | 4 | +0.00 | [+0.00, +0.00] | +0.00 | -0.047 |
| babilong | data/qa9/8k | sempointer_vs_full_context | 4 | +50.00 | [+0.00, +100.00] | -5720.75 | -5.568 |
| babilong | data/qa9/8k | sempointer_vs_pointer_off | 4 | +0.00 | [+0.00, +0.00] | +251.50 | +0.169 |
| infinite | code_debug | sempointer_vs_bm25_2 | 26 | +7.69 | [-7.69, +23.08] | +6676.42 | +6.136 |
| infinite | code_debug | sempointer_vs_bm25_4 | 26 | -7.69 | [-23.17, +11.54] | +5652.12 | +5.315 |
| infinite | code_debug | sempointer_vs_bm25_matched_budget | 26 | -23.08 | [-46.15, +0.00] | +0.35 | -0.352 |
| infinite | code_debug | sempointer_vs_pointer_off | 26 | -7.69 | [-23.17, +11.54] | +5716.96 | +5.345 |
| infinite | code_run | sempointer_vs_bm25_2 | 25 | +0.00 | [+0.00, +0.00] | +4032.48 | +3.525 |
| infinite | code_run | sempointer_vs_bm25_4 | 25 | +0.00 | [+0.00, +0.00] | +2992.20 | +2.728 |
| infinite | code_run | sempointer_vs_bm25_matched_budget | 25 | +0.00 | [+0.00, +0.00] | +0.88 | -5.185 |
| infinite | code_run | sempointer_vs_pointer_off | 25 | +0.00 | [+0.00, +0.00] | +4032.48 | +3.522 |
| infinite | kv_retrieval | sempointer_vs_bm25_2 | 25 | +4.00 | [+0.00, +12.00] | +7161.48 | +6.632 |
| infinite | kv_retrieval | sempointer_vs_bm25_4 | 25 | +4.00 | [+0.00, +12.00] | +6121.76 | +6.088 |
| infinite | kv_retrieval | sempointer_vs_bm25_matched_budget | 25 | +0.00 | [+0.00, +0.00] | +0.00 | -0.313 |
| infinite | kv_retrieval | sempointer_vs_pointer_off | 25 | +4.00 | [+0.00, +12.00] | +7119.84 | +6.568 |
| infinite | longbook_choice_eng | sempointer_vs_bm25_2 | 25 | +8.00 | [-12.00, +32.00] | +7385.32 | +6.883 |
| infinite | longbook_choice_eng | sempointer_vs_bm25_4 | 25 | -8.00 | [-24.00, +4.00] | +6344.32 | +6.068 |
| infinite | longbook_choice_eng | sempointer_vs_bm25_matched_budget | 25 | -8.00 | [-24.00, +4.00] | +0.04 | -0.492 |
| infinite | longbook_choice_eng | sempointer_vs_pointer_off | 25 | -4.00 | [-16.00, +8.00] | +6427.64 | +6.126 |
| infinite | longbook_qa_chn | sempointer_vs_bm25_2 | 24 | +2.36 | [-7.37, +15.07] | +1691.62 | +0.434 |
| infinite | longbook_qa_chn | sempointer_vs_bm25_4 | 24 | +0.22 | [-0.21, +0.88] | +653.25 | +0.569 |
| infinite | longbook_qa_chn | sempointer_vs_bm25_matched_budget | 24 | +0.22 | [-0.21, +0.88] | +0.00 | -0.094 |
| infinite | longbook_qa_chn | sempointer_vs_pointer_off | 24 | -0.70 | [-2.02, +0.00] | +739.88 | +0.657 |
| infinite | longbook_qa_eng | sempointer_vs_bm25_2 | 25 | +2.88 | [-2.23, +7.73] | +6146.12 | +5.930 |
| infinite | longbook_qa_eng | sempointer_vs_bm25_4 | 25 | -9.03 | [-21.70, +1.48] | +5105.20 | +5.132 |
| infinite | longbook_qa_eng | sempointer_vs_bm25_matched_budget | 25 | -7.92 | [-21.65, +3.38] | +0.24 | -0.440 |
| infinite | longbook_qa_eng | sempointer_vs_pointer_off | 25 | +2.24 | [-2.59, +7.24] | +5771.32 | +5.575 |
| infinite | longbook_sum_eng | sempointer_vs_bm25_2 | 25 | +1.06 | [+0.21, +2.23] | +5928.76 | +7.482 |
| infinite | longbook_sum_eng | sempointer_vs_bm25_4 | 25 | +0.27 | [-1.01, +1.70] | +4887.88 | +5.844 |
| infinite | longbook_sum_eng | sempointer_vs_bm25_matched_budget | 25 | -0.91 | [-2.11, +0.46] | +0.04 | -1.168 |
| infinite | longbook_sum_eng | sempointer_vs_pointer_off | 25 | +0.70 | [-0.50, +2.01] | +5762.36 | +6.905 |
| infinite | longdialogue_qa_eng | sempointer_vs_bm25_2 | 25 | -2.00 | [-18.00, +16.00] | +4926.96 | +4.183 |
| infinite | longdialogue_qa_eng | sempointer_vs_bm25_4 | 25 | -4.00 | [-20.00, +12.00] | +3886.32 | +3.420 |
| infinite | longdialogue_qa_eng | sempointer_vs_bm25_matched_budget | 25 | -4.00 | [-12.00, +0.00] | +0.08 | -0.242 |
| infinite | longdialogue_qa_eng | sempointer_vs_pointer_off | 25 | -2.00 | [-18.00, +16.00] | +4593.92 | +3.853 |
| infinite | math_calc | sempointer_vs_bm25_2 | 25 | +0.00 | [+0.00, +0.00] | +3161.12 | +3.541 |
| infinite | math_calc | sempointer_vs_bm25_4 | 25 | +0.00 | [+0.00, +0.00] | +2124.20 | +0.041 |
| infinite | math_calc | sempointer_vs_bm25_matched_budget | 25 | +0.00 | [+0.00, +0.00] | +0.32 | -1.684 |
| infinite | math_calc | sempointer_vs_pointer_off | 25 | +0.00 | [+0.00, +0.00] | +2124.20 | +0.048 |
| infinite | math_find | sempointer_vs_bm25_2 | 25 | -4.00 | [-20.00, +12.00] | +5791.80 | +5.161 |
| infinite | math_find | sempointer_vs_bm25_4 | 25 | -8.00 | [-24.00, +8.00] | +4755.20 | +4.372 |
| infinite | math_find | sempointer_vs_bm25_matched_budget | 25 | -8.00 | [-24.00, +4.00] | +0.08 | -0.213 |
| infinite | math_find | sempointer_vs_pointer_off | 25 | -8.00 | [-24.00, +8.00] | +4755.20 | +4.364 |
| infinite | number_string | sempointer_vs_bm25_2 | 25 | +0.00 | [+0.00, +0.00] | +3303.52 | +2.762 |
| infinite | number_string | sempointer_vs_bm25_4 | 25 | +0.00 | [+0.00, +0.00] | +2265.52 | +1.962 |
| infinite | number_string | sempointer_vs_bm25_matched_budget | 25 | +0.00 | [+0.00, +0.00] | +0.00 | -0.162 |
| infinite | number_string | sempointer_vs_pointer_off | 25 | +0.00 | [+0.00, +0.00] | +3303.52 | +2.760 |
| infinite | passkey | sempointer_vs_bm25_2 | 25 | +0.00 | [+0.00, +0.00] | +3303.32 | +2.762 |
| infinite | passkey | sempointer_vs_bm25_4 | 25 | +0.00 | [+0.00, +0.00] | +2265.32 | +1.966 |
| infinite | passkey | sempointer_vs_bm25_matched_budget | 25 | +0.00 | [+0.00, +0.00] | +0.00 | -0.170 |
| infinite | passkey | sempointer_vs_pointer_off | 25 | +0.00 | [+0.00, +0.00] | +3303.32 | +2.754 |
| leval | codeU | sempointer_vs_bm25_2 | 17 | +0.00 | [+0.00, +0.00] | +1701.88 | +2.236 |
| leval | codeU | sempointer_vs_bm25_4 | 17 | +0.00 | [+0.00, +0.00] | +664.00 | -4.519 |
| leval | codeU | sempointer_vs_bm25_matched_budget | 17 | -5.88 | [-17.65, +0.00] | +0.24 | -3.271 |
| leval | codeU | sempointer_vs_pointer_off | 17 | +0.00 | [+0.00, +0.00] | +1030.53 | -0.013 |
| leval | coursera | sempointer_vs_bm25_2 | 17 | -5.88 | [-17.65, +0.00] | +886.06 | +0.570 |
| leval | coursera | sempointer_vs_bm25_4 | 17 | -5.88 | [-17.65, +0.00] | -134.76 | -0.269 |
| leval | coursera | sempointer_vs_bm25_matched_budget | 17 | +0.00 | [+0.00, +0.00] | +0.06 | -0.075 |
| leval | coursera | sempointer_vs_full_context | 17 | +0.00 | [-17.65, +17.65] | -4714.00 | -4.466 |
| leval | coursera | sempointer_vs_pointer_off | 17 | +0.00 | [+0.00, +0.00] | +214.00 | +0.065 |
| leval | financial_qa | sempointer_vs_bm25_2 | 17 | +0.35 | [-3.32, +3.43] | +593.47 | +0.170 |
| leval | financial_qa | sempointer_vs_bm25_4 | 17 | -7.74 | [-15.75, -0.39] | -427.12 | -1.769 |
| leval | financial_qa | sempointer_vs_bm25_matched_budget | 17 | -3.32 | [-8.84, +2.18] | +0.00 | -1.189 |
| leval | financial_qa | sempointer_vs_full_context | 17 | -7.33 | [-14.76, +0.38] | -4020.94 | -8.092 |
| leval | financial_qa | sempointer_vs_pointer_off | 17 | -1.28 | [-3.91, +0.36] | +183.53 | -1.007 |
| leval | gov_report_summ | sempointer_vs_bm25_2 | 11 | +0.15 | [-1.64, +1.88] | +429.36 | -1.083 |
| leval | gov_report_summ | sempointer_vs_bm25_4 | 11 | -1.16 | [-3.24, +0.83] | -608.73 | -2.172 |
| leval | gov_report_summ | sempointer_vs_bm25_matched_budget | 11 | +0.23 | [-2.33, +2.58] | +0.00 | -1.366 |
| leval | gov_report_summ | sempointer_vs_full_context | 10 | -1.25 | [-2.82, +0.21] | -3882.20 | -6.868 |
| leval | gov_report_summ | sempointer_vs_pointer_off | 11 | -0.04 | [-1.70, +1.53] | +240.82 | -1.114 |
| leval | gsm100 | sempointer_vs_bm25_2 | 17 | +0.00 | [+0.00, +0.00] | +751.88 | +0.925 |
| leval | gsm100 | sempointer_vs_bm25_4 | 17 | +0.00 | [+0.00, +0.00] | -282.18 | -0.256 |
| leval | gsm100 | sempointer_vs_bm25_matched_budget | 17 | +0.00 | [+0.00, +0.00] | +0.00 | -0.347 |
| leval | gsm100 | sempointer_vs_full_context | 17 | +0.00 | [+0.00, +0.00] | -3348.94 | -3.876 |
| leval | gsm100 | sempointer_vs_pointer_off | 17 | +0.00 | [+0.00, +0.00] | +145.47 | +0.417 |
| leval | legal_contract_qa | sempointer_vs_bm25_2 | 17 | +9.12 | [-3.08, +21.60] | +1819.18 | +4.069 |
| leval | legal_contract_qa | sempointer_vs_bm25_4 | 17 | -2.63 | [-10.52, +4.97] | +779.88 | +1.905 |
| leval | legal_contract_qa | sempointer_vs_bm25_matched_budget | 17 | -0.97 | [-4.80, +2.96] | +0.12 | +4.550 |
| leval | legal_contract_qa | sempointer_vs_full_context | 9 | -10.98 | [-24.09, +1.15] | -10155.56 | -7.949 |
| leval | legal_contract_qa | sempointer_vs_pointer_off | 17 | -3.24 | [-10.24, +2.52] | +963.18 | +2.858 |
| leval | meeting_summ | sempointer_vs_bm25_2 | 16 | -1.70 | [-5.37, +0.59] | +1058.38 | +1.432 |
| leval | meeting_summ | sempointer_vs_bm25_4 | 16 | -0.55 | [-5.11, +3.23] | +19.31 | -0.095 |
| leval | meeting_summ | sempointer_vs_bm25_matched_budget | 16 | -2.06 | [-5.82, +0.95] | +0.06 | -0.087 |
| leval | meeting_summ | sempointer_vs_full_context | 8 | -0.45 | [-7.03, +4.83] | -9475.62 | -12.203 |
| leval | meeting_summ | sempointer_vs_pointer_off | 16 | -0.76 | [-4.73, +2.20] | +668.69 | +0.929 |
| leval | multidoc_qa | sempointer_vs_bm25_2 | 16 | +4.18 | [-1.50, +10.18] | +225.94 | +0.118 |
| leval | multidoc_qa | sempointer_vs_bm25_4 | 16 | -7.89 | [-17.92, +1.66] | -812.06 | -0.435 |
| leval | multidoc_qa | sempointer_vs_bm25_matched_budget | 16 | -3.61 | [-13.63, +6.19] | +0.12 | +0.377 |
| leval | multidoc_qa | sempointer_vs_full_context | 16 | -9.54 | [-20.02, +0.88] | -1924.44 | -2.308 |
| leval | multidoc_qa | sempointer_vs_pointer_off | 16 | +2.16 | [-3.20, +6.76] | +96.19 | -0.024 |
| leval | narrative_qa | sempointer_vs_bm25_2 | 16 | +8.15 | [+1.32, +16.86] | +2576.94 | +1.900 |
| leval | narrative_qa | sempointer_vs_bm25_4 | 16 | -3.11 | [-14.04, +6.50] | +1537.25 | +1.407 |
| leval | narrative_qa | sempointer_vs_bm25_matched_budget | 16 | -6.51 | [-15.87, +0.75] | +0.25 | -0.023 |
| leval | narrative_qa | sempointer_vs_full_context | 2 | -23.53 | [-47.06, +0.00] | -7765.00 | -7.494 |
| leval | narrative_qa | sempointer_vs_pointer_off | 16 | +4.69 | [-0.33, +10.59] | +2252.00 | +1.756 |
| leval | natural_question | sempointer_vs_bm25_2 | 16 | +2.58 | [-18.75, +24.46] | +911.44 | +0.633 |
| leval | natural_question | sempointer_vs_bm25_4 | 16 | -11.90 | [-31.10, +5.37] | -127.31 | -0.025 |
| leval | natural_question | sempointer_vs_bm25_matched_budget | 16 | -6.70 | [-22.32, +6.25] | +0.19 | +0.051 |
| leval | natural_question | sempointer_vs_full_context | 8 | -5.68 | [-44.35, +30.01] | -7459.50 | -7.685 |
| leval | natural_question | sempointer_vs_pointer_off | 16 | +2.58 | [-18.75, +24.46] | +781.62 | +0.536 |
| leval | news_summ | sempointer_vs_bm25_2 | 8 | +1.04 | [-0.57, +2.91] | +681.88 | +0.968 |
| leval | news_summ | sempointer_vs_bm25_4 | 8 | -1.18 | [-2.45, +0.15] | -356.75 | -1.535 |
| leval | news_summ | sempointer_vs_bm25_matched_budget | 8 | -0.84 | [-2.23, +0.41] | +0.12 | -2.035 |
| leval | news_summ | sempointer_vs_full_context | 8 | -1.60 | [-3.13, -0.03] | -3580.25 | -7.371 |
| leval | news_summ | sempointer_vs_pointer_off | 8 | -0.37 | [-1.79, +1.03] | +162.38 | -0.531 |
| leval | paper_assistant | sempointer_vs_bm25_2 | 16 | +0.69 | [-0.08, +1.48] | +688.81 | +2.443 |
| leval | paper_assistant | sempointer_vs_bm25_4 | 16 | -0.83 | [-2.41, +0.43] | -349.69 | -0.769 |
| leval | paper_assistant | sempointer_vs_bm25_matched_budget | 16 | +0.39 | [-0.53, +1.21] | +0.00 | +0.731 |
| leval | paper_assistant | sempointer_vs_full_context | 15 | -1.96 | [-4.54, +0.06] | -7022.73 | -13.268 |
| leval | paper_assistant | sempointer_vs_pointer_off | 16 | +0.35 | [-0.64, +1.32] | +364.38 | +0.502 |
| leval | patent_summ | sempointer_vs_bm25_2 | 9 | +0.70 | [-0.68, +2.10] | +445.44 | +1.142 |
| leval | patent_summ | sempointer_vs_bm25_4 | 9 | -2.73 | [-7.35, +0.76] | -592.11 | -1.410 |
| leval | patent_summ | sempointer_vs_bm25_matched_budget | 9 | -0.75 | [-3.17, +1.35] | +0.11 | +1.153 |
| leval | patent_summ | sempointer_vs_full_context | 9 | -2.30 | [-9.72, +2.28] | -4755.33 | -8.169 |
| leval | patent_summ | sempointer_vs_pointer_off | 9 | +0.27 | [-1.19, +1.83] | +215.33 | -0.331 |
| leval | quality | sempointer_vs_bm25_2 | 16 | +12.50 | [+0.00, +31.25] | +872.94 | +0.606 |
| leval | quality | sempointer_vs_bm25_4 | 16 | -12.50 | [-31.25, +0.00] | -153.19 | -0.146 |
| leval | quality | sempointer_vs_bm25_matched_budget | 16 | -6.25 | [-18.75, +0.00] | +0.31 | -0.052 |
| leval | quality | sempointer_vs_full_context | 16 | -12.50 | [-31.25, +0.00] | -3794.81 | -3.639 |
| leval | quality | sempointer_vs_pointer_off | 16 | -6.25 | [-18.75, +0.00] | +170.94 | +0.143 |
| leval | review_summ | sempointer_vs_bm25_2 | 16 | +2.38 | [-0.17, +5.08] | +1017.88 | +1.170 |
| leval | review_summ | sempointer_vs_bm25_4 | 16 | -0.11 | [-2.24, +2.06] | -21.50 | -1.415 |
| leval | review_summ | sempointer_vs_bm25_matched_budget | 16 | -0.68 | [-2.71, +1.63] | +0.00 | -1.195 |
| leval | review_summ | sempointer_vs_pointer_off | 16 | +1.37 | [-0.30, +3.11] | +693.06 | +0.637 |
| leval | sci_fi | sempointer_vs_bm25_2 | 16 | +31.25 | [+12.50, +56.25] | +1298.88 | +1.047 |
| leval | sci_fi | sempointer_vs_bm25_4 | 16 | +12.50 | [+0.00, +31.25] | +262.06 | +0.228 |
| leval | sci_fi | sempointer_vs_bm25_matched_budget | 16 | +6.25 | [-12.50, +25.00] | +1.25 | -0.089 |
| leval | sci_fi | sempointer_vs_full_context | 9 | +0.00 | [+0.00, +0.00] | -10608.56 | -10.774 |
| leval | sci_fi | sempointer_vs_pointer_off | 16 | +12.50 | [+0.00, +31.25] | +586.62 | +0.483 |
| leval | scientific_qa | sempointer_vs_bm25_2 | 16 | +2.60 | [-9.95, +18.56] | +113.69 | +0.136 |
| leval | scientific_qa | sempointer_vs_bm25_4 | 16 | -10.64 | [-31.05, +9.45] | -901.12 | -0.946 |
| leval | scientific_qa | sempointer_vs_bm25_matched_budget | 16 | -2.90 | [-6.04, -0.56] | +0.00 | +0.698 |
| leval | scientific_qa | sempointer_vs_full_context | 16 | -16.68 | [-31.76, -4.30] | -2477.44 | -2.866 |
| leval | scientific_qa | sempointer_vs_pointer_off | 16 | +2.60 | [-9.95, +18.56] | +113.69 | +0.166 |
| leval | topic_retrieval_longchat | sempointer_vs_bm25_2 | 16 | +1.29 | [+0.23, +2.90] | +418.50 | +0.301 |
| leval | topic_retrieval_longchat | sempointer_vs_bm25_4 | 16 | -5.61 | [-22.01, +7.50] | -600.38 | -0.484 |
| leval | topic_retrieval_longchat | sempointer_vs_bm25_matched_budget | 16 | +3.67 | [-0.70, +9.15] | +1.56 | -0.063 |
| leval | topic_retrieval_longchat | sempointer_vs_full_context | 16 | -71.33 | [-78.68, -60.91] | -9729.25 | -9.656 |
| leval | topic_retrieval_longchat | sempointer_vs_pointer_off | 16 | +1.29 | [+0.23, +2.90] | +418.50 | +0.303 |
| leval | tpo | sempointer_vs_bm25_2 | 16 | +0.00 | [+0.00, +0.00] | +584.69 | +0.442 |
| leval | tpo | sempointer_vs_bm25_4 | 16 | -6.25 | [-18.75, +0.00] | -375.06 | -0.322 |
| leval | tpo | sempointer_vs_bm25_matched_budget | 16 | +0.00 | [+0.00, +0.00] | +0.12 | -0.083 |
| leval | tpo | sempointer_vs_full_context | 16 | -6.25 | [-18.75, +0.00] | -1927.19 | -1.634 |
| leval | tpo | sempointer_vs_pointer_off | 16 | +0.00 | [+0.00, +0.00] | +94.00 | +0.040 |
| leval | tv_show_summ | sempointer_vs_bm25_2 | 11 | -0.67 | [-1.91, +0.57] | +610.64 | -0.471 |
| leval | tv_show_summ | sempointer_vs_bm25_4 | 11 | -2.69 | [-5.10, -0.38] | -427.36 | -1.720 |
| leval | tv_show_summ | sempointer_vs_bm25_matched_budget | 11 | -1.28 | [-2.72, +0.22] | +0.09 | -0.532 |
| leval | tv_show_summ | sempointer_vs_full_context | 11 | -2.80 | [-5.19, +0.86] | -7520.64 | -13.203 |
| leval | tv_show_summ | sempointer_vs_pointer_off | 11 | -0.94 | [-2.31, +0.54] | +327.64 | -0.666 |
| locomo | qa | sempointer_vs_bm25_2 | 300 | +2.37 | [+0.54, +4.70] | +1088.59 | +0.652 |
| locomo | qa | sempointer_vs_bm25_4 | 300 | -0.60 | [-2.81, +1.84] | +54.78 | -0.132 |
| locomo | qa | sempointer_vs_bm25_matched_budget | 300 | +0.04 | [-2.52, +2.51] | +0.02 | -0.219 |
| locomo | qa | sempointer_vs_pointer_off | 300 | +1.24 | [-0.51, +3.40] | +795.29 | +0.366 |
| longbench | hotpotqa | sempointer_vs_bm25_2 | 27 | +18.95 | [+6.91, +32.72] | +817.00 | +0.451 |
| longbench | hotpotqa | sempointer_vs_bm25_4 | 27 | +5.09 | [-3.75, +16.12] | -216.89 | -0.314 |
| longbench | hotpotqa | sempointer_vs_bm25_matched_budget | 27 | +5.15 | [-5.65, +17.59] | +0.04 | -0.360 |
| longbench | hotpotqa | sempointer_vs_full_context | 15 | +2.78 | [-4.50, +15.00] | -7764.33 | -7.684 |
| longbench | hotpotqa | sempointer_vs_pointer_off | 27 | +11.26 | [+1.38, +22.61] | +472.15 | +0.193 |
| longbench | lcc | sempointer_vs_bm25_2 | 24 | -1.60 | [-4.37, +1.24] | +1007.75 | +1.498 |
| longbench | lcc | sempointer_vs_bm25_4 | 24 | +1.06 | [+0.05, +2.30] | +78.58 | -1.293 |
| longbench | lcc | sempointer_vs_bm25_matched_budget | 24 | +0.72 | [-0.71, +2.38] | +0.25 | -1.184 |
| longbench | lcc | sempointer_vs_full_context | 23 | +0.72 | [-0.80, +2.57] | -934.30 | -2.048 |
| longbench | lcc | sempointer_vs_pointer_off | 24 | +1.06 | [+0.05, +2.30] | +78.58 | -1.295 |
| longbench | lsht | sempointer_vs_bm25_2 | 13 | -15.38 | [-38.46, +0.00] | +1326.31 | +1.059 |
| longbench | lsht | sempointer_vs_bm25_4 | 13 | +0.00 | [+0.00, +0.00] | +296.62 | +0.291 |
| longbench | lsht | sempointer_vs_bm25_matched_budget | 13 | +0.00 | [+0.00, +0.00] | +0.00 | -0.136 |
| longbench | lsht | sempointer_vs_full_context | 8 | -12.50 | [-37.50, +0.00] | -7540.12 | -7.513 |
| longbench | lsht | sempointer_vs_pointer_off | 13 | +0.00 | [+0.00, +0.00] | +296.62 | +0.280 |
| longbench | multi_news | sempointer_vs_bm25_2 | 17 | +0.03 | [-2.05, +1.91] | +847.59 | +0.751 |
| longbench | multi_news | sempointer_vs_bm25_4 | 17 | -0.12 | [-1.40, +1.53] | +98.00 | -0.097 |
| longbench | multi_news | sempointer_vs_bm25_matched_budget | 17 | -0.69 | [-2.57, +1.24] | +0.06 | -0.464 |
| longbench | multi_news | sempointer_vs_full_context | 17 | -3.53 | [-5.94, -1.49] | -2084.29 | -4.482 |
| longbench | multi_news | sempointer_vs_pointer_off | 17 | -0.12 | [-1.40, +1.53] | +98.00 | -0.097 |
| longbench | multifieldqa_en | sempointer_vs_bm25_2 | 8 | -4.98 | [-15.78, +4.64] | +638.25 | +0.649 |
| longbench | multifieldqa_en | sempointer_vs_bm25_4 | 8 | -1.15 | [-11.18, +7.47] | -400.50 | -1.684 |
| longbench | multifieldqa_en | sempointer_vs_bm25_matched_budget | 8 | +0.23 | [-6.42, +7.15] | +0.00 | -0.350 |
| longbench | multifieldqa_en | sempointer_vs_full_context | 8 | +3.69 | [-4.54, +11.92] | -5607.88 | -5.483 |
| longbench | multifieldqa_en | sempointer_vs_pointer_off | 8 | -2.08 | [-13.97, +7.47] | +248.50 | -0.947 |
| longbench | multifieldqa_zh | sempointer_vs_bm25_2 | 34 | +3.85 | [-5.22, +12.71] | +911.76 | +0.799 |
| longbench | multifieldqa_zh | sempointer_vs_bm25_4 | 34 | -5.60 | [-14.12, +1.00] | -39.09 | -0.245 |
| longbench | multifieldqa_zh | sempointer_vs_bm25_matched_budget | 34 | -0.97 | [-5.55, +2.57] | +0.03 | -0.217 |
| longbench | multifieldqa_zh | sempointer_vs_full_context | 34 | -14.03 | [-24.06, -6.19] | -1923.35 | -1.928 |
| longbench | multifieldqa_zh | sempointer_vs_pointer_off | 34 | -3.52 | [-9.57, +0.78] | +80.18 | -0.081 |
| longbench | musique | sempointer_vs_bm25_2 | 23 | +1.14 | [-3.11, +6.52] | +1037.61 | +0.668 |
| longbench | musique | sempointer_vs_bm25_4 | 23 | +1.09 | [-10.83, +13.88] | +26.09 | +0.060 |
| longbench | musique | sempointer_vs_bm25_matched_budget | 23 | +1.21 | [-4.12, +6.94] | +0.04 | -0.227 |
| longbench | musique | sempointer_vs_full_context | 6 | +0.00 | [+0.00, +0.00] | -10806.67 | -11.085 |
| longbench | musique | sempointer_vs_pointer_off | 23 | +3.12 | [-6.82, +15.02] | +599.35 | +0.337 |
| longbench | passage_count | sempointer_vs_bm25_2 | 13 | +0.00 | [+0.00, +0.00] | +1616.23 | -0.943 |
| longbench | passage_count | sempointer_vs_bm25_4 | 13 | +0.00 | [+0.00, +0.00] | +578.08 | -1.824 |
| longbench | passage_count | sempointer_vs_bm25_matched_budget | 13 | +0.00 | [+0.00, +0.00] | +0.00 | -3.520 |
| longbench | passage_count | sempointer_vs_full_context | 8 | +0.00 | [+0.00, +0.00] | -10806.12 | -16.660 |
| longbench | passage_count | sempointer_vs_pointer_off | 13 | +0.00 | [+0.00, +0.00] | +578.08 | -1.800 |
| longbench | passage_retrieval_en | sempointer_vs_bm25_2 | 11 | +0.00 | [+0.00, +0.00] | +486.45 | -1.842 |
| longbench | passage_retrieval_en | sempointer_vs_bm25_4 | 11 | +0.00 | [+0.00, +0.00] | -497.73 | -2.094 |
| longbench | passage_retrieval_en | sempointer_vs_bm25_matched_budget | 11 | +0.00 | [+0.00, +0.00] | +0.55 | -1.301 |
| longbench | passage_retrieval_en | sempointer_vs_full_context | 11 | +0.00 | [+0.00, +0.00] | -11367.00 | -14.198 |
| longbench | passage_retrieval_en | sempointer_vs_pointer_off | 11 | +0.00 | [+0.00, +0.00] | +486.45 | -1.883 |
| longbench | passage_retrieval_zh | sempointer_vs_bm25_2 | 26 | +7.69 | [+0.00, +19.23] | +556.54 | -0.594 |
| longbench | passage_retrieval_zh | sempointer_vs_bm25_4 | 26 | +7.69 | [+0.00, +19.23] | -481.38 | -1.200 |
| longbench | passage_retrieval_zh | sempointer_vs_bm25_matched_budget | 26 | +7.69 | [+0.00, +19.23] | +0.00 | -1.227 |
| longbench | passage_retrieval_zh | sempointer_vs_full_context | 26 | +3.85 | [-7.69, +15.48] | -2645.85 | -5.578 |
| longbench | passage_retrieval_zh | sempointer_vs_pointer_off | 26 | +7.69 | [+0.00, +19.23] | +117.46 | -0.874 |
| longbench | qasper | sempointer_vs_bm25_2 | 8 | +24.33 | [+1.42, +54.45] | +291.12 | -0.111 |
| longbench | qasper | sempointer_vs_bm25_4 | 8 | +13.08 | [+0.36, +35.45] | -729.88 | -0.941 |
| longbench | qasper | sempointer_vs_bm25_matched_budget | 8 | +10.68 | [+0.00, +31.78] | +0.00 | -0.222 |
| longbench | qasper | sempointer_vs_full_context | 8 | +15.79 | [-1.34, +42.38] | -3614.62 | -4.709 |
| longbench | qasper | sempointer_vs_pointer_off | 8 | +11.83 | [+0.53, +33.02] | +161.38 | -0.158 |
| longbench | qmsum | sempointer_vs_bm25_2 | 15 | +0.22 | [-1.95, +1.94] | +799.53 | +0.895 |
| longbench | qmsum | sempointer_vs_bm25_4 | 15 | -1.30 | [-3.91, +1.20] | -239.67 | +0.103 |
| longbench | qmsum | sempointer_vs_bm25_matched_budget | 15 | -2.31 | [-6.08, +0.79] | +0.00 | +0.207 |
| longbench | qmsum | sempointer_vs_full_context | 8 | -2.63 | [-9.95, +4.69] | -9669.62 | -11.616 |
| longbench | qmsum | sempointer_vs_pointer_off | 15 | -0.08 | [-2.29, +1.90] | +591.53 | +0.534 |
| longbench | repobench-p | sempointer_vs_bm25_2 | 26 | +2.77 | [-7.77, +13.76] | +1413.19 | +1.587 |
| longbench | repobench-p | sempointer_vs_bm25_4 | 26 | +0.64 | [-1.46, +2.86] | +400.73 | +0.309 |
| longbench | repobench-p | sempointer_vs_bm25_matched_budget | 26 | +3.75 | [-1.54, +11.10] | +0.42 | -0.393 |
| longbench | repobench-p | sempointer_vs_full_context | 16 | -1.24 | [-14.25, +7.85] | -5027.44 | -6.581 |
| longbench | repobench-p | sempointer_vs_pointer_off | 26 | +0.28 | [-1.82, +2.54] | +440.65 | +0.307 |
| longbench | samsum | sempointer_vs_bm25_2 | 17 | -1.44 | [-6.11, +3.93] | +1172.41 | +0.747 |
| longbench | samsum | sempointer_vs_bm25_4 | 17 | +1.79 | [-2.98, +6.59] | +158.18 | +0.054 |
| longbench | samsum | sempointer_vs_bm25_matched_budget | 17 | +0.08 | [-5.68, +6.64] | +0.06 | -0.391 |
| longbench | samsum | sempointer_vs_full_context | 16 | -1.50 | [-8.21, +5.10] | -5895.56 | -6.488 |
| longbench | samsum | sempointer_vs_pointer_off | 17 | +1.80 | [-2.98, +6.59] | +280.24 | +0.218 |
| longbench | trec | sempointer_vs_bm25_2 | 10 | -10.00 | [-30.00, +0.00] | +1200.90 | +1.009 |
| longbench | trec | sempointer_vs_bm25_4 | 10 | +0.00 | [+0.00, +0.00] | +163.20 | +0.140 |
| longbench | trec | sempointer_vs_bm25_matched_budget | 10 | +0.00 | [+0.00, +0.00] | +0.10 | -0.090 |
| longbench | trec | sempointer_vs_full_context | 10 | -10.00 | [-30.00, +0.00] | -3589.60 | -3.335 |
| longbench | trec | sempointer_vs_pointer_off | 10 | +0.00 | [+0.00, +0.00] | +163.20 | +0.141 |
| longbench | triviaqa | sempointer_vs_bm25_2 | 10 | +0.00 | [+0.00, +0.00] | +1503.50 | +1.309 |
| longbench | triviaqa | sempointer_vs_bm25_4 | 10 | +0.00 | [+0.00, +0.00] | +477.00 | +0.424 |
| longbench | triviaqa | sempointer_vs_bm25_matched_budget | 10 | +0.00 | [+0.00, +0.00] | +0.70 | -0.158 |
| longbench | triviaqa | sempointer_vs_full_context | 6 | -1.11 | [-3.33, +0.00] | -6705.50 | -6.816 |
| longbench | triviaqa | sempointer_vs_pointer_off | 10 | +0.00 | [+0.00, +0.00] | +477.00 | +0.418 |
| longbench | vcsum | sempointer_vs_bm25_2 | 18 | -1.36 | [-5.09, +2.04] | +1191.44 | +0.354 |
| longbench | vcsum | sempointer_vs_bm25_4 | 18 | -3.74 | [-6.45, -1.34] | +153.44 | -0.822 |
| longbench | vcsum | sempointer_vs_bm25_matched_budget | 18 | -1.73 | [-5.54, +2.31] | +0.00 | +0.367 |
| longbench | vcsum | sempointer_vs_full_context | 18 | -6.28 | [-10.92, -1.60] | -4677.50 | -8.000 |
| longbench | vcsum | sempointer_vs_pointer_off | 18 | -3.74 | [-6.45, -1.34] | +153.44 | -0.855 |
| longbench_v2 | data | sempointer_vs_bm25_2 | 300 | -2.00 | [-7.33, +3.00] | +4667.10 | +4.245 |
| longbench_v2 | data | sempointer_vs_bm25_4 | 300 | +2.33 | [-1.33, +6.00] | +3627.29 | +3.411 |
| longbench_v2 | data | sempointer_vs_bm25_matched_budget | 300 | -2.00 | [-6.67, +2.67] | +0.22 | -0.332 |
| longbench_v2 | data | sempointer_vs_full_context | 13 | -7.69 | [-30.77, +15.38] | -10831.23 | -11.211 |
| longbench_v2 | data | sempointer_vs_pointer_off | 300 | +2.00 | [-1.33, +5.67] | +3658.46 | +3.425 |
| longmem | longmemeval_s_cleaned | sempointer_vs_bm25_2 | 300 | +8.01 | [+4.61, +11.58] | +5527.72 | +5.099 |
| longmem | longmemeval_s_cleaned | sempointer_vs_bm25_4 | 300 | +0.40 | [-1.56, +2.46] | +4486.66 | +4.128 |
| longmem | longmemeval_s_cleaned | sempointer_vs_bm25_matched_budget | 300 | -0.95 | [-4.66, +2.95] | +0.07 | -1.343 |
| longmem | longmemeval_s_cleaned | sempointer_vs_pointer_off | 300 | +0.73 | [-1.05, +2.66] | +4507.48 | +4.127 |
| loogle | longdep_qa | sempointer_vs_bm25_2 | 75 | +2.90 | [+0.95, +5.18] | +1832.61 | +1.268 |
| loogle | longdep_qa | sempointer_vs_bm25_4 | 75 | -0.68 | [-5.56, +3.21] | +803.75 | +0.757 |
| loogle | longdep_qa | sempointer_vs_bm25_matched_budget | 75 | -2.16 | [-6.45, +1.26] | +0.04 | -0.509 |
| loogle | longdep_qa | sempointer_vs_full_context | 5 | -0.79 | [-5.33, +3.21] | -12161.60 | -15.114 |
| loogle | longdep_qa | sempointer_vs_pointer_off | 75 | +0.45 | [-2.50, +2.85] | +1288.84 | +1.107 |
| loogle | shortdep_cloze | sempointer_vs_bm25_2 | 75 | +7.22 | [+0.33, +13.89] | +2372.69 | +2.668 |
| loogle | shortdep_cloze | sempointer_vs_bm25_4 | 75 | -0.84 | [-6.44, +4.89] | +1334.23 | +1.406 |
| loogle | shortdep_cloze | sempointer_vs_bm25_matched_budget | 75 | +1.07 | [-4.00, +6.40] | +0.15 | -0.078 |
| loogle | shortdep_cloze | sempointer_vs_pointer_off | 75 | +1.16 | [-4.89, +7.78] | +1402.75 | +1.467 |
| loogle | shortdep_qa | sempointer_vs_bm25_2 | 75 | +3.24 | [-1.81, +7.99] | +1030.48 | +0.583 |
| loogle | shortdep_qa | sempointer_vs_bm25_4 | 75 | -0.49 | [-5.08, +4.51] | -9.15 | -0.192 |
| loogle | shortdep_qa | sempointer_vs_bm25_matched_budget | 75 | -1.63 | [-7.46, +4.45] | +0.20 | -0.250 |
| loogle | shortdep_qa | sempointer_vs_full_context | 22 | -5.66 | [-20.45, +8.64] | -13302.36 | -15.632 |
| loogle | shortdep_qa | sempointer_vs_pointer_off | 75 | +3.28 | [-1.14, +7.70] | +836.43 | +0.430 |
| loogle | summarization | sempointer_vs_bm25_2 | 75 | +1.08 | [+0.25, +1.94] | +1399.41 | +2.018 |
| loogle | summarization | sempointer_vs_bm25_4 | 75 | -1.18 | [-2.08, -0.34] | +361.33 | -1.128 |
| loogle | summarization | sempointer_vs_bm25_matched_budget | 75 | -2.05 | [-3.12, -1.01] | +0.12 | -1.712 |
| loogle | summarization | sempointer_vs_full_context | 25 | -1.48 | [-2.85, -0.15] | -13114.80 | -22.225 |
| loogle | summarization | sempointer_vs_pointer_off | 75 | -0.14 | [-0.88, +0.55] | +804.31 | +0.196 |
| ruler | cwe | sempointer_vs_bm25_2 | 47 | +0.00 | [-6.38, +6.38] | +305.38 | +0.160 |
| ruler | cwe | sempointer_vs_bm25_4 | 47 | -11.70 | [-21.28, -4.26] | -724.66 | -0.438 |
| ruler | cwe | sempointer_vs_bm25_matched_budget | 47 | +0.85 | [-2.55, +5.96] | +0.11 | +0.127 |
| ruler | cwe | sempointer_vs_full_context | 38 | -10.53 | [-23.68, +2.63] | -5161.32 | -5.154 |
| ruler | cwe | sempointer_vs_pointer_off | 48 | +0.00 | [-6.25, +6.25] | +304.29 | +0.110 |
| ruler | niah_multikey_1 | sempointer_vs_bm25_2 | 22 | +0.00 | [+0.00, +0.00] | +1047.23 | +0.819 |
| ruler | niah_multikey_1 | sempointer_vs_bm25_4 | 22 | +0.00 | [+0.00, +0.00] | +21.82 | +0.032 |
| ruler | niah_multikey_1 | sempointer_vs_bm25_matched_budget | 22 | +0.00 | [+0.00, +0.00] | +0.00 | -0.092 |
| ruler | niah_multikey_1 | sempointer_vs_full_context | 15 | +0.00 | [+0.00, +0.00] | -4904.60 | -4.628 |
| ruler | niah_multikey_1 | sempointer_vs_pointer_off | 22 | +0.00 | [+0.00, +0.00] | +338.64 | +0.268 |
| ruler | niah_single_3 | sempointer_vs_bm25_2 | 1 | +0.00 | [+0.00, +0.00] | +1628.00 | +1.343 |
| ruler | niah_single_3 | sempointer_vs_bm25_4 | 1 | +0.00 | [+0.00, +0.00] | +589.00 | +0.452 |
| ruler | niah_single_3 | sempointer_vs_bm25_matched_budget | 1 | +0.00 | [+0.00, +0.00] | +1.00 | -0.095 |
| ruler | niah_single_3 | sempointer_vs_pointer_off | 1 | +0.00 | [+0.00, +0.00] | +589.00 | +0.504 |
| ruler | vt | sempointer_vs_bm25_2 | 1 | +0.00 | [+0.00, +0.00] | +194.00 | -0.348 |
| ruler | vt | sempointer_vs_bm25_4 | 1 | -20.00 | [-20.00, -20.00] | -844.00 | -1.292 |
| ruler | vt | sempointer_vs_bm25_matched_budget | 1 | +0.00 | [+0.00, +0.00] | +0.00 | -0.278 |
| ruler | vt | sempointer_vs_full_context | 1 | -40.00 | [-40.00, -40.00] | -6457.00 | -6.561 |
| ruler | vt | sempointer_vs_pointer_off | 1 | +0.00 | [+0.00, +0.00] | +194.00 | -0.327 |
| scrolls | contract_nli | sempointer_vs_bm25_2 | 1 | +0.00 | [+0.00, +0.00] | +0.00 | -0.002 |
| scrolls | contract_nli | sempointer_vs_bm25_4 | 1 | +0.00 | [+0.00, +0.00] | +0.00 | -0.004 |
| scrolls | contract_nli | sempointer_vs_bm25_matched_budget | 1 | +0.00 | [+0.00, +0.00] | +0.00 | -0.001 |
| scrolls | contract_nli | sempointer_vs_full_context | 1 | +0.00 | [+0.00, +0.00] | +0.00 | -0.009 |
| scrolls | contract_nli | sempointer_vs_pointer_off | 1 | +0.00 | [+0.00, +0.00] | +0.00 | -0.005 |
| scrolls | gov_report | sempointer_vs_bm25_2 | 1 | +2.25 | [+2.25, +2.25] | +274.00 | +3.071 |
| scrolls | gov_report | sempointer_vs_bm25_4 | 1 | -1.00 | [-1.00, -1.00] | -764.00 | -2.902 |
| scrolls | gov_report | sempointer_vs_bm25_matched_budget | 1 | -0.14 | [-0.14, -0.14] | +0.00 | -1.337 |
| scrolls | gov_report | sempointer_vs_full_context | 1 | -3.61 | [-3.61, -3.61] | -6023.00 | -8.400 |
| scrolls | gov_report | sempointer_vs_pointer_off | 1 | +2.25 | [+2.25, +2.25] | +274.00 | +2.972 |
| scrolls | narrative_qa | sempointer_vs_bm25_2 | 1 | -6.28 | [-6.28, -6.28] | +9997.00 | +13.874 |
| scrolls | narrative_qa | sempointer_vs_bm25_4 | 1 | -11.23 | [-11.23, -11.23] | +8955.00 | +12.720 |
| scrolls | narrative_qa | sempointer_vs_bm25_matched_budget | 1 | -32.69 | [-32.69, -32.69] | +0.00 | +3.042 |
| scrolls | narrative_qa | sempointer_vs_pointer_off | 1 | -6.28 | [-6.28, -6.28] | +9997.00 | +13.840 |
| scrolls | qmsum | sempointer_vs_bm25_2 | 1 | +0.00 | [+0.00, +0.00] | +1028.00 | +0.787 |
| scrolls | qmsum | sempointer_vs_bm25_4 | 1 | +0.31 | [+0.31, +0.31] | -12.00 | -0.272 |
| scrolls | qmsum | sempointer_vs_bm25_matched_budget | 1 | +0.31 | [+0.31, +0.31] | +0.00 | -0.238 |
| scrolls | qmsum | sempointer_vs_pointer_off | 1 | +0.00 | [+0.00, +0.00] | +1028.00 | +0.725 |
