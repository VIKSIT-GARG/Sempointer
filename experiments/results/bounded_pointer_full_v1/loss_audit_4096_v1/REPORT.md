# Audit of all 229 bounded-pointer losses at the 4K input budget

CPU-only analysis of existing generations. No model weights loaded; no new GPU calls. Original run evidence is preserved.

## Method and interpretation

Loss = bounded score below budgeted BM25 on the same question (tolerance 1e-12). Expanded text was extracted from saved prompts, excluding the pointer section and question, and every full block was checked against the frozen source with the actual tokenizer. Support spans are mapped through source-token offsets to the actual visible payload blocks, including conservative handling of BM25 partial blocks.

Evidence tiers: generated RULER facts and publisher dialogue/paragraph annotations are fine-grained checks; LongMemEval support sessions are coarse; short answer mentions are weak proxies. Summary references, option letters, abstention labels and code targets are not treated as gold evidence. All annotated text present does not prove sufficient evidence, pointer distraction, or a reasoning-free task. Annotations may be redundant, and dialogue dates can matter beyond the annotated utterances.

## Benchmark breakdown

| Family | Losses | Summary tasks | Annotated units omitted, BM25 retains | Annotated units present | Pointer-off reaches BM25 |
| --- | --- | --- | --- | --- | --- |
| ruler | 14 | 0 | 9 | 2 | 1 |
| longbench | 31 | 16 | 0 | 0 | 11 |
| locomo | 35 | 0 | 6 | 6 | 5 |
| longmem | 11 | 0 | 0 | 0 | 2 |
| babilong | 8 | 0 | 0 | 0 | 1 |
| longbench_v2 | 5 | 0 | 0 | 0 | 0 |
| infinite | 18 | 6 | 0 | 0 | 6 |
| loogle | 29 | 19 | 0 | 0 | 4 |
| leval | 35 | 18 | 0 | 0 | 4 |
| scrolls | 43 | 32 | 3 | 1 | 13 |

## Evidence diagnoses

| Diagnosis | Cases |
| --- | --- |
| reference_not_evidence_manual_review_required | 134 |
| answer_mention_present_not_evidence_proof | 10 |
| answer_mention_omitted_bm25_retains_proxy_only | 9 |
| answer_mention_absent_both_cause_unresolved | 22 |
| annotated_units_omitted_bm25_retains | 18 |
| annotated_units_present_cause_unresolved | 9 |
| annotated_units_incomplete_both | 16 |
| support_session_coverage_requires_review | 11 |

## Saved pointer-off counterfactual

Identical expanded payload text in 229/229 cases. Removing pointers improves the score in 94 and reaches or exceeds BM25 in 47. These are single saved greedy outputs, not repeated deterministic trials; this is evidence of sensitivity to the pointer-section intervention, not proof of a unique cognitive cause such as distraction. Other losses remain when pointer text is removed.

## Budget and task limits

0 bounded payloads were cut by the 4K input cap. Mean bounded input on these losses: 1764.8 tokens. Only 1 bounded answers hit the output cap. 91 losses are summary tasks (including SAMSum, whose local adapter uses QA F1). A larger evidence selection could use unused room within the existing 4K allowance; raising the input ceiling alone does not make the two/four-block router fetch more. This audit does not establish that every omission would be fixed by more text, nor that longer prompts improve all summaries.

## Files

- [cases.csv](cases.csv): all 229 cases and diagnosis flags.
- [cases.jsonl](cases.jsonl): detailed annotations, support token coverage, saved answers, expanded blocks and source pointers for every loss.
- [casebook.md](casebook.md): human-readable per-loss packets with all three arms' payloads.
- [summary.json](summary.json): machine-readable aggregate counts.
- [provenance.json](provenance.json): source checksums and zero-generation execution record.

## What remains unresolved

Cases without mappable evidence annotations require semantic/manual review; answer absence is not proof of retrieval failure. The casebook covers every loss but does not falsely claim complete gold-evidence labeling where the dataset supplies no such labels. A controlled rerun would be needed to test proposed repairs. No configuration or original result was changed.

## Manual review and validation

See [MANUAL_REVIEW.md](MANUAL_REVIEW.md) for source-grounded examples, block-boundary failures, lexical-scoring artifacts, abstention issues, and repair priorities. [review_index.csv](review_index.csv) includes each question, reference, three scores, support status, and diagnosis. Full non-GPU checks: 129 passed, 7 failed, 21 setup errors, 7 deselected. Prior failure names and traces are in [full_tests.log](full_tests.log).
