# Reviewer response draft — not submitted

We thank the reviewers for their careful assessment. We have revised the manuscript to distinguish the bounded evaluation from the earlier full-index experiments, report both benefits and unfavorable outcomes, and describe the remaining limits of the evidence. The responses below summarize the evidence available for this revision; they do not imply that every concern has been fully resolved.

## Reviewer 2

**Development validation.** We now explain how the router settings were selected and distinguish that choice from the later descriptor-budget pilot. On 36 RULER development questions, all three router settings scored 77.5, compared with 73.9 for BM25-4. The selected loose setting used 1,815 mean input tokens, compared with 1,865 for the other settings. Separately, a 30-question development pilot selected a 128-token descriptor allowance over 256 tokens at each of the three input budgets. These small development comparisons guided configuration; they are not independent confirmation of the final system's performance.

**Pointer-off control.** We added a retrospective fixed-payload comparison using saved generations. The expanded text, identifiers, question, references, source hash, output cap, and configuration agree between conditions; removing the descriptor section reproduces the pointer-off prompt byte for byte. Among 990 pairs with an active descriptor intervention, pointer-on produced 134 wins, 730 ties, and 126 losses. The net win balance was +0.81 percentage points, with an exploratory context-clustered 95% interval of [−2.47, 3.96]. Ten additional pairs had identical prompts and answers. The overall effect is therefore inconclusive.

The control also shows a narrower, exploratory result on 17 single-chain variable-tracking questions: pointer-on scored 74.12 versus 37.65, with 12 wins, five ties, and no losses. The Holm-adjusted value is 0.026 across 53 inference-eligible groups and 0.060 across all 123 inspected groups. Descriptors exposed gold variable names absent from the expanded payload in 14 of the 17 cases. Because every assigned variable in these cases is a gold answer name, this observation supports entity recovery in this setup and does not establish general multi-chain reasoning. Budget-filling BM25 scored 96.47 on the same questions.

**Matched-token baseline.** We have made the distinction between the historical and bounded comparisons explicit. In the older full-index Qwen3.5 experiment, SemPointer and matched-token BM25 had approximately equal mean input token counts over 936 shared questions (the difference rounds to +0.00%). SemPointer used 6.0% less generation time, with 140 wins, 611 ties, and 185 losses. This is a historical operating point, not a matched-token evaluation of the final bounded design. The bounded study compares methods under a common maximum input allowance; their realized token counts differ. A realized-token-matched BM25 comparison for the final bounded design remains outstanding.

The separate Kaggle report snapshot adds a matched-budget comparison on 2,400 shared questions across eight complete family subsets. Mean input is 4,071.4 versus 4,071.3 tokens and reported latency is 8.9% lower for SemPointer. Across 125 task groups, mean native scores are higher/tied/lower in 33/54/38 groups. This T4 NF4/FP16 configuration is distinct from the bounded local NF4/BF16 study, and the comparison is drawn from report aggregates pending raw-archive validation. It therefore adds evidence under approximately matched mean input without resolving the remaining bounded-configuration control.

The manuscript also includes five illustrative task comparisons with paired intervals and retains the complete task inventory in its source package. The examples were selected after inspection; all intervals include zero and are unadjusted for multiplicity, so they do not establish quality superiority or equivalence.

**Coverage and low scores.** The bounded study covers 1,000 questions from ten benchmark families at the 4,096-token limit. Relative to budget-filling BM25, it uses 54.2% fewer input tokens and 42.7% less generation time, but the paired quality outcomes are 123 wins, 648 ties, and 229 losses. At that limit, the bounded design evaluates all 1,000 requests while the historical full index fits 700; coverage here means an evaluated request, not a correct answer. The manuscript reports the quality–cost tradeoff and retains low or unfavorable outcomes alongside positive estimates. We also describe the restrictions imposed by source and input guards, the use of one quantized checkpoint, exposed tests, possible source overlap, and lexical metric artifacts. These qualifications limit generalization beyond the evaluated setting.

**CWE output cap.** We added the 32-versus-512-token output-cap audit for common-word extraction. On 18 paired questions, the 32-token cap was reached in 17 SemPointer outputs and all 18 BM25-4 outputs; neither arm reached the 512-token cap. SemPointer's score changed from 0.00 to 0.56, while BM25-4 changed from 15.00 to 18.33. This small, task-specific audit identifies truncation under the shorter cap and does not establish a general accuracy improvement. The main study uses a 512-token CWE output cap.

## Reviewer 3

We appreciate the reviewer's assessment and have clarified the empirical tradeoffs, controls, and limitations in the revision. The review did not identify an additional concrete request requiring a separate response.

## Internal provenance — remove before submission

- Router selection: `experiments/results/local_ten_qwen35_nf4_100_v1/results/threshold_selection.json`; development and pilot descriptions: `paper/extended.tex`.
- Fixed-payload control and variable tracking: `experiments/results/bounded_pointer_full_v1/fixed_payload_pointer_control_v1/summary.json`, `per_task.csv`, and `vt_mechanism.csv`; interpretation checks: `paper/REVISION.md`.
- Historical matched-token comparison: `paper/generated/revision_original_controls.tex`; bounded comparison: `paper/generated/submission_budgets.tex`.
- CWE cap audit: `experiments/results/local_ten_qwen35_nf4_100_v1/results/cwe_cap_audit.json`; score summary: `paper/extended.tex`.

- Kaggle snapshot: `kaggle_upload/publication_2026-11-11/report_data.json` and `source_report_configuration.json`; included source-package copy: `kaggle_evidence/`.
