# Fixed-payload pointer-on versus pointer-off control

## Status and scope

Completed verification and secondary analysis of the already-executed 4K control in run `c73abaf595dc68e55696`. **No new inference was launched.** This report is not a fresh independent replication.

All 1,000 pairs have byte-identical expanded payload strings, selected IDs, question, gold references, native framing outside the descriptor section, and output caps. Removing only the entire descriptor section from the pointer-on prompt produces the pointer-off prompt exactly. Both arms share the same frozen checkpoint, NF4 precision, greedy non-thinking decoding, routing settings and stage configuration.

The 1K and 2K experiments often trade pointer space against payload space; they are not used as this control. The 4K budget was selected for the exact intervention match, not by filtering favorable score outcomes. All 1,000 questions, not only the 229 BM25 losses, are included.

## Overall descriptive results

| Pairs | n | Pointer-on wins | Ties | Losses | Identical answer texts |
| --- | --- | --- | --- | --- | --- |
| All fixed-payload pairs | 1000 | 134 | 740 | 126 | 607 |
| Descriptor section actually present | 990 | 134 | 730 | 126 | 597 |
| No descriptors; identical input prompts | 10 | 0 | 10 | 0 | 10 |

On the 990 actual interventions, the net win fraction is 0.81 percentage points (wins minus losses divided by question count). A family-stratified same-context cluster bootstrap gives a 95% interval of [-2.47, 3.96] points. This measures question-wise win balance, not mean accuracy or score equivalence. The interval is exploratory; shared documents may remain correlated beyond exact-context hashes, and LoCoMo has only four source conversations. A question-independent sign test is retained in summary.json with its invalid-independence caveat.

## Per-family results

| Family | n | Actual descriptor pairs | Source contexts | Wins | Ties | Losses | Extra input tokens, mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ruler | 100 | 100 | 100 | 12 | 87 | 1 | 107.1 |
| longbench | 100 | 91 | 100 | 10 | 69 | 21 | 92.7 |
| locomo | 100 | 100 | 4 | 12 | 71 | 17 | 112.6 |
| longmem | 100 | 100 | 100 | 10 | 80 | 10 | 116.7 |
| babilong | 100 | 100 | 100 | 6 | 92 | 2 | 112.1 |
| longbench_v2 | 100 | 100 | 91 | 8 | 91 | 1 | 114.4 |
| infinite | 100 | 100 | 95 | 1 | 93 | 6 | 113.3 |
| loogle | 100 | 100 | 91 | 24 | 54 | 22 | 112.1 |
| leval | 100 | 100 | 86 | 26 | 56 | 18 | 108.5 |
| scrolls | 100 | 99 | 94 | 25 | 47 | 28 | 101.8 |

Counts are within-question comparisons under each task's own metric; heterogeneous score means are not pooled.

## Per-task inference and multiplicity

All 123 active task/metric groups are exported in [per_task.csv](per_task.csv). Within each group, report native-score means/differences. Groups with at least five exact source contexts receive paired context-cluster bootstrap intervals and cluster sign-flip p-values. Smaller groups are descriptive only. Holm correction covers all eligible groups; no task is selected for reporting because its p-value is favorable.

Eligible groups: 53; Holm-adjusted p < 0.05: 1 positive, 0 negative. This exploratory analysis does not establish a preregistered confirmatory result.

### Task-specific positive result: RULER variable tracking

The 17 variable-tracking questions score 74.12 with pointers and 37.65 without pointers on a 0–100 display scale: a +36.47-point paired mean difference, exploratory 95% context-cluster bootstrap interval [+22.35, +51.76]. There are 12 wins, five ties and no losses. The raw cluster sign-flip p-value is 0.000488; Holm adjustment across the 53 inference-eligible task groups gives 0.0259. A more conservative sensitivity treating all 123 inspected groups as the hypothesis family gives 0.0601. Thus the multiplicity conclusion depends on the justified hypothesis family; do not omit this qualification.

A source-based mechanism probe finds 14/17 cases where descriptors expose gold variable names absent from the expanded payload; 11 of the 12 pointer-on wins have such added names. All 17 examples have one chain, and every source assignment variable is a gold name. This is evidence that compact descriptors can recover otherwise unseen answer entities in this synthetic setup. It does not establish general multi-chain reasoning, learned addressing, or a model-issued dereference mechanism. Gold names were used only for retrospective analysis, never to construct descriptors. Raw probe rows are in [vt_mechanism.csv](vt_mechanism.csv).

## Cost and mechanism

Across all 1,000 pairs, pointer-on adds 109.135 input tokens on average (6.31% more input). Recorded generation time is 1.62% lower. Pointer-on was generated before pointer-off in 742/1,000 cases under the existing four-arm rotation; the pair is not perfectly counterbalanced. The small timing difference is descriptive, not a reliable speed advantage.

This intervention tests adding the whole descriptor section, including its extra tokens and positional effects. It does not isolate descriptor semantics from length/placement; that would require additional matched-length or shuffled controls. Saved greedy outputs are single observations, not repeated trials. The ten identical-input checks agree here, but that small check does not prove deterministic behavior for every prompt.

## Claim supported by this control

The existing experiment contains a verified fixed-payload comparison. Descriptor text changes some answers and has dataset-dependent benefits and harms, but the overall win balance is small and its clustered interval includes no effect. **No clear overall quality benefit from displaying pointers is demonstrated.** The variable-tracking result supports a narrower synthetic-task benefit with the stated multiplicity and task-design caveats. This does not prove that the effect is zero elsewhere or that pointers never help. Efficiency claims should be attributed to the adaptive evidence policy unless a stronger descriptor-specific effect is established. Do not infer cross-model robustness, official leaderboard standing, or guaranteed publication readiness.

## Files and reproduction

- [verified_pairs.csv](verified_pairs.csv): all pairs, intervention/payload hashes, answer texts, scores, timing and null-intervention flags.
- [per_task.csv](per_task.csv): every homogeneous task group, including negative and inconclusive outcomes.
- [summary.json](summary.json): aggregate results, uncertainty settings and eligible task findings.
- [provenance.json](provenance.json): preserved source/config hashes and zero-new-inference declaration.

From experiments/, create a new analysis directory when reproducing:

```bash
CUDA_VISIBLE_DEVICES='' .venv-local-suite/bin/python -m analysis.fixed_payload_control --output /absolute/new/analysis-directory
```

No model is imported or loaded by this analysis. It reads the existing SQLite records in read-only mode. Neither completed test data nor frozen inputs are changed.

## Software verification

Eight fixed-control guard tests and ten evidence-audit tests pass (18 total). The broader offline non-GPU suite has 137 passes, 7 failures, 21 setup errors and 7 deselected tests. The failures match the prior legacy-cache and paper-assets issues; all 28 names and traces are in [full_tests.log](full_tests.log). [validation.json](validation.json) records the final artifact checks.
