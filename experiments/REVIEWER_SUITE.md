# Run the reviewer experiments before the deadline

From any directory:

```bash
bash /home/viksit/Projects/tokens2/experiments/run_reviewer_suite.sh --hours 8
```

The default output is `experiments/results/reviewer_revision/`. Run the same
command to resume. Successful stages and committed predictions are preserved.
The hours allowance restarts on each invocation; it is not cumulative across
resumes. Keep this terminal open, or run the command inside your own persistent
terminal session. One GPU worker runs stages sequentially; there is no training.
The project Python, local Qwen2.5-3B checkpoint, and datasets are already present.
Inference stays offline and never downloads assets implicitly.

**Expected duration: 5–8 hours on the current RTX 5060 Laptop GPU, Qwen2.5-3B
NF4 setup.** The validation smoke completed 12 real generations, averaging
6.14 generation seconds per pair; the separate existing throughput pilot averaged
5.18 seconds across 250 scored pairs. These are throughput observations, not a
guarantee: output caps, task mix, 16K full-context prefill, other GPU activity,
and preprocessing change elapsed time. The suite prints a smoke-based projection
and saves it in `runtime_estimate.json`.

The eight-hour allowance reserves ten minutes internally for reporting. At its
compute deadline the active child process is interrupted, committed SQLite
evidence is exported, and pending stages are marked unstarted. A timeout, error,
empty test stage, or failed control invariant produces exit code 2. Partial
results remain usable but must be described as partial. Reporting usually takes
minutes; this is not a hard real-time guarantee. Leave the remaining two hours
of the ten-hour window for interpreting results, revising the manuscript, and
checking the submission PDF.

## What runs

| Stage | Scope | Maximum generation pairs |
| --- | --- | ---: |
| Model preflight | Local model/config/tokenizer and weight availability | 0 |
| Smoke | One 4K NIAH and one 4K CWE example, six systems | 12 |
| Development | Three prespecified threshold candidates; six RULER tasks, 4K/8K, three examples per cell; SemPointer and BM25-4 | 216 |
| Fresh RULER test | Six tasks, 4K/8K/16K, ten examples per cell; six systems | 1,080 |
| CWE cap audit | The same 30 test CWE examples and six systems, with a 32-token cap | 180 |
| Additional LongBench | Up to 50 eligible questions per each of six QA tasks; five systems | 1,500 |
| Additional LoCoMo | Up to 30 questions per unused conversation; five systems | 1,050 |

The upper planning total is about **4,038 generations**. Exclusions and available
records can reduce it. LongBench/LoCoMo counts are additional evaluated subsets,
not full benchmark coverage. The five core systems are SemPointer, pointer-off,
BM25-2, BM25-4, and matched-budget BM25. RULER additionally includes full context.
Full context is skipped explicitly if it exceeds the model/prompt guard; the
RULER prompt allowance is 20,000 tokens to permit the intended 16K comparisons.

At implementation-time inspection, the conservative exposure filters left 197
scheduled additional LongBench questions (14 2WikiMQA, 50 HotpotQA, 33
MultiFieldQA, 50 MuSiQue, and 50 QASPER), plus 210 questions from seven unused
LoCoMo conversations. NarrativeQA had no remaining unseen document groups and
will be explicitly marked as having no eligible examples. These counts can
change if more retained exposure logs appear before the first suite invocation.
Under that inspected coverage the total is approximately 3,523 generation pairs.

Smoke, development, and test seeds are `20261109`, `20261110`, and `20261111`.
Generated RULER identities include the seed so different generated samples cannot
collide. Development uses no test examples. It compares the following triples:

| Candidate | Expansion coverage | Ambiguity coverage | Third/second score ratio |
| --- | ---: | ---: | ---: |
| loose | 0.70 | 0.90 | 0.80 |
| paper | 0.80 | 0.95 | 0.85 |
| cautious | 0.90 | 0.98 | 0.90 |

The prespecified development rule chooses the lowest mean token count among
candidates whose paired mean score is within 0.02 of development BM25-4. If no
candidate meets that heuristic, it chooses highest development score. Ties favor
higher score, fewer tokens, then name. This is a small development selection
rule, not an inferential non-inferiority claim. All three candidates must have
complete, identically paired development data. The selected parameters are
written to `threshold_selection.json` **before** any test run; test results never
select parameters. State any change from paper thresholds in the revised paper.
Development is synthetic only; cross-domain transfer remains an empirical limit.

Known exposure exclusions remove entire document/conversation groups associated
with the paper's first 100 LongBench records per task, its first three LoCoMo
conversations, and scored records in retained newer prediction/event logs.
`excluded_groups.json` and `exposure_manifest.json` record this scope. Unknown
historical exposure cannot be ruled out. LoCoMo sampling caps each eligible
conversation separately, rather than filling the sample from one conversation.

The new runner uses strict chunk boundaries, session dates for LoCoMo, 128-token
QA/non-CWE RULER output limits, and 512 tokens for CWE. The cap audit changes only
CWE's output allowance to 32, preserving the test seed and evidence policy. Run
all compared arms under this new protocol; do not compare new control scores
directly against old paper scores and attribute the difference to pointers.

## Results and interpretation

- `REPORT.md`: stage completion, per-task paired contrasts, and CWE cap audit.
- `comparisons.csv` / `comparisons.json`: paired mean quality, token, and latency
  differences; score differences use the 0–1 scale.
- `checks.json`: pointer-off selected-ID identity, token-budget enforcement,
  scored/group counts, output-cap hits, and development/test context overlap.
- `REVIEWER_RESPONSE.md`: a worksheet for writing the response from real results.
- Each stage: raw prompts, answers, IDs, timestamps, model/data/code provenance,
  SQLite records, exports, and a per-task report. Logs are under `logs/`.

Pointer-off shares the exact selected block IDs; it removes unresolved pointer
text. Matched-budget BM25 uses the same total prompt-token upper allowance and
logs slack/partial blocks. It does not promise exactly identical token counts.
Negative/null pointer effects are valid results and must be reported honestly.
Longer output limits can remove a bottleneck without improving CWE quality.
LoCoMo remains a token-F1 adaptation, and CIs resample examples rather than
independent conversation clusters. The script does not modify the uploaded PDF
or automatically rewrite manuscript claims.

## Preview, recovery, and shorter runs

```bash
# Print commands without loading a model or creating results.
bash /home/viksit/Projects/tokens2/experiments/run_reviewer_suite.sh --dry-run

# Rebuild reports from committed data after interruption.
bash /home/viksit/Projects/tokens2/experiments/run_reviewer_suite.sh --report-only

# Smaller scope: use a NEW output directory because the evidence protocol changes.
bash /home/viksit/Projects/tokens2/experiments/run_reviewer_suite.sh \
  --hours 5 --ruler-samples 5 --longbench-limit 25 --locomo-per-conversation 15 \
  --output /home/viksit/Projects/tokens2/experiments/results/reviewer_revision_small
```

Model/data/code/settings changes are rejected on resume; choose a new output
directory if changing the protocol. Changing only the invocation time allowance
is allowed. CPU tests and the small validation GPU run are software/inference
checks, not the completed reviewer experiment suite.

## Implementation verification

The 38 focused CPU checks pass, including a full orchestrator run with an
explicitly fake external generator, a completed resume with an expired compute
allowance, timeout recovery, selection independence, group filtering, and real
control construction. The separate real GPU smoke at
`results/reviewer_suite_validation/` completed 12/12 generations and passed
pointer selected-ID identity and actual-token budget checks. This validates
inference and controls, not the full reviewer study or its scientific claims.

The broader non-GPU test command was also run: 71 passed, seven failed, and 21
had setup errors (seven GPU tests deselected). The failures match the previously
documented uncached GPT-2/Qwen/embedding dependencies and
`tests/test_paper_assets.py::test_partial_stage_preserves_scored_rows_and_marks_missing_examples`.
They were not suppressed or rewritten. Details are in
`/tmp/tokens2-reviewer-tests-final.log`.
