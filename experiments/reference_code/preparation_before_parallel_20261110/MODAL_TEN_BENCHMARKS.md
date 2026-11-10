# Detached Modal evaluation: ten benchmark families

The launcher is `/home/viksit/Projects/tokens2/experiments/run_modal_suite.sh`.
The project-local Modal CLI recognizes workspace `viksitg427`; no new login was
needed. No paid build, deployment or cloud GPU run was started during setup.

## One-time billing prerequisite

In the **same Modal workspace**, open Settings → Usage & Billing and set:

- **Workspace usage limit: $25 or less**, before credits.
- **Out-of-pocket spend limit: $0**, to use only applicable free credits.

This follows [Modal's budget documentation](https://modal.com/docs/guide/budgets).
The script requires `--workspace-budget-confirmed` to record your confirmation.
It **cannot read, verify or set these server-side billing limits**. Do not pass
the flag until you have checked them. If credits are already exhausted, the $0
net limit may stop the run immediately. Other workspace workloads also consume
the workspace allowance. The $25 target leaves $5 below the requested ceiling.

The in-run ledger is a conservative resource-time allowance, not a billing API.
It reserves the worst-case worker interval before each chunk; uncertain attempts
retain the entire reservation. It also reserves $1 per submission for setup,
image building and reporting. The workspace cap is the final invoice guard.
The ledger does not guarantee completion or replace the provider's billing cap.

## Start with one command

After checking those two limits:

```bash
bash /home/viksit/Projects/tokens2/experiments/run_modal_suite.sh start --workspace-budget-confirmed
```

The command performs these steps automatically:

1. Scan local releases and prepare a fixed, balanced sample of at most 300
   eligible questions **in total per family**, with prior-exposure exclusions.
2. Freeze the examples and exact runtime source into a checksummed bundle.
3. Upload that bundle into a persistent Modal Volume. It does not upload your
   credentials, virtual environments, historical results, or complete datasets.
4. Build the pinned cloud image and submit a detached CPU setup function.
5. Download the pinned public Qwen3.5-4B checkpoint into the cloud Volume.
6. Run three small routing-development configurations and freeze the winner.
7. Run a representative cost pilot on the already selected test questions.
8. Continue through the remaining test questions and paired CWE audit **only if
   the conservative projection fits the remaining allowance**.
9. Export raw evidence, summaries and a report. Completed predictions are reused.

Preparation scans large local files and can take substantial CPU time. The cloud
does not begin until preparation and upload finish. Leave the terminal/laptop
running until it prints **`detached_submission` and a setup call ID**. After that,
you can close the terminal or switch off the laptop; the cloud continuation chain
submits subsequent chunks without a local controller.

A cost rejection or exhausted allowance yields a **partial** run and stops; it
does not silently reduce the sample count or substitute easier examples. A paused
cost gate is not bypassed by `resume`. No-context is deferred.

## Inspect, download, stop, or resume

```bash
bash /home/viksit/Projects/tokens2/experiments/run_modal_suite.sh status
bash /home/viksit/Projects/tokens2/experiments/run_modal_suite.sh download
bash /home/viksit/Projects/tokens2/experiments/run_modal_suite.sh cancel
bash /home/viksit/Projects/tokens2/experiments/run_modal_suite.sh resume --workspace-budget-confirmed
```

`cancel` asks the worker to stop at a checkpoint after the active generation.
`resume` checks the provider's recorded call before releasing a stale lease;
it refuses to start a duplicate worker. Ambiguous handoff state fails closed and
requires inspecting the Modal app rather than guessing that the old call died.
An interrupted attempt may retain a full reservation, reducing the remaining
allowance. A CPU report may still be running after inference ends; `status`
shows the active lease until reporting releases it.

Local bundle: `experiments/modal_runs/ten_qwen35_v1/bundle/`.
Downloaded results: `experiments/modal_runs/ten_qwen35_v1/downloaded_results/`.
Remote Volume: `sempointer-ten-benchmarks-v1`, under `runs/<run_id>/`.
Raw SQLite records and event logs are checkpointed even if the final report is
interrupted. Downloaded results stay separate from historical `results/` runs.
Use `SEMPTR_MODAL_RUN_DIR=/absolute/new/run-directory` for a deliberately new
bundle after changing source/configuration; existing bundles keep their frozen
code and are not silently rewritten.

## Sampling scope and interpretation

Families: RULER, LongBench v1, LoCoMo, LongMemEval, BABILong, LongBench v2,
InfiniteBench, LooGLE, L-Eval and SCROLLS. The maximum is 3,000 main questions
and 18,000 system-question pairs, including full-context attempts that may be
explicitly skipped. Fewer eligible questions produce a recorded shortfall.

- LongBench uses its 21 base tasks, balanced across the six task categories;
  overlapping LongBench-E variants are excluded.
- LongMemEval uses **S**, excluding the privileged Oracle and larger M variant.
- BABILong uses its **4K–256K** published configurations, balanced by task and
  length bucket. The 0K diagnostic and longer configurations are excluded.
- SCROLLS uses public validation answers, with a separate document-group hash
  test partition. Missing-label test releases are not scored.
- The other families balance their task/category metadata. Fixed hash priorities
  select questions without reading their scores; exact duplicates within a
  family are removed. Existing document/conversation groups and protocol labels
  survive materialization.
- Main sources must be labeled, nonempty, in the local test partition, outside
  known exposure groups, at most 2,000,000 characters and 262,144 model tokens.
  Source history is not truncated. The final prompt guard is 16,384 tokens.

Equal-category sampling measures a balanced distribution, not original corpus
frequencies. Related questions and overlapping benchmark sources remain correlated.
The manifest records selections, exclusions and category shortfalls. Scores use
the existing explicitly adapted per-task metrics, not official model-based judges.
No overall mixed-metric benchmark score is manufactured.

All six systems share the frozen Qwen3.5-4B model, NF4 precision and native
non-thinking chat format. Controls are SemPointer, pointer-off, BM25-2, BM25-4,
matched-budget BM25 and full context when it fits. QA/non-CWE RULER allow 128
output tokens, CWE 512; other tasks retain their task-specific allowances.

Threshold development uses 36 fresh synthetic questions and two systems across
three candidates (216 generations), separate from test selection. The CWE audit
uses at most 30 selected main-test CWE questions at 32 tokens (180 extra
generations). Pilot predictions are reused in the main test, so the pilot does
not add a duplicate test set. The projection uses the slowest observed pilot
question per family plus 50% time headroom and chunk overhead.

## Software and preservation

The local Modal SDK is isolated in `.venv-modal/` (version 1.6.3). The cloud
targets one L40S, two CPU cores and 16 GiB RAM. The image pins CUDA 12.8,
PyTorch 2.10 and Transformers 5.2, and installs the DeltaNet/convolution kernels.
The cloud image build and real cloud throughput have **not** been verified by a
paid execution; the first launch may expose build/runtime issues. The pilot and
allowance stop work rather than assuming performance from local smoke timings.

Offline tests exercise the complete phase sequence using an explicitly fake
model boundary, real storage/checkpointing, family quotas, immutable bundles,
allowance recovery, cap pairing and detached handoff races. Those fixture results
are software checks only and are not empirical benchmark evidence.

Historical results were not modified. The earlier Qwen3.5 source was recovered
by removing only the new cloud hooks and verified against its smoke run's exact
recorded code hash in `reference_code/qwen35_before_modal_20261110/manifest.json`.
The earlier Qwen2.5 snapshot remains in `reference_code/before_qwen35_20261109T180506Z/`.
Every new cloud bundle includes its own frozen source snapshot for future resume.
