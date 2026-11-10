# Kaggle backup evaluation

Import **SemPointer_Kaggle_T4x2_AttentionFix.ipynb** into Kaggle. The notebook embeds the full
launcher; **sempointer_kaggle.py** is provided separately for inspection.
Attach the existing private Dataset containing **sempointer_qwen35_frozen.bin**.
Select GPU T4 x2, enable Internet, and enable file persistence if available.

The first settings cell defaults to `RUN_MODE = "pilot"` and `HOURS = 2.0`.
Run all cells. Existing working files are reused, and a fresh runtime is set up
automatically with pip-less venv creation. The package installation uses the
notebook's existing pip to avoid Kaggle's failing ensurepip bootstrap.

The pilot command runs the frozen development phase (36 questions, three
candidates, two systems), freezes its selected thresholds, and runs the original
representative pilot on two independent model copies. Each question and all six
of its systems belong to exactly one GPU. It then exports results and stops.

Inspect `pilot_projection.json` or the final notebook output. To continue, set
`RUN_MODE = "run"`, choose an allowance below the remaining session time, and
rerun. The pilot predictions are reused. For a new saved background full-run
session, nine launcher hours leaves headroom within Kaggle's 12-hour maximum.
`HOURS` starts when the launcher starts, not when Kaggle creates the session;
it does not read the provider's timer or remaining weekly quota.

For unattended work, use the clean notebook's Save Version / Save & Run All.
Download `sempointer_kaggle_results.tar.gz` after each invocation. For a later
session, attach the saved output as Input. A single result archive is restored
automatically; choose `RESTORE_ARCHIVE` explicitly when several are present.
All phases stop without changing question counts or token caps when the time
allowance expires. A hard provider interruption can lose unsaved working files;
file persistence alone is not the durable checkpoint.

## Evidence and provenance

The corrected launcher explicitly expands shared KV heads and requires efficient
SDPA for every generation. Both T4s passed the user's direct 16,384-token kernel
check at 832 MiB peak allocation each. This is a diagnostic, not a full-model
pilot or throughput result.

Corrected results use `kaggle_results_efficient/`; the earlier automatic-SDPA run
remains under `kaggle_results/`. A complete earlier development choice is reused
only after model/source/package identity, exact paired IDs, and its selection
rule are validated. Raw SQLite snapshots and the original backend provenance
are retained in `development_reference/`. Earlier main-test predictions are not
imported across the backend change. If no valid saved development run is present,
development runs normally with the corrected backend. Existing result archives
are backed up before the corrected run exports its own archive.

- Original bundle: `8ef26bfd68509bae3b6b`, 3,000 main questions in ten families.
- Model: `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, thinking disabled.
- Precision: NF4 weights; explicit FP16 compute and non-quantized components.
  This is separately labeled from L40S BF16 results. Batch size is one per GPU.
- Original source and examples are checksum-verified and unchanged. A runtime
  copy changes two loader precision expressions. The adapter adds a finite-logits
  guard; its hash, derived source hashes, packages, model weight hashes and GPU
  names are frozen in the output config and checked before any resume.
- Raw evidence lives in each worker's `results/*/results.sqlite`. `attempts`
  retains every raw attempt; `latest` retains the current response/status for each
  question and system. Prompts, answers, gold references, metrics and timings are
  included. The combined report reads both databases without duplicating them.
- Reports retain adapted task metrics and paired comparisons. There is no pooled
  score across different metrics. Full-context skips and errors are not scores.
  Any evaluation error stops the workers and retains the failed attempt.
- The export uses SQLite backups, including committed WAL records. It excludes
  model weights, environments, input examples and duplicate raw JSONL copies.
  Worker logs, thresholds, manifests, configurations and state are retained.

## Local verification

From this directory:

```bash
../experiments/warden/bin/python -m pytest -q test_kaggle_runner.py
../experiments/warden/bin/python build_notebook.py
```

Tests use a fake model boundary with the real frozen evaluator, metrics and
storage. They are software checks, not benchmark evidence. The earlier one-T4 NF4/FP16 smoke passed. The retained Kaggle report snapshot now covers eight complete 300-question subsets, with RULER and SCROLLS partial. Its downloaded raw archive is incomplete and has not been independently audited; see `publication_2026-11-11/README.md` and the project evidence map.
