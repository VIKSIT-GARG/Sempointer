# Dataset downloads

This directory holds the full downloadable releases listed in
`../docs/STEP1_DATASET_AUDIT.md`, rather than only the paper's evaluation subsets.
No model weights are included and no benchmark is executed.

## Layout

- `huggingface/<owner>/<dataset>/`: every file from each pinned dataset revision,
  including all published splits/configurations in that release.
- `originals/`: original HotpotQA, 2WikiMultihopQA, MuSiQue, QASPER, LoCoMo,
  and the publicly downloadable BEIR corpus archives.
- `sources/<owner>/<repository>/`: pinned upstream source archives, including
  RULER and Needle in a Haystack generators and bundled assets.
- `status.json`: per-file URLs, revisions, expected sizes, checksums, progress,
  failures, disk deferrals, and any access restrictions found during discovery.
- `download.log`: transfer completion and error messages.

The existing verified LongBench ZIP is reused by symlink. Existing experiment
caches and results are preserved. Compressed archives stay compressed; extraction
would require additional disk space. RULER and Needle in a Haystack are generators:
downloading their source does not create every possible synthetic test set.

The manager allows three BEIR transfers plus one independent transfer from other
sources, so slow BEIR transfers cannot block the Hugging Face queue. It checks upstream SHA-256/MD5 when
available, records SHA-256 for completed files, and retains `.part` transfers for
resumption. It leaves at least 10 GiB free on the filesystem and records files that
would cross that reserve as `deferred_disk_space`. Unavailable or access-restricted
sources are recorded explicitly, not silently counted as downloaded.

## Inspect progress

From the project root:

```bash
tail -f datasets/download.log
```

Open `datasets/status.json` for the exact per-file state. `complete` and
`verified_existing` mean that file has been checked; `pending`, `downloading`,
`retrying`, `failed`, and `deferred_disk_space` do not mean completion.

## Resume after the manager exits

```bash
experiments/warden/bin/python -u datasets/download_all.py
```

A lock prevents two managers from running together. Completed files are checked
and reused, and partial downloads use HTTP Range where supported. An upstream
error or insufficient disk space may require resolving the recorded issue before
retrying. The process ID and overall phase are in `status.json`.

All manager writes are scoped to this directory. Source code from downloaded
archives is not executed. Downloading a dataset does not integrate it with the
existing experiment loaders.
