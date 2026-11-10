# SemPointer

SemPointer selects exact source blocks with an adaptive lexical router and exposes short descriptors of unexpanded candidates. The current paper evaluates its quality–cost tradeoff and separates evidence selection from the effect of displaying descriptors.

## Current paper and submission

- [Camera-ready PDF](output/pdf/SemPointer_camera_ready.pdf): **11 pages including references**, with the separate Kaggle resource and illustrative quality comparisons.
- [Editable LaTeX](submission/camera_ready/main.tex): text, native tables, and equations. Edit this file for the current paper.
- [Self-contained source ZIP](output/submission/SemPointer_source.zip): bibliography, template, vector figures, figure data, and Kaggle aggregate evidence.
- [Submission checks and outstanding items](output/submission/SUBMISSION_NOTES.md): the supplied `sn-jnl` template was used; its ICT Series applicability and the author's Turnitin report remain external checks.
- [Original submitted PDF](main.pdf): preserved unchanged; not the current camera-ready version.

Build the current manuscript without inference:

```bash
cd submission/camera_ready
mkdir -p build
tectonic main.tex --outdir build --keep-logs --keep-intermediates --reruns 3
python tools/check_kaggle_snapshot.py
```

The compiled document is `submission/camera_ready/build/main.pdf`. The separate [source-package README](submission/camera_ready/README.txt) also gives a pdfLaTeX/BibTeX build sequence. Figures are vector PDFs; tables and equations are editable LaTeX.

## Research layout

| Directory | Purpose |
| --- | --- |
| [`experiments/`](experiments/README.md) | Model interface, router, baselines, task adapters, runners, CPU tests, and retained experiment evidence |
| [`experiments/results/`](experiments/results/README.md) | Eighteen labeled local result groups, each with its own README |
| [`docs/EVIDENCE.md`](docs/EVIDENCE.md) | Mapping from manuscript claims to original, bounded, and Kaggle evidence |
| [`docs/EXPERIMENT_INVENTORY.md`](docs/EXPERIMENT_INVENTORY.md) | File counts, sizes, and provenance inventory; does not infer completion from file presence |
| [`kaggle_upload/`](kaggle_upload/README.md) | T4 notebook, frozen launcher, reports, and aggregate snapshots |
| [`paper/`](paper/README.md) | Historical 10-page LNCS source, 22-page extended report, and derivation scripts |
| [`submission/`](submission/README.md) | Current camera-ready source tree |
| [`output/`](output/README.md) | Canonical delivery PDFs, source ZIP, checks, and author notes |
| [`datasets/`](datasets/README.md) | Download manager and acquisition manifests; downloaded releases stay local |
| [`models/`](models/README.md) | Local pinned checkpoint caches; weights stay local |
| [`tools/`](tools/README.md) | Read-only inventory and document-verification utilities |

## Evidence and interpretation

The bounded local study reuses 1,000 questions at three input ceilings; these are not 3,000 independent observations. At 4K, it saves input tokens and generation time against budget-filling BM25 while recording 123 quality wins, 648 ties, and 229 losses. The overall fixed-payload descriptor intervention is inconclusive. Metrics differ by task, and some adapters differ from official benchmark protocols.

The separate Kaggle snapshot uses the same checkpoint on two T4 workers with FP16 compute. Its matched-budget comparison covers 2,400 shared questions across eight complete family subsets; RULER and SCROLLS remain partial. Reported request latency includes selector cost and is not the local generation-only measure. The raw archive is incomplete and not independently audited. Illustrative task comparisons were selected after inspection and are not confirmatory evidence of superiority.

## Local storage and Git policy

Scientific code, tests, manuscript sources, aggregate reports, metadata, and compact historical predictions are versioned. Large SQLite files, full source contexts, duplicate event streams, cloud transfer bundles, downloaded datasets, model weights, runtimes, and caches are retained locally and excluded from Git. [The evidence manifest](docs/manifests/evidence_files.csv) records retained raw-file paths, sizes, and SHA-256 hashes; it is an index, not a remote backup of those bytes.

AI tool instructions, skills, chat/state, browser traces, scratch files, and agent plans are ignored at every directory level. A scientific file is not ignored merely because an AI tool edited it. [`.gitignore`](.gitignore) defines the publishing boundaries.

The repository now starts at the project root. The previous `experiments/` Git history was preserved through an explicit path-migration commit. This camera-ready cleanup is published on `camera-ready-cleanup-2026-11-11`; the separate concept-paper rewrite on GitHub's `main` is preserved.

See [cleanup and verification records](docs/CLEANUP_2026-11-11.md) for retained evidence, removed scratch, compatibility links, and validation.
