# SemPointer evidence map

The first section preserves the 2026-11-08 submitted-paper provenance. The dated camera-ready update below covers the current study. Git has since moved from `experiments/.git` to the project root; the earlier statement describes that historical consolidation.

## Original submitted paper (historical record)

The canonical `main.pdf` is the exact uploaded `/home/viksit/sempointer.pdf`, SHA-256 `8a41ed056d5e4498c34436b240e1d668a4b107889f4e2060788ed702a3ff3e24`.

| Result directory under experiments/results | Role |
| --- | --- |
| `v2_ruler` | Current SemPointer, BM25-2, and BM25-4; 180 examples |
| `v2_longbench` | Current SemPointer and BM25-4; 600 examples |
| `v2_locomo` | Current SemPointer and BM25-4; 497 examples with adapted token F1 |
| `paper_ruler` | Earlier implementation and full-context comparisons on matching example IDs |
| `paper_longbench` | Earlier implementation comparisons on matching example IDs |
| `paper_locomo` | Earlier implementation comparisons on matching example IDs |
| `paper_ablation` | Earlier selector diagnosis motivating lexical routing |
| `paper_kv` | Earlier KV controls; not evidence for the current text router having a KV advantage |

`paper/build_results.py` requires all six main/comparison runs. It validates scored counts and errors and computes bootstrap intervals from per-example predictions. `paper/build_figures.py` reads the three current runs. Preserve every file in each retained run, including configuration and environment metadata. The ablation and KV runs are retained as supporting evidence for the diagnosis and limitations, rather than discarded based on their age.

The model, benchmark loaders, metrics, baselines, datasets, and existing `experiments/.git` remain in place. No benchmark was rerun and no result values were changed during consolidation.

The uploaded PDF and retained source have matching title and key abstract results. The source includes author details omitted by the uploaded export; exact full-text equivalence has not been established. Compiling writes a separate authored export under `paper/build/` and preserves the uploaded reference.

## Evidence map update — 2026-11-11

The canonical manuscript source is `submission/camera_ready/main.tex`; the camera-ready PDF is `output/pdf/SemPointer_camera_ready.pdf`. The original uploaded reference and its provenance above are retained separately. The present manuscript includes the following distinct evidence, without pooling scores or timings across runs.

| Run or record | Scope and status |
| --- | --- |
| `experiments/results/v2_{ruler,longbench,locomo}` and `experiments/results/paper_{ruler,longbench,locomo,ablation,kv}` | Original Qwen2.5 study and historical comparisons and diagnostics described above. |
| `experiments/results/local_ten_qwen35_nf4_100_v1` | Intermediate local full-index Qwen3.5 test, router development records and CWE 32-versus-512 output-cap audit. |
| `experiments/results/bounded_pointer_pilot_v1` | Separate 30-question descriptor-cap development pilot and 100-question pilot test. |
| `experiments/results/bounded_pointer_full_v1` | Completed local bounded 1,000-question study at three budgets, including `fixed_payload_pointer_control_v1` and `loss_audit_4096_v1`; run `c73abaf595dc68e55696`. |
| `experiments/results/reviewer_qwen35_4b_nf4_qa128_cwe512`, `reviewer_suite_validation`, `downloaded_suite`, `suite_preflight`, `suite_reference_checks`, `throughput_pilot` | Reviewer controls, smoke, validation, preflight or pilot records. Individual records may be partial; they are not full benchmark results. |
| `experiments/results/dataset_counts` | Dataset availability and counts, not benchmark completion or performance. |
| `experiments/modal_runs` | Preparation, precision pilots, measured and recovered bundles, and launch/transfer logs. Preserve partial and failed attempts as provenance. |
| `kaggle_upload/publication_2026-11-11` and `submission/camera_ready/kaggle_evidence` | Separate Kaggle run `8ef26bfd68509bae3b6b` combined-report aggregates and provenance. Eight families have complete recorded coverage; RULER and SCROLLS are partial. The downloaded raw archive was incomplete and raw-database validation remains pending. |

The root repository versions code and summaries; heavy SQLite databases, input contexts and large event streams can remain locally ignored and are cataloged by the SHA-256 evidence manifest. Ignoring files does not authorize deleting them. Keep predictions, databases and SQLite WAL/SHM sidecars, frozen inputs and code, configurations, manifests, logs, and derived control/audit records together. See `paper/REVISION.md` for the detailed local table-to-evidence mapping and limitations.

### Latest Downloads check — 2026-11-11

The latest nonempty Kaggle download was inspected read-only after the camera-ready integration. Its combined `metrics.json` SHA-256 is `8ff2dde050f242a8de09434d270f10bc3a0d989aa92f3cea02fab8e17aaedf74`, identical to the snapshot already used by both current Kaggle tables. It is still a truncated `.tar.gz.part`; the zero-byte final-name file contains no results. The older complete archive has an earlier, smaller report and does not supersede the integrated snapshot. No newer score values were found, no duplicate observations were added, and raw-database validation remains pending. `docs/manifests/kaggle_download_verification.json` records archive sizes and hashes. Downloads were not modified.
