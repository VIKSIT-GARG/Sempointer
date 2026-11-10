# Retained experiment inventory

This inventory records files available in the author workspace. Presence, size,
or a matching checksum does not establish that a run completed or that its
results were independently validated. See [the evidence map](EVIDENCE.md) and
[per-run guides](../experiments/results/README.md) for interpretation.

| Evidence group | Files | Logical bytes |
| --- | ---: | ---: |
| `experiments/modal_runs` | 3 | 145,454 |
| `experiments/modal_runs/precision_pilot` | 13 | 1,118,816 |
| `experiments/modal_runs/preparation_validation` | 5 | 302,439 |
| `experiments/modal_runs/ten_qwen35_measured_v1` | 3,090 | 821,000,919 |
| `experiments/modal_runs/ten_qwen35_recovered_v1` | 3,087 | 598,494,197 |
| `experiments/modal_runs/ten_qwen35_v1` | 3,093 | 603,551,423 |
| `experiments/reference_code` | 1 | 619 |
| `experiments/reference_code/before_bounded_pointer_20261110` | 1 | 18,500 |
| `experiments/reference_code/before_precision_pilot_20261110` | 9 | 66,369 |
| `experiments/reference_code/before_qwen35_20261109T180506Z` | 60 | 457,558 |
| `experiments/reference_code/preparation_before_parallel_20261110` | 4 | 21,761 |
| `experiments/reference_code/qwen35_before_modal_20261110` | 42 | 315,145 |
| `experiments/results` | 2 | 887 |
| `experiments/results/bounded_pointer_full_v1` | 1,116 | 957,711,420 |
| `experiments/results/bounded_pointer_pilot_v1` | 258 | 125,719,070 |
| `experiments/results/dataset_counts` | 4 | 201,436 |
| `experiments/results/downloaded_suite` | 25 | 282,465,454 |
| `experiments/results/local_ten_qwen35_nf4_100_v1` | 1,157 | 1,572,756,306 |
| `experiments/results/paper_ablation` | 6 | 1,216,321 |
| `experiments/results/paper_kv` | 6 | 306,849 |
| `experiments/results/paper_locomo` | 6 | 1,881,037 |
| `experiments/results/paper_longbench` | 6 | 2,370,538 |
| `experiments/results/paper_ruler` | 6 | 1,050,227 |
| `experiments/results/reviewer_qwen35_4b_nf4_qa128_cwe512` | 14 | 914,309 |
| `experiments/results/reviewer_suite_validation` | 14 | 960,345 |
| `experiments/results/suite_preflight` | 4 | 316,806 |
| `experiments/results/suite_reference_checks` | 2 | 2,074 |
| `experiments/results/throughput_pilot` | 14 | 23,332,958 |
| `experiments/results/v2_locomo` | 6 | 1,009,974 |
| `experiments/results/v2_longbench` | 5 | 1,270,075 |
| `experiments/results/v2_ruler` | 5 | 471,670 |
| `kaggle_upload/publication_2026-11-11` | 20 | 2,434,309 |
| `kaggle_upload/reference_before_attention_fix` | 2 | 44,247 |
| `kaggle_upload/report_review_2026-11-10` | 12 | 2,701,368 |
| `kaggle_upload/sempointer_qwen35_frozen.bin` | 1 | 196,418,923 |
| `submission/camera_ready` | 6 | 1,082,730 |

Total: 12,105 files, 5,202,132,533 logical bytes.

The [file manifest](manifests/evidence_files.csv) retains each relative path,
size, and SHA-256 fingerprint. Large raw databases, source examples and cloud
bundles remain local and are ignored by Git; a manifest is not a backup.

To regenerate from available local files, run `python tools/inventory_evidence.py`.
The command reads bytes without loading models or modifying databases.

The separate Kaggle combined report is an aggregate snapshot. Its raw archive
download is incomplete; raw-worker-database validation remains pending.
