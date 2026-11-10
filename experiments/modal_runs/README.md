# Retained cloud execution artifacts (local only)

These directories remain at their original paths to preserve configurations, frozen inputs, source snapshots, logs, recovery records and transfers. Large bundles, examples and runtime copies are not uploaded to Git. Their files are indexed with SHA-256 hashes in `docs/manifests/evidence_files.csv` from the project root.

| Directory | Role and interpretation |
| --- | --- |
| `precision_pilot/` | Precision selection, candidate inputs, attempts, and diagnostic results; a pilot, not final test evidence |
| `preparation_validation/` | Source preparation/tokenization diagnostics, not answer-quality evaluation |
| `ten_qwen35_v1/` | Original ten-family preparation bundle and cache; presence does not establish complete model evaluation |
| `ten_qwen35_measured_v1/` | Measured-run preparation and transfer bundle; retain launch/transfer metadata |
| `ten_qwen35_recovered_v1/` | Recovered cloud artifacts and recovery procedure; do not infer benchmark completion from successful transfer |
| `selected_inference_bundle.json` | Provenance pointer to the selected frozen bundle, not a scored result |

The separate Kaggle snapshot is under `kaggle_upload/` at the project root. It is not a raw Modal database validation. No failed/partial run was deleted during cleanup and no cloud job was launched.
