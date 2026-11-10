# Local GPU ten-benchmark evaluation

Run: `2146b7900c244e92e6f3`; status: **complete**.

Only recorded predictions are evidence. A budget stop means partial coverage, not benchmark completion.

| Family | Selected questions |
| --- | ---: |
| ruler | 100 |
| longbench | 100 |
| locomo | 100 |
| longmem | 100 |
| babilong | 100 |
| longbench_v2 | 100 |
| infinite | 100 |
| loogle | 100 |
| leval | 100 |
| scrolls | 100 |

## Recorded stage counts

{
  "cwe_cap32": {
    "scored": 100,
    "skipped": 8
  },
  "dev_cautious": {
    "scored": 72
  },
  "dev_loose": {
    "scored": 72
  },
  "dev_paper": {
    "scored": 72
  },
  "test": {
    "scored": 5105,
    "skipped": 895
  }
}

See each stage's REPORT.md, metrics.json, raw predictions and attempts. No overall mixed-metric score is computed.

CWE cap comparisons are in cwe_cap_audit.json. The frozen development choice is in threshold_selection.json.

These are selected benchmark subsets with explicitly adapted metrics; they are not official leaderboard submissions.
Confidence intervals over questions do not remove document/conversation clustering or source overlap.

Execution: local CUDA, Qwen3.5-4B NF4, batch one. No cloud inference.
