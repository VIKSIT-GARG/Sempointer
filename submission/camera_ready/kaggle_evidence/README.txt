Kaggle evidence included with the camera-ready manuscript

Run: 8ef26bfd68509bae3b6b
Source: retained combined report snapshot used to build the separate
sempointer_kaggle_results_draft.pdf (not appended to the manuscript).

report_data.json retains the aggregate snapshot, including partial families.
source_report_configuration.json retains checkpoint, precision, worker and
runtime provenance reported by the source. Its source description is the
report producer's description, not independent raw-database verification.
all_task_scores.csv and paired_comparisons.csv retain the full task inventory.

IMPORTANT: the raw archive download was incomplete. The manuscript table
reconciles with these report aggregates; the raw worker databases have not
been independently audited. RULER and SCROLLS are partial and are excluded
from the headline paired results. Quality directions count task-group means,
not individual-question wins or a common-scale accuracy statistic.

The 622-pair full-context SemPointer means are reconstructed report aggregates
for the eligible shared subset, not the overall 2,400-question means.
Latency includes generation and measured selector cost, excluding shared
ingestion and outer prompt preparation. It has no repeat-run interval.
These T4 NF4/FP16 results are separate from the bounded local NF4/BF16 study.

Run python tools/check_kaggle_snapshot.py from the package directory to
reproduce the headline percentages and quality directions from this snapshot.
The command uses only the Python standard library and performs no inference.
