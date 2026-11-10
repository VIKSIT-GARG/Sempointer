# Counted evaluation samples

All 276 configured generation sources were scanned without loader errors. No model inference was run.

- Total adapted entries across dev and test: **209,131**.
- Labeled, nonempty entries across dev and test: **154,473**.
- Labeled, nonempty entries in the configured test partition: **124,770**.
- Test entries within the 2,000,000-character source limit: **122,420**.
- Test entries exceeding that character limit: **2,350**.
- Test entries with hidden/missing answers: **43,803**; empty contexts: **18**.

Counts include overlapping original and packaged releases and are not deduplicated unique questions. The local test partition is a document-group hash split of available non-training sources, including official validation releases. RULER contributes the configured 240 synthetic examples, of which 187 fall in the local test partition.

The 122,420 figure is before tokenizer, prompt-length, model-context and GPU checks. The exact number successfully evaluated is known only after those checks and model runs. Seven comparison arms produce up to 856,940 attempted system-example pairs before further skips.

| Family | Labeled test entries | Within character limit |
|---|---:|---:|
| 2wikimultihop | 30,225 | 30,225 |
| babilong | 11,205 | 9,351 |
| hotpot_hf | 11,848 | 11,848 |
| hotpot_original | 11,796 | 11,796 |
| infinite | 3,163 | 3,106 |
| leval | 1,839 | 1,839 |
| locomo | 1,793 | 1,793 |
| longbench | 6,686 | 6,686 |
| longbench_v2 | 402 | 361 |
| longmem | 1,194 | 796 |
| loogle | 5,372 | 5,372 |
| musique | 5,814 | 5,814 |
| narrative_original | 11,624 | 11,624 |
| qasper_original | 1,929 | 1,929 |
| ruler | 187 | 187 |
| scrolls | 19,693 | 19,693 |

BEIR is separate: **44,929 query/split entries** in currently complete archives. This is not a deduplicated query count and excludes archives still downloading. It is not included in the QA totals above.

Machine-readable details: `sample_counts.json`.
