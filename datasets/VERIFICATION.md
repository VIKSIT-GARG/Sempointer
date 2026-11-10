# Dataset verification snapshot

Checked: 2026-11-08T07:51:13+00:00. Downloads continue; this report is a point-in-time snapshot.

**Everything downloaded: NO.**

- Completed or reused: **252 / 295 files**.
- Pending: **36**; downloading: **3**; failed URLs: **4**.
- Independently rehashed: **247 files**, **10.03 GiB**, **zero verification failures**.
- All **11 pinned Hugging Face inventories** match the queue, with no missing filenames, unexpected filenames, size differences, or upstream SHA-256 metadata differences.
- Downloader active: **True**, PID **1909738**.
- Free disk space: **62.61 GiB**.

## Integrity checks

Every file marked complete at the start of the independent integrity pass was reread and SHA-256 checked against the download record and the upstream hash where available. Available upstream MD5 values were also checked. ZIP archives passed member CRC checks; gzip/tar archives were streamed without extraction. Parquet footers/schemas were readable, and JSON/JSONL files smaller than 32 MiB were syntax checked. This is not a full semantic validation of every dataset record or every Parquet data page. Files completed after the audit snapshot are checked by the downloader but are not included in the independent 247-file pass.

## Collection coverage

Counts include metadata and source archives as well as data files; a completed source archive does not mean generated benchmark examples exist.

| Collection | Complete / planned |
| --- | --- |
| xiaowu0162/longmemeval-cleaned | 3/5 |
| RMT-team/babilong | 122/137 |
| bigai-nlco/LooGLE | 6/7 |
| L4NLP/LEval | 24/24 |
| tau/scrolls | 12/14 |
| allenai/qasper | 5/5 |
| zai-org/LongBench-v2 | 2/3 |
| zai-org/LongBench | 4/4 |
| xinrongzhang2022/InfiniteBench | 10/14 |
| deepmind/narrativeqa | 36/39 |
| NVIDIA/RULER | 1/1 |
| snap-research/locomo | 2/2 |
| gkamradt/needle-in-a-haystack | 1/1 |
| beir-cellar/beir | 1/1 |
| Alab-NII/2wikimultihop | 1/1 |
| StonyBrookNLP/musique | 1/1 |
| google-deepmind/narrativeqa | 1/1 |
| hotpotqa | 0/4 |
| qasper | 2/2 |
| 2wikimultihop | 4/4 |
| musique | 0/1 |
| beir | 5/15 |
| hotpotqa/hotpot_qa | 9/9 |

## Outstanding access/source issues

- Four original HotpotQA URLs fail because `curtis.ml.cmu.edu` does not resolve. The publisher-hosted Hugging Face replacement is complete, but the original JSON URLs remain recorded as failed; no byte-equivalence between the original raw JSON and packaged Parquet is claimed.
- BioASQ, Signal-1M, TREC-NEWS, and Robust04 are listed by BEIR without public direct downloads. They are explicit access gaps outside the 295-file public queue.
- Large remaining dataset files are queued or transferring. No disk-space deferrals were observed at this snapshot.

Detailed evidence: `verification.json`, `inventory_verification.json`, and live `status.json`. The last of these changes while downloads run.
