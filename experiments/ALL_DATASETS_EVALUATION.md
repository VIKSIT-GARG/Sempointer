# Evaluate the downloaded releases with SemPointer

The main script is `run_all_datasets.py`; `run_downloaded_suite.sh` provides
one-command stages. The default model is **Qwen/Qwen2.5-3B-Instruct, NF4**.
Weights stay frozen. This suite evaluates inference-time retrieval; it does not
train or fine-tune a language model. No complete GPU evaluation was run while
building the scripts.

## Run commands

From `/home/viksit/Projects/tokens2/experiments`:

```bash
# Installed into the existing project environment during setup:
uv pip install --python warden/bin/python -r requirements-evaluation.txt

# One-time explicit model download into this project (if not already cached).
# This stage needs network and several GiB; it does not run inference.
bash run_downloaded_suite.sh model

# Catalog and inspect one real record per source. No model weights are loaded.
bash run_downloaded_suite.sh check

# Seven real generations on one 4K RULER example. You launch this yourself.
bash run_downloaded_suite.sh smoke

# All available evaluation sources and separate BEIR retrieval diagnostics.
# Runs sequentially on the laptop; --limit 0 means all eligible examples.
bash run_downloaded_suite.sh full

# Rebuild reports from committed records, including an interrupted run.
bash run_downloaded_suite.sh report
```

The full suite can involve very many examples and substantial output/index space;
it is not a promise that every published context length fits an 8 GB GPU. Model
and tokenizer must already be cached or present under `../models/`: network
downloads are disabled in evaluation stages. The model stage above explicitly
downloads the official checkpoint, which is quantized to NF4 during loading. Keep using
the paper model initially; another model needs engine compatibility verification
and a separate output directory. The engine's incremental-cache generation path
has not been validated for Qwen3.5's hybrid architecture.

For a smaller broad run:

```bash
warden/bin/python run_all_datasets.py run --limit 10 \
  --partition test --output results/downloaded_pilot
```

Select specific families using the names in `catalog.json`, e.g.
`--families longbench,longbench_v2,locomo`. `--tasks` accepts exact task names.
The cap is **per source/shard**, not per whole family. Resume by repeating the
same command. Changed code, package versions, source content, model/tokenizer
revision, or experimental settings require a new output directory.
For the final complete-release run, wait for downloads to finish. Catalogs are
snapshots of available files; late downloads can add sources and therefore change
the run configuration. BEIR explicitly records missing archives as skipped.

## What each system is compared with

| System | Evidence sent to the same frozen model | Question tested |
| --- | --- | --- |
| `sempointer` | Protected BM25 top two, optional expansion to four; unresolved pointers retained | Main method |
| `pointer_off` | Exactly the same selected block IDs, with unresolved pointers removed | Do pointer entries add answer value or only token cost? |
| `bm25_2` | Fixed top two blocks, no pointers | Is the adaptive method worth more than the cheaper fixed budget? |
| `bm25_4` | Fixed top four blocks, no pointers | Does adaptation reduce cost at similar observed answer quality? |
| `bm25_matched_budget` | BM25 ranking filled under SemPointer's total per-example prompt-token allowance | Does adaptive routing help at a comparable prompt budget? |
| `full_context` | All source blocks, when the complete prompt fits | What quality is lost by restricting evidence? |
| `no_context` | Question and instructions only | How much can the model answer without the supplied history? |

Matched-budget selection may include a partial final block. Tokenizer/framing
effects can leave slack, which is logged. This is an identical **upper allowance**,
not a claim that every pair has exactly the same token count. The budget includes
instructions, labels, pointers, and query, using the same tokenizer accounting as
generation. Full context is skipped rather than truncated when it does not fit.

Router thresholds stay fixed: 0.80 weighted coverage; ambiguity expansion below
0.95 coverage when the third/second score ratio is at least 0.85; at most four
resolved blocks and 2,560 resolved tokens. Semantic rescue is off. All systems use
one shared engine and identical task-specific output caps. Execution order rotates
deterministically per example. Real selection IDs and actual prompts are retained.

## Data coverage

The current local catalog has 276 source/task configurations (this can grow when
additional downloads finish). It includes all 34 LongBench v1 archive members
(base and length-balanced variants), LongBench v2, all three LongMemEval variants,
135 BABILong task/length files, InfiniteBench, L-Eval, LooGLE, labeled SCROLLS
evaluation splits, all LoCoMo conversations, original HotpotQA, MuSiQue, 2Wiki,
QASPER and NarrativeQA evaluation releases, and eight supported RULER tasks at
three configured lengths. Shards remain individual catalog entries, not distinct
benchmark claims.

Training splits are excluded by default. `--include-train` enables explicit
training-data diagnostics; do not describe those scores as held-out evaluation.
Hidden-label test sets are recorded as `missing_gold`. Duplicate original and
packaged releases stay separate; do not average overlapping datasets together.
SCROLLS consecutive references are merged into one question; a non-contiguous
repeat raises a recorded loader error rather than silently double-counting it.

RULER is generated with seed 143 by default and target-tokenizer length counting
at evaluation time. Its cached essay/word-list deviations from upstream remain.
`--ruler-samples` defaults to ten per task/length; synthetic generators have no
finite universal "all examples" set. Needle-in-a-Haystack/RULER source archives
are recorded as generation assets, not falsely counted as model evaluations.

The new suite uses strict 512-token chunk boundaries. Historical project chunking
merged short tails and could exceed that cap. Session dates are now included for
LoCoMo. Output caps are larger than the paper's 32/64-token shared caps. Thus these
are **new, improved evaluation runs**, not byte-identical reproductions of the
uploaded paper. Existing `run_v2_suite.sh` remains the historical reproduction path.

## Scores and protocol limits

- English QA uses the existing normalized token-F1 or normalized exact match.
- Multiple-choice tasks use extracted option-set accuracy; summaries use local
  word-token LCS ROUGE-L (Chinese VCSUM uses character tokens); Chinese QA
  answers use a character-F1 adaptation.
- Code completion uses first-line string similarity; code-output tasks use
  numeric-string matching. Downloaded code is never executed.
- LongMemEval exports answers/golds for later judging and reports a clearly
  labeled token-F1 **surrogate**, not official judge accuracy.
- LoCoMo uses the declared token-F1 adaptation and records question categories.
- Other adapted metrics/prompts are named in each example's `protocol` field.
  These scores must not be presented as official leaderboard submissions.

Development/test partitioning hashes document/conversation groups: 20% development,
80% test, with all questions from one group together. The default is `test`;
`--partition dev` supports development work in a separate run. This new partition
does not erase previous exposure to the paper's datasets, or make overlapping
benchmark releases independent. Thresholds are not automatically tuned.

## BEIR is a separate retrieval experiment

BEIR contains document relevance labels rather than reference answers. The `beir`
mode creates disk-backed SQLite FTS5 indices and retrieves 20 candidates by default.
The SemPointer router then selects 512-token blocks from that pool. Report linear-
gain nDCG@10, Recall@10, selected-document recall, and shared candidate recall.
Candidate retrieval never uses qrels. Pointer-off has the same retrieval ranking
by construction; its causal value is tested in the generation experiments above.

This is a **two-stage retrieval diagnostic**, not official full-corpus SemPointer
QA. SQLite FTS5's BM25 differs from the block router's BM25 implementation.
`--beir-candidates` changes candidate depth and must be held fixed across runs.
Indexing can require substantial extra disk space; it stops at 10 GiB free.
Completed generated indices are removed by default, keeping one corpus index at
a time alongside the results. `completed_corpora.json` prevents completed corpora
from being indexed again on resume. Use `--keep-beir-indices` only if you want to
retain all generated indices and have enough disk space. Original ZIPs stay intact.
Restricted BioASQ, Signal-1M, TREC-NEWS, and Robust04 are not publicly downloaded
and cannot be evaluated until their data/access is supplied. Incomplete `.part`
archives are not treated as complete datasets.

## Evidence written to every run directory

| Artifact | Contents |
| --- | --- |
| `catalog.json` | Selected tasks, archive members, splits, decoding caps, source revisions, download coverage |
| `config.json` | Exact settings, source SHA-256 values, code fingerprint, package versions, resolved model/tokenizer provenance |
| `environments.jsonl` | QA generation engine and hardware/package environment on each invocation |
| `results.sqlite` | Authoritative durable latest result and every attempt; resumable by example/system |
| `events.jsonl` | Live per-attempt log, including prompts, questions, references, responses, IDs and errors |
| `predictions.jsonl` | Export of latest per-example results; produced at completion or by `summarize` |
| `attempts.jsonl` | Full attempt history exported from SQLite |
| `metrics.json` | Per-task/system counts, scores, CIs, tokens/timing/memory and paired differences |
| `summary.csv`, `REPORT.md` | Tables for reviewing what was tested and the measured results |

Errors are retryable on resume. Scored/skipped pairs are not rerun under unchanged
settings. A nonzero exit code indicates recorded execution/loader errors; errors
and skipped examples are not assigned fake zero answer scores. SQLite commits
every result; re-export after interruption with `summarize`. Two managers cannot
write the same run directory concurrently.

Latency is generation plus measured selector cost. Shared ingestion and total
prompt-suite preparation are logged separately, so do not claim these timings
are complete production request latency. Bootstrap CIs resample examples, not
independent document clusters; correlated questions limit inference. Quality
differences that include zero are not proof of equivalence.

Default laptop guards: 8,192 input tokens, 262,144 source tokens and 2,000,000
source characters. Oversize histories, including most 10M-token BABILong examples,
are explicitly skipped. Override guards only when your hardware/serving setup can
support them; unresolved pointers also consume prompt tokens. Results stop at a
2 GiB free-space reserve. Source data and paper results are never overwritten.

For a full schema scan, without generation:

```bash
warden/bin/python run_all_datasets.py validate --validate-all \
  --output results/downloaded_full_schema_scan
```

Sample-level validation and CPU tests are software checks, not empirical model
results. Run the smoke stage before committing the laptop to the full experiment.
