# experiments/MANIFEST.md — checksum + provenance manifest for the final runs

Recorded 2026-09-29 by the REPRO worker (CPU-only; no inference executed).
All paths relative to `experiments/`. Re-verify any entry with
`sha256sum <path>`.

## 1. Final-run artifacts (sha256 at record time)

| File | sha256 |
|---|---|
| `results/final_ruler/metrics.json` | `736391be92a068294854aa194798460a6b489969d73ea69ec3bc005c2b9484bd` |
| `results/final_ruler/config.json` | `6441d399f9a66e2aaca7e5d547e95ba575285b513af60a0e999edd4c4a044996` |
| `results/final_longbench/metrics.json` | `6cc82d3812697145f428ebf18a796799c84ffc21c1defdb1a52696096a4cca57` |
| `results/final_longbench/config.json` | `5d00627467464144611802e88e313b43f457c522ab13ab3a009e89f0edf30378` |
| `results/final_locomo/metrics.json` | `ad32ca78231e1324e3fdf583b90c34f620df0c000c00660f125b085e0b188994` |
| `results/final_locomo/config.json` | `b60ec6cf523a9c93fc8ffe8b5ee058cda18ff98e39bceecdddf520017e220ceb` |
| `results/final_kvcompare/metrics.json` | `9b1725cc6e100683f13a2b235f56eedef17ae59604d0765233061beb805842c3` |
| `results/final_kvcompare/config.json` | `81e7076650bb4faae46f1c0b7e1e038914ff6fcfa37e40be84d8e964647a1564` |

Each `results/final_*/` dir additionally holds `environment.json` and
`predictions.jsonl` (per-(system, example) raw records; failures recorded as
error rows, never hidden). Current HEAD pin: `4e212efb33dbb5e8903418235680a7da3be522ff`
(`experiments/`, branch `paper-rewrite`). Note: the four `config.json` files were
written during an earlier run and still embed the snapshot
`"git_commit": "3f314f8d3016e736b7e1973fff9547b703b37f55"`; the manifest pin above
supersedes them.

## 1b. Pilot-run artifacts (sha256 at record time)

30-example LongBench `musique` pilots (CPU). Each dir holds `metrics.json`,
`config.json`, `environment.json`, `predictions.jsonl`.

| File | sha256 |
|---|---|
| `results/pilot_multihop/metrics.json` | `d335972f29574b8aeee3d73be3d88db1f34cb724cdaa9d70827ac149950c87b9` |
| `results/pilot_multihop/config.json` | `62ca3a8ae45307545fd2f10f39a6916582a3723935c7c05cf58686a006b5d9f8` |
| `results/pilot_multihop_v2/metrics.json` | `8669a01c82b7284e0383b3a3af039567cb6e76c04b1a4c96d2097e0ffa56801a` |
| `results/pilot_multihop_v2/config.json` | `da099937b72f820e12dc6098973e21b5fb3feda3d284896ce48a08d68fbfdd4c` |
| `results/pilot_address/metrics.json` | `8be14c6219ef134766b531934e355b0834f6131def1a64c046ed3486acba1410` |
| `results/pilot_address/config.json` | `ddb66cb4b824e13e22dc8a8392a432222c6c55c5b5e0236059f4aea8973d154b` |
| `results/pilot_routing_v2/metrics.json` | `8adc109818a3b29064c9d18ecf859475f21a98df36c26df675bd62b037adff40` |
| `results/pilot_routing_v2/config.json` | `30dcf3f2150fade8ca2299a21376466e366646d01a8e73bcda889a462db0c6b7` |

Rerunnable CPU-only. `results/envcheck/` is a 1-example environment check;
`results/smoke_{ruler,longbench,locomo,kvpress,3b_nf4}/` and `results/_dryrun/`
are single-task smoke/validation runs — none are benchmark evidence and none are
covered by the checksums above.

## 2. Derived statistics — provenance (generator scripts + seed)

| File | sha256 | Generator | Seed / params |
|---|---|---|---|
| `results/stats.json` | `9f620267c266459462cd46785882276283b5edaf927585bf8e803609be01ac76` | `stats_real.py` (paired bootstrap CIs, McNemar, Wilcoxon, Cohen's d; reads `final_*/{metrics.json,config.json,predictions.jsonl}`) | `SEED = 42`, `N_BOOT = 10_000` |
| `results/error_breakdown.json` | `06ec8859d471cd9fc4a1151e07dd9b974e52023fbd23e37a803c3ee36271ba3b` | `error_taxonomy.py` (retrieval-miss vs block-hit-wrong-answer join over regenerated seed-42 contexts; reads `final_*/predictions.jsonl` + dataset caches) | seed 42; miss := score<1.0 containment, score<0.5 F1 (lenient, exact-zero subcounts reported) |
| `results/config_table.json` | `c405dbc386b877c655ce22ec4dbb9f16a388d6589ae2757f75f903009b8617fc` | `error_taxonomy.py` (True-B table: mean n_blocks, mean LLM-tokens/block, k=8, m=1, kappa) | same run as error_breakdown |

Both generators are CPU-only (tokenizer + cached data; never load weights or
run inference) and rerunnable in <5 min. LoCoMo metric throughout is the
documented **token-F1 adaptation** (no GPT-4 judge available), recorded per
example in `meta["metric_adaptation"]`.

## 3. Dataset cache identifiers

| Cache | sha256 (full) | Size |
|---|---|---|
| `benchmarks/data/longbench_cache/data.zip` (THUDM/LongBench `data.zip`, 35 files) | `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64` | 113,932,529 bytes |
| `benchmarks/data/locomo_cache/locomo10.json` (official snap-research file) | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` | 2,805,274 bytes |
| `benchmarks/data/ruler_cache/english_words.json` (cwe/fwe fallback pool) | `affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca` | 8,564,991 bytes |
| `benchmarks/data/ruler_cache/wordlists/` (`adjectivelist.txt` 912 / `nounlist.txt` 6782 / `verblist.txt` 1042 lines; wonderwords 3.0.1 vendored) | — (line counts verified) | — |
| `benchmarks/data/ruler_cache/essays/` (first 5 Paul Graham essays + siblings) | — (presence verified) | — |

RULER examples are generated synthetically at run time (seed 42); the caches
above plus `nltk` `punkt` (`~/nltk_data/tokenizers/punkt`, presence verified)
are the only static inputs. Fetch commands: BENCHMARK_PROVENANCE.md §§1–3,
RUNNING_EXPERIMENTS.md §3.

## 4. Model / decode / environment record (per `environment.json`)

- Model: `Qwen/Qwen2.5-3B-Instruct` (public/ungated) · dtype `nf4` · greedy
  decoding · `attn_implementation: eager` · seed **42** · block 512 · k=8 · m=1.
- Dense embedder: `sentence-transformers/all-MiniLM-L6-v2` (public/ungated).
- Python 3.11.16 · torch `2.14.0+cu130` · transformers 5.2.0 ·
  sentence-transformers 6.1.0 · datasets 5.0.1 · bitsandbytes 0.50.2 ·
  numpy 2.4.6 · kvpress 0.5.5 / rank-bm25 0.2.2 (from venv; the
  `environment.json` recorder writes `null` for these two fields — known
  recorder gap, values confirmed via `uv pip freeze`).
- GPU: `NVIDIA GeForce RTX 5060 Laptop GPU` (Blackwell `sm_120a`) ·
  7776.6 MB · CUDA runtime 13.0 · driver `615.71.09` ·
  OS `Linux-7.2.7-1-cachyos-x86_64-with-glibc2.44`.
- Full dependency pins: `experiments/requirements_locked.txt` (104 packages).

## 5. Git placeholder

- Repo root `/home/viksit/Projects/token-opencode` is **not** a git repository.
- Nested repo `experiments/` — branch: `paper-rewrite` — HEAD:
  `4e212efb33dbb5e8903418235680a7da3be522ff`
  (`11-page swarm: theory trims … title reframe, structural softened`).
- The legacy-tree move is **committed**, not pending: `archive/legacy_analysis/`,
  `archive/legacy_data/`, `archive/legacy_runners/` are tracked and
  `git -C experiments status --short` is clean (the only working-tree edits at
  this record were `README.md` and this `MANIFEST.md`, by the doc-fix worker).
  Re-run `git -C experiments log --oneline -1 && git -C experiments status --short`
  and paste the output alongside this manifest when handing results on.
