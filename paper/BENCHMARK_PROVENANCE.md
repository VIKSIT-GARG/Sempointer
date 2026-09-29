# Benchmark Provenance — SemPointer final runs

Scope: `experiments/results/{final_ruler,final_longbench,final_locomo,final_kvcompare}/`,
produced by `experiments/run_benchmark.py` + `experiments/benchmarks/{ruler,longbench,locomo}.py`
+ `experiments/metrics/official_metrics.py`.
Code refs below are paths inside `experiments/`. All loader docstrings quoted here were
verified against the files on disk (git commit `3f314f8`, per `config.json` in each final run).

---

## 1. RULER (synthetic haystacks)

- **Official source.** Repo: https://github.com/NVIDIA/RULER · Paper: arXiv:2404.06654
  ("RULER: What's the Real Context Size of Your Long-Context Language Models?").
- **What the local loader implements** (`benchmarks/ruler.py`).
  Faithful re-implementation of the official generators for
  `niah_single_1/2/3`, `niah_multikey_1/2`, `vt`, `cwe`, `fwe`:
  verbatim official prompt templates (`_OFFICIAL_TASKS`, from
  `scripts/data/synthetic/constants.py`, composed as `template + answer_prefix` exactly as
  `scripts/data/prepare.py` does for `model_template_type='base'`),
  verbatim `TASK_ARGS` (from `scripts/data/synthetic.yaml`),
  the official binary-search haystack sizing, the 40 official needle depths,
  the official vt 1-shot ICL example (with its tokens counted in the budget),
  and the official cwe few-shot block. Prompt layout is byte-reconstructible:
  each example stores `meta["prompt_before_context"]` / `meta["prompt_after_context"]`
  such that `before + context + after` equals the official model prompt.
- **Exact metric implemented** (`metrics/official_metrics.py::ruler_containment`):
  per-sample `string_match_all` from `scripts/eval/synthetic/constants.py` —
  `1.0 if ref.lower() in pred.lower() else 0.0`, averaged over refs.
  Official task score = mean over samples × 100; the runner stores the per-sample
  value in [0, 1] (`metrics.json: score_mean`), so multiply by 100 to compare with
  official RULER tables. No answer extraction is applied, matching the official
  evaluator (`evaluate.py::postprocess_pred` strips control characters only).
- **Documented deviations** (all stated in the `benchmarks/ruler.py` docstring; none hidden):
  1. **Essay corpus truncated.** Only the first 5 Paul Graham essays (sorted order,
     matching the official prefix) are cached vs the official 275-essay corpus; the
     official repetition logic is kept for longer requests.
  2. **Word lists vendored.** `wonderwords` is not installed; its v3.0.1 assets are
     cached and read identically. NIAH keys sample adjective and noun independently —
     distribution-identical to the official uniform choice over the adj–noun product.
  3. **Token counting is approximate IN THE FINAL RUNS.** `run_benchmark.py::build_examples`
     calls the RULER builder **without** `tokenizer_fn`
     (`builder(task=task, context_length=clen, n_samples=..., seed=...)`), so haystack
     sizing falls back to the documented word-based estimates (prose ≈ 0.75 BPE
     tokens/word; cwe ≈ 1 token/entry; fwe ≈ 2 tokens/coded word). Realized lengths can
     deviate up to ~20% from the nominal 4096/8192/16384 budgets.
  4. **Official per-task jsonl bookkeeping fields** (`length`, `token_position_answer`, …)
     are not reproduced; golds are recorded at generation time from the seeded RNG.
  5. **Generation budget is smaller than official.** Final runs use `--max-new-tokens 32`
     while official `tokens_to_generate` is 128 (niah), 30 (vt), 120 (cwe), 50 (fwe).
     Only vt matches; niah/cwe/fwe generations are cut short, which can only hurt recall
     of multi-answer tasks (cwe expects 10 words, fwe 3).
  6. **Official prompt is reconstructed, not fed verbatim.** Systems chunk the context
     into 512-token blocks and use their own prompts
     (`sempointer/pipeline.py: DEFAULT_SYSTEM_PROMPT`), so the byte-exact official prompt
     in `meta` documents provenance but is not what the model sees.
- **Dataset fetch commands.** No dataset download: RULER examples are generated
  synthetically at run time. The two cached resources a reproducer needs:
  - Paul Graham essays (first 5, sorted): from https://github.com/gkamradt/LLMTest_NeedleInAHaystack
    → `experiments/benchmarks/data/ruler_cache/essays/` (files present: `addiction.txt`,
    `aord.txt`, `apple.txt`, `avg.txt`, `before.txt`, …).
  - `wonderwords` 3.0.1 word lists → `experiments/benchmarks/data/ruler_cache/wordlists/`
    (`adjectivelist.txt` 912 lines, `nounlist.txt` 6782, `verblist.txt` 1042);
    `english_words.json` fallback pool (sha256 `affcd6d4…bc2eca`, see §5).
  - Pipeline: `nltk` `punkt` sentence tokenizer must be available (used by the vt/niah essay branch).
- **Gated-model access.** None. RULER is fully synthetic; no HF approval needed for data.

## 2. LongBench (real-document QA)

- **Official source.** Dataset: https://huggingface.co/datasets/THUDM/LongBench ·
  Repo: https://github.com/THUDM/LongBench · Paper: arXiv:2308.14586 (ACL 2024).
- **What the local loader implements** (`benchmarks/longbench.py`).
  Official data path replicated exactly: downloads the same `data.zip` the official
  `LongBench.py` loading script fetches, caches it at
  `benchmarks/data/longbench_cache/data.zip`, and parses the same per-task jsonl files
  with the same fields (`input`, `context`, `answers`, `length`, `dataset`, `language`,
  `all_classes`, `_id`). Data is byte-identical to what `load_dataset` would return.
  Six tasks: `hotpotqa`, `2wikimqa`, `musique`, `multifieldqa_en`, `narrativeqa`, `qasper`
  (exact official config names). Official prompt templates (`dataset2prompt.json`) and
  per-task max-gen lengths (`dataset2maxlen.json`: 32/32/32/64/128/128) are preserved
  verbatim in each example's `meta`.
- **Exact metric implemented** (`metrics/official_metrics.py::qa_f1_score`):
  verbatim `LongBench/metrics.py::normalize_answer` (lowercase → remove punctuation →
  remove articles → whitespace fix) + Counter-based token F1, MAXed over ground truths
  exactly as `LongBench/eval.py::scorer` does. (`qa_em_score` in the same file is a
  project supplement on the identical normalization; LongBench itself reports F1 only.)
- **Documented deviations:**
  1. **`datasets` 5.x drops script datasets**, so the official `load_dataset("THUDM/LongBench")`
     path no longer works; the manual `data.zip` download replicates the script's fetch
     byte-for-byte (deviation is in tooling, not data).
  2. **Prefix subsetting when `--limit` is set**: first-n of the official test split, no
     shuffling, recorded as `meta["subset"]="first_n"`. **Not active in the final run**
     (`limit: null` → full splits: 200/200/200/150/200/200 = 1150 examples).
  3. **Prompts not fed verbatim** (same block-chunking note as RULER §1.6): official
     templates are kept in `meta` for provenance; systems ingest chunked blocks with their
     own system prompt. This is a harness adaptation, not a data deviation.
  4. **Generation budget differs from official per-task maxima**: final run uses a flat
     `--max-new-tokens 64` vs official 32 (hotpotqa/2wikimqa/musique), 64
     (multifieldqa_en — matches), 128 (narrativeqa/qasper).
- **Dataset fetch commands** (reproducer runs exactly this once):
  ```bash
  cd experiments/benchmarks/data/longbench_cache
  curl -L https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip -o data.zip
  ```
  Verified cache: `data.zip` sha256 `cb45b11a…575857f7f64` (113,932,529 bytes), containing
  35 files; per-task counts inside the zip are exactly the official test sizes
  (hotpotqa 200, 2wikimqa 200, musique 200, multifieldqa_en 150, narrativeqa 200, qasper 200).
- **Gated-model access.** None for data — `THUDM/LongBench` is a public, ungated HF dataset.

## 3. LoCoMo (long-term conversational memory QA)

- **Official source.** Repo: https://github.com/snap-research/locomo ·
  Paper: arXiv:2402.17753 (ACL 2024) · Data:
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
  (10 synthetic multi-session conversations).
- **What the local loader implements** (`benchmarks/locomo.py`).
  Official file read directly (cached at `benchmarks/data/locomo_cache/locomo10.json`).
  10 conversations verified on disk; conversation rendering orders sessions numerically,
  keeps turns in file order as `Speaker: text` lines, drops image-only turns, adds no
  date/time lines (stated in `render_conversation` docstring). Gold handling follows the
  official `task_eval/evaluation.py`: categories 1–4 use file `answer` (category 3
  truncated at the first `;`, exactly as official); category 5 (adversarial, which has
  `adversarial_answer` but no `answer`) records gold `["not mentioned"]` and keeps the
  adversarial statement in `meta["adversarial_answer"]`. QA order = official file order;
  category filter + `n_samples` cap applied deterministically (no shuffling).
- **Exact metric implemented** (`metrics/official_metrics.py::locomo_f1` = `qa_f1_score`,
  LongBench-style token F1, metric id `"locomo_f1"`).
- **DOCUMENTED ADAPTATION — metric (must accompany every LoCoMo table):**
  the official protocol scores open-domain (cat 3) and temporal (cat 2) answers with an
  **LLM judge (GPT-4)** per the paper, and the official repo's `evaluation.py` uses
  Porter-stemmed token F1 (whose normalization also removes `and` and commas) for
  categories 1–4 plus a **binary adversarial rule** for category 5 (score 1 iff the
  prediction contains "no information available" / "not mentioned"). No judge API is
  available in this environment, so ALL categories are scored with plain token F1.
  Consequences: (a) cat-2/3 scores are not comparable to published LoCoMo numbers;
  (b) cat-5 "not mentioned"-style predictions get partial instead of binary credit
  (e.g. `locomo_f1("not mentioned in the conversation", ["not mentioned"]) > 0`
  but < 1); (c) stemming is NOT applied (`running` vs `runs` → 0). Every LoCoMo result
  must be labeled **"LoCoMo (token-F1 adaptation)"**. The adaptation is also recorded
  per-example in `meta["metric_adaptation"]`.
- **Dataset fetch commands** (reproducer runs exactly this once):
  ```bash
  mkdir -p experiments/benchmarks/data/locomo_cache
  curl -L https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
    -o experiments/benchmarks/data/locomo_cache/locomo10.json
  ```
  Verified cache: sha256 `79fa87e9…51ea698ff4` (2,805,274 bytes).
- **Gated-model access.** None — raw GitHub file, no approval needed.

## 4. Model access (all final runs)

Single model everywhere: `Qwen/Qwen2.5-3B-Instruct`, `dtype: nf4`, greedy decoding,
`attn_implementation: eager` (per `metrics.json: engine`). **No HF access approval
required**: Qwen2.5-Instruct weights are public/ungated; the suite preflight
(`run_final_suite.sh`) pulls them via anonymous `snapshot_download`. The only
credentials-gated component in the whole design space is the *absent* LoCoMo LLM judge
(GPT-4 API key) — which is exactly why the token-F1 adaptation exists. Dense-RAG
embedder `sentence-transformers/all-MiniLM-L6-v2` (see `sempointer/pipeline.py`) is
likewise public/ungated.

## 5. LongMemEval existence check — VERDICT: real, suitable only as future work

- **Real? Yes.** LongMemEval (Wu et al., ICLR 2025; arXiv:2410.10813):
  repo https://github.com/xiaowu0162/LongMemEval (~1.1k stars),
  cleaned data on Hugging Face `xiaowu0162/longmemeval-cleaned`, also released via
  Google Drive. The repo README was fetched and its instructions verified 2026-09-29.
- **URL / fetch:**
  ```bash
  mkdir -p data && cd data
  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
  ```
  (public URLs per the official README; no approval step documented).
- **Task shape.** 500 curated questions testing five memory abilities (information
  extraction, multi-session reasoning, temporal reasoning, knowledge updates,
  abstention) over scalable user–assistant chat histories. Two standard settings:
  **S** ≈ 115k tokens/question (~40 sessions; needs a ~128k-context reader) and
  **M** ≈ 500 sessions (~1.5M tokens). Instances carry evidence-session labels, so
  retrieval Recall@k/NDCG is computable; QA correctness is graded by a **GPT-4o LLM
  judge** (`src/evaluation/evaluate_qa.py`, requires `OPENAI_API_KEY`).
- **Suitability verdict — future work, not a drop-in addition.**
  (+) Topically ideal: conversational memory like LoCoMo, plus knowledge-update and
  abstention coverage LoCoMo lacks; oracle-retrieval split gives a retrieval-free
  evaluation mode.
  (−) Blockers for this rig: (1) QA eval requires paid OpenAI API (same judge-gap that
  forced the LoCoMo adaptation); (2) S histories (~115k tokens) far exceed the 8 GB
  rig's proven full-context envelope (~14.5k tokens mean in `final_locomo`), so only
  the oracle or retrieval-augmented settings are runnable — full-history S/M is not;
  (3) no local loader exists yet (`benchmarks/` has no `longmemeval.py`). Recommend
  citing as future work with the oracle setting as the concrete next step.

## 6. Traceability index (local paths)

| Artifact | Path |
|---|---|
| Runner (config, scoring dispatch, bootstrap CIs) | `experiments/run_benchmark.py` (`score_example`, `bootstrap_ci`, `build_examples`) |
| RULER loader + deviations 1–4 | `experiments/benchmarks/ruler.py` (module docstring, `TASK_ARGS`, `_OFFICIAL_TASKS`) |
| LongBench loader + subset policy | `experiments/benchmarks/longbench.py` (module docstring, `OFFICIAL_PROMPT_TEMPLATES`, `OFFICIAL_MAX_GEN_LEN`) |
| LoCoMo loader + metric adaptation | `experiments/benchmarks/locomo.py` (module docstring, `render_conversation`, `build_locomo`) |
| Metric implementations + self-test | `experiments/metrics/official_metrics.py` (`qa_f1_score`, `ruler_containment`, `locomo_f1`, `_self_test`) |
| Uniform example schema | `experiments/benchmarks/common.py` (`BenchExample`, `SUPPORTED_METRICS`) |
| Final-run invocation | `experiments/run_final_suite.sh` |
| Per-run configs transcribed in `notes/datasets.md` | `experiments/results/final_*/config.json` |
