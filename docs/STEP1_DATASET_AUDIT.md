# Step 1 — Paper, datasets, and laptop feasibility

Checked on 2026-11-08. Scope: understand the reference, identify its exact evaluation data, locate official downloads, and assess local feasibility. No architecture changes, model loading, or experiments were performed. This is a broad, task-relevant inventory, not a claim to enumerate every long-context dataset in existence.

Evidence labels: **[ESTABLISHED]** identifies mechanisms documented in primary papers; **[REASONED]** identifies recommendations or deductions. **[LOCAL]** identifies direct inspection of this workspace or hardware; **[SOURCE]** identifies information checked in an official repository or dataset release. Download pages were inspected; only the explicitly listed direct endpoints were HTTP-checked. Remote file contents were not downloaded and compared with the caches.

## 1. Reference paper and architectural target

**[LOCAL]** The canonical reference is `main.pdf`: **SemPointer: Adaptive Lexical Routing for Token-Efficient Long-Context Question Answering**, sections 3–4. SHA-256: `8a41ed056d5e4498c34436b240e1d668a4b107889f4e2060788ed702a3ff3e24`. It is a local manuscript; this audit does not establish an external publication venue or public paper URL.

**[LOCAL]** Its measured path is:

1. Divide history into blocks of at most **512 target-model tokens**.
2. Store exact text under stable block IDs, with readable pointers capped at **16 tokens** each.
3. Rank blocks with **BM25**, protecting the top **two**.
4. Expand the requested set to **four** if weighted query coverage is below **0.80**, or if coverage is below **0.95** and the third BM25 score is at least **0.85** times the second.
5. Resolve selected IDs to exact text under a **2,560-token payload cap**. Retain unresolved pointers in the mixed prompt.
6. Generate with **Qwen2.5-3B-Instruct, NF4**. Semantic rescue is disabled in the reported configuration.

**[LOCAL]** Relevant implementation: `experiments/sempointer/v2_pipeline.py`; paper reproduction settings: `experiments/run_v2_suite.sh`. These expose the above parameters. This preliminary reading identifies the target for a later architecture audit; it does not establish full implementation correctness. The older `sempointer/pipeline.py` is also present and has different selection behavior.

**[REASONED]** The first experiment package needs evaluation datasets and pretrained model weights. The paper does not require pretraining a transformer or fitting a new neural pointer encoder. Changing the base model or quantization would produce an extension rather than an exact reproduction.

## 2. Laptop resources actually detected

**[LOCAL]** Resource snapshot from `free -h`, `nvidia-smi`, and `df -h .`:

| Resource | Observed |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU VRAM | 8,151 MiB total; 5,789 MiB free at a later check |
| System RAM | Approximately 30 GiB total; approximately 12 GiB available at the initial check |
| Swap | Approximately 30 GiB total, 10 GiB used |
| Workspace filesystem | Approximately 71 GiB free |

**[REASONED]** The paper's 3B NF4 inference configuration is an appropriate starting point for this hardware class. Runtime compatibility and peak memory still need your smoke run. Free memory changes with desktop applications and other processes. The existing runner requires 5,000 MiB free VRAM for normal stages and 7,000 MiB for its holdout stage; the latter threshold was not met in this snapshot.

**[REASONED]** Process one example at a time and load dataset files incrementally where possible. Keep the text registry and lexical index in system RAM. NF4 reduces model-weight memory; it does not by itself quantize the KV cache or eliminate activation memory.

**[SOURCE]** The released model configuration has `max_position_embeddings=32768`. Therefore, 100K–1M source histories cannot be sent unmodified through this base model under its default configuration. Retrieval can select a smaller prompt, but the resulting experiment must be described as retrieval over a long history. [Official model configuration](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json).

## 3. Exact benchmark collections used by the paper

| Collection | What it tests | Exact paper subset [LOCAL] | Official download / source [SOURCE] |
| --- | --- | --- | --- |
| **RULER** | Controlled retrieval, variable tracking, and aggregation | Six tasks × 10 examples × 3 lengths = **180** examples, seed 42; lengths **4,096, 8,192, 16,384** | [Official repository and generation instructions](https://github.com/NVIDIA/RULER); [source ZIP](https://github.com/NVIDIA/RULER/archive/refs/heads/main.zip) |
| **LongBench v1** | Natural single-document and multi-document QA | First **100** items from each of six tasks = **600** examples | [Official dataset](https://huggingface.co/datasets/zai-org/LongBench); [direct data.zip](https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip); [official v1 instructions](https://github.com/THUDM/LongBench/tree/main/LongBench) |
| **LoCoMo** | Memory QA across conversation sessions | Conversations **0, 1, 2**, containing **497** QA items | [Official repository](https://github.com/snap-research/locomo); [direct locomo10.json](https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json) |

**[SOURCE]** The former LongBench location `THUDM/LongBench` currently redirects to `zai-org/LongBench`. The core ZIP endpoint returned HTTP 200 and advertised **113,932,529 bytes**, approximately 114 MB decimal. The LoCoMo JSON endpoint returned HTTP 200 and advertised **2,805,274 bytes**, approximately 2.8 MB decimal.

**[ESTABLISHED]** RULER is configurable synthetic evaluation, rather than one fixed universal download; its published evaluation contains 13 representative tasks. Its official generator depends on the tokenizer, task, length, template, and seed. Hsieh, Sun, Kriman, Acharya, Rekesh, Jia, Zhang, and Ginsburg, **RULER: What's the Real Context Size of Your Long-Context Language Models?**, COLM 2024. [Primary paper](https://arxiv.org/abs/2404.06654).

**[LOCAL]** The six paper tasks are `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `vt`, and `cwe`. The local generator documents deviations: a five-essay prefix instead of the full essay corpus, cached word-list assets, approximate token counts when no tokenizer is supplied, and omitted upstream bookkeeping fields. Do not assume byte-equivalence to a newly generated upstream RULER set. Reproduction and upstream-protocol evaluation must be recorded separately.

**[ESTABLISHED]** LongBench v1 comprises 21 datasets across six task categories. Bai, Lv, Zhang, Lyu, Tang, Huang, Du, Liu, Zeng, Hou, Dong, Tang, and Li, **LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding**, ACL 2024. [Primary paper](https://arxiv.org/abs/2308.14508).

**[ESTABLISHED]** LoCoMo evaluates very long-term conversation memory. Maharana, Lee, Tulyakov, Bansal, Barbieri, and Fang, **Evaluating Very Long-Term Conversational Memory of LLM Agents**, ACL 2024. [Primary paper](https://arxiv.org/abs/2402.17753).

## 4. All six LongBench QA datasets used in this paper

**[LOCAL]** These files are already included inside the single LongBench ZIP. The available counts below were read from the local archive; they describe the LongBench versions, not the complete original datasets.

| Dataset / configuration | Why it matters [REASONED] | Available locally / paper uses | Original release or download portal [SOURCE] |
| --- | --- | --- | --- |
| **HotpotQA** / `hotpotqa` | Multi-hop evidence retrieval across documents | 200 / 100 | [Official downloads](https://hotpotqa.github.io/) |
| **2WikiMultihopQA** / `2wikimqa` | Compositional multi-hop reasoning | 200 / 100 | [Official repository and data links](https://github.com/Alab-NII/2wikimultihop) |
| **MuSiQue** / `musique` | Tests whether routing retrieves several necessary facts | 200 / 100 | [Official repository and data links](https://github.com/StonyBrookNLP/musique) |
| **MultiFieldQA-English** / `multifieldqa_en` | QA across diverse document domains | 150 / 100 | [LongBench data.zip](https://huggingface.co/datasets/zai-org/LongBench/resolve/main/data.zip) |
| **NarrativeQA** / `narrativeqa` | QA over long stories | 200 / 100 | [Original dataset and story retrieval instructions](https://github.com/google-deepmind/narrativeqa) |
| **QASPER** / `qasper` | Scientific-document QA, including unanswerable questions | 200 / 100 | [Official Ai2 dataset](https://huggingface.co/datasets/allenai/qasper) |

**[REASONED]** Use the LongBench versions first: replacing them with the full original datasets changes context construction, IDs, splits, and evaluation. You do not need six additional original-corpus downloads to reproduce this paper.

**[LOCAL]** For completeness, the cached LongBench v1 archive contains these 21 base task files:

| Category | Configuration names |
| --- | --- |
| Single-document QA | `narrativeqa`, `qasper`, `multifieldqa_en`, `multifieldqa_zh` |
| Multi-document QA | `hotpotqa`, `2wikimqa`, `musique`, `dureader` |
| Summarization | `gov_report`, `qmsum`, `multi_news`, `vcsum` |
| Few-shot tasks | `trec`, `triviaqa`, `samsum`, `lsht` |
| Synthetic tasks | `passage_count`, `passage_retrieval_en`, `passage_retrieval_zh` |
| Code completion | `lcc`, `repobench-p` |

**[LOCAL]** The existing loader supports the six QA tasks only. Other tasks require task-specific prompts, metrics, and adapters. They are available data, not currently integrated experiments.

## 5. Related benchmark extensions and download links

**[REASONED]** Priorities below reflect relevance to this architecture and laptop, not measured performance or a quantitative ranking of benchmark popularity. These are additional evaluation options, not datasets used in the reference paper.

| Benchmark | Relevance | Official data download | Laptop assessment / integration status |
| --- | --- | --- | --- |
| **LongMemEval** | Strong next test for persistent conversational memory, updates, temporal reasoning, and abstention | [Dataset files](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/main); [S direct JSON](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json); [oracle direct JSON](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json); [official protocol](https://github.com/xiaowu0162/LongMemEval) | Prioritize S with bounded retrieved prompts. The S endpoint returned HTTP 200, approximately 277 MB. Oracle is a diagnostic with privileged evidence, not a replacement for full-history retrieval. New adapter required. |
| **BABILong** | Multi-fact reasoning hidden in distractor text; useful against lexical shortcuts | [Official 100-example-per-task/length release](https://huggingface.co/datasets/RMT-team/babilong); [official repository](https://github.com/booydar/babilong) | Start with 4K/8K/16K configurations. Download selected lengths rather than the whole release. New adapter required. |
| **LongBench v2** | More demanding long-context QA and reasoning | [Official dataset](https://huggingface.co/datasets/zai-org/LongBench-v2); [direct data.json](https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json); [official repository](https://github.com/THUDM/LongBench) | Approximately 465 MB per official file listing. Use retrieval with a bounded prompt; inspect source lengths. New adapter and multiple-choice evaluation required. |
| **InfiniteBench / ∞Bench** | Retrieval and QA beyond 100K tokens | [Official files](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/tree/main); [official repository](https://github.com/OpenBMB/InfiniteBench) | Later stress test. Official listing is approximately 2.5 GB; select files. Full-context evaluation exceeds this base model's default window. New adapter required. |
| **L-Eval** | Diverse long-document tasks, including closed-ended QA | [Official dataset files](https://huggingface.co/datasets/L4NLP/LEval/tree/main); [official repository](https://github.com/OpenLMLab/LEval) | Start with selected closed-ended tasks. Scoring varies across tasks. New adapters required. |
| **LooGLE** | Short- and long-dependency QA and summarization | [Official dataset files](https://huggingface.co/datasets/bigai-nlco/LooGLE/tree/main); [official repository](https://github.com/bigai-nlco/LooGLE) | QA is the closest fit; source-length checks and new adapters required. |
| **SCROLLS** | Long-document QA, summarization, and inference | [Official dataset files](https://huggingface.co/datasets/tau/scrolls/tree/main) | Select QA configurations; account for overlap with NarrativeQA and QASPER. New adapters and task-specific metrics required. |
| **Needle in a Haystack** | Controlled retrieval by context length and evidence position | [Official repository, corpus, and generator](https://github.com/gkamradt/needle-in-a-haystack) | Useful lightweight diagnostic at 4K/8K/16K. A generator/corpus download, not a fixed universal test file. RULER already offers richer retrieval tests. |
| **BEIR** | Separates retrieval quality from generated answer quality | [Official benchmark and per-dataset download table](https://github.com/beir-cellar/beir#beers-available-datasets) | Optional retrieval-only extension. Begin with a small corpus; full-corpus indexing costs differ from per-history routing. Requires a separate evaluation adapter. |

**[REASONED]** Suggested order after paper reproduction: held-out RULER seeds and unused LoCoMo conversations; then LongMemEval-S and short BABILong; then selected LongBench v2 tasks; finally longer stress tests. Additional datasets do not replace the paper's missing pointer-off and matched-token controls.

**[REASONED]** Preserve the complete source history when evaluating retrieval over long inputs. Truncating it changes the problem and can discard gold evidence. Also account for unresolved pointers: 100K source tokens produce about 196 blocks, whose 16-token pointers alone could occupy about 3,136 tokens; 1M tokens could produce about 31,264 pointer tokens, before selected text, labels, question, and instructions. Thus the 2,560-token resolved-text cap is not a cap on the complete prompt.

## 6. Existing local data and verification

**[LOCAL]** No new dataset download is necessary to begin reproducing the three paper collections:

| Existing path, relative to project root | Verified state |
| --- | --- |
| `experiments/benchmarks/data/longbench_cache/data.zip` | ZIP CRC check passed; all six QA files parse and contain `input`, `context`, and `answers` fields |
| `experiments/benchmarks/data/locomo_cache/locomo10.json` | JSON parses; **10 conversations**, **1,986 QA items**, **497** in conversations 0–2 |
| `experiments/benchmarks/data/ruler_cache/` | Approximately **8.4 MiB** of generator assets; five essays, word lists, and English-word assets present; no new examples generated |

**[LOCAL]** Cache directories occupy approximately 459 MiB for LongBench, 2.7 MiB for LoCoMo, and 8.4 MiB for RULER. The LongBench ZIP itself is approximately 109 MiB; directory usage includes other local cache contents.

| File | Local SHA-256 |
| --- | --- |
| LongBench ZIP | `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64` |
| LoCoMo JSON | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` |

**[LOCAL]** The local LongBench loader contains an incorrect paper identifier, `2308.14586`. The verified LongBench identifier is **2308.14508**. This report records the issue; implementation files were left unchanged during step 1.

## 7. Protocol details to carry into your later runs

- **[LOCAL]** The paper caps generation at 32 tokens for RULER and 64 for LongBench/LoCoMo. The local LongBench table records official caps of 128 for NarrativeQA and QASPER. Reproduction settings and official-protocol extensions should have different run IDs.
- **[LOCAL]** LoCoMo currently uses adapted normalized token-F1 rather than its official category-specific evaluation. Its local rendering omits session date/time headers; this can affect temporal QA. Preserve the rendering for reproduction and evaluate any correction separately.
- **[SOURCE]** LongMemEval includes a model-based evaluator in its official protocol. A local judge or lexical substitute changes that protocol; report it explicitly. [Official evaluation instructions](https://github.com/xiaowu0162/LongMemEval).
- **[REASONED]** Fix thresholds before held-out evaluation, retain example IDs and dataset hashes, and apply identical model/generation settings to compared systems. Measure indexing separately from per-answer latency and include its amortized cost for repeated queries.
- **[LOCAL]** No benchmarks were run, no model was loaded, no paid service was invoked, and no existing data, results, paper, or architecture files were modified. Only this report was added.

Step 1 deliverable: verified paper dataset inventory, official download locations, local cache checks, and hardware-aware evaluation recommendations. Architecture remodeling and empirical verification remain subsequent work.
