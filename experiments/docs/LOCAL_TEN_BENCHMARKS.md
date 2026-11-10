# Local GPU evaluation: 100 samples per family

The user's replacement plan runs 100 questions from each of the ten agreed benchmark families on the RTX 5060 Laptop GPU. No Modal execution or API calls are used by the local runner. The previous Modal runs and their evidence remain separate.

The frozen local run is `2146b7900c244e92e6f3`, stored in `results/local_ten_qwen35_nf4_100_v1`. Its `input/manifest.json` contains exactly 100 selected questions per family, 1,000 total: RULER, LongBench, LoCoMo, LongMemEval, BABILong, LongBench v2, InfiniteBench, LooGLE, L-Eval and SCROLLS. Selection is deterministic and balanced over the existing strata within each family. It is a subset of the frozen 300-per-family samples; no scores, answers or pilot runtimes influence selection. All context hashes, exposure exclusions and source references are retained.

The six systems remain SemPointer, pointer-off, BM25-2, BM25-4, matched-budget BM25 and full context where it fits. This schedules up to 6,000 main evaluations. An independent 36-question development set is rerun for all three routing configurations under NF4 (216 additional evaluations), and 18 selected CWE questions have the paired 32-token audit (108 additional evaluations). There is no financial pilot on the local GPU.

The model remains the pinned Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. NF4, batch one and an 8,192-token prompt guard fit the local 8 GiB GPU. Existing task-specific output caps are unchanged, including the 512-token CWE cap. Sources retain their existing size guards and are not truncated. A full-context prompt beyond the local prompt guard is explicitly skipped; selected sample counts are not claims of complete scored coverage in every system. Results must not be silently pooled with the previous BF16 run or compared as though precision and prompt guards were identical.

All inference uses the local CUDA device; there is no CPU or cloud inference fallback. CPU preprocessing uses eight PyTorch threads and eight tokenizer worker threads. `fla-core==0.6.0` and `einops==0.8.2` are installed in `.venv-local-suite`; a local `.pth` file reuses the existing `warden` PyTorch CUDA 13 / Transformers runtime without duplicating its large packages or modifying that environment. Fast DeltaNet and recurrent kernels are available. The optional causal-convolution extension is absent; its convolution uses the existing PyTorch implementation. The global upstream warning about an incomplete fast path does not mean the installed DeltaNet kernels are disabled.

## Run controls

From `experiments/`:

```bash
bash run_local_suite.sh status
tail -f results/local_ten_qwen35_nf4_100_v1/worker.log
bash run_local_suite.sh stop
bash run_local_suite.sh start
```

`start` resumes the same frozen run, skips recorded finished evaluations and refuses changed sample/model/prompt settings. `stop` takes effect at a safe checkpoint. One file lock protects the whole run from concurrent local workers using the same output directory. Run control does not import or execute the Modal SDK. The frozen code is launched from `input/code` to isolate this long job from later working-tree edits.

The process runs in the background and survives closing its launching terminal. Keep the laptop on, plugged in and awake: local computation pauses during sleep and stops at shutdown. A later `start` resumes saved progress. No system power, clock or sleep settings were changed.

## Saved evidence

Each stage retains SQLite predictions/attempts and append-only raw events. Progress is saved after each question in `state.json`; completed system evaluations are committed as they are recorded. The final or paused report is written automatically in `results/REPORT.md`, with per-stage metrics and predictions, threshold selection and the CWE cap audit. No mixed-metric aggregate is invented. `worker.json` and `worker.log` record the local process and startup details. Original model files, dataset files, frozen Modal bundles and historical results are preserved.

## Validation

Three new CPU tests cover deterministic family-wide quotas/category balance, resuming past the omitted financial-pilot phase and retaining a stopped checkpoint. The tiny CPU model test explicitly disables optional CUDA kernels for its CPU-only fixture so it also works in the new GPU-enabled environment.

All 78 relevant evaluation tests passed in the local runtime (one existing deprecation warning). The selected input bundle and all 1,036 example files verified successfully, including exactly 100 selected main examples per family and unchanged model-weight hashes. Shell syntax and whitespace checks passed. A live process on the RTX 5060 produced and saved development scores, with approximately 5.2 GiB total GPU memory use and 100% GPU utilization at the observation point. These measurements establish that the job started; they are not a claim that the entire evaluation completed.

The full offline non-GPU test run returned 111 passed, 7 failed, 21 errors and 7 deselected. Full traces are recorded in `results/local_ten_qwen35_nf4_100_v1/full_tests.log`. Existing legacy model/tokenizer cache failures and the paper-assets assertion are tracked there by name; the local evaluation's focused green checks do not imply that the whole repository is green.
