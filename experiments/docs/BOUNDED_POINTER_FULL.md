# Bounded pointers: 100 questions per family

The user authorized expanding the completed bounded-pointer pilot to the ten agreed benchmark families with 100 samples from each. This is a new local evaluation, run `c73abaf595dc68e55696`, in `results/bounded_pointer_full_v1`.

There are exactly 1,000 fresh selected question IDs: 100 each from RULER, LongBench, LoCoMo, LongMemEval, BABILong, LongBench v2, InfiniteBench, LooGLE, L-Eval and SCROLLS. The questions are a deterministic stratified selection from the remaining frozen 300-per-family pool. Previously evaluated local/pilot questions, development questions and Modal cost-pilot IDs are excluded. Pointer-development source groups are also excluded; LoCoMo had 108 eligible fresh questions after this restriction, allowing the requested 100 without relaxing it.

New question IDs do not establish unseen-document generalization. In particular, the manifest records older test-source group overlap for LoCoMo (100 questions), BABILong (81), L-Eval (27), InfiniteBench (22), SCROLLS (18), LooGLE (15), LongBench v2 (7) and LongBench (1). Document/conversation clustering limits example-bootstrap inference. This evaluation does not become an official leaderboard submission by covering more families.

## Fixed method and stages

- Pinned Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, NF4, batch one, local RTX 5060 Laptop GPU.
- Input budgets: 1,024, 2,048 and 4,096 tokens, including question, instruction, pointer labels, retrieved payload and native chat framing.
- Same four systems: bounded pointers, adaptive retrieval with no index, BM25 filling the external token budget, and the original SemPointer prompt where it fits.
- Pointer allowance remains 128 tokens for every budget, inherited exactly from the completed pilot's development choice. A checksummed copy of that source selection is frozen with the new inputs. Routing thresholds and output caps are unchanged.
- No new development calls or parameter selection. All three input budgets run, rather than choosing one based on pilot test outcomes.
- The same 1,000 questions are evaluated at each budget, scheduling 12,000 system–question–budget entries. These are not 3,000 independent questions. Explicit skips reduce actual inference calls.
- Original oversized prompts and questions whose framing alone exceeds the input budget are recorded as skips, not truncated questions or invented zero scores. The last retrieved evidence block may be partially included, as in the pilot.

Preparation verifies that the bounded prompt builder, core runner, controls, model engine and pointer pipeline match the frozen pilot versions. Only full-set sampling, test-only orchestration and report descriptions change. The complete new runtime is frozen under `input/code`; later working-tree edits cannot change the active process. Model weights are verified against the retained hashes before inference.

## Run controls

From `experiments/`:

```bash
bash run_bounded_full.sh status
tail -f results/bounded_pointer_full_v1/worker.log
bash run_bounded_full.sh stop
bash run_bounded_full.sh start
```

The process runs independently of its launching terminal. Keep the laptop powered on, plugged in and awake. A stop request takes effect at a safe checkpoint, and start resumes committed progress. A file lock prevents duplicate workers for this output root. The original pilot launcher continues to refer to its completed pilot, and the original local/Modal runs are not repointed or overwritten.

There are no Modal calls, cloud GPUs, model downloads or training in this run. The saved prior Modal experiments remain separate.

## Evidence and analysis

The three stages are `test_1024`, `test_2048` and `test_4096`, each with 4,000 scheduled records. SQLite attempts, raw events and question checkpoints preserve progress. Stage exports and final reports are generated automatically when execution completes or pauses. The report includes metric-specific paired differences/confidence intervals, descriptive question win/tie/loss counts, token and generation-time differences on shared scored cases, and coverage.

Input-budget equality means equal upper limits, not equal actual prompt length. Generation timing includes prefill and decoding; ingestion and shared prompt preparation are separate. Larger savings against budget-filling BM25 do not by themselves establish pointer-specific value, end-to-end serving speedups, quality equivalence or publication novelty. Report all three budgets and negative/null findings without tuning on this test set.

## Verification

The two new tests failed before implementation and now pass: deterministic balanced full-set selection/exclusions, and real test-only stage execution that reuses frozen choices and resumes without extra model calls. The broader relevant evaluation selection passed all 86 tests (one existing deprecation warning). The full offline suite returned 119 passed, 7 failed, 21 errors and 7 deselected; missing legacy model/tokenizer caches and the existing paper-assets assertion remain. Full traces and names are in `results/bounded_pointer_full_v1/full_tests.log`.

The frozen 1,000-question bundle verified successfully, with no overlap in excluded question IDs or pointer-development groups. All 135 hashed prior result/provenance files remained unchanged. Shell syntax and whitespace checks passed. A real local CUDA worker generated and saved test scores at the 1K budget, rather than merely being submitted. Implementation and launch are complete; the background evaluation itself is still in progress at this documentation point.

Final review was an author self-review because sub-agents are prohibited in this side conversation. It checked frozen settings, default pilot behavior, report counts, data/group isolation, exact family counts and resume behavior; it is not an independent review.
