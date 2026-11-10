# Qwen3.5-4B evaluation configuration

The requested next model is `Qwen/Qwen3.5-4B`, with thinking explicitly disabled.
Use the new launcher below. Historical Qwen2.5 scripts and result directories are
retained for reference; they are not overwritten or resumed with the new model.

```bash
cd /home/viksit/Projects/tokens2/experiments
bash run_qwen35_suite.sh model     # explicit checkpoint download; no inference
bash run_qwen35_suite.sh check
bash run_qwen35_suite.sh smoke     # 12 real generations, two 4K RULER examples
bash run_qwen35_suite.sh dry-run   # inspect reviewer scope without creating results
bash run_qwen35_suite.sh run --hours 8
```

The default precision remains NF4 for the 8 GB laptop. BF16 is selectable for a
roomier GPU, for example:

```bash
SEMPTR_QWEN35_DTYPE=bf16 bash run_qwen35_suite.sh smoke
```

Compare measured time and memory before choosing precision. NF4 stores linear
weights at four bits and computes at higher precision; it reduces memory, but is
not guaranteed to improve inference speed. BF16 requires substantially more
weight memory. Precision changes are separate experiments, with separate outputs.
No speed or accuracy improvement is assumed in advance.

Default output roots:

- `results/reviewer_qwen35_4b_nf4_qa128_cwe512/`
- `results/reviewer_qwen35_4b_bf16_qa128_cwe512/`

`SEMPTR_QWEN35_OUTPUT` or `--output` can select another **new** directory. The
eight-hour allowance is a runtime cutoff, not a dollar limit. This script does
not create a Modal deployment, configure billing, or start paid cloud resources.

## Output caps and comparisons

QA and non-CWE RULER allow 128 output tokens, versus 64/32 in the paper. CWE
allows 512 tokens. The paired CWE audit retains 32 tokens on the same examples.
Source-context guards, 512-token blocks, two-to-four-block selection, and the
2,560-token resolved-evidence cap are unchanged. Summarization and other tasks
retain their own task caps in the broad runner. All arms share model, precision,
chat formatting and output allowance within each paired comparison.

The implemented reviewer suite has six distinct systems:

1. SemPointer.
2. Pointer-off, with identical selected evidence IDs (pointer ablation).
3. BM25-2 (fixed smaller evidence budget).
4. BM25-4 (fixed larger evidence budget).
5. BM25 under SemPointer's total prompt-token upper allowance.
6. Full context, on RULER when it fits.

Natural QA runs use the first five. Development additionally compares three
threshold configurations (loose, paper, cautious); these are configurations,
not three additional systems. The output-cap audit is another experiment, not
a seventh system. The general downloaded-data runner also offers no-context,
bringing its selectable system count to seven. No standalone original reviewer
letter was available in the inspected project; these counts describe the
implemented reviewer protocol, not a verified exhaustive reviewer checklist.

## Compatibility and provenance

Qwen3.5 uses its native conditional-generation loader and native hybrid cache.
The legacy custom chunked generation loop is bypassed. The native non-thinking
chat template is rendered before every control's token budget is counted.
This includes pointer-off and matched-budget BM25, so chat framing cannot be
silently omitted from their comparison. The model is frozen; no training occurs.

The pre-change code snapshot is
`reference_code/before_qwen35_20261109T180506Z/`. Its manifest records source
hashes and the original result-file inventory. Restoring the snapshotted source
files is necessary to reproduce the old code fingerprint when resuming historical
runs. Keep existing raw predictions, configs, model weights and datasets at their
original paths; do not mix model generations in one score table without labels.
The new held-out suite excludes previously exposed document/conversation groups.
Consequently its aggregate scores are not a paired model-only comparison with
the old paper subset. A separate cross-model comparison must use identical
examples and generation settings, clearly distinguished from held-out testing.

Software checks use tiny randomly initialized weights only as a compatibility
fixture. They are not benchmark evidence. Real-model smoke results, if produced,
belong in the new output directory and do not establish full-suite completion.

## Verification on 2026-11-09

The official checkpoint was downloaded into `../models/Qwen3.5-4B/`. A real
NF4 smoke completed all 12 generations (six systems on one 4K single-needle
example and one 4K CWE example). Output caps were 128 and 512 respectively;
all answers stopped at EOS, using 10 and 45 generated tokens. The measured
peak allocated VRAM was approximately 4,178 MiB, and summed generation time was
43.04 seconds, excluding setup, model loading, and indexing.

The single-needle item scored 1.0 for every arm; CWE scored 0.0 for every arm.
These two examples verify execution only. They do not establish benchmark
accuracy, a benefit from the higher cap, or a speed comparison against BF16.
Optimized DeltaNet/convolution kernels were unavailable in this local environment,
so Transformers used its PyTorch fallback. Profile an appropriately configured
Modal environment before estimating cloud throughput or spending.

The old selector tests additionally require uncached Qwen2.5-1.5B and embedding
assets in offline mode; their missing-cache failures were not repaired as part
of the model switch. No historical result file was modified.
