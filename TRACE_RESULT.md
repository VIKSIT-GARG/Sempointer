# TRACE RESULT — MiniLM `(... > 256)` warning hunt

**Guilty site:** `experiments/sempointer/pointer_generator.py:45`
```python
ids = tok.encode(text, add_special_tokens=False)   # inside embed_long()
```
Full guilty stack (representative hit #22, `dense_rag.ingest` path):
```
trace_warning.py:196
baselines/dense_rag.py:72 in ingest
sempointer/pointer_generator.py:45 in embed_long
transformers/tokenization_utils_base.py:2222 in encode
```
Same line is reached via `sempointer/pipeline.py:96 ingest → registry.py:34/36 register →
pointer_generator.py:191/213 generate/generate_token_ids → :179 _embed → :45 embed_long`
(hits #9–#20), `baselines/hybrid.py:66` (hits #26–#29) and
`baselines/iterative_rag.py:57` (hits #30–#33).

**Mechanism.**
1. Blocks are 512 *LLM*-token chunks. Re-tokenized with MiniLM's WordPiece
   tokenizer they measure ~450–570 tokens (tracer: 561/565/562/448 for 512-Qwen-token
   blocks; production logs: 478/508/511). So nearly every full block is over-length.
2. `embed_long()` first *measures* the block with a bare `tok.encode(...)` (no
   truncation) to decide whether to window. That bare call hits
   `PreTrainedTokenizerBase._eventual_warn_about_too_long_sequence`, which logs
   `Token indices sequence length is longer than the specified maximum sequence
   length for this model (... > 256)`.
3. The warn is **measurement-only noise**: `tok.encode` does not truncate, so the
   returned ids are complete and the subsequent windowing math is exact; the real
   `SentenceTransformer.encode` calls only ever receive ≤256-token windows or short
   queries (tracer ST-ENCODE spy: **0 over-length model inputs** across all paths,
   incl. iterative round-2 expanded query at 236 MiniLM tokens). Embeddings — and
   therefore scores — are unaffected.
4. Stock transformers warns **once per tokenizer instance**
   (`deprecation_warnings` flag), which is why 1150-example runs show ~2 warnings.
   Count audit matches exactly: ruler/locomo/longbench (`sempointer` + `rag` each own
   one MiniLM instance) → 2 warnings per process; kvcompare (only `sempointer` owns
   MiniLM) → 1 warning.

**Why SemPointer "seemingly" doesn't trigger it: it does — it triggers it FIRST.**
`run_benchmark.py` runs one system over all examples before the next, in CLI order
with `sempointer` first, so SemPointer's tokenizer instance always wins the
once-per-instance race. Tracer hits #9–#20 prove `sempointer.ingest` fires the hook
before `dense_rag.ingest` (hits #22+) ever runs. The opposite impression comes from
stderr/stdout interleave in the tee'd suite logs (unbuffered warning vs
block-buffered score lines), which makes the warning appear next to unrelated
(`full`/`rag`) progress lines, e.g. `071516.log:1386`, `083935.log:2729`.
Secondary same-issue line: `sempointer/registry.py:43` (`len(tokenizer.tokenize(text))`,
also found via `:72` pattern) — same instance, currently shadowed by line 45 firing first.

**Fix proposal (NOT applied — production files untouched):**
```diff
--- a/experiments/sempointer/pointer_generator.py
+++ b/experiments/sempointer/pointer_generator.py
@@ -42,7 +42,7 @@
     out = []
     for text in items:
-        ids = tok.encode(text, add_special_tokens=False)
+        ids = tok.encode(text, add_special_tokens=False, verbose=False)
--- a/experiments/sempointer/registry.py
+++ b/experiments/sempointer/registry.py
@@ register() and register_batch()
-            token_count=len(generator.tokenizer.tokenize(text)),
+            token_count=len(generator.tokenizer.encode(text, add_special_tokens=False, verbose=False)),
```
Rationale: `verbose=False` is plumbed to the warn hook; verified on a fresh offline
tokenizer instance — silent, and ids byte-identical to the verbose call; also verified
`len(tokenize(t)) == len(encode(t, no specials))`. Both lines must change together:
fixing only line 45 would promote `registry.py:43` to first-toucher on SemPointer's
instance and the warning would simply move there. (Half of the diff is the actual fix;
the registry line is its shadow.)

**Scores/paper numbers affected: NONE.** The warned call returns complete ids
(warning only, no truncation); windowing and all downstream embeddings are bit-exact
either way. Pure log-hygiene fix. Repro: `HF_HUB_OFFLINE=1 ./warden/bin/python -u
experiments/trace_warning.py` (CPU-only, cached MiniLM + Qwen tokenizer only, no LLM
weights; 35 hook hits with the spy vs a handful in stock logs due to the
once-per-instance dedup).
