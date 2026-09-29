#!/usr/bin/env python
"""TRACE — hunt the MiniLM `Token indices sequence length is longer than the
specified maximum sequence length for this model (... > 256)` warning.

CPU-ONLY. No LLM weights are loaded (no engine/llm import anywhere).
Only cached models are used (HF_HUB_OFFLINE=1):
  - sentence-transformers/all-MiniLM-L6-v2  (the embedder under test)
  - Qwen/Qwen2.5-3B-Instruct *tokenizer only* (production chunking tokenizer;
    falls back to cached gpt2 if unavailable)

Method:
  (1) Monkeypatch the EXACT warning source
      transformers.PreTrainedTokenizerBase._eventual_warn_about_too_long_sequence
      to record + print a stack trace on every over-length touch (the stock
      code warns only ONCE per tokenizer instance, which is why the benchmark
      logs show the warning only a handful of times).
  (2) Monkeypatch SentenceTransformer.encode to record the WordPiece length of
      every input batch (measured via tokenizer.tokenize, which does NOT hit
      the warning hook, so no recursion) + print a stack trace when an input
      exceeds the embedder's max_seq_length.
  (3) Drive the REAL code paths, device='cpu', in benchmark order, over a
      self-constructed context whose 512-token blocks land in the 400-550
      MiniLM-token range (like the 478/508/511-token offenders in the logs):
        - pipeline.chunk_text
        - embed_long (sempointer.pointer_generator)
        - SemPointerPipeline.ingest + PointerScorer.score + active_prompt
        - DenseRAGBaseline.ingest + retrieve
        - HybridRRFBaseline.ingest + retrieve
        - IterativeRAGBaseline.ingest + round-2 expanded-query retrieve
        - SemanticIDPointerGenerator.generate_token_ids (registry path)

Run:  HF_HUB_OFFLINE=1 ./warden/bin/python -u experiments/trace_warning.py
      (append 2>/dev/null to hide safetensors progress bars; TRACE hits print
      to STDOUT, unbuffered.)
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

torch.set_num_threads(4)

CURRENT_STEP = {"name": "startup"}
HITS = []  # list of dicts: {n, kind, step, ntok, limit, stack}


def _repo_frames(extracted, limit=14):
    keep = [f for f in extracted if "experiments/" in (f.filename or "")]
    tail = keep[-limit:] if keep else list(extracted)[-4:]
    return [(f.filename, f.lineno, f.name) for f in tail]


def _record(kind, ntok, lim):
    HITS.append(
        {
            "n": len(HITS) + 1,
            "kind": kind,
            "step": CURRENT_STEP["name"],
            "ntok": ntok,
            "limit": lim,
            "stack": _repo_frames(traceback.extract_stack()[:-2]),
        }
    )
    h = HITS[-1]
    print(
        f"\n[TRACE-HIT #{h['n']}] {h['kind']} | step={h['step']} "
        f"| tokens={h['ntok']} > limit={h['limit']}",
        flush=True,
    )
    for fn, ln, name in h["stack"]:
        print(f"    at {fn}:{ln} in {name}", flush=True)


# ---- spy 1: the exact warning source in transformers -------------------- #
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_orig_warn = PreTrainedTokenizerBase._eventual_warn_about_too_long_sequence


def _spy_warn(self, ids, max_length, verbose):
    try:
        lim = self.model_max_length or 0
        if max_length is None and verbose and lim and len(ids) > lim:
            _record("TOKENIZER-WARN (bare tok.encode/__call__, no truncation)", len(ids), lim)
    finally:
        return _orig_warn(self, ids, max_length, verbose)


PreTrainedTokenizerBase._eventual_warn_about_too_long_sequence = _spy_warn

# ---- spy 2: SentenceTransformer.encode inputs --------------------------- #
from sentence_transformers import SentenceTransformer

_orig_encode = SentenceTransformer.encode


def _spy_encode(self, sentences, *args, **kwargs):
    try:
        items = [sentences] if isinstance(sentences, str) else list(sentences)
        lim = int(getattr(self, "max_seq_length", 0) or 256)
        worst = 0
        for s in items:
            if isinstance(s, str):
                worst = max(worst, len(self.tokenizer.tokenize(s)))
        if worst > lim:
            _record("ST-ENCODE over-length input (truncated internally, safe)", worst, lim)
    except Exception as e:  # never break the real path
        print(f"[TRACE] spy2 measurement failed: {e}", flush=True)
    return _orig_encode(self, sentences, *args, **kwargs)


SentenceTransformer.encode = _spy_encode

print("[TRACE] spies armed. CURRENT warning-dedup behaviour of stock code: once per tokenizer instance.",
      flush=True)

# ---------------------------------------------------------------- driver -- #
from transformers import AutoTokenizer

for _tok_id in ("Qwen/Qwen2.5-3B-Instruct", "gpt2"):
    try:
        llm_tok = AutoTokenizer.from_pretrained(_tok_id, trust_remote_code=False)
        print(f"[TRACE] chunking tokenizer: {_tok_id}", flush=True)
        break
    except Exception as e:
        print(f"[TRACE] could not load {_tok_id} offline: {e}", flush=True)
else:
    raise RuntimeError("no cached chunking tokenizer available offline")

from sempointer.pipeline import SemPointerPipeline, chunk_text
from sempointer.pointer_generator import embed_long

UNIT = (
    "Memory retrieval depends on compact pointer representations that address "
    "long-range evidence without expanding the active context window. Each block "
    "of the document is compressed into a dense embedding and a small set of "
    "discrete semantic tokens. At query time the system scores every pointer "
    "against the question embedding and resolves only the top-ranked blocks "
    "into full text, keeping the remaining context as lightweight addresses. "
    "This mechanism preserves answer quality while holding inference cost flat. "
)
# Adapt repeat count so at least one 512-token block exceeds 300 MiniLM tokens.
probe = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
context, blocks = "", []
for repeats in (24, 48, 96):
    context = UNIT * repeats
    blocks = chunk_text(context, 512, llm_tok)
    wp = [len(probe.tokenizer.tokenize(b)) for b in blocks]
    qt = [len(llm_tok.encode(b, add_special_tokens=False)) for b in blocks]
    print(f"[TRACE] repeats={repeats} -> {len(blocks)} blocks; "
          f"Qwen-tokens={qt}; MiniLM-tokens={wp}", flush=True)
    if max(wp) > 300:
        break
assert max(wp) > 256, "test context failed to produce an over-length block"
print(f"[TRACE] using {len(blocks)} blocks; offender range present (max MiniLM tokens={max(wp)}).",
      flush=True)
del probe

QUERY = "How does the system keep inference cost flat while preserving answer quality?"
LONG_QUESTION = ("In the context of the memory retrieval architecture described above, "
                 "explain in detail ") + "precisely " * 40 + "how inference cost stays flat?"

print("\n[TRACE] step 1: embed_long on raw blocks (shared helper)", flush=True)
CURRENT_STEP["name"] = "embed_long(blocks)"
st_probe = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
e = embed_long(st_probe, blocks)
print(f"[TRACE] embed_long ok: {tuple(e.shape)}", flush=True)

print("\n[TRACE] step 2: SemPointerPipeline.ingest + scorer + active_prompt", flush=True)
CURRENT_STEP["name"] = "sempointer.ingest"
pipe = SemPointerPipeline(tokenizer=llm_tok, k=8, m=1, block_size=512, device="cpu")
pipe.ingest(context)
CURRENT_STEP["name"] = "sempointer.scorer.score"
ranked, scores = pipe.scorer.score(QUERY, pipe.registry, m=1)
print(f"[TRACE] sempointer ranked={ranked}", flush=True)
CURRENT_STEP["name"] = "sempointer.active_prompt"
_ = pipe.active_prompt(QUERY)
print("[TRACE] sempointer active_prompt ok", flush=True)
CURRENT_STEP["name"] = "sempointer.generate_token_ids"
_ = pipe.generator.generate_token_ids(blocks[0])
print("[TRACE] generate_token_ids ok", flush=True)

print("\n[TRACE] step 3: DenseRAGBaseline.ingest + retrieve", flush=True)
from baselines.dense_rag import DenseRAGBaseline

CURRENT_STEP["name"] = "dense_rag.ingest"
rag = DenseRAGBaseline(tokenizer=llm_tok, m=1, block_size=512, device="cpu")
rag.ingest(context)
CURRENT_STEP["name"] = "dense_rag.retrieve(short query)"
print(f"[TRACE] dense retrieve -> {rag.retrieve(QUERY)[0]}", flush=True)
CURRENT_STEP["name"] = "dense_rag.retrieve(long question)"
print(f"[TRACE] dense retrieve(long) -> {rag.retrieve(LONG_QUESTION)[0]}", flush=True)

print("\n[TRACE] step 4: HybridRRFBaseline.ingest + retrieve", flush=True)
from baselines.hybrid import HybridRRFBaseline

CURRENT_STEP["name"] = "hybrid.ingest"
hyb = HybridRRFBaseline(tokenizer=llm_tok, m=1, block_size=512, device="cpu")
hyb.ingest(context)
CURRENT_STEP["name"] = "hybrid.retrieve"
print(f"[TRACE] hybrid retrieve -> {hyb.retrieve(QUERY)[0]}", flush=True)

print("\n[TRACE] step 5: IterativeRAGBaseline.ingest + round-2 expanded-query retrieve", flush=True)
from baselines.iterative_rag import IterativeRAGBaseline

CURRENT_STEP["name"] = "iterative.ingest"
it = IterativeRAGBaseline(tokenizer=llm_tok, m=1, block_size=512, device="cpu")
it.ingest(context)
draft_snippet = ("draft answer text with retrieved evidence phrasing " * 20)[:1000]
expanded = f"{LONG_QUESTION} {draft_snippet}".strip()
print(f"[TRACE] expanded query chars={len(expanded)}; "
      f"MiniLM tokens={len(it._rag.embedder.tokenizer.tokenize(expanded))}", flush=True)
CURRENT_STEP["name"] = "iterative.round2.retrieve(expanded)"
print(f"[TRACE] iterative round-2 retrieve -> {it._rag.retrieve(expanded)[0]}", flush=True)

print("\n[TRACE] step 6: fix-equivalence check (tokenize+convert vs encode)", flush=True)
tok = st_probe.tokenizer
with_hide = blocks[int(len(blocks) / 2)]
a = tok.convert_tokens_to_ids(tok.tokenize(with_hide))
import logging
logging.disable(logging.CRITICAL)
b = tok.encode(with_hide, add_special_tokens=False)
logging.disable(logging.NOTSET)
print(f"[TRACE] convert_tokens_to_ids(tokenize(t)) == encode(t, no specials): {a == b}",
      flush=True)

# ---------------------------------------------------------------- summary - #
print("\n================ TRACE SUMMARY ================", flush=True)
warns = [h for h in HITS if h["kind"].startswith("TOKENIZER-WARN")]
sts = [h for h in HITS if h["kind"].startswith("ST-ENCODE")]
print(f"[TRACE] TOKENIZER-WARN hits (these print in benchmark logs): {len(warns)}", flush=True)
for h in warns:
    print(f"  #{h['n']} step={h['step']} tokens={h['ntok']} > {h['limit']}", flush=True)
    for fn, ln, name in h["stack"]:
        print(f"      at {fn}:{ln} in {name}", flush=True)
print(f"[TRACE] ST-ENCODE over-length inputs (silently truncated, never warn): {len(sts)}", flush=True)
for h in sts:
    print(f"  #{h['n']} step={h['step']} tokens={h['ntok']} > {h['limit']}", flush=True)
    for fn, ln, name in h["stack"]:
        print(f"      at {fn}:{ln} in {name}", flush=True)
if warns:
    h = warns[0]
    print(f"\n[TRACE] GUILTY SITE (first benchmark-log warning): {h['stack'][-1][0]}:"
          f"{h['stack'][-1][1]} during step {h['step']}", flush=True)
print("[TRACE] done.", flush=True)
