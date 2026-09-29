"""B4: KV-cache-compression baseline via NVIDIA kvpress 0.5.5 (uniform interface).

Confirmed API (read from installed source at
site-packages/kvpress/{__init__,pipeline}.py and site-packages/kvpress/presses/):

- `from kvpress import KVPressTextGenerationPipeline, SnapKVPress, PyramidKVPress,
  KnormPress, TOVAPress` (all present in kvpress 0.5.5 `__all__`; there is NO
  H2OPress in this version).
- Presses are dataclasses exposing `compression_ratio: float = 0.0`
  (SnapKVPress/PyramidKVPress/TOVAPress declare it; KnormPress inherits it from
  ScorerPress). `__post_init__` asserts 0 <= compression_ratio < 1.
- Usage: `pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)`
  then `out = pipeline(context_str, question=q, press=press, max_new_tokens=n,
  cache=cache)`. NOTE: `context` is the first POSITIONAL argument —
  `KVPressTextGenerationPipeline._sanitize_parameters` has no `context` keyword
  (unknown call kwargs are ignored), so it must be passed positionally.
  Returns `{"answer": str}` for a single question.
- Decoding inside `_forward.generate_answer` is greedy argmax with EOS stop —
  matches do_sample=False of the shared engine.
- Preprocessing applies the tokenizer chat template to the context (with
  add_generation_prompt=True) and appends the question after the assistant
  header; there is no separate system role, so this adapter prepends the
  system prompt to the context (recorded in extra["system_prompt_location"]).
- The pipeline does NOT return token counts, latency, or VRAM. This adapter
  measures them honestly:
    * latency/peak-VRAM: wall clock + torch.cuda.max_memory_allocated around
      the pipeline call (same methodology as engine.llm.GenResult);
    * post-compression context KV length: pass our own `DynamicCache` and read
      `cache.get_seq_length()` after the call (kvpress compresses the cache
      in-place during prefill and removes the answer tokens afterwards);
    * prefill context tokens / question tokens: recompute the exact strings
      the pipeline tokenizes (same chat-template recipe as its `preprocess`).
  active_tokens = post-compression context KV tokens + question tokens —
  the tokens that actively participate in decoding (analogous to SemPointer's
  L_active = (N-m)k + m*B_res + L_Q).

Because the press modifies attention during prefill, this system loads its OWN
model instance (same model_id, dtype, device, eager attention, greedy decoding
and max_new_tokens as the shared LLMEngine). This is recorded in info().
The model MUST be loaded with attn_implementation="eager" (kvpress requirement).
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import (
    KVPressTextGenerationPipeline,
    KnormPress,
    PyramidKVPress,
    SnapKVPress,
    TOVAPress,
)

from ._text_common import DEFAULT_SYSTEM_PROMPT

PRESS_REGISTRY = {
    "snapkv": SnapKVPress,
    "pyramidkv": PyramidKVPress,
    "knorm": KnormPress,
    "tova": TOVAPress,
}


class KVPressBaseline:
    """kvpress-<press> at a fixed prefill compression ratio."""

    def __init__(
        self,
        press_name: str,
        compression_ratio: float = 0.3,
        model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        dtype: str = "bf16",
        device: str = "cuda",
        max_new_tokens: int = 64,
        seed: int = 42,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        trust_remote_code: bool = False,
    ):
        if press_name not in PRESS_REGISTRY:
            raise ValueError(
                f"press_name must be one of {sorted(PRESS_REGISTRY)}, got {press_name!r}"
            )
        if dtype not in ("bf16", "nf4"):
            raise ValueError(f"dtype must be 'bf16' or 'nf4', got {dtype!r}")

        self.press_name = press_name
        self.compression_ratio = float(compression_ratio)
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.press = PRESS_REGISTRY[press_name](compression_ratio=self.compression_ratio)
        self._context: Optional[str] = None
        self.ingestion: Dict[str, Any] = {}
        self.model = None
        self.pipeline = None

    def _ensure_loaded(self) -> None:
        """Lazy model load: kvpress runs its own engine instance, so on an 8GB
        GPU it must not be resident while the shared text-engine is."""
        if self.model is not None:
            return

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": False,
            "attn_implementation": "eager",  # required by kvpress
        }
        if self.dtype == "nf4":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        if self.device == "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        torch.manual_seed(self.seed)
        self.pipeline = KVPressTextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)

    # ------------------------------------------------------------------ #
    def ingest(self, context: str) -> None:
        """Stores the raw context string (kvpress prefills per query, so the
        full context is re-prefilled and re-compressed inside each answer())."""
        self._context = context
        self.ingestion = {
            "ingestion_tokens": len(self.tokenizer.encode(context, add_special_tokens=False)),
            "ingestion_latency_ms": 0.0,  # no separate indexing step in kvpress
        }

    def _pipeline_input(self, query: str) -> Dict[str, str]:
        """Returns the raw context to feed the pipeline (its `preprocess`
        applies the chat template itself) plus the exact strings kvpress will
        tokenize, so token counts are measured over the real inputs (not
        estimates)."""
        context = self._require_context()
        if self.system_prompt:
            context = f"{self.system_prompt.strip()}\n\n{context}"
        tok = self.tokenizer
        if tok.chat_template is None:
            bos = getattr(tok, "bos_token", "") or ""
            return {
                "raw_context": context,
                "context_text": bos + context,
                "suffix": "\n",
                "question_text": query,
                "full_input_text": bos + context + "\n" + query,
            }
        separator = "#" * (len(context) + 10)
        templated = tok.apply_chat_template(
            [{"role": "user", "content": context + separator}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        templated_ctx, suffix = templated.split(separator)
        return {
            "raw_context": context,  # pipeline applies the template to this
            "context_text": templated_ctx,
            "suffix": suffix,
            "question_text": query,
            "full_input_text": templated_ctx + suffix + query,
        }

    def answer(self, query: str, llm=None) -> Dict[str, Any]:
        """Prefill + press-compress + greedy decode via the kvpress pipeline.

        `llm` (shared engine) is accepted for interface parity and used only to
        take max_new_tokens from, so the decoding budget matches other systems.
        """
        self._ensure_loaded()
        if llm is not None and getattr(llm, "max_new_tokens", None):
            max_new_tokens = int(llm.max_new_tokens)
            max_new_tokens_source = "shared_engine"
        else:
            max_new_tokens = self.max_new_tokens
            max_new_tokens_source = "kvpress_baseline"

        texts = self._pipeline_input(query)
        cache = DynamicCache()  # ours => post-compression length is measurable

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.pipeline(
            texts["raw_context"],
            question=texts["question_text"],
            press=self.press,
            max_new_tokens=max_new_tokens,
            cache=cache,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2) if self.device == "cuda" else 0.0
        )

        answer = out["answer"]  # pipeline postprocess -> {"answer": str} for one question

        # --- honest token accounting (see module docstring) ----------------
        context_prefill_tokens = len(self.tokenizer.encode(texts["context_text"], add_special_tokens=False))
        question_tokens_raw = len(self.tokenizer.encode(texts["question_text"], add_special_tokens=False))
        question_tokens_with_suffix = len(
            self.tokenizer.encode(texts["suffix"] + texts["question_text"], add_special_tokens=False)
        )
        try:
            compressed_context_tokens = int(cache.get_seq_length())
        except TypeError:  # transformers 5.x requires a layer index
            compressed_context_tokens = int(cache.get_seq_length(layer_idx=0))
        completion_tokens = len(self.tokenizer.encode(answer, add_special_tokens=False))

        return {
            "answer": answer,
            "prompt": texts["full_input_text"],  # exact chat-templated input the pipeline ran on
            "active_tokens": compressed_context_tokens + question_tokens_raw,
            "selected_ids": None,  # KV compression prunes cache entries, not text blocks
            "generation_latency_ms": latency_ms,
            "completion_tokens": completion_tokens,
            "peak_vram_mb": peak_vram_mb,
            "n_blocks": 0,
            "stop_reason": "eos" if completion_tokens < max_new_tokens else "max_new_tokens",
            "extra": {
                "system": f"kvpress:{self.press_name}",
                "press": self.press_name,
                "press_class": type(self.press).__name__,
                "compression_ratio": self.compression_ratio,
                "compressed_context_tokens": compressed_context_tokens,  # MEASURED from cache
                "context_tokens_prefill": context_prefill_tokens,  # before compression
                "question_tokens_raw": question_tokens_raw,
                "question_tokens_with_template_suffix": question_tokens_with_suffix,
                "active_tokens_note": "compressed context KV tokens + raw question tokens",
                "max_new_tokens": max_new_tokens,
                "max_new_tokens_source": max_new_tokens_source,
                "decoding": "greedy (kvpress generate_answer argmax)",
                "chat_template_applied": True,
                "system_prompt_location": (
                    "prepended to context (kvpress pipeline has no system role)"
                    if self.system_prompt
                    else "none"
                ),
                "engine": "separate KVPressTextGenerationPipeline model instance "
                "(same model_id/dtype/eager attention/greedy decoding as shared engine)",
                "ingestion": self.ingestion,
            },
        }

    # ------------------------------------------------------------------ #
    def _require_context(self) -> str:
        if self._context is None:
            raise RuntimeError("KVPressBaseline.ingest() must be called before use")
        return self._context

    def cleanup(self) -> None:
        del self.pipeline
        self.pipeline = None
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def info(self) -> Dict[str, Any]:
        return {
            "system": f"kvpress:{self.press_name}",
            "press": self.press_name,
            "press_class": type(self.press).__name__,
            "compression_ratio": self.compression_ratio,
            "model_id": self.model_id,
            "dtype": self.dtype,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "decoding": "greedy",
            "seed": self.seed,
            "attn_implementation": "eager",
            "engine": "separate model instance (press modifies attention); identical "
            "model/precision/decoding to the shared LLMEngine",
            "ingestion_tokens": self.ingestion.get("ingestion_tokens"),
            "has_system_prompt": bool(self.system_prompt),
        }
