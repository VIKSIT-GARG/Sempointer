"""Real causal-LM inference engine.

Every evaluated system (SemPointer and all baselines) generates answers through
this engine, so model, dtype, device and generation settings are held constant
across systems. Only the context mechanism differs between systems.

Measured per generation (MEASURED, not derived):
  - wall-clock latency (prefill + decode)
  - TTFT: time to first generated token, incl. prefill (None when not
    separately measurable — see GenResult field docs)
  - decode throughput after the first token (None when undefined)
  - prompt/completion token counts from the real tokenizer
  - peak CUDA memory allocated during the generation call
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.generation.streamers import BaseStreamer

    _HAS_STREAMER = True
except Exception:  # pragma: no cover - very old transformers without streamers
    BaseStreamer = object  # type: ignore[assignment,misc]
    _HAS_STREAMER = False

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_SMOKE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DTYPE_CHOICES = ("bf16", "nf4")

PREFILL_CHUNK = 4096  # max tokens per prefill forward (MLP transient cap)


@dataclass
class GenResult:
    text: str
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    peak_vram_mb: float
    stop_reason: str = "max_new_tokens"
    # -- additive latency split (None = not separately measurable) -------- #
    # ttft_ms: wall clock from generate() entry to the first decoded token,
    #   including prefill. Chunked-prefill path: stamped after the prefill
    #   pass (+1 cuda sync). Short-prompt path: stamped by a first-token
    #   streamer tap during model.generate (+1 cuda sync). Neither changes
    #   decoding: outputs are identical with/without the stamp.
    # decode_tokens_per_sec: 1/TPOT where TPOT = (latency_ms - ttft_ms) /
    #   (completion_tokens - 1); None when completion_tokens < 2 or the split
    #   is unmeasurable/degenerate. Never averaged or estimated — both inputs
    #   are measured wall-clock stamps from the same call.
    ttft_ms: Optional[float] = None
    decode_tokens_per_sec: Optional[float] = None


class _FirstTokenTimer(BaseStreamer):
    """Streamer tap that only records when the first token arrives.

    Passed to model.generate on the short-prompt path so TTFT (prefill +
    first decode step) is MEASURED without changing decoding: greedy outputs
    are identical with or without a streamer. Only instantiated when
    transformers exposes BaseStreamer; otherwise TTFT stays None.
    """

    def __init__(self, device: str):
        super().__init__()
        self._device = device
        self.t_first: Optional[float] = None

    def put(self, value) -> None:
        if self.t_first is None:
            if self._device == "cuda":
                torch.cuda.synchronize()
            self.t_first = time.perf_counter()

    def end(self) -> None:
        return None


def _decode_rate(
    completion_tokens: int, latency_ms: float, ttft_ms: Optional[float]
) -> Optional[float]:
    """1/TPOT over post-first-token decode; None when undefined/degenerate."""
    if ttft_ms is None or completion_tokens < 2:
        return None
    denom_ms = latency_ms - ttft_ms
    if denom_ms <= 0:
        return None
    return (completion_tokens - 1) / (denom_ms / 1000.0)


class LLMEngine:
    """Wraps one causal LM. All systems share a single engine instance so that
    model, precision and decoding are experimental constants."""

    def __init__(
        self,
        model_id: str = DEFAULT_SMOKE_MODEL,
        dtype: str = "bf16",
        device: str = "cuda",
        max_new_tokens: int = 64,
        seed: int = 42,
        trust_remote_code: bool = False,
    ):
        if dtype not in DTYPE_CHOICES:
            raise ValueError(f"dtype must be one of {DTYPE_CHOICES}, got {dtype!r}")
        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device == "cuda":
            # sdpa (memory-efficient attention) keeps long-context prefill
            # feasible on 8GB; kvpress systems load their OWN engine instance
            # with eager attention (kvpress requirement)
            load_kwargs["attn_implementation"] = "sdpa"
        placed_by_device_map = False
        if dtype == "nf4":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            if device == "cuda":
                # bitsandbytes only quantizes during from_pretrained when the
                # model is placed on the device via device_map; without it the
                # weights materialize in bf16 and OOM on 8GB GPUs.
                load_kwargs["device_map"] = {"": device}
                placed_by_device_map = True
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        else:
            # transformers >= 4.56 renamed torch_dtype -> dtype; support both.
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, dtype=torch.bfloat16, **load_kwargs
                )
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16, **load_kwargs
                )
        if device == "cuda" and not placed_by_device_map:
            # never .to() a bnb-quantized model — it is already on-device
            self.model = self.model.to("cuda")
        self.model.eval()
        torch.manual_seed(seed)

    # ------------------------------------------------------------------ #
    def chat_prompt(self, system: str, user: str) -> str:
        """Renders the prompt through the model's own chat template."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return (f"{system}\n\n" if system else "") + user

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenResult:
        """Greedy decoding. Peak-VRAM measurement is per-call (reset before, read after).

        Prompts longer than PREFILL_CHUNK tokens are prefilled in chunks with
        an incremental KV cache: a single 16k-token forward needs a >1.2 GiB
        MLP transient that does not fit an 8GB GPU next to weights+KV, while
        8k chunks do. Decoding stays greedy; results are identical."""
        max_new_tokens = max_new_tokens or self.max_new_tokens
        full_prompt = self.chat_prompt(system, prompt) if system is not None else prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        n_prompt = int(inputs["input_ids"].shape[1])

        if self.device == "cuda":
            # release cached blocks from previous, differently-sized calls —
            # fragmentation was the difference between a 16k prefill fitting
            # and OOMing on the shared 8GB GPU
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if n_prompt <= PREFILL_CHUNK:
            # TTFT via a first-token streamer tap: decoding is unchanged
            # (greedy, same kwargs); without BaseStreamer ttft stays None.
            timer = _FirstTokenTimer(self.device) if _HAS_STREAMER else None
            gen_kwargs: Dict[str, Any] = {}
            if timer is not None:
                gen_kwargs["streamer"] = timer
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                logits_to_keep=1,
                pad_token_id=self.tokenizer.pad_token_id,
                **gen_kwargs,
            )
            generated_ids = out[0][n_prompt:]
            finish = out[0][-1].item()
            ttft_ms = (
                (timer.t_first - t0) * 1000.0
                if timer is not None and timer.t_first is not None
                else None
            )
        else:
            generated_ids, finish, ttft_ms = self._chunked_prefill_generate(
                inputs["input_ids"], max_new_tokens, t0
            )

        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2) if self.device == "cuda" else 0.0
        )
        stop_reason = "eos" if finish == self.tokenizer.eos_token_id else "max_new_tokens"

        return GenResult(
            text=text,
            prompt=full_prompt,
            prompt_tokens=n_prompt,
            completion_tokens=int(generated_ids.shape[0]),
            latency_ms=latency_ms,
            peak_vram_mb=peak_vram_mb,
            stop_reason=stop_reason,
            ttft_ms=ttft_ms,
            decode_tokens_per_sec=_decode_rate(
                int(generated_ids.shape[0]), latency_ms, ttft_ms
            ),
        )

    def _chunked_prefill_generate(
        self, input_ids: torch.Tensor, max_new_tokens: int, t0: float
    ):
        """Prefills `input_ids` in PREFILL_CHUNK-token chunks into an
        incremental cache, then greedy-decodes. No padding is involved, so
        attention_mask=None (implicit full-causal attention) is exact.

        `logits_to_keep=1` restricts the lm_head to the final position —
        full-chunk logits would need a ~2.5 GB (8192 x vocab) tensor.

        Returns (generated_ids, finish_token_id, ttft_ms): ttft_ms is wall
        clock from t0 (generate() entry, incl. the full prefill) to the first
        decoded token. The first token is argmaxed directly from the final
        prefill logits — no extra forward — so the stamp costs one
        cuda.synchronize() and changes nothing about the output."""
        past = None
        seq_len = input_ids.shape[1]
        pos = 0
        logits = None
        while pos < seq_len:
            chunk = input_ids[:, pos : pos + PREFILL_CHUNK]
            cache_position = torch.arange(
                pos, pos + chunk.shape[1], device=input_ids.device
            )
            with torch.no_grad():
                out = self.model(
                    chunk,
                    attention_mask=None,
                    past_key_values=past,
                    use_cache=True,
                    cache_position=cache_position,
                    logits_to_keep=1,
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            pos += chunk.shape[1]

        generated = []
        finish = None
        next_token = logits.argmax(dim=-1, keepdim=True)
        if self.device == "cuda":
            torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000.0
        for step in range(max_new_tokens):
            generated.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                finish = next_token.item()
                break
            with torch.no_grad():
                out = self.model(
                    next_token,
                    attention_mask=None,
                    past_key_values=past,
                    use_cache=True,
                    cache_position=torch.tensor(
                        [seq_len + step], device=input_ids.device
                    ),
                    logits_to_keep=1,
                )
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if finish is None:
            finish = generated[-1] if generated else self.tokenizer.eos_token_id
        return torch.tensor(generated, device=input_ids.device), finish, ttft_ms

    # ------------------------------------------------------------------ #
    def prefill_footprint(self, context: str) -> Dict[str, float]:
        """Measured memory/latency of a single prefill forward over `context`
        (no generation). Used by efficiency experiments: MEASURED peak VRAM."""
        inputs = self.tokenizer(context, return_tensors="pt").to(self.model.device)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            self.model(**inputs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return {
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "prefill_latency_ms": (time.perf_counter() - t0) * 1000.0,
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated() / (1024 ** 2) if self.device == "cuda" else 0.0
            ),
        }

    def model_vram_mb(self) -> float:
        """MEASURED resident weight memory."""
        if self.device != "cuda":
            return 0.0
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated() / (1024 ** 2)

    def cleanup(self) -> None:
        del self.model
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        cfg = {
            "model_id": self.model_id,
            "dtype": self.dtype,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "decoding": "greedy",
            "attn_implementation": "sdpa",
            "seed": self.seed,
            "attn_implementation": "eager",
        }
        try:
            from huggingface_hub import hf_hub_download

            import json as _json

            rev = hf_hub_download(self.model_id, "config.json")
            cfg["model_config_revision"] = _json.load(open(rev)).get("_name_or_path", "")
        except Exception:
            pass
        return cfg

    def save_info(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.info(), f, indent=2)
