"""Real causal-LM inference engine.

Every evaluated system (SemPointer and all baselines) generates answers through
this engine, so model, dtype, device and generation settings are held constant
across systems. Only the context mechanism differs between systems.

Measured per generation (MEASURED, not derived):
  - wall-clock latency (prefill + decode)
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

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_SMOKE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DTYPE_CHOICES = ("bf16", "nf4")


def nf4_compute_dtype(device: str) -> torch.dtype:
    """Use FP16 on pre-Ampere GPUs such as Kaggle T4; keep BF16 elsewhere."""
    if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16

PREFILL_CHUNK = 4096  # max tokens per prefill forward (MLP transient cap)
LONG_PREFILL_THRESHOLD = 12000
LONG_PREFILL_CHUNK = 2048  # smaller activation peak for 16k full-context prompts


@dataclass
class GenResult:
    text: str
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    peak_vram_mb: float
    ttft_ms: Optional[float] = None
    stop_reason: str = "max_new_tokens"


class _FirstTokenTimer:
    """Record the first generated token callback from Transformers.generate.

    Transformers sends the input prompt to a streamer first, then generated
    tokens. The callback receives CPU tensors, so CUDA work has synchronized.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.first_token_at: Optional[float] = None

    def put(self, token_ids) -> None:
        self.calls += 1
        if self.calls == 2:
            self.first_token_at = time.perf_counter()

    def end(self) -> None:
        pass


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
        self.nf4_compute_dtype = nf4_compute_dtype(device) if dtype == "nf4" else None

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
                bnb_4bit_compute_dtype=self.nf4_compute_dtype,
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
        an incremental KV cache. Above 12k tokens, 2k chunks lower the
        temporary activation peak on the 8GB GPU. Decoding stays greedy."""
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
            timer = _FirstTokenTimer()
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                logits_to_keep=1,
                pad_token_id=self.tokenizer.pad_token_id,
                streamer=timer,
            )
            generated_ids = out[0][n_prompt:]
            finish = out[0][-1].item()
            first_token_at = timer.first_token_at
        else:
            chunk_size = LONG_PREFILL_CHUNK if n_prompt > LONG_PREFILL_THRESHOLD else PREFILL_CHUNK
            generated_ids, finish, first_token_at = self._chunked_prefill_generate(
                inputs["input_ids"], max_new_tokens, chunk_size
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
            ttft_ms=(first_token_at - t0) * 1000.0 if first_token_at is not None else None,
            stop_reason=stop_reason,
        )

    def _chunked_prefill_generate(self, input_ids: torch.Tensor, max_new_tokens: int,
                                  chunk_size: int):
        """Prefills `input_ids` in `chunk_size`-token chunks into an
        incremental cache, then greedy-decodes. No padding is involved, so
        attention_mask=None (implicit full-causal attention) is exact.

        `logits_to_keep=1` restricts the lm_head to the final position —
        full-chunk logits would need a ~2.5 GB (8192 x vocab) tensor."""
        past = None
        seq_len = input_ids.shape[1]
        pos = 0
        logits = None
        while pos < seq_len:
            chunk = input_ids[:, pos : pos + chunk_size]
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
        first_token_at = time.perf_counter()
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
        return torch.tensor(generated, device=input_ids.device), finish, first_token_at

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
            "nf4_compute_dtype": str(getattr(self, "nf4_compute_dtype", None)) if self.dtype == "nf4" else None,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "decoding": "greedy",
            "attn_implementation": "sdpa",
            "seed": self.seed,
        }
        # Transformers attaches the Hub commit to the loaded config. Reading
        # it here avoids a network request and does not mistake _name_or_path
        # (a model identifier) for an immutable revision.
        cfg["model_config_revision"] = (
            getattr(getattr(self.model, "config", None), "_commit_hash", None)
            or getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        )
        return cfg

    def save_info(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.info(), f, indent=2)
