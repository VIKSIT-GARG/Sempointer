"""Opt-in Qwen3.5 engine: native hybrid-cache generation, direct answers."""
import time

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from engine.llm import LLMEngine, GenResult, _FirstTokenTimer, nf4_compute_dtype


class Qwen35Engine(LLMEngine):
    def __init__(self, model_id="Qwen/Qwen3.5-4B", dtype="nf4", device="cuda",
                 max_new_tokens=128, seed=42):
        if dtype not in {"nf4", "bf16"}:
            raise ValueError("Qwen3.5 precision must be nf4 or bf16")
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        if config.model_type != "qwen3_5":
            raise ValueError("--engine qwen35 requires a Qwen3.5 checkpoint")
        self.model_id, self.dtype, self.device = model_id, dtype, device
        self.max_new_tokens, self.seed = max_new_tokens, seed
        self.nf4_compute_dtype = nf4_compute_dtype(device) if dtype == "nf4" else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = dict(config=config, local_files_only=True, dtype=torch.bfloat16)
        if device.startswith("cuda"):
            kwargs.update(device_map={"": device}, attn_implementation="sdpa")
        if dtype == "nf4":
            if not device.startswith("cuda"):
                raise ValueError("This NF4 evaluation path requires a CUDA device")
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.nf4_compute_dtype,
                bnb_4bit_use_double_quant=True)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if not device.startswith("cuda"):
            self.model = self.model.to(device)
        self.model.eval()
        torch.manual_seed(seed)

    def format_prompt(self, prompt):
        # Apply once, before control-budget accounting; generate consumes this
        # exact string. A second template would invalidate token matching.
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=None):
        cap = max_new_tokens or self.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        n_prompt = int(inputs["input_ids"].shape[1])
        cuda = self.device.startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        timer = _FirstTokenTimer()
        started = time.perf_counter()
        # Transformers owns both recurrent DeltaNet state and attention KV
        # state. Do not reuse the legacy engine's manual chunk/cache loop.
        output = self.model.generate(**inputs, max_new_tokens=cap, do_sample=False,
            use_cache=True, logits_to_keep=1, pad_token_id=self.tokenizer.pad_token_id,
            streamer=timer)
        if cuda:
            torch.cuda.synchronize()
        latency = (time.perf_counter() - started) * 1000
        generated = output[0][n_prompt:]
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        return GenResult(text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            prompt=prompt, prompt_tokens=n_prompt, completion_tokens=int(generated.numel()),
            latency_ms=latency, peak_vram_mb=torch.cuda.max_memory_allocated() / 1024**2 if cuda else 0.,
            ttft_ms=(timer.first_token_at - started) * 1000 if timer.first_token_at is not None else None,
            stop_reason="eos" if generated.numel() and int(generated[-1]) in eos else "max_new_tokens")

    def info(self):
        return {**super().info(), "engine": "qwen35_native_generate", "enable_thinking": False,
                "prompt_format": "native_chat_template", "cache": "transformers_native_hybrid"}
