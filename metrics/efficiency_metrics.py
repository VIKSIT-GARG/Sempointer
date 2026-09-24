"""Computational Efficiency, FLOPs, and VRAM Profiling Metrics.

Provides analytical FLOP counters, torch.profiler hooks, GPU memory trackers,
and throughput instruments for transformer context processing.
"""

import time
from typing import Dict, Tuple, Any, List, Optional
from contextlib import contextmanager
import torch


@contextmanager
def gpu_timer(device: str = "cuda"):
    """Context manager for GPU-synchronized or wall-clock timing."""
    if torch.cuda.is_available() and "cuda" in device:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        yield lambda: (end_event.record(), torch.cuda.synchronize(), start_event.elapsed_time(end_event))[2]
    else:
        start_time = time.perf_counter()
        yield lambda: (time.perf_counter() - start_time) * 1000.0


def analytical_flops_transformer(
    seq_len: int,
    d_model: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    d_head: Optional[int] = None,
    include_mlp: bool = True,
) -> Dict[str, float]:
    """Computes exact analytical FLOP counts for standard autoregressive transformer prefill.

    Formulation:
    - Self-Attention QKV & Output Projections: 4 * seq_len * d_model^2 per layer
    - Self-Attention QK^T & Softmax-V Matrix Products: 4 * seq_len^2 * d_model per layer
    - SwiGLU / MLP Feed-Forward (hidden dimension ~ 8/3 * d_model): ~ 16 * seq_len * d_model^2 per layer
    - Standard MLP (4 * d_model intermediate): 16 * seq_len * d_model^2 per layer
    """
    if d_head is None:
        d_head = d_model // n_heads

    # Attention core: QK^T and AV attention FLOPs
    attn_quadratic_flops = 4.0 * (seq_len ** 2) * d_model * n_layers
    attn_linear_flops = 4.0 * seq_len * (d_model ** 2) * n_layers

    mlp_flops = 16.0 * seq_len * (d_model ** 2) * n_layers if include_mlp else 0.0

    total_flops = attn_quadratic_flops + attn_linear_flops + mlp_flops

    return {
        "total_flops": total_flops,
        "attention_quadratic_flops": attn_quadratic_flops,
        "attention_linear_flops": attn_linear_flops,
        "mlp_flops": mlp_flops,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_layers": n_layers,
    }


class FLOPsProfiler:
    """Measures model execution FLOPs using torch.profiler and analytical calibration."""

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device

    def profile_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_warmup: int = 2,
        n_trials: int = 5,
    ) -> Dict[str, Any]:
        """Profiles forward pass latency and FLOPs."""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                self.model(input_ids, attention_mask=attention_mask)
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.synchronize()

        # Timing
        latencies = []
        for _ in range(n_trials):
            start = time.perf_counter()
            with torch.no_grad():
                self.model(input_ids, attention_mask=attention_mask)
            if torch.cuda.is_available() and "cuda" in self.device:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)

        mean_latency = sum(latencies) / len(latencies)
        std_latency = float(torch.tensor(latencies).std().item()) if len(latencies) > 1 else 0.0

        # Extract architectural properties from config
        config = getattr(self.model, "config", None)
        seq_len = input_ids.shape[1]
        d_model = getattr(config, "hidden_size", 4096)
        n_layers = getattr(config, "num_hidden_layers", 32)
        n_heads = getattr(config, "num_attention_heads", 32)

        # Profile with torch.profiler if supported
        profiler_flops = 0.0
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available() and "cuda" in self.device:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                with torch.no_grad():
                    self.model(input_ids, attention_mask=attention_mask)
            profiler_flops = float(sum(evt.flops for evt in prof.key_averages() if getattr(evt, "flops", None) is not None))
        except Exception:
            profiler_flops = 0.0

        # Analytical FLOP computation
        analytical = analytical_flops_transformer(
            seq_len=seq_len,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
        )

        effective_flops = profiler_flops if profiler_flops > 0 else analytical["total_flops"]

        return {
            "effective_flops": effective_flops,
            "profiler_measured_flops": profiler_flops,
            "analytical_flops": analytical["total_flops"],
            "attention_flops": analytical["attention_quadratic_flops"] + analytical["attention_linear_flops"],
            "mlp_flops": analytical["mlp_flops"],
            "mean_latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "seq_len": seq_len,
        }


class MemoryProfiler:
    """Tracks GPU VRAM allocation, peak usage, and KV cache scaling."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def measure_peak_vram(self, fn, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Measures peak GPU memory allocated during execution of callable."""
        if torch.cuda.is_available() and "cuda" in self.device:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            start_alloc = torch.cuda.memory_allocated(self.device)
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
            return result, {
                "peak_vram_bytes": peak_alloc,
                "peak_vram_mb": peak_alloc / (1024 * 1024),
                "allocated_bytes_delta": peak_alloc - start_alloc,
            }
        else:
            start_time = time.perf_counter()
            result = fn(*args, **kwargs)
            return result, {
                "peak_vram_bytes": 0,
                "peak_vram_mb": 0.0,
                "allocated_bytes_delta": 0,
                "wall_time_ms": (time.perf_counter() - start_time) * 1000.0,
            }

    def measure_kv_cache_size(self, model, seq_len: int, dtype_bytes: int = 2) -> Dict[str, Any]:
        """Calculates exact theoretical KV cache bytes for sequence length."""
        config = getattr(model, "config", None)
        n_layers = getattr(config, "num_hidden_layers", 32)
        n_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 32))
        hidden_size = getattr(config, "hidden_size", 4096)
        total_heads = getattr(config, "num_attention_heads", 32)
        d_head = hidden_size // total_heads

        # 2 tensors (Key, Value) * n_layers * seq_len * n_kv_heads * d_head * dtype_bytes
        kv_bytes = 2 * n_layers * seq_len * n_heads * d_head * dtype_bytes
        return {
            "kv_cache_bytes": kv_bytes,
            "kv_cache_mb": kv_bytes / (1024 * 1024),
            "bytes_per_token": kv_bytes / max(1, seq_len),
            "n_layers": n_layers,
            "seq_len": seq_len,
        }

    def estimate_n_max_before_oom(
        self,
        model,
        B: int = 512,
        k: int = 8,
        regime: str = "pointer",
        reserve_headroom_mb: float = 1024.0,
    ) -> int:
        """Estimates max memory registry size N before exceeding available VRAM."""
        if not torch.cuda.is_available() or "cuda" not in self.device:
            return 1024

        total_vram = torch.cuda.get_device_properties(self.device).total_memory
        current_alloc = torch.cuda.memory_allocated(self.device)
        usable_vram = max(0, total_vram - current_alloc - int(reserve_headroom_mb * 1024 * 1024))

        config = getattr(model, "config", None)
        n_layers = getattr(config, "num_hidden_layers", 32)
        n_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 32))
        hidden_size = getattr(config, "hidden_size", 4096)
        total_heads = getattr(config, "num_attention_heads", 32)
        d_head = hidden_size // total_heads
        dtype_bytes = 2

        bytes_per_token = 2 * n_layers * n_heads * d_head * dtype_bytes

        if regime == "pointer":
            # SemPointer stores only k tokens per memory in fast KV memory
            tokens_per_unit = k
        else:
            # Full context stores all B tokens per memory in fast KV memory
            tokens_per_unit = B

        unit_bytes = tokens_per_unit * bytes_per_token
        return max(1, usable_vram // max(1, unit_bytes))


def compute_compression_ratio(N: int, k: int, m: int, B_res: int, L_Q: int, B: int) -> float:
    """Calculates active sequence compression ratio: L_linear / L_active."""
    numerator = N * B + L_Q
    denominator = (N - m) * k + m * B_res + L_Q
    return float(numerator / max(1, denominator))
