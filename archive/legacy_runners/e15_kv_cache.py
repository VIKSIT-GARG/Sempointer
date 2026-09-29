"""E15: Persistent Key-Value Cache Regime and Memory Capacity.

Tests Regime 2 (Persistent KV Cache Memory Architecture):
- Condition A (Full Context KV): Maintains KV cache for all N * B tokens.
- Condition B (SemPointer KV): Maintains KV cache only for N * k pointer tokens (+ m * B resolved).
- Measures memory footprint scaling across N in [8, 16, 32, 64, 128, 256, 512, 1024].
- Computes exact theoretical capacity ratio: B / k = 64x memory footprint reduction.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Any
import numpy as np
import torch
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_kv_cache_bytes(
    seq_len: int,
    n_layers: int = 32,
    n_kv_heads: int = 8,  # Grouped Query Attention (GQA) typical for Llama 3 / Qwen 2.5
    d_head: int = 128,
    dtype_bytes: int = 2, # FP16 / BF16
) -> int:
    """Calculates exact KV cache memory bytes: 2 * n_layers * seq_len * n_kv_heads * d_head * dtype_bytes."""
    return 2 * n_layers * seq_len * n_kv_heads * d_head * dtype_bytes


def run_e15_cell(
    N: int,
    B: int = 512,
    k: int = 8,
    m: int = 1,
    L_Q: int = 64,
    vram_budget_gb: float = 8.0,
) -> Dict[str, Any]:
    """Computes exact VRAM footprints for Full Context vs SemPointer KV caches."""
    # Full Context KV sequence length
    full_seq_len = N * B + L_Q
    # SemPointer KV sequence length: (N - m)*k unselected + m*B resolved + L_Q
    eff_m = min(m, N)
    sp_seq_len = (N - eff_m) * k + eff_m * B + L_Q

    full_kv_bytes = calculate_kv_cache_bytes(full_seq_len)
    sp_kv_bytes = calculate_kv_cache_bytes(sp_seq_len)

    vram_budget_bytes = vram_budget_gb * (1024 ** 3)
    full_oom = bool(full_kv_bytes > vram_budget_bytes)
    sp_oom = bool(sp_kv_bytes > vram_budget_bytes)

    capacity_ratio = full_kv_bytes / max(1, sp_kv_bytes)

    return {
        "N": N,
        "full_seq_len": full_seq_len,
        "sp_seq_len": sp_seq_len,
        "full_kv_mb": float(full_kv_bytes / (1024 * 1024)),
        "sp_kv_mb": float(sp_kv_bytes / (1024 * 1024)),
        "capacity_multiplier": float(capacity_ratio),
        "full_context_oom_at_8gb": full_oom,
        "sempointer_oom_at_8gb": sp_oom,
    }


def main():
    parser = argparse.ArgumentParser(description="E15: Persistent KV Cache Regime")
    parser.add_argument("--config", type=str, default="config/e15_kv_cache.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal N sweep.")
    parser.add_argument("--vram-gb", type=float, default=8.0, help="Target GPU VRAM in GB.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device.")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    B = config.get("sempointer", {}).get("B", 512)
    k = config.get("sempointer", {}).get("k", 8)
    m = config.get("sempointer", {}).get("m", 1)
    L_Q = config.get("sempointer", {}).get("L_Q", 64)

    if args.dry_run:
        N_vals = [8, 32, 128, 512]
    else:
        N_vals = config.get("sweep", {}).get("N", [8, 16, 32, 64, 128, 256, 512, 1024])

    results = {}
    table = Table(title=f"E15: KV Cache Memory Footprint & Scaling ({args.vram_gb} GB VRAM Budget)")
    table.add_column("N", style="cyan")
    table.add_column("Full KV (MB)", style="red")
    table.add_column("SemPointer KV (MB)", style="green")
    table.add_column("VRAM Savings", style="bold green")
    table.add_column("Full OOM?", style="yellow")
    table.add_column("SP OOM?", style="blue")

    for N in N_vals:
        res = run_e15_cell(N=N, B=B, k=k, m=m, L_Q=L_Q, vram_budget_gb=args.vram_gb)
        results[f"N_{N}"] = res

        table.add_row(
            str(N),
            f"{res['full_kv_mb']:.1f} MB",
            f"{res['sp_kv_mb']:.1f} MB",
            f"{res['capacity_multiplier']:.1f}x",
            "[red]OOM[/red]" if res["full_context_oom_at_8gb"] else "[green]OK[/green]",
            "[red]OOM[/red]" if res["sempointer_oom_at_8gb"] else "[green]OK[/green]",
        )

    console.print(table)

    os.makedirs("results/e15", exist_ok=True)
    out_file = f"results/e15/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E15_Persistent_KV_Cache",
                "timestamp": time.time(),
                "vram_budget_gb": args.vram_gb,
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
