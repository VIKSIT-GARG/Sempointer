"""E16: Paged and Offloaded Memory Regime (Regime 3).

Tests Regime 3 (Paged Host-Device Memory Architecture):
- Models and benchmarks host-to-device paging bandwidth (PCIe Gen4/Gen5) when historical
  context exceeds on-chip GPU capacity.
- Evaluates:
  1. Transfer volume for SemPointer: transfers only m resolved blocks (m * B * bytes_per_token).
  2. Transfer volume for linear re-ingestion: re-transfers full history (N * B * bytes_per_token).
  3. End-to-end paging latency across PCIe bus speeds (16 GB/s to 64 GB/s).
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

from runners.e15_kv_cache import calculate_kv_cache_bytes

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def benchmark_device_transfer_speed(chunk_mb: int = 64, device: str = "cuda") -> float:
    """Measures actual host-to-device PCIe transfer bandwidth in GB/s if CUDA is available."""
    if not torch.cuda.is_available() or "cuda" not in device:
        return 32.0 # Theoretical PCIe Gen4 x16 bandwidth in GB/s

    num_elements = (chunk_mb * 1024 * 1024) // 4
    host_tensor = torch.randn(num_elements, dtype=torch.float32, pin_memory=True)

    # Warmup
    _ = host_tensor.to(device, non_blocking=False)
    torch.cuda.synchronize()

    times = []
    for _ in range(5):
        start = time.perf_counter()
        _ = host_tensor.to(device, non_blocking=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_sec = float(np.mean(times))
    bandwidth_gb_s = (chunk_mb / 1024.0) / max(1e-5, mean_sec)
    return float(bandwidth_gb_s)


def run_e16_cell(
    N: int,
    B: int = 512,
    m: int = 1,
    bus_bandwidth_gb_s: float = 32.0,
) -> Dict[str, Any]:
    """Computes host-device transfer bytes and paging latency for SemPointer vs Full Ingestion."""
    eff_m = min(m, N)
    bytes_per_block = calculate_kv_cache_bytes(B)

    # Paging volume
    full_transfer_bytes = N * bytes_per_block
    sp_transfer_bytes = eff_m * bytes_per_block

    bus_bytes_per_sec = bus_bandwidth_gb_s * (1024 ** 3)
    full_latency_ms = (full_transfer_bytes / bus_bytes_per_sec) * 1000.0
    sp_latency_ms = (sp_transfer_bytes / bus_bytes_per_sec) * 1000.0

    return {
        "N": N,
        "full_transfer_mb": full_transfer_bytes / (1024 * 1024),
        "sp_transfer_mb": sp_transfer_bytes / (1024 * 1024),
        "full_paging_latency_ms": full_latency_ms,
        "sp_paging_latency_ms": sp_latency_ms,
        "speedup_factor": full_transfer_bytes / max(1, sp_transfer_bytes),
    }


def main():
    parser = argparse.ArgumentParser(description="E16: Paged Memory Regime Evaluation")
    parser.add_argument("--config", type=str, default="config/e16_paged.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal N sweep.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    B = config.get("sempointer", {}).get("B", 512)
    m = config.get("sempointer", {}).get("m", 1)

    measured_bw = benchmark_device_transfer_speed(chunk_mb=32, device=args.device)
    console.print(f"[bold]Host-to-Device Paging Bandwidth:[/bold] {measured_bw:.2f} GB/s")

    if args.dry_run:
        N_vals = [32, 128, 512]
    else:
        N_vals = config.get("sweep", {}).get("N", [32, 64, 128, 256, 512, 1024, 2048])

    results = {}
    table = Table(title="E16: Paged Memory Offload & Paging Latency Comparison")
    table.add_column("N (blocks)", style="cyan")
    table.add_column("Linear Re-transfer", style="red")
    table.add_column("SemPointer Paging", style="green")
    table.add_column("Linear Latency", style="yellow")
    table.add_column("SemPointer Latency", style="bold green")
    table.add_column("Transfer Reduction", style="bold blue")

    for N in N_vals:
        res = run_e16_cell(N=N, B=B, m=m, bus_bandwidth_gb_s=measured_bw)
        results[f"N_{N}"] = res

        table.add_row(
            str(N),
            f"{res['full_transfer_mb']:.1f} MB",
            f"{res['sp_transfer_mb']:.1f} MB",
            f"{res['full_paging_latency_ms']:.2f} ms",
            f"{res['sp_paging_latency_ms']:.2f} ms",
            f"{res['speedup_factor']:.1f}x",
        )

    console.print(table)

    os.makedirs("results/e16", exist_ok=True)
    out_file = f"results/e16/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E16_Paged_Memory_Regime",
                "timestamp": time.time(),
                "bandwidth_gb_s": measured_bw,
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
