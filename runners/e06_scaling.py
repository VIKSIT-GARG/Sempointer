"""E06: Scaling with Registry Size N.

Tests Axiom 5, Selector Complexity O(L_Q*d + N*d), and Hypothesis H6 (Graceful Scaling):
- Fits linear regression: selector_latency = a*N + b (requires R^2 > 0.95).
- Fits log degradation: Recall@1 ~ alpha - beta * log10(N).
- Falsification condition: Recall@1 drops below 0.70 before N=256 with k=8.
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Any
import numpy as np
import torch
import yaml
from rich.console import Console
from rich.table import Table

from sempointer.pointer_generator import PoolingPointerGenerator
from sempointer.registry import PointerRegistry
from sempointer.scorer import PointerScorer
from stats.statistical_tests import fit_linear, fit_log_degradation
from metrics.retrieval_metrics import recall_at_k

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_scaling_cell(
    passages: List[Dict[str, Any]],
    N: int,
    k: int,
    m: int,
    B: int,
    L_Q: int,
    generator: PoolingPointerGenerator,
    device: str = "cpu",
    n_queries: int = 20,
) -> Dict[str, Any]:
    """Measures selector latency and recall for registry size N."""
    registry = PointerRegistry(k=k, B=B, max_size=max(2048, N + 10))
    selected = passages[:N]
    for p in selected:
        registry.register(p["text"], generator, metadata={"id": p["id"]})

    scorer = PointerScorer(generator, device=device)

    latencies = []
    r1_scores = []

    test_queries = selected[: min(n_queries, len(selected))]
    for q_item in test_queries:
        query = q_item["target_query"]
        target_id = q_item["id"]

        start = time.perf_counter()
        ranked_ids, _ = scorer.score_all(query, registry)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)

        r1_scores.append(recall_at_k(ranked_ids, [target_id], k=1))

    mean_lat = float(np.mean(latencies))
    mean_r1 = float(np.mean(r1_scores))
    l_active = (N - m) * k + m * B + L_Q

    return {
        "N": N,
        "selector_latency_ms": mean_lat,
        "recall_at_1": mean_r1,
        "l_active": l_active,
        "h6_supported": bool(mean_r1 >= 0.70 or N >= 256),
    }


def main():
    parser = argparse.ArgumentParser(description="E06: Scaling with Registry Size N")
    parser.add_argument("--config", type=str, default="config/e06_scaling.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal N sweep.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    embed_model = config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    generator = PoolingPointerGenerator(model_name=embed_model, k=8, device=args.device)

    if args.dry_run:
        N_vals = [8, 16, 32, 64]
        n_queries = 10
    else:
        N_vals = config.get("sweep", {}).get("N", [8, 16, 32, 64, 128, 256])
        n_queries = 20

    from runners.e01_retrieval import generate_benchmark_passages
    passages = generate_benchmark_passages(max(N_vals) + 10, seed=seed)

    results = {}
    table = Table(title="E06: Scaling with Registry Size N")
    table.add_column("N", style="cyan")
    table.add_column("L_active (tok)", style="yellow")
    table.add_column("Selector Lat (ms)", style="blue")
    table.add_column("Recall@1", style="bold green")
    table.add_column("H6 Status", style="bold")

    lat_list = []
    r1_list = []

    for N in N_vals:
        res = run_scaling_cell(
            passages=passages,
            N=N,
            k=8,
            m=1,
            B=512,
            L_Q=64,
            generator=generator,
            device=args.device,
            n_queries=n_queries,
        )
        results[f"N_{N}"] = res
        lat_list.append(res["selector_latency_ms"])
        r1_list.append(res["recall_at_1"])

        table.add_row(
            str(N),
            str(res["l_active"]),
            f"{res['selector_latency_ms']:.3f}",
            f"{res['recall_at_1']:.3f}",
            "[green]PASS[/green]" if res["h6_supported"] else "[red]FAIL[/red]",
        )

    console.print(table)

    # Linear regression fit for latency: latency = a*N + b
    lin_fit = fit_linear(N_vals, lat_list)
    # Log degradation fit: recall = alpha - beta*log10(N)
    log_fit = fit_log_degradation(N_vals, r1_list)

    console.print(f"\n[bold]Selector Latency Linear Fit:[/bold] slope={lin_fit['slope']:.5f} ms/item, R^2={lin_fit['r_squared']:.4f}")
    console.print(f"[bold]Recall@1 Log Degradation Fit:[/bold] beta={log_fit['beta']:.4f}, R^2={log_fit['r_squared']:.4f}")

    os.makedirs("results/e06", exist_ok=True)
    out_file = f"results/e06/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E06_Scaling_with_N",
                "timestamp": time.time(),
                "linear_fit": lin_fit,
                "log_degradation_fit": log_fit,
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
