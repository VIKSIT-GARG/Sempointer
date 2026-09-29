"""E07: Pointer Length (k) Ablation.

Tests Axioms 1 & 2, Proposition 4 (k >= 4 bound), and Hypothesis H5:
- H5 (Compactness-Fidelity Tradeoff): Monotonic increase of Recall with k,
  saturating at k_crit <= 8 for B=512.
- Evaluates k in [2, 4, 8, 16, 32, 64] with fixed N=64, B=512, m=1.
- Falsification condition: Non-monotone Recall vs k across k in [2, 4, 8, 16].
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
from metrics.retrieval_metrics import recall_at_k, collision_rate
from stats.statistical_tests import bootstrap_ci

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_e07_cell(
    passages: List[Dict[str, Any]],
    N: int,
    k: int,
    generator: PoolingPointerGenerator,
    device: str = "cpu",
    n_queries: int = 25,
) -> Dict[str, Any]:
    """Evaluates retrieval and collision properties for pointer length k."""
    registry = PointerRegistry(k=k, B=512, max_size=max(2048, N + 10))
    selected = passages[:N]
    for p in selected:
        registry.register(p["text"], generator, metadata={"id": p["id"]})

    scorer = PointerScorer(generator, device=device)

    r1_list = []
    latencies = []

    test_queries = selected[: min(n_queries, len(selected))]
    for q_item in test_queries:
        query = q_item["target_query"]
        target_id = q_item["id"]

        start = time.perf_counter()
        ranked_ids, _ = scorer.score_all(query, registry)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)

        r1_list.append(recall_at_k(ranked_ids, [target_id], k=1))

    all_embs = registry.all_pointer_embeddings()
    if all_embs.dim() == 3:
        pooled = all_embs.mean(dim=1)
    else:
        pooled = all_embs
    coll = collision_rate(pooled, threshold=0.95)

    r1_mean, r1_low, r1_high = bootstrap_ci(r1_list, n_bootstrap=1000)
    l_active = (N - 1) * k + 1 * 512 + 64

    return {
        "k": k,
        "N": N,
        "recall_at_1": float(r1_mean),
        "recall_at_1_ci": [float(r1_low), float(r1_high)],
        "collision_rate": float(coll),
        "l_active_tokens": l_active,
        "selector_latency_ms": float(np.mean(latencies)),
    }


def main():
    parser = argparse.ArgumentParser(description="E07: Pointer Length Ablation")
    parser.add_argument("--config", type=str, default="config/e07_pointer_length.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal k sweep.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    embed_model = config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    generator = PoolingPointerGenerator(model_name=embed_model, k=8, device=args.device)

    N = 64
    if args.dry_run:
        k_values = [4, 8, 16]
        n_queries = 10
    else:
        k_values = config.get("sweep", {}).get("k", [2, 4, 8, 16, 32, 64])
        n_queries = 25

    from runners.e01_retrieval import generate_benchmark_passages
    passages = generate_benchmark_passages(N + 10, seed=seed)

    results = {}
    table = Table(title="E07: Pointer Length (k) Ablation Results")
    table.add_column("k", style="cyan")
    table.add_column("Recall@1", style="bold green")
    table.add_column("Collision Rate", style="yellow")
    table.add_column("L_active (tok)", style="blue")
    table.add_column("Latency (ms)", style="white")

    r1_prev = 0.0
    is_monotone = True

    for k in k_values:
        res = run_e07_cell(
            passages=passages, N=N, k=k, generator=generator, device=args.device, n_queries=n_queries
        )
        results[f"k_{k}"] = res

        if res["recall_at_1"] < r1_prev - 0.05: # allow small statistical noise margin
            is_monotone = False
        r1_prev = res["recall_at_1"]

        table.add_row(
            str(k),
            f"{res['recall_at_1']:.3f}",
            f"{res['collision_rate']:.4f}",
            str(res["l_active_tokens"]),
            f"{res['selector_latency_ms']:.2f}",
        )

    console.print(table)
    console.print(f"\n[bold]H5 Monotonicity Check:[/bold] {'PASSED (monotone)' if is_monotone else 'FAILED (non-monotone)'}")

    os.makedirs("results/e07", exist_ok=True)
    out_file = f"results/e07/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E07_Pointer_Length_Ablation",
                "timestamp": time.time(),
                "is_monotone": is_monotone,
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
