"""E08: Memory Unit Payload Size (B) Ablation.

Tests k/B Ratio Dependence and Hypotheses H3, H4:
- Evaluates B in [128, 256, 512, 1024, 2048] with fixed N=32, k=8, m=1, L_Q=64.
- Analyzes compression ratio scaling: as B increases, compression ratio increases and K* decreases.
- Measures retrieval accuracy and latency across varied block lengths.
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

from sempointer.cost_model import SemPointerCostModel
from sempointer.pointer_generator import PoolingPointerGenerator
from sempointer.registry import PointerRegistry
from sempointer.scorer import PointerScorer
from metrics.retrieval_metrics import recall_at_k

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_e08_cell(
    passages: List[Dict[str, Any]],
    N: int,
    B: int,
    k: int,
    m: int,
    L_Q: int,
    generator: PoolingPointerGenerator,
    device: str = "cpu",
    n_queries: int = 20,
) -> Dict[str, Any]:
    """Runs payload size ablation for a given block length B."""
    cost_model = SemPointerCostModel(d=4096, B=B, k=k, m=m, B_res=B, L_Q=L_Q, substrate="M_kv")
    k_star = cost_model.k_star(N)

    # Truncate or expand passages to approximately B words/tokens
    adjusted_passages = []
    for p in passages[:N]:
        text = p["text"]
        words = text.split()
        if len(words) < B:
            words = (words * (B // len(words) + 1))[:B]
        else:
            words = words[:B]
        adj_text = " ".join(words)
        adjusted_passages.append({"id": p["id"], "text": adj_text, "query": p["target_query"]})

    registry = PointerRegistry(k=k, B=B, max_size=max(2048, N + 10))
    for p in adjusted_passages:
        registry.register(p["text"], generator, metadata={"id": p["id"]})

    scorer = PointerScorer(generator, device=device)

    r1_list = []
    latencies = []
    test_queries = adjusted_passages[: min(n_queries, len(adjusted_passages))]
    for q_item in test_queries:
        query = q_item["query"]
        target_id = q_item["id"]

        start = time.perf_counter()
        ranked_ids, _ = scorer.score_all(query, registry)
        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000.0)
        r1_list.append(recall_at_k(ranked_ids, [target_id], k=1))

    linear_tokens = N * B + L_Q
    active_tokens = (N - m) * k + m * B + L_Q
    comp_ratio = linear_tokens / max(1, active_tokens)

    return {
        "B": B,
        "N": N,
        "k": k,
        "compression_ratio": float(comp_ratio),
        "k_star_theoretical": float(k_star),
        "recall_at_1": float(np.mean(r1_list)),
        "selector_latency_ms": float(np.mean(latencies)),
        "linear_tokens": linear_tokens,
        "active_tokens": active_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="E08: Payload Size Ablation")
    parser.add_argument("--config", type=str, default="config/e08_payload_size.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal B sweep.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    embed_model = config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    generator = PoolingPointerGenerator(model_name=embed_model, k=8, device=args.device)

    N = 32
    k = 8
    m = 1
    L_Q = 64

    if args.dry_run:
        B_values = [128, 512, 1024]
        n_queries = 10
    else:
        B_values = config.get("sweep", {}).get("B", [128, 256, 512, 1024, 2048])
        n_queries = 20

    from runners.e01_retrieval import generate_benchmark_passages
    passages = generate_benchmark_passages(N + 10, seed=seed)

    results = {}
    table = Table(title="E08: Memory Unit Payload Size (B) Ablation")
    table.add_column("B (tokens)", style="cyan")
    table.add_column("Comp. Ratio", style="bold green")
    table.add_column("K* Theory", style="bold magenta")
    table.add_column("Recall@1", style="yellow")
    table.add_column("Latency (ms)", style="white")

    for B in B_values:
        res = run_e08_cell(
            passages=passages,
            N=N,
            B=B,
            k=k,
            m=m,
            L_Q=L_Q,
            generator=generator,
            device=args.device,
            n_queries=n_queries,
        )
        results[f"B_{B}"] = res

        table.add_row(
            str(B),
            f"{res['compression_ratio']:.2f}x",
            f"{res['k_star_theoretical']:.4f}",
            f"{res['recall_at_1']:.3f}",
            f"{res['selector_latency_ms']:.2f}",
        )

    console.print(table)

    os.makedirs("results/e08", exist_ok=True)
    out_file = f"results/e08/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E08_Payload_Size_Ablation",
                "timestamp": time.time(),
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
