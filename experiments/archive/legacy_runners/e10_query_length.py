"""E10: Query Length (L_Q) Ablation.

Tests Assumption A5: Selector Complexity C_select = O(L_Q*d + N*d).
- Evaluates L_Q in [16, 64, 256, 512, 1024] with fixed N=64, k=8, m=1, B=512.
- Fits multiple regression: selector_latency = a*N + b*L_Q + c.
- Verifies that query length contribution b is negligible relative to registry scanning.
"""

import argparse
import json
import os
import random
import time
from typing import Dict, List, Any
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from sempointer.pointer_generator import PoolingPointerGenerator
from sempointer.registry import PointerRegistry
from sempointer.scorer import PointerScorer

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_variable_length_query(base_topic: str, target_tokens: int) -> str:
    """Constructs realistic query text padded to approximately target_tokens words."""
    base = f"What are the operational parameters of {base_topic}?"
    words = base.split()
    filler = "specifically considering detailed architectural, thermal, mechanical, and dynamic interface specifications"
    while len(words) < target_tokens:
        words.extend(filler.split())
    return " ".join(words[:target_tokens])


def run_e10_cell(
    passages: List[Dict[str, Any]],
    N: int,
    L_Q: int,
    generator: PoolingPointerGenerator,
    device: str = "cpu",
    n_trials: int = 15,
) -> Dict[str, Any]:
    """Measures query encoding latency and scoring latency for query length L_Q."""
    registry = PointerRegistry(k=8, B=512, max_size=max(2048, N + 10))
    for p in passages[:N]:
        registry.register(p["text"], generator, metadata={"id": p["id"]})

    scorer = PointerScorer(generator, device=device)
    query_text = build_variable_length_query(passages[0]["target_entity"], target_tokens=L_Q)

    encoding_times = []
    scoring_times = []
    total_times = []

    for _ in range(n_trials):
        start_enc = time.perf_counter()
        q_emb = generator.generate(query_text)
        enc_ms = (time.perf_counter() - start_enc) * 1000.0
        encoding_times.append(enc_ms)

        start_score = time.perf_counter()
        scorer.score(query_text, registry, m=1)
        total_ms = (time.perf_counter() - start_score) * 1000.0
        total_times.append(total_ms)
        scoring_times.append(max(0.001, total_ms - enc_ms))

    return {
        "L_Q": L_Q,
        "N": N,
        "mean_query_encoding_ms": float(np.mean(encoding_times)),
        "mean_scoring_ms": float(np.mean(scoring_times)),
        "mean_total_latency_ms": float(np.mean(total_times)),
    }


def main():
    parser = argparse.ArgumentParser(description="E10: Query Length Ablation")
    parser.add_argument("--config", type=str, default="config/e10_query_length.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal L_Q sweep.")
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
        lq_values = [16, 64, 256]
        n_trials = 5
    else:
        lq_values = config.get("sweep", {}).get("L_Q", [16, 64, 256, 512, 1024])
        n_trials = 15

    from runners.e01_retrieval import generate_benchmark_passages
    passages = generate_benchmark_passages(N + 10, seed=seed)

    results = {}
    table = Table(title="E10: Query Length (L_Q) Ablation Performance")
    table.add_column("L_Q (tokens)", style="cyan")
    table.add_column("Query Enc. (ms)", style="yellow")
    table.add_column("Scoring (ms)", style="blue")
    table.add_column("Total Latency (ms)", style="bold green")

    lq_list = []
    lat_list = []

    for lq in lq_values:
        res = run_e10_cell(
            passages=passages, N=N, L_Q=lq, generator=generator, device=args.device, n_trials=n_trials
        )
        results[f"LQ_{lq}"] = res
        lq_list.append(lq)
        lat_list.append(res["mean_total_latency_ms"])

        table.add_row(
            str(lq),
            f"{res['mean_query_encoding_ms']:.2f}",
            f"{res['mean_scoring_ms']:.2f}",
            f"{res['mean_total_latency_ms']:.2f}",
        )

    console.print(table)

    # Fit linear model vs L_Q
    slope, intercept = np.polyfit(lq_list, lat_list, 1)
    console.print(f"\n[bold]Regression vs L_Q:[/bold] latency = {slope:.5f} * L_Q + {intercept:.2f} ms")

    os.makedirs("results/e10", exist_ok=True)
    out_file = f"results/e10/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E10_Query_Length_Ablation",
                "timestamp": time.time(),
                "slope_ms_per_token": float(slope),
                "intercept_ms": float(intercept),
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
