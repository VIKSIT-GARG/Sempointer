"""E09: Number of Selected Pointers (m) Ablation.

Tests Axiom 5 (Sparsity) and Hypotheses H2, H4:
- Evaluates m in [1, 2, 4, 8, 16] with fixed N=64, k=8, B=512, L_Q=64.
- Verifies active sequence scaling: L_active = (N - m)*k + m*B_res + L_Q.
- Evaluates whether accuracy plateaus when m >= m_true on multi-target queries.
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
from sempointer.cost_model import SemPointerCostModel
from metrics.retrieval_metrics import recall_at_k

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_composite_items(n_items: int, m_true: int = 2, seed: int = 42) -> List[Dict[str, Any]]:
    """Builds test items where answering the query requires exactly m_true memory blocks."""
    random.seed(seed)
    items = []
    for i in range(n_items):
        topic = f"Mission_Subsystem_{i}"
        part1 = f"In {topic}, Component Alpha controls primary propulsion ignition."
        part2 = f"In {topic}, Component Beta regulates reactant chamber pressure."
        query = f"What are the functions of Component Alpha and Component Beta in {topic}?"

        items.append({
            "id": i,
            "topic": topic,
            "query": query,
            "passages": [part1, part2],
            "m_true": m_true,
        })
    return items


def run_e09_cell(
    composite_items: List[Dict[str, Any]],
    distractor_texts: List[str],
    N: int,
    m: int,
    k: int,
    B: int,
    L_Q: int,
    generator: PoolingPointerGenerator,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluates multi-target recall and active length scaling for selection width m."""
    cost_model = SemPointerCostModel(d=4096, B=B, k=k, m=m, B_res=B, L_Q=L_Q, substrate="M_kv")
    pred_l_active = cost_model.l_active(N)

    scorer = PointerScorer(generator, device=device)
    completeness_scores = []
    latencies = []

    for item in composite_items:
        registry = PointerRegistry(k=k, B=B, max_size=max(2048, N + 10))
        target_ids = []
        for p_txt in item["passages"]:
            mid = registry.register(p_txt, generator, metadata={"target": True})
            target_ids.append(mid)

        num_distractors_needed = max(0, N - len(target_ids))
        for d_txt in distractor_texts[:num_distractors_needed]:
            registry.register(d_txt, generator, metadata={"target": False})

        start = time.perf_counter()
        ranked_ids, _ = scorer.score(item["query"], registry, m=m)
        latencies.append((time.perf_counter() - start) * 1000.0)

        # Fraction of required target IDs retrieved in top-m
        retrieved_targets = sum(1 for tid in target_ids if tid in ranked_ids)
        completeness = retrieved_targets / len(target_ids)
        completeness_scores.append(completeness)

    mean_completeness = float(np.mean(completeness_scores))
    mean_lat = float(np.mean(latencies))
    c_query = cost_model.c_query_sempointer(N)

    return {
        "m": m,
        "N": N,
        "target_completeness": mean_completeness,
        "l_active_predicted": pred_l_active,
        "c_query_gflops": c_query / 1e9,
        "latency_ms": mean_lat,
        "plateau_reached": bool(m >= 2 and mean_completeness >= 0.85),
    }


def main():
    parser = argparse.ArgumentParser(description="E09: Number of Selected Pointers (m) Ablation")
    parser.add_argument("--config", type=str, default="config/e09_selected_pointers.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal m sweep.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    embed_model = config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    generator = PoolingPointerGenerator(model_name=embed_model, k=8, device=args.device)

    N = 64
    k = 8
    B = 512
    L_Q = 64

    if args.dry_run:
        m_values = [1, 2, 4]
        n_items = 5
    else:
        m_values = config.get("sweep", {}).get("m", [1, 2, 4, 8, 16])
        n_items = 15

    from runners.e02_qa import build_distractor_passages
    composite_items = build_composite_items(n_items, m_true=2, seed=seed)
    distractors = build_distractor_passages(N + 10, seed=seed + 1)

    results = {}
    table = Table(title="E09: Number of Selected Pointers (m) Ablation")
    table.add_column("m", style="cyan")
    table.add_column("Target Completeness (m_true=2)", style="bold green")
    table.add_column("L_active (tok)", style="yellow")
    table.add_column("C_query (GFLOPs)", style="blue")
    table.add_column("Latency (ms)", style="white")

    for m in m_values:
        res = run_e09_cell(
            composite_items=composite_items,
            distractor_texts=distractors,
            N=N,
            m=m,
            k=k,
            B=B,
            L_Q=L_Q,
            generator=generator,
            device=args.device,
        )
        results[f"m_{m}"] = res

        table.add_row(
            str(m),
            f"{res['target_completeness'] * 100:.1f}%",
            str(res["l_active_predicted"]),
            f"{res['c_query_gflops']:.2f}",
            f"{res['latency_ms']:.2f}",
        )

    console.print(table)

    os.makedirs("results/e09", exist_ok=True)
    out_file = f"results/e09/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E09_Selected_Pointers_Ablation",
                "timestamp": time.time(),
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
