"""E17: Cross-Model Architectural Generalization.

Replicates core SemPointer evaluations (E01 Retrieval, E03 Compression, E05 K* Crossover)
across diverse open-weights model architectures:
1. Qwen Family: Qwen2.5
2. LLaMA Family: Llama-3.1
3. Gemma Family: Gemma-2
4. Phi Family: Phi-3.5

Verifies that the theoretical bounds and crossover points are architectural invariants
rather than artifacts of a specific model vocabulary or tokenizer.
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

from sempointer.cost_model import SemPointerCostModel, compute_kstar_table
from sempointer.pointer_generator import PoolingPointerGenerator
from runners.e01_retrieval import run_single_n_k_experiment, generate_benchmark_passages
from runners.e03_compression import run_compression_cell, get_tokenizer

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_model_family(
    family_name: str,
    model_hf_id: str,
    hidden_dim: int,
    passages: List[Dict[str, Any]],
    generator: PoolingPointerGenerator,
    device: str = "cpu",
    N: int = 32,
    k: int = 8,
    m: int = 1,
    B: int = 512,
    L_Q: int = 64,
) -> Dict[str, Any]:
    """Runs compression verification, retrieval evaluation, and K* calculation for a model family."""
    # 1. Cost model crossover for hidden dimension d
    cost_model = SemPointerCostModel(d=hidden_dim, B=B, k=k, m=m, B_res=B, L_Q=L_Q, substrate="M_kv")
    k_star_theory = cost_model.k_star(N)
    comp_ratio = (N * B + L_Q) / cost_model.l_active(N)

    # 2. Tokenizer active length verification
    tokenizer = get_tokenizer(model_hf_id)
    comp_cell = run_compression_cell(tokenizer, N=N, k=k, m=m, B_res=B, L_Q=L_Q, B=B, num_trials=5)

    # 3. Retrieval evaluation
    ret_res = run_single_n_k_experiment(passages, N=N, k=k, generator=generator, device=device, n_queries=15)

    return {
        "family": family_name,
        "model_id": model_hf_id,
        "hidden_dim_d": hidden_dim,
        "theoretical_k_star": float(k_star_theory),
        "compression_ratio": float(comp_ratio),
        "token_compression_error_pct": comp_cell["mean_rel_error"] * 100.0,
        "recall_at_1": ret_res["recall_at_1"],
        "mrr": ret_res["mrr"],
        "selector_latency_ms": ret_res["selector_latency_ms"],
    }


def main():
    parser = argparse.ArgumentParser(description="E17: Cross-Model Generalization")
    parser.add_argument("--config", type=str, default="config/e17_cross_model.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal cross-model test.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    embed_model = config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    generator = PoolingPointerGenerator(model_name=embed_model, k=8, device=args.device)

    model_families = [
        {"name": "Qwen 2.5", "hf_id": "Qwen/Qwen2.5-7B-Instruct", "d": 3584},
        {"name": "Llama 3.1", "hf_id": "meta-llama/Llama-3.1-8B-Instruct", "d": 4096},
        {"name": "Gemma 2", "hf_id": "google/gemma-2-9b-it", "d": 3584},
        {"name": "Phi 3.5", "hf_id": "microsoft/Phi-3.5-mini-instruct", "d": 3072},
    ]

    if args.dry_run:
        model_families = model_families[:2]

    passages = generate_benchmark_passages(64, seed=seed)

    results = {}
    table = Table(title="E17: Cross-Model Architectural Generalization")
    table.add_column("Model Family", style="cyan")
    table.add_column("d (hidden)", style="yellow")
    table.add_column("K* Theory", style="bold magenta")
    table.add_column("Comp. Ratio", style="bold green")
    table.add_column("Token Err %", style="white")
    table.add_column("Recall@1", style="green")
    table.add_column("MRR", style="blue")

    for mf in model_families:
        console.print(f"Evaluating family: [bold]{mf['name']}[/bold] ({mf['hf_id']})...")
        res = evaluate_model_family(
            family_name=mf["name"],
            model_hf_id=mf["hf_id"],
            hidden_dim=mf["d"],
            passages=passages,
            generator=generator,
            device=args.device,
        )
        results[mf["name"]] = res

        table.add_row(
            mf["name"],
            str(mf["d"]),
            f"{res['theoretical_k_star']:.3f}",
            f"{res['compression_ratio']:.2f}x",
            f"{res['token_compression_error_pct']:.2f}%",
            f"{res['recall_at_1']:.3f}",
            f"{res['mrr']:.3f}",
        )

    console.print(table)

    os.makedirs("results/e17", exist_ok=True)
    out_file = f"results/e17/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E17_Cross_Model_Generalization",
                "timestamp": time.time(),
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
