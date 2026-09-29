"""E12: Pointer Address Collision and Generalized Birthday Bound.

Tests Proposition 3: Address Collision Probability
P_collision(N) ~ 1 - exp(- N^2 / (2 * |P|))

Evaluates empirical pairwise cosine similarities over N in [8, 16, 32, 64, 128, 256]
across difficulty conditions (Easy, Medium, Hard Near-Paraphrase).
Compares observed collisions against analytical birthday bound.
"""

import argparse
import json
import os
import random
import time
import math
from typing import Dict, List, Any
import numpy as np
import torch
import yaml
from rich.console import Console
from rich.table import Table

from sempointer.pointer_generator import PoolingPointerGenerator
from sempointer.registry import PointerRegistry
from metrics.retrieval_metrics import collision_rate, pairwise_cosine_similarities

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_difficulty_passages(N: int, difficulty: str, seed: int = 42) -> List[str]:
    """Synthesizes text sets with controlled semantic overlap."""
    random.seed(seed)
    if difficulty == "easy":
        # Broadly divergent scientific domains
        from runners.e01_retrieval import generate_benchmark_passages
        raw = generate_benchmark_passages(N, seed=seed)
        return [r["text"] for r in raw]
    elif difficulty == "medium":
        # Same broad domain (Astrophysics), different sub-topics
        topics = ["exoplanet atmospheric spectra", "pulsar timing arrays", "quasar emission lines", "stellar nucleosynthesis"]
        passages = []
        for i in range(N):
            t = topics[i % len(topics)]
            passages.append(f"Astrophysics report {i}: Detailed observational analysis of {t} demonstrates consistent luminosity calibration.")
        return passages
    elif difficulty == "hard":
        # Near-paraphrase adversarial conditions
        passages = []
        base = "The experimental reactor sustained fusion plasma equilibrium at"
        for i in range(N):
            temp = 100 + (i % 10)
            passages.append(f"{base} approximately {temp} million Kelvin under high magnetic confinement.")
        return passages
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")


def run_e12_cell(
    passages: List[str],
    N: int,
    k: int,
    threshold: float,
    generator: PoolingPointerGenerator,
) -> Dict[str, Any]:
    """Computes exact pairwise similarity distribution and collision rate."""
    selected = passages[:N]
    registry = PointerRegistry(k=k, B=512, max_size=max(2048, N + 10))
    for text in selected:
        registry.register(text, generator)

    all_embs = registry.all_pointer_embeddings()
    if all_embs.dim() == 3:
        pooled = all_embs.mean(dim=1)
    else:
        pooled = all_embs

    # Exact pairwise cosine similarities
    similarities = pairwise_cosine_similarities(pooled)
    total_pairs = similarities.numel()
    collisions = int((similarities > threshold).sum().item())
    emp_collision_rate = collisions / max(1, total_pairs)

    # Theoretical birthday bound approximation: P ~ 1 - exp(- N^2 / (2 * V_eff))
    # For unit sphere S^{d-1}, effective volume V_eff is very large for independent random draws
    d = pooled.shape[-1]
    # Cap exponent calculation safely
    theoretical_birthday_p = 1.0 - math.exp(-min(50.0, (N ** 2) / (2.0 * (2.0 ** min(d, 32)))))

    return {
        "N": N,
        "total_pairs": total_pairs,
        "collision_count": collisions,
        "empirical_collision_rate": float(emp_collision_rate),
        "mean_cosine_similarity": float(similarities.mean().item()),
        "max_cosine_similarity": float(similarities.max().item()),
        "std_cosine_similarity": float(similarities.std().item()),
        "theoretical_uniform_p": float(theoretical_birthday_p),
    }


def main():
    parser = argparse.ArgumentParser(description="E12: Pointer Address Collision")
    parser.add_argument("--config", type=str, default="config/e12_collision.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal collision test.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    embed_model = config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    generator = PoolingPointerGenerator(model_name=embed_model, k=8, device=args.device)

    threshold = 0.95
    if args.dry_run:
        N_vals = [8, 16, 32]
        difficulties = ["easy", "hard"]
    else:
        N_vals = config.get("sweep", {}).get("N", [8, 16, 32, 64, 128])
        difficulties = ["easy", "medium", "hard"]

    results = {}
    table = Table(title="E12: Pointer Collision Rates vs Difficulty")
    table.add_column("Difficulty", style="cyan")
    table.add_column("N", style="magenta")
    table.add_column("Total Pairs", style="yellow")
    table.add_column("Collisions (>0.95)", style="red")
    table.add_column("Collision Rate", style="bold green")
    table.add_column("Max Cosine", style="blue")

    for diff in difficulties:
        results[diff] = {}
        passages = build_difficulty_passages(max(N_vals) + 10, difficulty=diff, seed=seed)
        for N in N_vals:
            res = run_e12_cell(passages, N=N, k=8, threshold=threshold, generator=generator)
            results[diff][f"N_{N}"] = res

            table.add_row(
                diff.upper(),
                str(N),
                str(res["total_pairs"]),
                str(res["collision_count"]),
                f"{res['empirical_collision_rate'] * 100:.2f}%",
                f"{res['max_cosine_similarity']:.4f}",
            )

    console.print(table)

    os.makedirs("results/e12", exist_ok=True)
    out_file = f"results/e12/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E12_Pointer_Collision",
                "timestamp": time.time(),
                "threshold": threshold,
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
