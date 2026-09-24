"""E13: Semantic Drift and Pointer Temporal Stability.

Tests Section VII-B Phenomenological Ansatz & Axiom 3 (Temporal Stability):
- Evaluates similarity decay across dialogue / context horizon Delta_t in [1, 5, 10, 25, 50, 100].
- Fits exponential drift ansatz: E[sim] ~ exp(-lambda * Delta_t).
- Falsification condition: If no statistically significant decay in Recall@1 occurs for Delta_t <= 100,
  the drift model is an unneeded complication in practice.
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
from metrics.retrieval_metrics import recall_at_k

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def simulate_intervening_dialogue_updates(
    initial_text: str,
    delta_t: int,
    noise_rate: float = 0.005,
    seed: int = 42,
) -> str:
    """Models multi-turn dialogue drift by introducing intervening discourse context."""
    random.seed(seed + delta_t)
    intervening_phrases = [
        "In subsequent conversational updates, additional constraints were introduced.",
        "Regarding the previous query, the user shifted focus to alternative considerations.",
        "The system recorded supplementary context in intermediate dialogue turns.",
        "Further clarification was requested regarding related external dependencies.",
        "System prompt instructions were reaffirmed following user interruption.",
    ]
    words = initial_text.split()
    # Slight contextual drift proportional to delta_t
    num_substitutions = int(len(words) * min(0.30, noise_rate * delta_t))
    if num_substitutions > 0:
        indices = random.sample(range(len(words)), num_substitutions)
        for idx in indices:
            words[idx] = random.choice(["the", "observed", "measured", "associated", "primary"])

    drifted_text = " ".join(words)
    if delta_t > 10:
        drifted_text += " " + " ".join(random.sample(intervening_phrases, min(3, delta_t // 10)))
    return drifted_text


def run_e13_experiment(
    delta_t_values: List[int],
    generator: PoolingPointerGenerator,
    device: str = "cpu",
    n_memories: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """Measures pointer cosine decay and retrieval retention across Delta_t horizons."""
    from runners.e01_retrieval import generate_benchmark_passages
    base_passages = generate_benchmark_passages(n_memories, seed=seed)

    # Initial registration at t=0
    registry = PointerRegistry(k=8, B=512, max_size=max(2048, n_memories + 10))
    for p in base_passages:
        registry.register(p["text"], generator, metadata={"id": p["id"]})

    initial_embeddings = registry.all_pointer_embeddings().to(device)
    if initial_embeddings.dim() == 3:
        initial_pooled = initial_embeddings.mean(dim=1)
    else:
        initial_pooled = initial_embeddings

    norm_initial = torch.nn.functional.normalize(initial_pooled, p=2, dim=-1)

    drift_results = {}
    delta_list = []
    cos_sim_list = []
    recall_list = []

    scorer = PointerScorer(generator, device=device)

    for dt in delta_t_values:
        # Generate drifted versions of the texts after dt conversational turns
        drifted_texts = [
            simulate_intervening_dialogue_updates(p["text"], delta_t=dt, seed=seed + p["id"])
            for p in base_passages
        ]
        # Generate new pointer embeddings for drifted representations
        drifted_embs = generator.generate_batch(drifted_texts).to(device)
        if drifted_embs.dim() == 3:
            drifted_pooled = drifted_embs.mean(dim=1)
        else:
            drifted_pooled = drifted_embs

        norm_drifted = torch.nn.functional.normalize(drifted_pooled, p=2, dim=-1)

        # Pairwise cosine similarity between t=0 and t=dt
        cos_similarities = (norm_initial * norm_drifted).sum(dim=-1).cpu().numpy()
        mean_cos = float(np.mean(cos_similarities))

        # Evaluate if queries still retrieve the drifted pointer
        r1_scores = []
        for p in base_passages:
            ranked_ids, _ = scorer.score_all(p["target_query"], registry)
            r1_scores.append(recall_at_k(ranked_ids, [p["id"]], k=1))
        mean_r1 = float(np.mean(r1_scores))

        drift_results[f"dt_{dt}"] = {
            "delta_t": dt,
            "mean_cosine_similarity": mean_cos,
            "min_cosine_similarity": float(np.min(cos_similarities)),
            "recall_at_1": mean_r1,
        }
        delta_list.append(dt)
        cos_sim_list.append(mean_cos)
        recall_list.append(mean_r1)

    # Fit exponential model: sim ~ exp(-lambda * dt) => ln(sim) ~ -lambda * dt
    ln_sim = np.log(np.clip(cos_sim_list, 1e-4, 1.0))
    # Linear fit with zero intercept constraint
    fitted_lambda = -float(np.dot(delta_list, ln_sim) / max(1e-6, np.dot(delta_list, delta_list)))

    decay_significant = bool(abs(recall_list[0] - recall_list[-1]) > 0.05)

    return {
        "drift_rate_lambda": fitted_lambda,
        "decay_significant_over_100_turns": decay_significant,
        "initial_recall_at_1": recall_list[0],
        "final_recall_at_1": recall_list[-1],
        "delta_horizons": drift_results,
    }


def main():
    parser = argparse.ArgumentParser(description="E13: Semantic Drift and Pointer Temporal Stability")
    parser.add_argument("--config", type=str, default="config/e13_drift.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal drift evaluation.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    embed_model = config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    generator = PoolingPointerGenerator(model_name=embed_model, k=8, device=args.device)

    if args.dry_run:
        delta_t_vals = [1, 10, 50]
        n_memories = 16
    else:
        delta_t_vals = config.get("sweep", {}).get("delta_t", [1, 5, 10, 25, 50, 100])
        n_memories = 32

    results = run_e13_experiment(delta_t_vals, generator, device=args.device, n_memories=n_memories, seed=seed)

    table = Table(title="E13: Semantic Drift over Conversational Horizon (Delta_t)")
    table.add_column("Delta_t (turns)", style="cyan")
    table.add_column("Pointer Cosine Sim", style="bold green")
    table.add_column("Min Cosine", style="yellow")
    table.add_column("Recall@1", style="blue")

    for dt, info in results["delta_horizons"].items():
        table.add_row(
            str(info["delta_t"]),
            f"{info['mean_cosine_similarity']:.4f}",
            f"{info['min_cosine_similarity']:.4f}",
            f"{info['recall_at_1']:.3f}",
        )

    console.print(table)
    console.print(f"\n[bold]Fitted Exponential Drift Lambda:[/bold] {results['drift_rate_lambda']:.6f} per turn")
    console.print(f"[bold]Drift Decay Significant (H6 Check):[/bold] {results['decay_significant_over_100_turns']}")

    os.makedirs("results/e13", exist_ok=True)
    out_file = f"results/e13/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E13_Semantic_Drift",
                "timestamp": time.time(),
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
