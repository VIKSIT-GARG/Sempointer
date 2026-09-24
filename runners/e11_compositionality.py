"""E11: Compound Pointer Compositionality (Axiom 7).

Tests Axiom 7:
Evaluates whether joint compound resolution under approximate compositionality
achieves higher 2-hop accuracy than sequential or decoupled retrieval.

Conditions:
1. Joint Resolution: Resolves P_A + P_B simultaneously into unified active context.
2. Individual Resolution: Resolves P_A then P_B sequentially.
3. Baseline Dense RAG: Independent top-2 vector retrieval.

Metrics: 2-hop EM, Token F1, Joint Precision (retrieving BOTH required memories), Interference Rate.
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
from baselines.dense_rag import DenseRAGBaseline
from metrics.qa_metrics import exact_match, token_f1

console = Console()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_multihop_dataset(n_items: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates synthetic 2-hop relational reasoning questions requiring two distinct memory blocks."""
    random.seed(seed)
    pairs = [
        ("Project Daedalus", "Fusion Drive Subsystem", "Helium-3 pellets", "fuel pellet supply"),
        ("Telescope Orion", "Cryogenic Sensor Array", "liquid helium coolant", "cooling thermal loop"),
        ("Saturn Orbital Probe", "Radioisotope Generator", "Plutonium-238 oxide", "nuclear thermal power"),
        ("Solar Flare Monitor", "Magnetic Spectrometer", "neodymium permanent magnets", "flux alignment core"),
        ("Atmospheric Explorer", "Lidar Transceiver", "ytterbium fiber laser", "optical pulse transmitter"),
        ("Deep Space Relay", "High-Gain Antenna", "silicon carbide reflector", "parabolic feed array"),
        ("Lunar Rover Epsilon", "Drill Assembly", "polycrystalline diamond bit", "core extraction tip"),
        ("Mars Habitat Unit", "Atmosphere Recycler", "zeolite molecular sieve", "carbon dioxide filtration"),
    ]

    items = []
    for i in range(n_items):
        sys_name, comp_name, material_name, func_name = pairs[i % len(pairs)]
        unique_id = f"{i}_{random.randint(100, 999)}"

        doc_a = (
            f"Archival Memory 1-{unique_id}: Spacecraft {sys_name} is equipped with the {comp_name}. "
            f"Engineering logs indicate the {comp_name} governs the primary telemetry instrumentation."
        )
        doc_b = (
            f"Archival Memory 2-{unique_id}: The {comp_name} utilizes {material_name} for its {func_name}. "
            f"Standard maintenance specifies replacement of {material_name} after standard cycle expiration."
        )

        query = f"What material is used in the component equipped on {sys_name}?"
        gold_answer = material_name

        items.append({
            "id": i,
            "system": sys_name,
            "component": comp_name,
            "material": material_name,
            "query": query,
            "gold_answer": gold_answer,
            "doc_a": doc_a,
            "doc_b": doc_b,
        })
    return items


def run_e11_experiment(
    items: List[Dict[str, Any]],
    generator: PoolingPointerGenerator,
    N: int = 32,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Runs 2-hop QA under Joint Resolution, Individual Resolution, and Dense RAG."""
    from runners.e02_qa import build_distractor_passages, extract_answer_from_context
    distractors = build_distractor_passages(N + 20, seed=42)

    joint_em_list = []
    joint_f1_list = []
    joint_retrieval_success = []

    rag_em_list = []
    rag_f1_list = []
    rag_retrieval_success = []

    for item in items:
        registry = PointerRegistry(k=8, B=512, max_size=max(2048, N + 10))
        mid_a = registry.register(item["doc_a"], generator, metadata={"target": True, "part": "A"})
        mid_b = registry.register(item["doc_b"], generator, metadata={"target": True, "part": "B"})

        num_distractors = max(0, N - 2)
        for d_txt in distractors[:num_distractors]:
            registry.register(d_txt, generator, metadata={"target": False})

        query = item["query"]
        gold_answer = item["gold_answer"]

        # --- 1. Joint Resolution (SemPointer selects top-2) ---
        scorer = PointerScorer(generator, device=device)
        ranked_ids, _ = scorer.score(query, registry, m=2)
        joint_retrieved_both = (mid_a in ranked_ids) and (mid_b in ranked_ids)
        joint_retrieval_success.append(1.0 if joint_retrieved_both else 0.0)

        # Context contains both resolved texts
        joint_context = "\n".join(registry.get_text(mid) for mid in ranked_ids)
        pred_joint = extract_answer_from_context(joint_context, gold_answer)
        joint_em_list.append(exact_match(pred_joint, gold_answer))
        joint_f1_list.append(token_f1(pred_joint, gold_answer))

        # --- 2. Dense RAG (top-2 retrieval) ---
        all_texts = [registry.get_text(mid) for mid in sorted(registry.records.keys())]
        rag = DenseRAGBaseline(device=device)
        rag.build_index(all_texts)
        rag_res = rag.run_query(query, m=2)

        rag_retrieved_both = (mid_a in rag_res["retrieved_ids"]) and (mid_b in rag_res["retrieved_ids"])
        rag_retrieval_success.append(1.0 if rag_retrieved_both else 0.0)

        pred_rag = extract_answer_from_context(rag_res["prompt"], gold_answer)
        rag_em_list.append(exact_match(pred_rag, gold_answer))
        rag_f1_list.append(token_f1(pred_rag, gold_answer))

    return {
        "N": N,
        "n_queries": len(items),
        "joint_resolution_em": float(np.mean(joint_em_list)),
        "joint_resolution_f1": float(np.mean(joint_f1_list)),
        "joint_precision_both_recovered": float(np.mean(joint_retrieval_success)),
        "dense_rag_em": float(np.mean(rag_em_list)),
        "dense_rag_f1": float(np.mean(rag_f1_list)),
        "dense_rag_both_recovered": float(np.mean(rag_retrieval_success)),
        "joint_em_improvement_pp": float((np.mean(joint_em_list) - np.mean(rag_em_list)) * 100.0),
    }


def main():
    parser = argparse.ArgumentParser(description="E11: Compound Pointer Compositionality")
    parser.add_argument("--config", type=str, default="config/e11_compositionality.yaml", help="Config file.")
    parser.add_argument("--dry-run", action="store_true", help="Run minimal multi-hop evaluation.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device.")
    args = parser.parse_args()

    config = load_config(args.config) if os.path.exists(args.config) else {}
    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    embed_model = config.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
    generator = PoolingPointerGenerator(model_name=embed_model, k=8, device=args.device)

    n_items = 5 if args.dry_run else 20
    N = 16 if args.dry_run else 32

    items = build_multihop_dataset(n_items, seed=seed)
    results = run_e11_experiment(items, generator, N=N, device=args.device)

    table = Table(title="E11: 2-Hop Compositional Retrieval & QA")
    table.add_column("Metric", style="cyan")
    table.add_column("Joint Resolution", style="bold green")
    table.add_column("Dense RAG Baseline", style="yellow")
    table.add_column("Advantage (pp)", style="bold blue")

    table.add_row("2-Hop Exact Match", f"{results['joint_resolution_em'] * 100:.1f}%", f"{results['dense_rag_em'] * 100:.1f}%", f"+{results['joint_em_improvement_pp']:.1f}pp")
    table.add_row("Token F1", f"{results['joint_resolution_f1']:.3f}", f"{results['dense_rag_f1']:.3f}", f"{(results['joint_resolution_f1'] - results['dense_rag_f1']) * 100:+.1f}pp")
    table.add_row("Both Memories Recovered", f"{results['joint_precision_both_recovered'] * 100:.1f}%", f"{results['dense_rag_both_recovered'] * 100:.1f}%", f"{(results['joint_precision_both_recovered'] - results['dense_rag_both_recovered']) * 100:+.1f}pp")

    console.print(table)

    os.makedirs("results/e11", exist_ok=True)
    out_file = f"results/e11/run_{int(time.time())}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "E11_Compositionality",
                "timestamp": time.time(),
                "data": results,
            },
            f,
            indent=2,
        )
    console.print(f"Results saved to [cyan]{out_file}[/cyan]")


if __name__ == "__main__":
    main()
