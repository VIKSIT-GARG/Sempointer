import torch
from typing import List, Dict
import math

def recall_at_k(ranked_ids: List[int], target_ids: List[int], k: int) -> float:
    """Fraction of queries where ANY target_id appears in top-k."""
    top_k = ranked_ids[:k]
    for tid in target_ids:
        if tid in top_k:
            return 1.0
    return 0.0

def mean_reciprocal_rank(ranked_ids: List[int], target_ids: List[int]) -> float:
    """MRR: 1/rank of first correct."""
    for i, rid in enumerate(ranked_ids):
        if rid in target_ids:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(ranked_ids: List[int], target_ids: List[int], k: int = 10) -> float:
    """nDCG@k with binary relevance."""
    dcg = 0.0
    idcg = 0.0
    for i in range(min(k, len(ranked_ids))):
        if ranked_ids[i] in target_ids:
            dcg += 1.0 / math.log2(i + 2)
    for i in range(min(k, len(target_ids))):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def false_positive_rate(ranked_ids: List[int], target_ids: List[int]) -> float:
    """Fraction where rank-1 is non-target."""
    if not ranked_ids:
        return 0.0
    return 1.0 if ranked_ids[0] not in target_ids else 0.0

def collision_rate(pointer_embeddings: torch.Tensor, threshold: float = 0.95) -> float:
    """Fraction of pointer pairs with cosine similarity > threshold."""
    n = pointer_embeddings.size(0)
    if n <= 1:
        return 0.0
    
    norms = torch.nn.functional.normalize(pointer_embeddings, p=2, dim=-1)
    sim_matrix = torch.matmul(norms, norms.T)
    
    triu_indices = torch.triu_indices(n, n, offset=1)
    similarities = sim_matrix[triu_indices[0], triu_indices[1]]
    
    collisions = (similarities > threshold).sum().item()
    total_pairs = similarities.size(0)
    
    return collisions / total_pairs if total_pairs > 0 else 0.0

def pairwise_cosine_similarities(embeddings: torch.Tensor) -> torch.Tensor:
    """Returns upper-triangle similarities."""
    n = embeddings.size(0)
    norms = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    sim_matrix = torch.matmul(norms, norms.T)
    triu_indices = torch.triu_indices(n, n, offset=1)
    return sim_matrix[triu_indices[0], triu_indices[1]]

def compute_retrieval_metrics_batch(
    ranked_ids_list: List[List[int]],
    target_ids_list: List[List[int]],
    k_values: List[int] = None
) -> Dict[str, float]:
    """Compute all retrieval metrics over a batch."""
    if k_values is None:
        k_values = [1, 5, 10]
        
    num_queries = len(ranked_ids_list)
    if num_queries == 0:
        return {}
        
    metrics = {f'recall@{k}': 0.0 for k in k_values}
    metrics['mrr'] = 0.0
    metrics['ndcg@10'] = 0.0
    metrics['fpr'] = 0.0
    
    for ranked, targets in zip(ranked_ids_list, target_ids_list):
        for k in k_values:
            metrics[f'recall@{k}'] += recall_at_k(ranked, targets, k)
        metrics['mrr'] += mean_reciprocal_rank(ranked, targets)
        metrics['ndcg@10'] += ndcg_at_k(ranked, targets, 10)
        metrics['fpr'] += false_positive_rate(ranked, targets)
        
    for k in metrics:
        metrics[k] /= num_queries
        
    return metrics
