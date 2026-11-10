import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict
import time
from .registry import PointerRegistry

class PointerScorer:
    def __init__(self, generator, device: str = 'cuda'):
        self.generator = generator
        self.device = device

    def score(self, query: str, registry: PointerRegistry, m: int) -> Tuple[List[int], torch.Tensor]:
        memory_ids, scores = self.score_all(query, registry)
        if m >= len(memory_ids):
            return memory_ids, scores
        return memory_ids[:m], scores[:m]

    def score_all(self, query: str, registry: PointerRegistry) -> Tuple[List[int], torch.Tensor]:
        if len(registry) == 0:
            return [], torch.empty(0)

        query_emb = self.generator.generate(query).to(self.device)
        # Using the first token or mean pool for query representation
        if query_emb.dim() == 2:
            query_repr = query_emb.mean(dim=0)
        else:
            query_repr = query_emb
            
        memory_ids = sorted(registry.records.keys())
        # shape: (N, k, d)
        all_embs = registry.all_pointer_embeddings().to(self.device)
        
        # mean pool pointers
        if all_embs.dim() == 3:
            all_reprs = all_embs.mean(dim=1)
        else:
            all_reprs = all_embs

        scores = F.cosine_similarity(query_repr.unsqueeze(0), all_reprs, dim=1)
        
        sorted_scores, indices = torch.sort(scores, descending=True)
        sorted_memory_ids = [memory_ids[idx.item()] for idx in indices]
        
        return sorted_memory_ids, sorted_scores

    def measure_latency(self, query: str, registry: PointerRegistry, n_trials: int = 50) -> Dict:
        latencies = []
        for _ in range(n_trials):
            start = time.perf_counter()
            self.score_all(query, registry)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)
            
        import numpy as np
        return {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies))
        }
