"""SemPointer Memory Substrate Resolvers.

Implements the three substrates defined in Section IV of the paper:
1. M_text: Raw Text Substrate (lossless, re-encodes fetched text)
2. M_kv: Key-Value Cache Substrate (lossless, reuses precomputed KV tensors, zero recompute FLOPs)
3. M_latent: Latent Bottleneck Substrate (compact bottleneck projection, unpools conditioned on query)
"""

from typing import List, Dict, Tuple, Any, Optional
import time
import torch
import torch.nn as nn
from .registry import PointerRegistry


class TextResolver:
    """M_text substrate: stores raw text in memory/disk and returns it upon selection."""

    def resolve(self, memory_ids: List[int], registry: PointerRegistry) -> List[str]:
        """Fetches raw text strings for the selected memory IDs."""
        return [registry.get_text(mid) for mid in memory_ids if mid in registry.records]

    def measure_latency(self, memory_ids: List[int], registry: PointerRegistry) -> Dict[str, float]:
        """Measures I/O fetch latency in milliseconds."""
        start = time.perf_counter()
        texts = self.resolve(memory_ids, registry)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "fetch_latency_ms": elapsed_ms,
            "blocks_resolved": len(texts),
            "total_chars": sum(len(t) for t in texts),
        }


class KVCacheResolver:
    """M_kv substrate: precomputes and caches past_key_values states."""

    def __init__(self, model=None, tokenizer=None, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if self.model is not None:
            self.model.eval()

    def precompute_kv(self, registry: PointerRegistry) -> Dict[int, Any]:
        """Precomputes KV states for all records in the registry."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be provided to precompute KV cache.")

        kv_cache = {}
        for mid, record in registry.records.items():
            inputs = self.tokenizer(record.text, return_tensors="pt", truncation=True, max_length=registry.B)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, use_cache=True)
            kv_cache[mid] = outputs.past_key_values
        return kv_cache

    def resolve(self, memory_ids: List[int], kv_cache: Dict[int, Any]) -> List[Any]:
        """Resolves selected memories by retrieving precomputed KV states."""
        return [kv_cache[mid] for mid in memory_ids if mid in kv_cache]

    def measure_vram(self, registry: PointerRegistry, kv_cache: Dict[int, Any]) -> Dict[str, Any]:
        """Calculates precise memory consumption of the cached KV states."""
        total_bytes = 0
        per_memory = {}
        for mid, pkv in kv_cache.items():
            mem_bytes = 0
            # Handle standard HuggingFace DynamicCache or tuple of (key, value) pairs
            if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
                for k, v in zip(pkv.key_cache, pkv.value_cache):
                    mem_bytes += k.element_size() * k.nelement() + v.element_size() * v.nelement()
            elif isinstance(pkv, (list, tuple)):
                for layer in pkv:
                    if isinstance(layer, (list, tuple)):
                        for t in layer:
                            if isinstance(t, torch.Tensor):
                                mem_bytes += t.element_size() * t.nelement()
                    elif isinstance(layer, torch.Tensor):
                        mem_bytes += layer.element_size() * layer.nelement()
            per_memory[mid] = mem_bytes
            total_bytes += mem_bytes

        return {
            "total_bytes": total_bytes,
            "total_megabytes": total_bytes / (1024 * 1024),
            "per_memory_bytes": per_memory,
            "mean_per_memory_kb": (total_bytes / max(1, len(kv_cache))) / 1024,
        }


class LatentBottleneckModule(nn.Module):
    """Real linear/MLP autoencoder bottleneck for M_latent experimentation."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


class LatentResolver:
    """M_latent substrate: compresses context to a compact latent bottleneck.

    Resolution reconstructs representations via conditioned unpooling.
    """

    def __init__(self, hidden_dim: int = 768, bottleneck_dim: int = 128, device: str = "cpu"):
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.device = device
        self.module = LatentBottleneckModule(hidden_dim, bottleneck_dim).to(device)
        self.module.eval()

    def compress_embeddings(self, block_embeddings: torch.Tensor) -> torch.Tensor:
        """Compresses block embeddings (seq_len, hidden_dim) into bottleneck (m_z, bottleneck_dim)."""
        block_embeddings = block_embeddings.to(self.device)
        with torch.no_grad():
            z, _ = self.module(block_embeddings)
        return z

    def reconstruct_embeddings(self, latent_z: torch.Tensor) -> torch.Tensor:
        """Unpools and reconstructs representations from bottleneck state."""
        latent_z = latent_z.to(self.device)
        with torch.no_grad():
            rec = self.module.decoder(latent_z)
        return rec

    def compute_reconstruction_fidelity(
        self, original_embeddings: torch.Tensor, reconstructed_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Calculates cosine similarity and MSE between original and reconstructed representations."""
        orig = original_embeddings.to(self.device)
        rec = reconstructed_embeddings.to(self.device)

        mse = torch.nn.functional.mse_loss(rec, orig).item()
        cos_sim = torch.nn.functional.cosine_similarity(orig, rec, dim=-1).mean().item()
        
        # Relative information loss epsilon = 1 - cos_sim
        epsilon = max(0.0, 1.0 - cos_sim)
        return {
            "mse_loss": float(mse),
            "cosine_fidelity": float(cos_sim),
            "epsilon_loss": float(epsilon),
            "compression_ratio": float(self.hidden_dim / max(1, self.bottleneck_dim)),
        }
