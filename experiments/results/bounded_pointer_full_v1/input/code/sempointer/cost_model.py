"""SemPointer Analytical Cost Model.

Grounded in SemPointer paper (Section VI, Propositions 1-2, Corollary 1).
Provides exact FLOP counts for:
- Linear Re-Ingestion (C_linear)
- SemPointer Initialization (C_init)
- Query-time Selection (C_select)
- Active Sequence Attention (C_query)
- Per-query Savings (Delta_C_step)
- Indirection Crossover Threshold (K*)
"""

from typing import Optional, List, Dict, Any
import pandas as pd


class SemPointerCostModel:
    """Exact computational complexity model for SemPointer indirection."""

    def __init__(
        self,
        d: int = 4096,
        B: int = 512,
        k: int = 8,
        m: int = 1,
        B_res: int = 512,
        L_Q: int = 64,
        generator_class: str = "pooling",  # "pooling" or "attention"
        substrate: str = "M_kv",           # "M_text", "M_kv", or "M_latent"
        c_g: float = 1.0,                  # generator coefficient for attention class
    ):
        self.d = d
        self.B = B
        self.k = k
        self.m = m
        self.B_res = B_res
        self.L_Q = L_Q
        self.generator_class = generator_class
        self.substrate = substrate
        self.c_g = c_g

    def l_active(self, N: int) -> int:
        """Active sequence length in tokens: (N - m)*k + m*B_res + L_Q."""
        eff_m = min(self.m, N)
        return (N - eff_m) * self.k + eff_m * self.B_res + self.L_Q

    def c_gen(self) -> float:
        """One-time pointer generation cost per memory unit."""
        if self.generator_class == "pooling":
            # Pooling-class: O(B*d) readout over precomputed representations
            return float(self.B * self.d)
        elif self.generator_class == "attention":
            # Attention-class readout: c_g * (2*d*B^2 + 4*d^2*B)
            return self.c_g * (2.0 * self.d * (self.B ** 2) + 4.0 * (self.d ** 2) * self.B)
        else:
            raise ValueError(f"Unknown generator class: {self.generator_class}")

    def c_init(self, N: int) -> float:
        """One-time upfront ingestion and pointer generation cost across N blocks."""
        per_block_ingest = 2.0 * self.d * (self.B ** 2) + 4.0 * (self.d ** 2) * self.B
        return float(N * (per_block_ingest + self.c_gen()))

    def c_select(self, N: int) -> float:
        """Per-query pointer scoring cost: L_Q*d (query projection) + N*d (inner products)."""
        return float(self.L_Q * self.d + N * self.d)

    def c_res(self, N: int) -> float:
        """Per-query memory resolution cost depending on substrate."""
        eff_m = min(self.m, N)
        if self.substrate == "M_kv":
            # Zero recompute FLOPs: pages cached KV tensors into active tables
            return 0.0
        elif self.substrate == "M_text":
            # Re-tokenizes and re-encodes fetched text blocks: m * (2*d*B_res^2 + 4*d^2*B_res)
            return float(eff_m * (2.0 * self.d * (self.B_res ** 2) + 4.0 * (self.d ** 2) * self.B_res))
        elif self.substrate == "M_latent":
            # Cross-attention unpooling over k pointer tokens and bottleneck state
            return float(eff_m * self.B_res * self.k * self.d)
        else:
            raise ValueError(f"Unknown substrate: {self.substrate}")

    def c_query_linear(self, N: int) -> float:
        """Per-query linear re-ingestion cost over full history (NB + L_Q)."""
        seq_len = N * self.B + self.L_Q
        return float(2.0 * self.d * (seq_len ** 2) + 4.0 * (self.d ** 2) * seq_len)

    def c_query_sempointer(self, N: int) -> float:
        """Per-query SemPointer cost: active attention + selection + resolution."""
        la = self.l_active(N)
        attention_flops = 2.0 * self.d * (la ** 2) + 4.0 * (self.d ** 2) * la
        return float(attention_flops + self.c_select(N) + self.c_res(N))

    def delta_c_step(self, N: int) -> float:
        """Marginal computational savings per query."""
        return self.c_query_linear(N) - self.c_query_sempointer(N)

    def k_star(self, N: int) -> float:
        """Algebraic crossover threshold K* = C_init / Delta_C_step.

        Returns float('inf') if Delta_C_step <= 0 (indirection is strictly dominated).
        """
        delta = self.delta_c_step(N)
        if delta <= 0:
            return float("inf")
        return float(self.c_init(N) / delta)

    def cumulative_linear(self, N: int, K: int) -> float:
        """Total cumulative FLOPs for linear re-ingestion across K queries."""
        return float(K * self.c_query_linear(N))

    def cumulative_sempointer(self, N: int, K: int) -> float:
        """Total cumulative FLOPs for SemPointer across K queries: C_init + K * C_query."""
        return float(self.c_init(N) + K * self.c_query_sempointer(N))

    def crossover_k_empirical(self, N: int, max_K: int = 1000) -> Optional[int]:
        """Finds smallest integer query count K where SemPointer cumulative FLOPs < linear."""
        for k_val in range(1, max_K + 1):
            if self.cumulative_sempointer(N, k_val) < self.cumulative_linear(N, k_val):
                return k_val
        return None

    def regime(self, N: int) -> str:
        """Categorizes operational regime based on K* and viability."""
        k_s = self.k_star(N)
        if k_s < 1.0:
            return "single_query_viable"  # N >= 2 in pooling class
        elif k_s != float("inf"):
            return "multi_turn_amortized" # Amortizes after K* queries
        else:
            return "dominated"            # Sparsity condition violated


def compute_kstar_table(
    d: int = 4096,
    B: int = 512,
    k: int = 8,
    N_values: Optional[List[int]] = None,
    m: int = 0,
    L_Q: int = 0,
    substrate: str = "M_kv",
    generator_class: str = "pooling",
) -> pd.DataFrame:
    """Computes exact K* table matching the paper's theoretical benchmarks."""
    if N_values is None:
        N_values = [1, 2, 4, 8, 16, 32, 64]

    model = SemPointerCostModel(
        d=d,
        B=B,
        k=k,
        m=m,
        B_res=B,
        L_Q=L_Q,
        substrate=substrate,
        generator_class=generator_class,
    )
    rows = []
    for N in N_values:
        c_init = model.c_init(N)
        c_lin = model.c_query_linear(N)
        c_sp = model.c_query_sempointer(N)
        delta = model.delta_c_step(N)
        k_s = model.k_star(N)
        k_int = model.crossover_k_empirical(N)
        rows.append({
            "N": N,
            "C_init": c_init,
            "C_linear_query": c_lin,
            "C_sp_query": c_sp,
            "Delta_C_step": delta,
            "K*_theory": round(k_s, 6) if k_s != float("inf") else float("inf"),
            "K*_int_crossover": k_int,
            "Regime": model.regime(N),
        })
    return pd.DataFrame(rows)
