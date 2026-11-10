"""SemPointer Prompt Synthesis and Active Sequence Token Verification.

Assembles active prompts conforming to Proposition 1:
L_active = (N - m)*k + m*B_res + L_Q

Measures exact token counts using the target tokenizer to verify
active sequence compression against linear re-ingestion.
"""

from typing import List, Dict, Any, Optional
from .registry import PointerRegistry
from .resolver import TextResolver


class PromptBuilder:
    """Assembles prompt sequences and verifies token-level compression."""

    def __init__(self, tokenizer, B_res: int = 512):
        self.tokenizer = tokenizer
        self.B_res = B_res

    def build_sempointer_prompt(
        self,
        query: str,
        selected_ids: List[int],
        registry: PointerRegistry,
        resolver: TextResolver,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Assembles prompt with unselected pointers, resolved blocks, and query.

        Structure:
        [System Prompt]
        [Pointers for unselected memories]
        [Resolved full text for selected memories]
        [Query]
        """
        sections = []
        if system_prompt:
            sections.append(f"System: {system_prompt.strip()}")

        # Unselected pointer tokens
        all_ids = sorted(registry.records.keys())
        unselected_ids = [mid for mid in all_ids if mid not in selected_ids]

        if unselected_ids:
            pointer_strings = []
            for mid in unselected_ids:
                p_ids = registry.get_pointer_token_ids(mid)
                decoded_p = self.tokenizer.decode(p_ids, skip_special_tokens=True).strip()
                pointer_strings.append(f"[PTR_{mid}: {decoded_p}]")
            sections.append("Memory Registry Pointers:\n" + " ".join(pointer_strings))

        # Resolved memory blocks
        resolved_texts = resolver.resolve(selected_ids, registry)
        if resolved_texts:
            resolved_blocks = []
            for mid, text in zip(selected_ids, resolved_texts):
                resolved_blocks.append(f"[RESOLVED MEMORY {mid}]:\n{text.strip()}")
            sections.append("Resolved Context:\n" + "\n\n".join(resolved_blocks))

        sections.append(f"Query: {query.strip()}\nAnswer:")
        return "\n\n".join(sections)

    def build_full_context_prompt(
        self,
        query: str,
        all_texts: List[str],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Assembles standard linear re-ingestion prompt containing all N unpruned blocks."""
        sections = []
        if system_prompt:
            sections.append(f"System: {system_prompt.strip()}")

        context_blocks = [f"[MEMORY {i}]:\n{t.strip()}" for i, t in enumerate(all_texts)]
        sections.append("Historical Context:\n" + "\n\n".join(context_blocks))
        sections.append(f"Query: {query.strip()}\nAnswer:")
        return "\n\n".join(sections)

    def measure_token_counts(
        self,
        query: str,
        selected_ids: List[int],
        registry: PointerRegistry,
        resolver: TextResolver,
    ) -> Dict[str, int]:
        """Calculates token counts partitioned by functional component."""
        all_ids = sorted(registry.records.keys())
        unselected_ids = [mid for mid in all_ids if mid not in selected_ids]

        # 1. Pointers token count
        pointer_tokens = 0
        for mid in unselected_ids:
            p_ids = registry.get_pointer_token_ids(mid)
            pointer_tokens += len(p_ids)

        # 2. Resolved token count
        resolved_tokens = 0
        for text in resolver.resolve(selected_ids, registry):
            enc = self.tokenizer.encode(text, add_special_tokens=False)
            resolved_tokens += len(enc)

        # 3. Query token count
        query_tokens = len(self.tokenizer.encode(query, add_special_tokens=False))

        # 4. Total synthesized prompt tokens
        prompt = self.build_sempointer_prompt(query, selected_ids, registry, resolver)
        total_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        framing_tokens = max(0, total_tokens - (pointer_tokens + resolved_tokens + query_tokens))

        return {
            "total_tokens": total_tokens,
            "pointer_tokens": pointer_tokens,
            "resolved_tokens": resolved_tokens,
            "query_tokens": query_tokens,
            "framing_tokens": framing_tokens,
            "core_active_tokens": pointer_tokens + resolved_tokens + query_tokens,
        }

    def verify_l_active_formula(
        self,
        N: int,
        k: int,
        m: int,
        B_res: int,
        L_Q: int,
        measured_tokens: int,
    ) -> Dict[str, Any]:
        """Compares measured active token count against the theoretical formula."""
        eff_m = min(m, N)
        predicted = (N - eff_m) * k + eff_m * B_res + L_Q
        residual = measured_tokens - predicted
        relative_error = abs(residual) / max(1, predicted)

        return {
            "predicted_l_active": predicted,
            "measured_l_active": measured_tokens,
            "residual": residual,
            "relative_error": float(relative_error),
            "percentage_error": float(relative_error * 100.0),
            "within_2pct_tolerance": relative_error <= 0.02,
        }
