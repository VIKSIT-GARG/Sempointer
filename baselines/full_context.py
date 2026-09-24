import time
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class FullContextBaseline:
    """B1: All N memories concatenated and re-ingested each query."""
    def __init__(self, model_name: str, tokenizer_name: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

    def run_query(
        self,
        query: str,
        memories: List[str],
        max_new_tokens: int = 100,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        # Concatenate all memories
        context = "\n".join(memories)
        
        # Build prompt
        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        prompt += f"Context:\n{context}\n\nQuery: {query}\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # Extract answer
        generated_ids = outputs[0][prompt_tokens:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        latency_ms = (time.time() - start_time) * 1000
        flops_estimate = self.measure_flops(prompt_tokens)
        
        return {
            'answer': answer,
            'prompt_tokens': prompt_tokens,
            'latency_ms': latency_ms,
            'flops_estimate': flops_estimate
        }
        
    def run_batch(
        self,
        queries: List[str],
        memories_per_query: List[List[str]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        results = []
        for q, m in zip(queries, memories_per_query):
            results.append(self.run_query(q, m, **kwargs))
        return results
        
    def measure_flops(self, prompt_token_count: int, d: int = 4096) -> float:
        # Returns estimated FLOPs: 2 * d * seq_len^2 (attention) + other terms
        seq_len = prompt_token_count
        # Rough estimate for forward pass FLOPs per token
        # C_forward = 2 * P (params) * seq_len + 2 * num_layers * seq_len^2 * d
        # Here we just use a simplified model as requested
        flops = 2 * d * (seq_len ** 2)
        # Add basic feed-forward approximation (e.g. 8 * d^2 * seq_len)
        flops += 8 * (d ** 2) * seq_len
        return float(flops)
