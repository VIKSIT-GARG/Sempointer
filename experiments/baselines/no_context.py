import time
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class NoContextBaseline:
    """B5: LLM answers query with no memory access."""
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.model.eval()

    def run_query(self, query: str, max_new_tokens: int = 100) -> Dict[str, Any]:
        start_time = time.time()
        
        prompt = f"Query: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][prompt_tokens:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'answer': answer,
            'latency_ms': latency_ms,
            'prompt_tokens': prompt_tokens
        }

    def run_batch(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        results = []
        for q in queries:
            results.append(self.run_query(q, **kwargs))
        return results
