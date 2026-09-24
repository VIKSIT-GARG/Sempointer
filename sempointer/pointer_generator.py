import torch
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from collections import Counter
import math

class PoolingPointerGenerator:
    def __init__(self, model_name: str, k: int, device: str = 'cuda'):
        self.k = k
        self.device = device
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            if device != 'cpu':
                import warnings
                warnings.warn(f"Failed to initialize SentenceTransformer on {device} ({e}). Falling back to CPU.")
                self.device = 'cpu'
                self.model = SentenceTransformer(model_name, device='cpu')
            else:
                raise e
        self.tokenizer = self.model.tokenizer

    def generate(self, text: str) -> torch.Tensor:
        try:
            embedding = self.model.encode(text, convert_to_tensor=True, device=self.device)
        except Exception:
            embedding = self.model.encode(text, convert_to_tensor=True, device='cpu')
        d = embedding.shape[-1]
        return embedding.unsqueeze(0).expand(self.k, d)

    def generate_batch(self, texts: List[str]) -> torch.Tensor:
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
        except Exception:
            embeddings = self.model.encode(texts, convert_to_tensor=True, device='cpu')
        d = embeddings.shape[-1]
        return embeddings.unsqueeze(1).expand(len(texts), self.k, d)

    def generate_token_ids(self, text: str, tokenizer) -> List[int]:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        counts = Counter(token_ids)
        special_ids = set(tokenizer.all_special_ids)
        valid_counts = {tid: count for tid, count in counts.items() if tid not in special_ids}
        
        sorted_ids = sorted(valid_counts.items(), key=lambda x: -x[1])
        top_k_ids = [tid for tid, _ in sorted_ids[:self.k]]
        
        while len(top_k_ids) < self.k:
            top_k_ids.append(tokenizer.eos_token_id or 0)
            
        return top_k_ids[:self.k]

class AttentionPointerGenerator:
    def __init__(self, model_name: str, k: int, device: str = 'cuda'):
        self.k = k
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModel.from_pretrained(model_name).to(device)
        except Exception as e:
            if device != 'cpu':
                import warnings
                warnings.warn(f"Failed to load AutoModel on {device} ({e}). Falling back to CPU.")
                self.device = 'cpu'
                self.model = AutoModel.from_pretrained(model_name).to('cpu')
            else:
                raise e
        self.model.eval()

    def generate(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_state = outputs.last_hidden_state[0]
        seq_len = hidden_state.shape[0]
        if seq_len >= self.k:
            return hidden_state[:self.k, :]
        else:
            pad_len = self.k - seq_len
            d = hidden_state.shape[-1]
            pad = torch.zeros((pad_len, d), device=self.device)
            return torch.cat([hidden_state, pad], dim=0)

    def generate_batch(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        batch_size, seq_len, d = hidden_states.shape
        if seq_len >= self.k:
            return hidden_states[:, :self.k, :]
        else:
            pad_len = self.k - seq_len
            pad = torch.zeros((batch_size, pad_len, d), device=self.device)
            return torch.cat([hidden_states, pad], dim=1)
