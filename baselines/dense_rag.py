"""B2: Dense RAG Baseline.

Sentence-BERT embedding + FAISS (or exact PyTorch normalized inner product) retrieval
followed by top-m passage concatenation into context.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    faiss = None


class DenseRAGBaseline:
    """B2: Dense Retrieval Augmented Generation baseline."""

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: Optional[str] = None,
        device: str = "cpu",
        embedder: Optional[Any] = None,
    ):
        self.device = device
        self.llm_model_name = llm_model_name
        if embedder is not None:
            self.embedder = embedder
        else:
            try:
                self.embedder = SentenceTransformer(embedding_model_name, device=device)
            except Exception as e:
                if device != "cpu":
                    import warnings
                    warnings.warn(f"Failed to load embedder on {device} ({e}). Falling back to CPU.")
                    self.device = "cpu"
                    self.embedder = SentenceTransformer(embedding_model_name, device="cpu")
                else:
                    raise e
        self.memories: List[str] = []
        self.memory_embeddings: Optional[torch.Tensor] = None
        self.faiss_index = None

        self.tokenizer = None
        self.model = None

    def _init_llm(self):
        """Lazy loads generation model when required."""
        if self.model is None and self.llm_model_name is not None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() and self.device != "cpu" else torch.float32,
                device_map=self.device,
            )
            self.model.eval()

    def build_index(self, memories: List[str]) -> None:
        """Encodes memories into normalized embeddings and creates index."""
        self.memories = memories
        if not memories:
            self.memory_embeddings = None
            self.faiss_index = None
            return

        with torch.no_grad():
            embs = self.embedder.encode(memories, convert_to_tensor=True, device=self.device)
            self.memory_embeddings = torch.nn.functional.normalize(embs, p=2, dim=-1)

        if faiss is not None:
            np_embs = self.memory_embeddings.cpu().numpy().astype(np.float32)
            d = np_embs.shape[1]
            self.faiss_index = faiss.IndexFlatIP(d)
            self.faiss_index.add(np_embs)
        else:
            self.faiss_index = None

    def retrieve(self, query: str, m: int) -> Tuple[List[int], List[float]]:
        """Retrieves top-m memories by inner product (cosine similarity)."""
        if self.memory_embeddings is None or len(self.memories) == 0:
            return [], []

        top_m = min(m, len(self.memories))

        if self.faiss_index is not None:
            q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
            faiss.normalize_L2(q_emb)
            scores, indices = self.faiss_index.search(q_emb, top_m)
            return indices[0].tolist(), scores[0].tolist()
        else:
            # Exact PyTorch normalized inner product
            with torch.no_grad():
                q_emb = self.embedder.encode([query], convert_to_tensor=True, device=self.device)
                q_norm = torch.nn.functional.normalize(q_emb, p=2, dim=-1)
                scores = torch.matmul(q_norm, self.memory_embeddings.T).squeeze(0)
                top_scores, top_indices = torch.topk(scores, k=top_m)
            return top_indices.cpu().tolist(), top_scores.cpu().tolist()

    def run_query(
        self,
        query: str,
        m: int = 1,
        max_new_tokens: int = 64,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieves top-m memories and evaluates context."""
        start_time = time.perf_counter()
        retrieved_ids, retrieval_scores = self.retrieve(query, m)
        retrieved_texts = [self.memories[i] for i in retrieved_ids]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        context_str = "\n\n".join(f"[DOC {i}]: {txt}" for i, txt in zip(retrieved_ids, retrieved_texts))

        prompt = ""
        if system_prompt:
            prompt += f"{system_prompt}\n\n"
        prompt += f"Context:\n{context_str}\n\nQuery: {query}\nAnswer:"

        answer = ""
        prompt_tokens = len(prompt.split())

        if self.llm_model_name is not None:
            self._init_llm()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_tokens = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated_ids = outputs[0][prompt_tokens:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "retrieved_ids": retrieved_ids,
            "retrieval_scores": retrieval_scores,
            "retrieved_texts": retrieved_texts,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "answer": answer,
            "retrieval_latency_ms": latency_ms,
        }
