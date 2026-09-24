"""B3: BM25 Lexical Retrieval Baseline.

Scores memory units using exact Okapi BM25 ranking, selects top-m blocks,
and constructs context for query resolution.
Includes self-contained BM25 algorithm to guarantee zero missing-dependency failures.
"""

import time
import math
import collections
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class PurePythonBM25Okapi:
    """Self-contained Okapi BM25 implementation adhering to standard IR formulation."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)

        # Document frequencies
        self.doc_freqs: Dict[str, int] = collections.defaultdict(int)
        self.term_freqs: List[Dict[str, int]] = []

        for doc in corpus:
            tf = collections.defaultdict(int)
            for word in doc:
                tf[word] += 1
            self.term_freqs.append(tf)
            for word in tf:
                self.doc_freqs[word] += 1

        # Inverse document frequency
        self.idf: Dict[str, float] = {}
        for word, freq in self.doc_freqs.items():
            # Standard Lucene/Robertson IDF
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=float)
        for q_term in query:
            if q_term not in self.idf:
                continue
            idf_val = self.idf[q_term]
            for i in range(self.corpus_size):
                tf_val = self.term_freqs[i].get(q_term, 0)
                if tf_val == 0:
                    continue
                denom = tf_val + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avg_doc_len))
                scores[i] += idf_val * (tf_val * (self.k1 + 1.0)) / denom
        return scores


class BM25Baseline:
    """B3: BM25 lexical retrieval + top-m concatenation."""

    def __init__(
        self,
        llm_model_name: Optional[str] = None,
        device: str = "cpu",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.device = device
        self.llm_model_name = llm_model_name
        self.k1 = k1
        self.b = b
        self.bm25: Optional[PurePythonBM25Okapi] = None
        self.memories: List[str] = []
        self.tokenizer = None
        self.model = None

    def _init_llm(self):
        """Lazy loads generation model when required."""
        if self.model is None and self.llm_model_name is not None:
            import torch
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
        """Indexes raw text memories into the BM25 structure."""
        self.memories = memories
        if not memories:
            self.bm25 = None
            return

        tokenized_corpus = [doc.lower().split() for doc in memories]
        self.bm25 = PurePythonBM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, m: int) -> Tuple[List[int], List[float]]:
        """Scores all indexed memories against query and returns top-m indices and scores."""
        if self.bm25 is None or not self.memories:
            return [], []

        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)

        top_n = min(m, len(self.memories))
        top_indices = np.argsort(doc_scores)[::-1][:top_n]
        return top_indices.tolist(), doc_scores[top_indices].tolist()

    def run_query(
        self,
        query: str,
        m: int = 1,
        max_new_tokens: int = 64,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieves top-m memories and evaluates or generates an answer."""
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

        # If LLM model is configured, run generation
        if self.llm_model_name is not None:
            self._init_llm()
            import torch
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
