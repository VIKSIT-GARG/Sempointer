import json
import uuid
import hashlib
import random
import os
from typing import List, Dict, Any, Optional
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

class SPBenchBuilder:
    def __init__(self, seed: int = 42, tokenizer_name: str = 'Qwen/Qwen2.5-7B-Instruct'):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def load_source_passages(
        self,
        source: str = 'wikipedia',
        n_passages: int = 10000,
        B_tokens: int = 512
    ) -> List[Dict[str, Any]]:
        passages = []
        if source == 'wikipedia':
            dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            for i, item in enumerate(dataset):
                if len(passages) >= n_passages:
                    break
                text = item['text']
                tokens = self.tokenizer.encode(text)
                
                # Chunk into B_tokens
                for j in range(0, len(tokens), B_tokens):
                    chunk_tokens = tokens[j:j+B_tokens]
                    if len(chunk_tokens) < B_tokens // 2:
                        continue # skip very short trailing chunks
                        
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                    passages.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "token_count": len(chunk_tokens),
                        "source_title": item.get('title', 'Unknown')
                    })
                    if len(passages) >= n_passages:
                        break
        else:
            with open(source, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(passages) >= n_passages:
                        break
                    data = json.loads(line)
                    text = data.get('text', '')
                    tokens = self.tokenizer.encode(text)
                    for j in range(0, len(tokens), B_tokens):
                        chunk_tokens = tokens[j:j+B_tokens]
                        chunk_text = self.tokenizer.decode(chunk_tokens)
                        passages.append({
                            "id": str(uuid.uuid4()),
                            "text": chunk_text,
                            "token_count": len(chunk_tokens),
                            "source_title": data.get('title', 'Unknown')
                        })
                        if len(passages) >= n_passages:
                            break
        return passages

    def compute_semantic_similarity(
        self,
        passages: List[Dict[str, Any]],
        embedding_model: str = 'sentence-transformers/all-mpnet-base-v2'
    ) -> np.ndarray:
        model = SentenceTransformer(embedding_model)
        texts = [p['text'] for p in passages]
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # Cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        return similarity_matrix

    def build_registry(
        self,
        target_passage: Dict[str, Any],
        all_passages: List[Dict[str, Any]],
        similarity_matrix: np.ndarray,
        N: int,
        difficulty: str,
        passage_idx: int
    ) -> List[Dict[str, Any]]:
        memories = []
        sim_scores = similarity_matrix[passage_idx]
        
        # Sort indices by similarity descending
        sorted_indices = np.argsort(sim_scores)[::-1]
        
        target_memory = {
            "memory_id": 0,
            "text": target_passage["text"],
            "token_count": target_passage["token_count"],
            "topic": target_passage["source_title"],
            "inserted_at": 0,
            "is_target": True,
            "is_near_paraphrase": False,
            "is_contradictory": False
        }
        memories.append(target_memory)
        
        # Pick distractors based on difficulty
        distractor_indices = []
        if difficulty == 'easy':
            # Skip top 100 similar, pick from the rest
            pool = sorted_indices[100:]
            if len(pool) > 0:
                distractor_indices = np.random.choice(pool, min(N-1, len(pool)), replace=False)
        else: # medium/hard
            # Pick highly similar documents
            pool = sorted_indices[1:N*2]
            if len(pool) > 0:
                distractor_indices = np.random.choice(pool, min(N-1, len(pool)), replace=False)
                
        for i, idx in enumerate(distractor_indices):
            p = all_passages[idx]
            memories.append({
                "memory_id": i + 1,
                "text": p["text"],
                "token_count": p["token_count"],
                "topic": p["source_title"],
                "inserted_at": i + 1,
                "is_target": False,
                "is_near_paraphrase": difficulty == 'hard',
                "is_contradictory": False
            })
            
        # Shuffle inserted_at
        positions = list(range(len(memories)))
        random.shuffle(positions)
        for m, pos in zip(memories, positions):
            m["inserted_at"] = pos
            
        # Sort by inserted_at
        memories.sort(key=lambda x: x["inserted_at"])
        
        return memories

    def generate_query(
        self,
        target_memory: Dict[str, Any],
        difficulty: str
    ) -> Dict[str, Any]:
        text = target_memory['text']
        words = text.split()
        if len(words) > 10:
            answer = " ".join(words[5:15])
            query_text = f"What is related to {' '.join(words[:5])}?"
        else:
            answer = text
            query_text = "What is the content of the target passage?"
            
        return {
            "query_id": 0,
            "text": query_text,
            "token_count": len(self.tokenizer.encode(query_text)),
            "target_memory_ids": [target_memory["memory_id"]],
            "requires_composition": difficulty == 'compositional',
            "answer": answer,
            "answer_span": {"memory_id": target_memory["memory_id"], "start": 0, "end": len(answer)}
        }

    def build_dataset(
        self,
        N_values: Optional[List[int]] = None,
        n_examples: int = 2500,
        output_path: str = 'data/spbench.jsonl'
    ) -> None:
        if N_values is None:
            N_values = [8, 16, 32, 64, 128, 256]
            
        passages = self.load_source_passages(n_passages=max(5000, n_examples * 2))
        sim_matrix = self.compute_semantic_similarity(passages)
        
        examples = []
        for i in range(n_examples):
            N = random.choice(N_values)
            diff = random.choice(['easy', 'medium', 'hard'])
            target_idx = i % len(passages)
            target_passage = passages[target_idx]
            
            memories = self.build_registry(target_passage, passages, sim_matrix, N, diff, target_idx)
            
            # Find target memory after shuffling
            t_mem = next(m for m in memories if m["is_target"])
            query = self.generate_query(t_mem, diff)
            
            ex = {
                "example_id": str(uuid.uuid4()),
                "difficulty": diff,
                "N": N,
                "memories": memories,
                "queries": [query],
                "metadata": {
                    "domain": "science",
                    "source": "wikipedia",
                    "split": "test" if i % 5 == 0 else "train"
                }
            }
            examples.append(ex)
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset

    def get_split(self, dataset: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
        return [ex for ex in dataset if ex.get("metadata", {}).get("split") == split]

    @staticmethod
    def compute_md5(path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
