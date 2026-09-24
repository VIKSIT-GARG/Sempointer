import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import threading
import pickle
import os

@dataclass
class MemoryRecord:
    memory_id: int
    text: str
    token_count: int
    pointer_embedding: torch.Tensor
    pointer_token_ids: List[int]
    inserted_at: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class PointerRegistry:
    def __init__(self, k: int, B: int, max_size: int = 2048):
        self.k = k
        self.B = B
        self.max_size = max_size
        self.records: Dict[int, MemoryRecord] = {}
        self.lock = threading.Lock()
        self._next_id = 0

    def register(self, text: str, generator, metadata: Dict = None) -> int:
        with self.lock:
            if len(self.records) >= self.max_size:
                raise RuntimeError("Registry reached max_size")
            memory_id = self._next_id
            self._next_id += 1

        embedding = generator.generate(text)
        try:
            token_ids = generator.generate_token_ids(text, generator.tokenizer)
        except AttributeError:
            token_ids = [0] * self.k

        record = MemoryRecord(
            memory_id=memory_id,
            text=text,
            token_count=len(generator.tokenizer.tokenize(text)),
            pointer_embedding=embedding.cpu(),
            pointer_token_ids=token_ids,
            inserted_at=memory_id,
            metadata=metadata or {}
        )
        with self.lock:
            self.records[memory_id] = record
        return memory_id

    def register_batch(self, texts: List[str], generator, metadata_list=None) -> List[int]:
        embeddings = generator.generate_batch(texts)
        ids = []
        for i, text in enumerate(texts):
            with self.lock:
                if len(self.records) >= self.max_size:
                    break
                memory_id = self._next_id
                self._next_id += 1
            
            try:
                token_ids = generator.generate_token_ids(text, generator.tokenizer)
            except AttributeError:
                token_ids = [0] * self.k
                
            meta = metadata_list[i] if metadata_list else {}
            record = MemoryRecord(
                memory_id=memory_id,
                text=text,
                token_count=len(generator.tokenizer.tokenize(text)),
                pointer_embedding=embeddings[i].cpu(),
                pointer_token_ids=token_ids,
                inserted_at=memory_id,
                metadata=meta
            )
            with self.lock:
                self.records[memory_id] = record
            ids.append(memory_id)
        return ids

    def get_pointer_embedding(self, memory_id: int) -> torch.Tensor:
        return self.records[memory_id].pointer_embedding

    def get_pointer_token_ids(self, memory_id: int) -> List[int]:
        return self.records[memory_id].pointer_token_ids

    def get_text(self, memory_id: int) -> str:
        return self.records[memory_id].text

    def get_record(self, memory_id: int) -> MemoryRecord:
        return self.records[memory_id]

    def all_pointer_embeddings(self) -> torch.Tensor:
        if not self.records:
            return torch.empty(0)
        embeddings = [self.records[mid].pointer_embedding for mid in sorted(self.records.keys())]
        return torch.stack(embeddings)

    def __len__(self) -> int:
        return len(self.records)

    def save(self, path: str):
        with self.lock:
            with open(path, 'wb') as f:
                pickle.dump({'records': self.records, 'next_id': self._next_id}, f)

    def load(self, path: str):
        with self.lock:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.records = data['records']
                self._next_id = data['next_id']
