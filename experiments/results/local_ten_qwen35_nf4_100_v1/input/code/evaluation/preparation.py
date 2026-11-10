"""Shared exact CPU token counting for local preparation, with a durable cache."""
from collections import Counter, OrderedDict
import hashlib
import sqlite3
import threading


class ExactTokenCounter:
    """Batch the pinned Rust tokenizer; store lengths rather than token-ID lists.

    The backend is cloned so concurrent read-only counting cannot race with the
    Transformers wrapper changing padding/truncation during RULER generation.
    One Rayon pool is shared by all calling threads, avoiding nested CPU pools.
    """
    def __init__(self, tokenizer, path, tokenizer_sha256):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
        self.backend.no_truncation()
        self.backend.no_padding()
        self.fingerprint = hashlib.sha256((tokenizer_sha256 + "\n" + self.backend.to_str()).encode()).hexdigest()
        self.lock = threading.Lock()
        self.memory = OrderedDict()
        self.metrics = Counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS counts (tokenizer TEXT, context TEXT, count INTEGER NOT NULL, PRIMARY KEY(tokenizer,context))")
        self.db.commit()

    def count_many(self, texts):
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        unique = dict(zip(keys, texts))
        known = {}
        with self.lock:
            for key in unique:
                if key in self.memory:
                    known[key] = self.memory[key]
                    self.memory.move_to_end(key)
                else:
                    row = self.db.execute("SELECT count FROM counts WHERE tokenizer=? AND context=?",
                                          (self.fingerprint, key)).fetchone()
                    if row is not None:
                        known[key] = row[0]
            self.metrics["cache_hits"] += len(known)
        missing = [key for key in unique if key not in known]
        if missing:
            # fast omits offset tracking; token IDs and normalization are unchanged.
            encode = getattr(self.backend, "encode_batch_fast", self.backend.encode_batch)
            encoded = encode([unique[key] for key in missing], add_special_tokens=False)
            known.update((key, len(value)) for key, value in zip(missing, encoded))
            del encoded
        with self.lock:
            self.db.executemany("INSERT OR IGNORE INTO counts VALUES (?,?,?)",
                                ((self.fingerprint, key, known[key]) for key in missing))
            self.db.commit()
            self.memory.update(known)
            while len(self.memory) > 65536:
                self.memory.popitem(last=False)
            self.metrics["encoded"] += len(missing)
            self.metrics["requests"] += len(texts)
        return [known[key] for key in keys]

    def stats(self):
        with self.lock:
            return dict(self.metrics)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.db.close()
