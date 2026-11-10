"""Lexical utilities: stopwords, word tokenization and an in-repo Okapi BM25.

BM25Index mirrors rank_bm25.BM25Okapi (and baselines/bm25_baseline.py's
validated `_InRepoBM25Okapi`): k1=1.5, b=0.75, negative IDF floored at
epsilon * mean_idf (epsilon=0.25). Unlike the baseline, tokenization removes
stopwords so that function words do not dominate short-block scoring.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence

import numpy as np

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are aren't as at be because been before being below
    between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
    few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
    him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just let's me more most
    mustn't my myself no nor not now of off on once only or other ought our ours ourselves out over own same
    shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves
    then there there's these they they'd they'll they're they've this those through to too under until up upon us
    very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who
    who's whom why why's will with won't would wouldn't you you'd you'll you're you've your yours yourself
    yourselves one two may might must shall s t d ll m re ve y get got like well even much many every
    """.split()
)

_WORD_RE = re.compile(r"\w+")


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Lowercase \\w+ tokens, optionally without stopwords."""
    toks = _WORD_RE.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


class BM25Index:
    """Okapi BM25 over `texts`. `scores(query)` -> np.ndarray of shape (N,)."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        corpus = [tokenize(t) for t in texts]
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            raise ValueError("BM25Index needs at least one text")
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_len = np.zeros(self.corpus_size, dtype=np.float64)
        nd: Dict[str, int] = {}
        for i, doc in enumerate(corpus):
            self.doc_len[i] = len(doc)
            tf: Dict[str, int] = {}
            for w in doc:
                tf[w] = tf.get(w, 0) + 1
            self.doc_freqs.append(tf)
            for w in tf:
                nd[w] = nd.get(w, 0) + 1
        self.avgdl = float(self.doc_len.mean()) or 1.0
        idf = {w: math.log(self.corpus_size - f + 0.5) - math.log(f + 0.5) for w, f in nd.items()}
        if idf:
            eps = self.epsilon * (sum(idf.values()) / len(idf))
            for w, v in idf.items():
                if v < 0:
                    idf[w] = eps
        self.idf = idf

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.corpus_size, dtype=np.float64)
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        for q in tokenize(query):
            idf_q = self.idf.get(q)
            if idf_q is None:
                continue
            f = np.array([tf.get(q, 0) for tf in self.doc_freqs], dtype=np.float64)
            out += idf_q * f * (self.k1 + 1.0) / (f + norm)
        return out
