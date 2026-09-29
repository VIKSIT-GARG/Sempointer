"""Statistics used by the paper artifacts (numpy only; scipy optional).

All tests are PAIRED on shared example ids:
  - bootstrap_mean_ci:        percentile bootstrap CI of a mean (10k resamples)
  - paired_bootstrap_diff:    bootstrap CI of mean(a - b) over paired examples
  - sign_flip_permutation_p:  two-sided paired permutation test (random sign
                              flips of the per-example differences; exact
                              enumeration when n <= 16, else Monte Carlo with
                              the (count+1)/(n_perm+1) correction)
  - holm:                     Holm-Bonferroni adjusted p-values
  - mcnemar_exact:            exact (binomial) McNemar test for binary outcomes
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

NAN = float("nan")


def _arr(x) -> np.ndarray:
    return np.asarray(list(x), dtype=np.float64)


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                      seed: int = 42):
    """Returns (mean, lo, hi). Empty input -> (nan, nan, nan)."""
    a = _arr(values)
    n = len(a)
    if n == 0:
        return NAN, NAN, NAN
    mean = float(a.mean())
    if n == 1 or np.all(a == a[0]):
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, int(2_000_000 // n))
    i = 0
    while i < n_boot:
        b = min(chunk, n_boot - i)
        idx = rng.integers(0, n, size=(b, n))
        means[i:i + b] = a[idx].mean(axis=1)
        i += b
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(lo), float(hi)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                          alpha: float = 0.05, seed: int = 42):
    a, b = _arr(a), _arr(b)
    if len(a) != len(b):
        raise ValueError("paired inputs must have equal length")
    return bootstrap_mean_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)


def sign_flip_permutation_p(diffs: Sequence[float], n_perm: int = 10000, seed: int = 42) -> float:
    d = _arr(diffs)
    n = len(d)
    if n == 0:
        return NAN
    if np.all(d == 0):
        return 1.0
    obs = abs(d.mean())
    tol = 1e-12
    if n <= 16:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        stats = np.abs((signs * d).mean(axis=1))
        return float(np.mean(stats >= obs - tol))
    rng = np.random.default_rng(seed)
    count = 0
    chunk = max(1, int(2_000_000 // n))
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        signs = rng.choice((-1.0, 1.0), size=(b, n))
        count += int(np.sum(np.abs((signs * d).mean(axis=1)) >= obs - tol))
        done += b
    return (count + 1) / (n_perm + 1)


def holm(pvals: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni adjusted p-values; None/nan entries are ignored and
    returned as None (they do not count towards the family size)."""
    idx = [i for i, p in enumerate(pvals) if p is not None and not math.isnan(p)]
    out: List[Optional[float]] = [None] * len(pvals)
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * float(pvals[i]))
        running = max(running, adj)
        out[i] = running
    return out


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict[str, float]:
    """Exact two-sided McNemar test. b01 = a right & b wrong; b10 = a wrong & b right."""
    a = [bool(x) for x in a_correct]
    b = [bool(x) for x in b_correct]
    if len(a) != len(b):
        raise ValueError("paired inputs must have equal length")
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"a_only": n01, "b_only": n10, "p_value": float(p)}


def paired_compare(a: Sequence[float], b: Sequence[float], binary: bool = False,
                   n_boot: int = 10000, n_perm: int = 10000, seed: int = 42) -> Dict[str, float]:
    """SemPointer-style comparison of paired score lists (a - b)."""
    a, b = _arr(a), _arr(b)
    d = a - b
    mean, lo, hi = bootstrap_mean_ci(d, n_boot=n_boot, seed=seed)
    out = {"n": int(len(d)), "delta": mean, "lo": lo, "hi": hi,
           "p": sign_flip_permutation_p(d, n_perm=n_perm, seed=seed)}
    if binary:
        mc = mcnemar_exact(a >= 1 - 1e-9, b >= 1 - 1e-9)
        out.update({"p_mcnemar": mc["p_value"], "a_only": mc["a_only"], "b_only": mc["b_only"]})
    return out
