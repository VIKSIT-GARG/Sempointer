"""Metric aggregation for run_benchmark.py (pure: operates on prediction rows).

aggregate(rows, system_names, seed) -> {"systems": {...}, "paired": {...}}

- per system: n_scored/n_errors, mean + bootstrap CI, token/latency means,
  optional-field means (selection_latency_ms, ttft_ms, n_hops, kv_bytes,
  active_tokens_core) when present, per-task and per-(task, context_length).
- paired: for every system vs each reference ("full" and every retrieval
  baseline rag*/bm25*/hybrid*), over the example ids both scored: mean
  difference, paired-bootstrap CI, Wilcoxon p (stats.statistical_tests).
  Also `matched_retrieval`: the retrieval baseline whose mean active tokens
  is closest to the system's (equal-budget comparison).
Older rows lacking optional fields are handled (fields just omitted).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

OPTIONAL_NUMERIC = ("selection_latency_ms", "ttft_ms", "n_hops", "kv_bytes",
                    "active_tokens_core", "n_selected")
RETRIEVAL_PREFIXES = ("rag", "bm25", "hybrid")


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 42):
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), (n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _mean(vals) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def is_retrieval(name: str) -> bool:
    return any(name == p or name.startswith(p + "_") for p in RETRIEVAL_PREFIXES)


def latest_scored(rows: List[Dict[str, Any]], system: str) -> Dict[str, Dict[str, Any]]:
    """example_id -> last scored row (resume may append retried rows)."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("system") == system and "score" in r:
            out[r["example_id"]] = r
    return out


def paired_comparison(a: Dict[str, Dict], b: Dict[str, Dict], seed: int = 42) -> Optional[Dict[str, Any]]:
    common = sorted(set(a) & set(b))
    if not common:
        return None
    sa = [float(a[i]["score"]) for i in common]
    sb = [float(b[i]["score"]) for i in common]
    diffs = np.asarray(sa) - np.asarray(sb)
    lo, hi = bootstrap_ci(diffs.tolist(), seed=seed)
    res = {"n_paired": len(common), "mean_a": float(np.mean(sa)), "mean_b": float(np.mean(sb)),
           "mean_diff": float(diffs.mean()), "diff_ci95": [lo, hi],
           "wins": int((diffs > 0).sum()), "losses": int((diffs < 0).sum()),
           "ties": int((diffs == 0).sum())}
    try:
        from stats.statistical_tests import wilcoxon_test

        w = wilcoxon_test(sa, sb)
        res["wilcoxon_p"] = w["p_value"]
    except Exception as e:  # noqa: BLE001 - statistics are supplementary
        res["wilcoxon_error"] = f"{type(e).__name__}: {e}"
    return res


def summarize_system(srows: List[Dict[str, Any]], seed: int = 42) -> Dict[str, Any]:
    scores = [float(r["score"]) for r in srows]
    lo, hi = bootstrap_ci(scores, seed=seed)
    out: Dict[str, Any] = {
        "n_scored": len(scores),
        "metric": srows[0].get("metric"),
        "score_mean": float(np.mean(scores)),
        "score_ci95": [lo, hi],
        "active_tokens_mean": _mean(r.get("active_tokens") for r in srows),
        "latency_ms_mean": _mean(r.get("latency_ms") for r in srows if r.get("latency_ms")),
        "peak_vram_mb_mean": _mean(r.get("peak_vram_mb") for r in srows if r.get("peak_vram_mb")),
    }
    for f in OPTIONAL_NUMERIC:
        m = _mean(r.get(f) for r in srows)
        if m is not None:
            out[f"{f}_mean"] = m
    per_task: Dict[str, Any] = {}
    for task in sorted({r.get("task") for r in srows}, key=str):
        ts = [r for r in srows if r.get("task") == task]
        per_task[str(task)] = {"n": len(ts), "score_mean": float(np.mean([r["score"] for r in ts])),
                               "active_tokens_mean": _mean(r.get("active_tokens") for r in ts)}
    out["per_task"] = per_task
    with_len = [r for r in srows if r.get("context_length") is not None]
    if with_len:
        per_tl: Dict[str, Any] = {}
        for key in sorted({(str(r.get("task")), int(r["context_length"])) for r in with_len}):
            ts = [r for r in with_len if (str(r.get("task")), int(r["context_length"])) == key]
            per_tl[f"{key[0]}@{key[1]}"] = {
                "task": key[0], "context_length": key[1], "n": len(ts),
                "score_mean": float(np.mean([r["score"] for r in ts])),
                "active_tokens_mean": _mean(r.get("active_tokens") for r in ts)}
        out["per_task_context_length"] = per_tl
    return out


def aggregate(rows: List[Dict[str, Any]], system_names: Sequence[str], seed: int = 42) -> Dict[str, Any]:
    scored_pairs = {(r["system"], r["example_id"]) for r in rows if "score" in r}
    systems: Dict[str, Any] = {}
    latest: Dict[str, Dict[str, Dict]] = {}
    for name in system_names:
        latest[name] = latest_scored(rows, name)
        errs = [r for r in rows if r.get("system") == name and "error" in r
                and (name, r["example_id"]) not in scored_pairs]
        srows = list(latest[name].values())
        if not srows:
            systems[name] = {"n_scored": 0, "n_errors": len(errs),
                             "errors": [e["error"] for e in errs[:5]]}
            continue
        systems[name] = {**summarize_system(srows, seed), "n_errors": len(errs)}

    refs = [n for n in system_names if (n == "full" or is_retrieval(n)) and latest[n]]
    paired: Dict[str, Any] = {}
    for name in system_names:
        if not latest[name]:
            continue
        comp: Dict[str, Any] = {}
        for ref in refs:
            if ref == name:
                continue
            pc = paired_comparison(latest[name], latest[ref], seed)
            if pc is not None:
                comp[f"vs_{ref}"] = pc
        tok = systems[name].get("active_tokens_mean")
        cands = [r for r in refs if is_retrieval(r) and r != name
                 and systems[r].get("active_tokens_mean") is not None]
        if tok is not None and cands:
            best = min(cands, key=lambda r: (abs(systems[r]["active_tokens_mean"] - tok), r))
            comp["matched_retrieval"] = best
        if comp:
            paired[name] = comp
    return {"systems": systems, "paired": paired}
