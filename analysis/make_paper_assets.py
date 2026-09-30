"""Generate LNCS tables, macros and figures from the new paper benchmark runs.

Only ``results/paper_*/{config,metrics,predictions}.json(l)`` are eligible.
Historical ``final_*``, ``smoke_*`` and ``results_archive`` runs are excluded.
Scored observations in partial runs remain eligible, with completeness and
errors disclosed. Absent observations render as ``\\textbf{TBD}``.

Run from anywhere::

    experiments/warden/bin/python experiments/analysis/make_paper_assets.py
    python experiments/analysis/make_paper_assets.py --results /path/to/results

Matplotlib is optional. Without it, the script writes tables/macros and reports
which figures could not be refreshed. No dependency is installed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.stats import holm, paired_bootstrap_diff, sign_flip_permutation_p  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
OUTPUT = ROOT / "paper" / "generated"
TBD = r"\textbf{TBD}"
MAIN_RUNS = {"ruler": "paper_ruler", "longbench": "paper_longbench",
             "locomo": "paper_locomo", "kv_compare": "paper_kv"}
BENCH_LABELS = {"ruler": "RULER", "longbench": "LongBench", "locomo": "LoCoMo",
                "kv_compare": "KV comparison"}
SYSTEM_LABELS = {"sempointer": "SemPointer", "full": "Full context",
                 "rag": "Dense RAG", "bm25": "BM25", "hybrid": "Hybrid retrieval"}
MAIN_SYSTEMS = {
    "ruler": ("sempointer", "full", "rag_m1", "rag_m4", "bm25_m4", "hybrid_m4"),
    "longbench": ("sempointer", "full", "rag_m4", "hybrid_m4"),
    "locomo": ("sempointer", "full", "rag_m4", "hybrid_m4"),
}
ABLATIONS = {
    "tab_ablation_pointer.tex": ("Pointer type", [("keyword", "sempointer"),
        ("random ID", "sempointer__ptype-random_id"), ("lead", "sempointer__ptype-lead"),
        ("none", "sempointer__ptype-none")]),
    "tab_ablation_selector.tex": ("Selector", [("hybrid + attention (default)", "sempointer"),
        ("dense", "sempointer__selector-dense"), ("bm25", "sempointer__selector-bm25"),
        ("attention", "sempointer__selector-attention")]),
    "tab_ablation_index.tex": ("Pointer index", [("not scheduled", "")]),
    "tab_ablation_k.tex": ("Pointer length $k$", [("8", "sempointer__k-8"),
        ("16 (default)", "sempointer"), ("32", "sempointer__k-32")]),
    "tab_ablation_m.tex": ("Resolved blocks $m$", [("1", "sempointer__m-1"),
        ("2 (default)", "sempointer"), ("adaptive", "sempointer__m-adaptive")]),
    "tab_ablation_hops.tex": ("Expansion", [("one hop (default)", "sempointer"),
        ("two hops", "sempointer__hops-2"), ("adaptive $m$", "sempointer__m-adaptive")]),
}
FIGURES = ("fig_accuracy_vs_length.pdf", "fig_pareto_tokens_accuracy.pdf",
           "fig_k_ablation.pdf", "fig_amortization.pdf", "fig_kv_memory.pdf",
           "fig_latency_vram.pdf")
_PAIRED_CACHE: dict[tuple[Path, str, str], dict | None] = {}


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def avg(rows: list[dict], key: str) -> float | None:
    vals = [v for row in rows if (v := finite_number(row.get(key))) is not None]
    return mean(vals) if vals else None


def fmt(value: float | None, kind: str = "score") -> str:
    if value is None:
        return TBD
    if kind == "score":
        return f"{value * 100:.1f}"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "p":
        return f"{value:.3g}"
    return f"{value:.1f}"


def latex(text: Any) -> str:
    s = str(text)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"),
                     ("}", r"\}")):
        s = s.replace(old, new)
    return s


@dataclass
class Run:
    path: Path
    config: dict
    metrics: dict
    rows: dict[tuple[str, str], dict]

    def system(self, name: str) -> list[dict]:
        return [r for (sys, _), r in self.rows.items() if sys == name]

    def names(self) -> set[str]:
        return {sys for sys, _ in self.rows}

    def expected(self) -> int:
        return self.metrics["n_examples"]

    def errors(self, name: str) -> int:
        return self.metrics.get("systems", {}).get(name, {}).get("n_errors", 0)


def load_runs(results: Path) -> tuple[dict[str, Run], list[str]]:
    runs: dict[str, Run] = {}
    notes: list[str] = []
    for path in sorted(results.glob("paper_*")):
        if not path.is_dir():
            continue
        try:
            config = json.loads((path / "config.json").read_text())
            metrics = json.loads((path / "metrics.json").read_text())
            if config.get("run_name") != path.name:
                raise ValueError("run_name in config does not match directory")
            if metrics.get("benchmark") != config.get("benchmark"):
                raise ValueError("benchmark differs between config and metrics")
            latest: dict[tuple[str, str], dict] = {}
            with (path / "predictions.jsonl").open() as handle:
                for line_no, line in enumerate(handle, 1):
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"row {line_no} is not an object")
                    score = finite_number(row.get("score"))
                    if score is None:
                        continue  # errors and invalid scores cannot enter any paper number
                    if not 0 <= score <= 1:
                        raise ValueError(f"row {line_no} score out of [0,1]")
                    key = (row["system"], row["example_id"])
                    latest[key] = row  # --resume can append a later success
            expected = metrics.get("n_examples")
            summaries = metrics.get("systems")
            if not isinstance(expected, int) or expected <= 0 or not isinstance(summaries, dict) or not summaries:
                raise ValueError("missing valid n_examples or system summaries")
            incomplete = []
            for sys_name, summary in summaries.items():
                actual = sum(key[0] == sys_name for key in latest)
                scored, errors = summary.get("n_scored"), summary.get("n_errors")
                if (not isinstance(scored, int) or not isinstance(errors, int)
                        or scored != actual or scored < 0 or errors < 0
                        or scored + errors > expected):
                    raise ValueError(f"inconsistent summary for {sys_name}: "
                                     f"{actual} raw scored, {scored} reported scored, "
                                     f"{errors} reported errors; expected {expected}")
                if scored != expected or errors:
                    incomplete.append(f"{sys_name}: {scored}/{expected} scored, {errors} errors")
            unknown = {sys_name for sys_name, _ in latest} - set(summaries)
            if unknown:
                raise ValueError(f"scored systems missing summaries: {sorted(unknown)}")
            runs[path.name] = Run(path, config, metrics, latest)
            notes.append(f"{path.name}: {len(latest)} latest scored predictions" +
                         ("; partial (" + "; ".join(incomplete) + ")" if incomplete else "; complete"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            notes.append(f"{path.name}: excluded: {exc}")
    return runs, notes


def main_run(runs: dict[str, Run], benchmark: str) -> Run | None:
    run = runs.get(MAIN_RUNS[benchmark])
    return run if run and run.config.get("benchmark") == benchmark else None


def system_rows(run: Run | None, name: str) -> list[dict]:
    if run is None:
        return []
    # Bare names survive earlier runner versions; explicit m is preferred.
    names = {name}
    if name in ("rag_m1", "bm25_m1", "hybrid_m1") and name not in run.names():
        names = {name.split("_m")[0]}
    return [r for (sys, _), r in run.rows.items() if sys in names]


def sample_count(run: Run | None, name: str) -> str:
    if run is None:
        return TBD
    n = len(system_rows(run, name))
    errors = run.errors(name)
    return f"{n}/{run.expected()}" + (f" (err {errors})" if errors else "")


def observed_score(run: Run | None, name: str, rows: list[dict] | None = None) -> str:
    rs = system_rows(run, name) if rows is None else rows
    value = fmt(avg(rs, "score"))
    if value != TBD and run is not None and len(rs) < run.expected():
        return value + r"\textsuperscript{*}"
    return value


def common_rows(run: Run | None, names: tuple[str, ...]) -> dict[str, list[dict]]:
    """Return each system's rows restricted to the same example IDs."""
    by_name = {name: {r["example_id"]: r for r in system_rows(run, name)} for name in names}
    ids = set.intersection(*(set(rows) for rows in by_name.values())) if by_name else set()
    return {name: [by_name[name][item] for item in sorted(ids)] for name in names}


def score(run: Run | None, name: str) -> float | None:
    return avg(system_rows(run, name), "score")


def field(run: Run | None, name: str, key: str) -> float | None:
    return avg(system_rows(run, name), key)


def retrieval_rows(run: Run | None, name: str) -> list[dict]:
    return [r for r in system_rows(run, name)
            if r.get("task", "").startswith("niah_")
            and r.get("retrieval_basis") == "gold string present in selected RULER NIAH block"
            and finite_number(r.get("retrieval_hit")) is not None]


def retrieval_score(run: Run | None, name: str) -> float | None:
    return avg(retrieval_rows(run, name), "retrieval_hit")


def ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b is not None and b != 0 else None


def delta(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def length_of(row: dict) -> int | None:
    val = row.get("context_length")
    if isinstance(val, int) and val > 0:
        return val
    # Some old-format rows omitted context_length although the example ID
    # encodes it. This is still directly recorded in the real prediction row.
    match = re.search(r"-(4096|8192|16384|32768)-\d+$", row.get("example_id", ""))
    return int(match.group(1)) if match else None


def subset(run: Run | None, name: str, pred: Callable[[dict], bool]) -> list[dict]:
    return [r for r in system_rows(run, name) if pred(r)]


def table(filename: str, caption: str, label: str, headers: list[str], data: list[list[str]], out: Path) -> None:
    cols = "l" + "r" * (len(headers) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(headers) + " \\\\", r"\midrule"]
    lines.extend(" & ".join(row) + " \\\\" for row in data)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out / filename).write_text("\n".join(lines))


def write_tables(runs: dict[str, Run], out: Path) -> None:
    # Scores are percentages; other columns retain their recorded units.
    rows = []
    for bench in ("ruler", "longbench", "locomo"):
        run = main_run(runs, bench)
        if run is None:
            rows.append([BENCH_LABELS[bench], TBD, TBD, TBD, TBD, TBD])
            continue
        for name in MAIN_SYSTEMS[bench]:
            rs = system_rows(run, name)
            label = SYSTEM_LABELS.get(name.split("_m")[0], name)
            if name.endswith("_m4"):
                label += " ($m=4$)"
            elif name.endswith("_m1"):
                label += " ($m=1$)"
            rows.append([BENCH_LABELS[bench], label, sample_count(run, name),
                         observed_score(run, name), fmt(avg(rs, "active_tokens"), "int"),
                         fmt(avg(rs, "latency_ms"), "int")])
    # Two textual columns require an explicit layout.
    _table_custom("tab_main.tex", "Main benchmark results. Score is the task metric (\\%). "
                  "Samples show scored/expected and outstanding errors. An asterisk marks a "
                  "score from fewer than expected examples; missing examples are not zeros. "
                  "Latency is per scored answer.", "tab:main",
                  "llrrrr", ["Benchmark", "System", "$n/N$ (errors)", "Score", "Tokens", "Latency (ms)"], rows, out,
                  compact=True)

    ruler = main_run(runs, "ruler")
    lengths = sorted({length_of(r) for r in ruler.rows.values() if length_of(r) is not None}) if ruler else []
    tasks = sorted({r.get("task", "") for r in ruler.rows.values()}) if ruler else []
    rows = [[TBD] * 6] if ruler is None else []
    if ruler:
        for task in tasks:
            for length in lengths:
                cells = [latex(task), str(length)]
                bucket_ids = {r["example_id"] for r in ruler.rows.values()
                              if r.get("task") == task and length_of(r) == length}
                for name in ("sempointer", "full", "rag_m4", "hybrid_m4"):
                    selected = subset(ruler, name, lambda r, t=task, n=length:
                                      r.get("task") == t and length_of(r) == n)
                    cell = fmt(avg(selected, "score"))
                    if cell != TBD and len(selected) < len(bucket_ids):
                        cell += rf"\textsuperscript{{*}} ({len(selected)}/{len(bucket_ids)})"
                    cells.append(cell)
                rows.append(cells)
    _table_custom("tab_ruler_breakdown.tex", "RULER score (\\%) by task and requested context length. "
                  "An asterisk marks a partial cell; its parentheses show scored/scheduled "
                  "examples. See the main table and provenance report for error counts.",
                  "tab:ruler-breakdown", "llrrrr", ["Task", "Tokens", "SemPointer", "Full", "RAG-4", "Hybrid-4"], rows, out)

    kv = main_run(runs, "kv_compare")
    kv_names = ["sempointer", "sempointer__substrate-kv", "full",
                "kvpress_snapkv_0.3", "kvpress_tova_0.3", "kvpress_knorm_0.3"]
    rows = [[latex(name), sample_count(kv, name), observed_score(kv, name),
             fmt(field(kv, name, "active_tokens"), "int"),
             fmt(field(kv, name, "latency_ms"), "int"), fmt(field(kv, name, "peak_vram_mb"), "int")]
            for name in kv_names] if kv else [[TBD] * 6]
    _table_custom("tab_kvcompare.tex", "KV comparison on scheduled examples. Samples show "
                  "scored/expected and outstanding errors; a score with no scored examples is TBD. "
                  "Peak VRAM is the recorded per-answer peak.", "tab:kvcompare", "llrrrr",
                  ["System", "$n/N$ (errors)", "Score (\\%)", "Tokens", "Latency (ms)", "VRAM (MiB)"],
                  rows, out, compact=True)

    abl = runs.get("paper_ablation")
    for filename, (heading, variants) in ABLATIONS.items():
        rows = [[latex(label), sample_count(abl, name), observed_score(abl, name),
                 fmt(avg((rs := system_rows(abl, name)), "active_tokens"), "int")]
                for label, name in variants] if abl else [[TBD] * 4]
        table(filename, f"{heading} ablation on RULER. Scores are percentages; "
              "samples show scored/expected and outstanding errors. An asterisk marks a partial score.",
              filename.removesuffix(".tex").replace("tab_", "tab:").replace("_", "-"),
              [heading, "$n/N$ (errors)", "Score (\\%)", "Tokens"], rows, out)

    sub_rows = [[label, sample_count(kv, name), observed_score(kv, name),
                 fmt(field(kv, name, "ttft_ms"), "int"),
                 fmt(field(kv, name, "peak_vram_mb"), "int")]
                for label, name in (("$M_{\\mathrm{text}}$", "sempointer"),
                                    ("$M_{\\mathrm{kv}}$", "sempointer__substrate-kv"))] if kv else [[TBD] * 5]
    _table_custom("tab_substrate.tex", "Substrate comparison; TTFT and VRAM are measured "
                  "per scored answer. Samples show scored/expected and outstanding errors.",
                  "tab:substrate", "llrrr",
                  ["Substrate", "$n/N$ (errors)", "Score (\\%)", "TTFT (ms)", "VRAM (MiB)"],
                  sub_rows, out)

    efficiency = runs.get("paper_efficiency")
    eff_rows = []
    for name in (("full", "sempointer", "sempointer__substrate-kv") if efficiency else ()):
        rs = system_rows(efficiency, name)
        eff_rows.append([latex(name), fmt(avg(rs, "ingest_ms"), "int"),
                         fmt(avg(rs, "kv_encode_ms"), "int"), fmt(avg(rs, "transfer_ms"), "int"),
                         fmt(avg(rs, "reused_tokens"), "int"), fmt(avg(rs, "ttft_ms"), "int")])
    if not efficiency:
        eff_rows = [[TBD] * 6]
    table("tab_efficiency.tex", "Recorded efficiency measurements; ingest and KV encoding are separate from "
          "per-answer TTFT. Each mean uses available prediction rows.",
          "tab:efficiency", ["System", "Ingest (ms)", "KV encode (ms)", "Transfer (ms)",
                             "Reused tok.", "TTFT (ms)"], eff_rows, out)

    model2 = runs.get("paper_model2")
    model_rows = [[latex(name), observed_score(model2, name), sample_count(model2, name)]
                  for name in ("sempointer", "full", "hybrid_m4")] \
                 if model2 else [[TBD] * 3]
    table("tab_model2.tex", "Second-model replication; samples show scored/expected and "
          "outstanding errors. An asterisk marks a partial score.",
          "tab:model2", ["System", "Score (\\%)", "$n/N$ (errors)"], model_rows, out)

    recall_rows = []
    for name in (("sempointer", "rag_m1", "rag_m4", "hybrid_m4", "bm25_m4") if ruler else ()):
        rs = retrieval_rows(ruler, name)
        recall_rows.append([latex(name), str(len(rs)) if rs else TBD,
                            fmt(avg(rs, "retrieval_hit"))])
    if not ruler:
        recall_rows = [[TBD] * 3]
    _table_custom("tab_retrieval.tex", "RULER NIAH gold-string block coverage among examples "
                  "whose answer string is identifiable in source blocks. This is not "
                  "general evidence recall.", "tab:retrieval", "lrr",
                  ["System", "$n$", "Coverage (\\%)"], recall_rows, out)

    sig_rows = []
    for bench in ("ruler", "longbench", "locomo"):
        run = main_run(runs, bench)
        if run is None:
            sig_rows.append([BENCH_LABELS[bench]] + [TBD] * 6)
            continue
        baselines = ("full", "rag_m1", "rag_m4", "hybrid_m4", "bm25_m4") \
                    if bench == "ruler" else ("full", "rag_m4", "hybrid_m4")
        comparisons = [paired_stats(run, "sempointer", baseline) for baseline in baselines]
        adjusted = holm([item["p"] if item else None for item in comparisons])
        for baseline, paired, adj in zip(baselines, comparisons, adjusted):
            interval = (f"[{paired['lo']:.1f}, {paired['hi']:.1f}]"
                        if paired is not None else TBD)
            sig_rows.append([BENCH_LABELS[bench], latex(baseline),
                             str(paired["n"]) if paired else TBD,
                             fmt(paired["delta"] if paired else None, "float"),
                             interval, fmt(paired["p"] if paired else None, "p"),
                             fmt(adj, "p")])
    _table_custom("tab_significance.tex", "Paired SemPointer-minus-baseline score difference "
                  "(percentage points), 95\\% bootstrap CI, two-sided sign-flip $p$, and Holm-adjusted "
                  "$p$. Each benchmark's scheduled baselines form one correction family.",
                  "tab:significance", "llrrlrr",
                  ["Benchmark", "Baseline", "$n$", "$\\Delta$", "95\\% CI", "$p$", "$p_{\\rm H}$"],
                  sig_rows, out, compact=True)


def _table_custom(filename: str, caption: str, label: str, cols: str,
                  headers: list[str], data: list[list[str]], out: Path,
                  compact: bool = False) -> None:
    lines = [r"\begin{table}[t]", r"\centering",
             r"\scriptsize" if compact else r"\small"]
    if compact:
        lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines += [f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(headers) + " \\\\", r"\midrule"]
    lines.extend(" & ".join(row) + " \\\\" for row in data)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out / filename).write_text("\n".join(lines))


def paired_stats(run: Run | None, a: str, b: str) -> dict | None:
    if run is None:
        return None
    cache_key = (run.path, a, b)
    if cache_key in _PAIRED_CACHE:
        return _PAIRED_CACHE[cache_key]
    ra = {r["example_id"]: r for r in system_rows(run, a)}
    rb = {r["example_id"]: r for r in system_rows(run, b)}
    ids = sorted(ra.keys() & rb.keys())
    if not ids:
        _PAIRED_CACHE[cache_key] = None
        return None
    diffs = [ra[i]["score"] - rb[i]["score"] for i in ids]
    point, lo, hi = paired_bootstrap_diff(
        [ra[i]["score"] for i in ids], [rb[i]["score"] for i in ids],
        n_boot=10000, seed=42)
    p = sign_flip_permutation_p(diffs, n_perm=10000, seed=42)
    result = {"n": len(ids), "delta": point * 100,
              "lo": lo * 100, "hi": hi * 100, "p": p}
    _PAIRED_CACHE[cache_key] = result
    return result


def write_macros(runs: dict[str, Run], out: Path) -> None:
    # The placeholder file is the stable, manuscript-facing macro inventory.
    placeholder = out / "macros_placeholder.tex"
    if not placeholder.exists():
        placeholder = OUTPUT / "macros_placeholder.tex"
    names = re.findall(r"\\providecommand\{\\([A-Za-z]+)\}", placeholder.read_text())
    values: dict[str, str] = {}
    for bench, suffix in (("ruler", "Ruler"), ("longbench", "LongBench"), ("locomo", "Locomo")):
        run = main_run(runs, bench)
        for prefix, name in (("SP", "sempointer"), ("Full", "full"), ("Hybrid", "hybrid_m4"),
                             ("Rag", "rag_m4"), ("BM", "bm25_m4"), ("NoCtx", "no_context")):
            values[f"{prefix}{suffix}Score"] = fmt(score(run, name))
        values[f"N{suffix if suffix != 'Ruler' else 'Ruler'}Examples"] = (
            str(len({r["example_id"] for r in run.rows.values()})) if run and run.rows else TBD)
        for prefix, name in (("SP", "sempointer"), ("Full", "full")):
            values[f"{prefix}ActiveTokens{suffix}"] = fmt(field(run, name, "active_tokens"), "int")
            values[f"{prefix}Latency{suffix}"] = fmt(field(run, name, "latency_ms"), "int")
        full_tok = field(run, "full", "active_tokens")
        sp_tok = field(run, "sempointer", "active_tokens")
        values[f"SPTokenReduction{'' if bench == 'ruler' else suffix}"] = fmt(
            100 * (1 - sp_tok / full_tok) if full_tok and sp_tok is not None else None, "float")
    ruler = main_run(runs, "ruler")
    for length, word in ((4096, "FourK"), (8192, "EightK"), (16384, "SixteenK")):
        for prefix, name in (("SP", "sempointer"), ("Full", "full")):
            values[f"{prefix}Ruler{word}"] = fmt(avg(subset(ruler, name,
                                                            lambda r, n=length: length_of(r) == n), "score"))
    values["SPTokenRatio"] = fmt(ratio(field(ruler, "full", "active_tokens"),
                                       field(ruler, "sempointer", "active_tokens")), "float")
    values["SPSpeedup"] = fmt(ratio(field(ruler, "full", "latency_ms"),
                                   field(ruler, "sempointer", "latency_ms")), "float")
    for bench, suffix in (("ruler", ""), ("longbench", "LongBench"), ("locomo", "Locomo")):
        run = main_run(runs, bench)
        for stem, name in (("Hybrid", "hybrid_m4"), ("Rag", "rag_m4"),
                           ("BM", "bm25_m4"), ("Full", "full")):
            p = paired_stats(run, "sempointer", name)
            values[f"SPvs{stem}Delta{suffix}"] = fmt(p["delta"] if p else None, "float")
            values[f"SPvs{stem}P{suffix}"] = fmt(p["p"] if p else None, "p")
        for stem, name in (("SP", "sempointer"), ("Hybrid", "hybrid_m4"),
                           ("Rag", "rag_m4"), ("BM", "bm25_m4")):
            values[f"{stem}Recall{'Ruler' if bench == 'ruler' else suffix}"] = fmt(
                retrieval_score(run, name) if bench == "ruler" else None)
        baselines = ("full", "rag_m1", "rag_m4", "hybrid_m4", "bm25_m4") \
                    if bench == "ruler" else ("full", "rag_m4", "hybrid_m4")
        comparisons = [paired_stats(run, "sempointer", name) for name in baselines]
        adjusted = holm([item["p"] if item else None for item in comparisons])
        adj_map = dict(zip(baselines, adjusted))
        for stem, name in (("Full", "full"), ("Rag", "rag_m4"), ("Hybrid", "hybrid_m4")):
            values[f"SPvs{stem}PHolm{suffix}"] = fmt(adj_map[name], "p")
    kv = main_run(runs, "kv_compare")
    for macro, name in (("SPKVCompareScore", "sempointer"), ("FullKVCompareScore", "full"),
                        ("SnapKVHalfScore", "kvpress_snapkv_0.5"), ("TovaHalfScore", "kvpress_tova_0.5"),
                        ("KnormHalfScore", "kvpress_knorm_0.5"), ("SnapKVQuarterScore", "kvpress_snapkv_0.25"),
                        ("TovaQuarterScore", "kvpress_tova_0.25")):
        values[macro] = fmt(score(kv, name))
    scores = [score(kv, name) for name in (kv.names() if kv else []) if name.startswith("kvpress_")]
    values["BestKVPressScore"] = fmt(max((v for v in scores if v is not None), default=None))
    for macro, name in (("SubTextScore", "sempointer"), ("SubKVScore", "sempointer__substrate-kv")):
        values[macro] = fmt(score(kv, name))
    for macro, name, key, kind in (("SubTextTTFT", "sempointer", "ttft_ms", "int"),
        ("SubKVTTFT", "sempointer__substrate-kv", "ttft_ms", "int"),
        ("SubTextVRAM", "sempointer", "peak_vram_mb", "int"),
        ("SubKVVRAM", "sempointer__substrate-kv", "peak_vram_mb", "int")):
        values[macro] = fmt(field(kv, name, key), kind)
    values["SubKVTTFTSpeedup"] = fmt(ratio(field(kv, "sempointer", "ttft_ms"),
                                            field(kv, "sempointer__substrate-kv", "ttft_ms")), "float")
    abl = runs.get("paper_ablation")
    for macro, name in (("AblPointerKeyword", "sempointer"), ("AblPointerRandom", "sempointer__ptype-random_id"),
                        ("AblPointerLead", "sempointer__ptype-lead"), ("AblPointerNone", "sempointer__ptype-none"),
                        ("AblSelDense", "sempointer__selector-dense"), ("AblSelBM", "sempointer__selector-bm25"),
                        ("AblSelHybrid", "sempointer__selector-hybrid"),
                        ("AblSelAttention", "sempointer__selector-attention"),
                        ("AblSelHybridAttention", "sempointer"),
                        ("AblHopsOne", "sempointer"), ("AblHopsTwo", "sempointer__hops-2"),
                        ("AblAdaptive", "sempointer__m-adaptive"),
                        ("AblKTwo", "sempointer__k-2"), ("AblKFour", "sempointer__k-4"),
                        ("AblKEight", "sempointer__k-8"), ("AblKSixteen", "sempointer"),
                        ("AblKThirtyTwo", "sempointer__k-32"), ("AblMOne", "sempointer__m-1"),
                        ("AblMTwo", "sempointer"), ("AblMThree", "sempointer__m-3"),
                        ("AblMFour", "sempointer__m-4")):
        values[macro] = fmt(score(abl, name))
    model2 = runs.get("paper_model2")
    for macro, name in (("MTwoSPRuler", "sempointer"), ("MTwoFullRuler", "full"),
                        ("MTwoHybridRuler", "hybrid_m4")):
        values[macro] = fmt(score(model2, name))
    values["NTotalPredictions"] = str(sum(len(run.rows) for run in runs.values())) if runs else TBD
    lines = ["% Generated from experiments/results/paper_* only. Missing observations are TBD."]
    names.extend(name for name in values if name not in names)
    lines.extend(f"\\providecommand{{\\{name}}}{{{values.get(name, TBD)}}}" for name in names)
    (out / "macros.tex").write_text("\n".join(lines) + "\n")


def write_figures(runs: dict[str, Run], out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        for name in FIGURES:
            (out / name).unlink(missing_ok=True)
        return ["matplotlib unavailable: figure PDFs omitted"]
    notes: list[str] = []
    def plot(name: str, title: str, xlabel: str, ylabel: str,
             series: list[tuple[str, list[tuple[float, float]]]], logx: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        populated = False
        for label, points in series:
            if points:
                x, y = zip(*sorted(points))
                ax.plot(x, y, marker="o", linewidth=1.6, label=label)
                populated = True
        if not populated:
            plt.close(fig)
            (out / name).unlink(missing_ok=True)
            notes.append(f"{name}: skipped (no measured points)")
            return
        if logx:
            ax.set_xscale("log")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(out / name)
        plt.close(fig)
        notes.append(f"{name}: measured points")
    ruler = main_run(runs, "ruler")
    series = []
    paired_ruler = common_rows(ruler, ("sempointer", "full", "rag_m4", "hybrid_m4"))
    for name in ("sempointer", "full", "rag_m4", "hybrid_m4"):
        rs = paired_ruler[name]
        lengths = sorted({length_of(r) for r in rs if length_of(r) is not None})
        series.append((name, [(n, v * 100) for n in lengths if (v := avg(
            [r for r in rs if length_of(r) == n], "score")) is not None]))
    plot("fig_accuracy_vs_length.pdf", "RULER by context length", "Requested tokens", "Score (%)", series)
    series = []
    for bench in ("ruler", "longbench", "locomo"):
        run = main_run(runs, bench)
        points = [(tok, sc * 100) for name in MAIN_SYSTEMS[bench]
                  if (tok := field(run, name, "active_tokens")) is not None
                  and (sc := score(run, name)) is not None]
        series.append((BENCH_LABELS[bench], points))
    plot("fig_pareto_tokens_accuracy.pdf", "Active tokens versus score", "Active tokens", "Score (%)", series, True)
    abl = runs.get("paper_ablation")
    series = [("Score", [(k, v * 100) for k in (2, 4, 8, 16, 32)
                         if (v := score(abl, "sempointer" if k == 16
                                        else f"sempointer__k-{k}")) is not None])]
    plot("fig_k_ablation.pdf", "Pointer-length ablation", "Pointer tokens $k$", "Score (%)", series)
    efficiency = runs.get("paper_efficiency")
    # Amortization requires measured ingest latency, not merely answer latency.
    amortized = []
    for name in ("full", "sempointer"):
        rs = system_rows(efficiency, name)
        ingest, answer = avg(rs, "ingest_ms"), avg(rs, "latency_ms")
        amortized.append((name, [(q, ingest + q * answer) for q in (1, 2, 4, 8, 16, 32)]
                          if ingest is not None and answer is not None else []))
    plot("fig_amortization.pdf", "Projected cumulative time from measured means",
         "Queries per context", "Time (ms)", amortized)
    memory = []
    for name in ("full", "sempointer", "sempointer__substrate-kv"):
        rs = system_rows(efficiency, name)
        buckets: dict[int, list[float]] = defaultdict(list)
        for row in rs:
            n, v = finite_number(row.get("n_blocks")), finite_number(row.get("kv_bytes"))
            if n is not None and v is not None:
                buckets[int(n)].append(v)
        memory.append((name, [(n, mean(v) / 2**20) for n, v in sorted(buckets.items())]))
    plot("fig_kv_memory.pdf", "Measured KV footprint", "Blocks", "KV memory (MiB)", memory)
    kv = main_run(runs, "kv_compare")
    latency = []
    for name in ("sempointer", "sempointer__substrate-kv", "full", "kvpress_snapkv_0.3"):
        vram, ttft = field(kv, name, "peak_vram_mb"), field(kv, name, "ttft_ms")
        latency.append((name, [(vram, ttft)] if vram is not None and ttft is not None else []))
    plot("fig_latency_vram.pdf", "TTFT versus peak VRAM", "Peak VRAM (MiB)", "TTFT (ms)", latency)
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs, notes = load_runs(args.results)
    if not runs:
        notes.append("No valid paper_* runs were eligible.")
    write_macros(runs, args.output)
    write_tables(runs, args.output)
    notes += write_figures(runs, args.output)
    (args.output / "REPORT.md").write_text(
        "# Paper asset provenance\n\nOnly `results/paper_*` was eligible. "
        "Other historical results were excluded. Valid scored observations from "
        "partial stages are included. TBD means a run, system score, or field was "
        "not measured; it never means zero. Scores average the latest scored row "
        "for each system and example. Tables show scored/expected and outstanding "
        "error counts; an asterisk marks a partial score. System means can therefore "
        "use different example sets. Paired tests and RULER length plots use only "
        "common scored example IDs. Invalid or inconsistent runs are excluded.\n\n" +
        "\n".join(f"- {line}" for line in notes) + "\n")
    print("\n".join(notes) if notes else "No paper_* runs found; wrote TBD assets.")


if __name__ == "__main__":
    main()
