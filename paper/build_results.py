"""Build SemPointer paper tables from completed prediction records, without inference."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analysis.stats import bootstrap_mean_ci, paired_bootstrap_diff, sign_flip_permutation_p  # noqa: E402

OUT = Path(__file__).resolve().parent / "generated"
RUNS = ROOT / "experiments" / "results"
BENCHES = {"ruler": 180, "longbench": 600, "locomo": 497}


def load(name: str) -> dict[tuple[str, str], dict]:
    folder = RUNS / name
    summary = json.loads((folder / "metrics.json").read_text())
    expected = BENCHES[name.removeprefix("v2_").removeprefix("paper_")]
    if summary["n_examples"] != expected:
        raise ValueError(f"{name}: expected {expected} examples")
    rows: dict[tuple[str, str], dict] = {}
    for line in (folder / "predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("score") is not None:
            rows[(row["system"], row["example_id"])] = row
    for system, metrics in summary["systems"].items():
        found = sum(key[0] == system for key in rows)
        if found != expected or metrics["n_scored"] != expected or metrics["n_errors"]:
            raise ValueError(f"{name}/{system}: incomplete or inconsistent ({found}/{expected})")
    return rows


def rows_for(rows: dict, system: str) -> list[dict]:
    return [row for (name, _), row in rows.items() if name == system]


def mean_ci(values: list[float], scale: float = 1) -> str:
    point, low, high = bootstrap_mean_ci(values, seed=42)
    return f"{point * scale:.1f} [{low * scale:.1f}, {high * scale:.1f}]"


def paired(a: dict, a_name: str, b: dict, b_name: str, field: str, scale: float = 1):
    ids = sorted({i for s, i in a if s == a_name} & {i for s, i in b if s == b_name})
    if not ids:
        raise ValueError(f"no shared examples: {a_name}, {b_name}")
    av = [a[(a_name, i)][field] for i in ids]
    bv = [b[(b_name, i)][field] for i in ids]
    p, lo, hi = paired_bootstrap_diff(av, bv, seed=42)
    return len(ids), p * scale, lo * scale, hi * scale, sign_flip_permutation_p(
        [x - y for x, y in zip(av, bv)], seed=42)


def table(name: str, caption: str, label: str, layout: str,
          header: list[str], body: list[list[str]]) -> None:
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{layout}}}", r"\toprule",
             " & ".join(header) + " \\\\", r"\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / name).write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    v2 = {b: load(f"v2_{b}") for b in BENCHES}
    v1 = {b: load(f"paper_{b}") for b in BENCHES}
    labels = {"ruler": "RULER", "longbench": "LongBench", "locomo": "LoCoMo"}

    main_rows = []
    for bench, data in v2.items():
        for system, label in (("sempointer_v2", "SemPointer"),
                              ("bm25_m2", "BM25-2"), ("bm25_m4", "BM25-4")):
            if system == "bm25_m2" and bench != "ruler":
                continue
            rs = rows_for(data, system)
            main_rows.append([labels[bench], label, str(len(rs)),
                              mean_ci([r["score"] for r in rs], 100),
                              f"{mean(r['active_tokens'] for r in rs):,.0f}",
                              f"{mean(r['latency_ms'] for r in rs):,.0f}",
                              f"{mean(r['ttft_ms'] for r in rs):,.0f}"])
    table("main.tex", "Same-run results. Score is mean [95\\% example-bootstrap CI] in percent; tokens, answer latency, and TTFT are per-query means. Ingestion is excluded. LoCoMo uses our token-F1 adaptation.",
          "tab:main", "llrlrrr", ["Task", "System", "$n$", "Score [95\\% CI]", "Tokens", "Answer ms", "TTFT ms"], main_rows)

    paired_rows = []
    for bench, data in v2.items():
        _, score, slo, shi, p = paired(data, "sempointer_v2", data, "bm25_m4", "score", 100)
        _, tok, tlo, thi, _ = paired(data, "sempointer_v2", data, "bm25_m4", "active_tokens")
        _, lat, llo, lhi, _ = paired(data, "sempointer_v2", data, "bm25_m4", "latency_ms")
        paired_rows.append([labels[bench], f"{score:+.2f} [{slo:+.2f}, {shi:+.2f}]",
                            f"{p:.3f}", f"{tok:+.0f} [{tlo:+.0f}, {thi:+.0f}]",
                            f"{lat:+.0f} [{llo:+.0f}, {lhi:+.0f}]"])
    table("paired.tex", "Paired SemPointer minus BM25-4 on identical example IDs. Intervals are 95\\% bootstrap CIs; $p$ is a two-sided sign-flip test of the score difference. Negative token and latency differences favor SemPointer.",
          "tab:paired", "lrlrr", ["Task", "$\\Delta$ score (pp) [CI]", "$p$", "$\\Delta$ tokens [CI]", "$\\Delta$ answer ms [CI]"], paired_rows)

    ruler_rows = []
    for task in ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "vt", "cwe"):
        line = [task.replace("_", r"\_")]
        for data, system in ((v2["ruler"], "sempointer_v2"),
                             (v2["ruler"], "bm25_m2"), (v2["ruler"], "bm25_m4"),
                             (v1["ruler"], "sempointer"), (v1["ruler"], "full")):
            rs = [r for r in rows_for(data, system) if r["task"] == task]
            line.append(f"{mean(r['score'] for r in rs) * 100:.1f}")
        ruler_rows.append(line)
    table("ruler.tex", "RULER score (\\%) by task, 30 examples each. V1 and full-context columns use the earlier V1 run on the same seeded IDs; timing comparisons are restricted to Table~\\ref{tab:main}.",
          "tab:ruler", "lrrrrr", ["Task", "SemPointer", "BM25-2", "BM25-4", "V1", "Full"], ruler_rows)

    route_rows = []
    for bench, data in v2.items():
        rs = rows_for(data, "sempointer_v2")
        two = sum(r["router_route"] == "lexical" for r in rs)
        four = sum(r["router_route"] == "expanded_lexical" for r in rs)
        if two + four != len(rs):
            raise ValueError(f"{bench}: unexpected route")
        route_rows.append([labels[bench], str(two), str(four), f"{two / len(rs) * 100:.1f}",
                           f"{mean(r['n_selected'] for r in rs):.2f}"])
    table("routes.tex", "Observed SemPointer routing decisions. The two-block route can still include a different block count if token-budget filtering applies.",
          "tab:routes", "lrrrr", ["Task", "2-block route", "4-block route", "2-block (\\%)", "Mean selected"], route_rows)

    for bench, data in v2.items():
        n, d, lo, hi, p = paired(data, "sempointer_v2", v1[bench], "sempointer", "score", 100)
        print(f"SemPointer minus V1 {bench}: n={n}, {d:+.2f} pp [{lo:+.2f}, {hi:+.2f}], p={p:.4f}")


if __name__ == "__main__":
    main()
