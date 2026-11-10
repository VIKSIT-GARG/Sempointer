"""Render revision tables from retained results; no model imports or new inference.

Writes only paper/generated/revision_*.tex and their source-hash manifest.
SQLite sources are opened read-only. Counts are checked against saved analyses.
"""
from __future__ import annotations

from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/results"
FULL = RUNS / "bounded_pointer_full_v1"
OUT = ROOT / "paper/generated"
SOURCES: set[Path] = set()
LABELS = {
    "ruler": "RULER", "longbench": "LongBench", "locomo": "LoCoMo",
    "longmem": "LongMemEval", "babilong": "BABILong", "longbench_v2": "LongBench v2",
    "infinite": "InfiniteBench", "loogle": "LooGLE", "leval": "L-Eval", "scrolls": "SCROLLS",
}


def read_json(path):
    SOURCES.add(path)
    return json.loads(path.read_text())


def read_stage(path, family=None):
    SOURCES.add(path)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        query = """SELECT example_id, system, score, active_tokens,
                            json_extract(payload, '$.generation_ms')
                            FROM latest WHERE status='scored'"""
        data = db.execute(query + (" AND family=?" if family else ""),
                          (family,) if family else ()).fetchall()
    return {(system, eid): (score, tokens, timing) for eid, system, score, tokens, timing in data}


def paired(data, on, off):
    ids = sorted({eid for system, eid in data if system == on} &
                 {eid for system, eid in data if system == off})
    if not ids:
        raise ValueError(f"No paired records: {on}/{off}")
    a, b = [data[on, eid] for eid in ids], [data[off, eid] for eid in ids]
    wins = sum(x[0] > y[0] + 1e-12 for x, y in zip(a, b))
    losses = sum(x[0] < y[0] - 1e-12 for x, y in zip(a, b))
    token_change = 100 * (sum(x[1] for x in a) / sum(x[1] for x in b) - 1)
    time_change = 100 * (sum(x[2] for x in a) / sum(x[2] for x in b) - 1)
    return len(ids), wins, len(ids) - wins - losses, losses, token_change, time_change


def table(name, caption, label, layout, header, body, font=r"\small"):
    placement = "H" if name.startswith("submission_") else "htbp"
    lines = ["% Derived from retained records; rebuild with paper/build_revision_tables.py.",
             f"\\begin{{table}}[{placement}]", r"\centering", font,
             r"\setlength{\tabcolsep}{4pt}", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{layout}}}",
             r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / name).write_text("\n".join(lines))


def task_inventory(control):
    path = FULL / "fixed_payload_pointer_control_v1/per_task.csv"
    SOURCES.add(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    positive = sum(float(r["mean_score_difference"]) > 1e-12 for r in rows)
    negative = sum(float(r["mean_score_difference"]) < -1e-12 for r in rows)
    if len(rows) != 123 or sum(int(r["n"]) for r in rows) != 990:
        raise ValueError("Task inventory does not partition the active control")
    if (positive, negative) != (25, 22):
        raise ValueError("Task-effect inventory changed")
    header = r"Family & Task/metric & $n/c$ & Off & On & $\Delta$ & $p_{\mathrm{Holm}}$ \\"
    lines = [r"\begingroup\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{lp{3.1cm}rrrrr}",
             r"\caption{Complete 4K descriptor task inventory: all 123 active task/metric groups, including 25 positive, 22 negative, and 76 zero mean differences. Scores are native task scores multiplied by 100; they are not pooled across tasks. Bold marks a positive point estimate, not statistical significance. $n/c$ gives questions/exact contexts. A dash means fewer than five contexts and descriptive reporting only. Holm values use the 53 eligible groups; variable tracking is 0.060 under the wider family.}\label{tab:task-inventory}\\",
             r"\toprule", header, r"\midrule", r"\endfirsthead",
             r"\multicolumn{7}{l}{Complete descriptor task inventory (continued)}\\",
             r"\toprule", header, r"\midrule", r"\endhead",
             r"\midrule\multicolumn{7}{r}{Continued on next page}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    for r in rows:
        delta = 100 * float(r["mean_score_difference"])
        effect = f"{delta:+.2f}"
        if delta > 1e-10:
            effect = r"\textbf{" + effect + "}"
        task = r["task"].replace("_", r"\_\allowbreak ").replace("/", r"/\allowbreak ")
        metric = r["metric"].replace("_", r"\_")
        p = f"{float(r['holm_adjusted_p']):.3f}" if r["holm_adjusted_p"] else "--"
        cells = [LABELS[r["family"]], r"\raggedright " + task + r" {\tiny (" + metric + ")}",
                 f"{r['n']}/{r['source_contexts']}", f"{100*float(r['off_mean_score']):.2f}",
                 f"{100*float(r['on_mean_score']):.2f}", effect, p]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\end{longtable}", r"\endgroup", ""]
    (OUT / "revision_task_inventory.tex").write_text("\n".join(lines))
    if control["task_inference"]["eligible_groups"] != 53:
        raise ValueError("Multiplicity family changed")


def expanded_benefit_tables(control, old):
    families, coverage, efficiency = [], [], []
    for family, name in LABELS.items():
        r = control["families"][family]
        families.append([name, r["wins"], r["ties"], r["losses"],
                         f"{r['extra_input_pct']:.2f}", f"{r['generation_time_reduction_pct']:+.2f}"])
        data = read_stage(FULL / "results/test_4096/results.sqlite", family)
        n, w, t, loss, tokens, timing = paired(data, "bounded_pointer", "bm25_budget")
        if n != 100:
            raise ValueError(f"Incomplete family comparison: {family}")
        efficiency.append([name, f"{w}/{t}/{loss}", f"{-tokens:.2f}", f"{-timing:.2f}"])
    table("revision_descriptor_families.tex",
          r"Complete family-level descriptor control at 4K, 100 pairs per family including identical-input checks. W/T/L compare pointer-on with pointer-off at fixed expanded text. Extra input is relative to pointer-off. Generation savings are descriptive and sensitive to execution order; positive values mean shorter recorded time. Family win balances are not significance tests.",
          "tab:descriptor-families", "lrrrrr", ["Family", "W", "T", "L", r"Extra input (\%)", r"Gen. saving (\%)"], families)
    table("revision_family_efficiency.tex",
          r"4K efficiency across every benchmark family: bounded SemPointer versus BM25 filling the same input allowance, 100 scored pairs each. Positive savings mean fewer input tokens or shorter generation time. W/T/L retain the associated task-score tradeoff.",
          "tab:family-efficiency", "lrrr", ["Family", "W/T/L", r"Token savings (\%)", r"Time savings (\%)"], efficiency)
    for budget in (1024, 2048, 4096):
        path = FULL / f"results/test_{budget}/results.sqlite"
        SOURCES.add(path)
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            counts = dict(db.execute("SELECT system,count(*) FROM latest WHERE status='scored' GROUP BY system"))
        coverage.append([f"{budget:,}", counts["bounded_pointer"], counts["adaptive_no_index"],
                         counts["bm25_budget"], counts["original_sempointer"]])
    table("revision_coverage.tex",
          r"Scored coverage out of the same 1,000 questions per input limit. The bounded version reaches 100\% coverage at 2K and 4K; coverage means an evaluated request, not a correct answer. Skips are reported explicitly and no inference errors were recorded.",
          "tab:coverage", "rrrrr", ["Input limit", "Bounded", "No index", "BM25 budget", "Full index"], coverage)
    original = []
    for system, name in [("bm25_2", "BM25-2"), ("bm25_4", "BM25-4"),
                         ("bm25_matched_budget", "Matched-token BM25"), ("pointer_off", "Pointer-off"),
                         ("full_context", "Full context")]:
        n, w, t, loss, tokens, timing = paired(old, "sempointer", system)
        original.append([name, n, f"{w}/{t}/{loss}", f"{tokens:+.2f}", f"{timing:+.2f}"])
    table("revision_original_controls.tex",
          r"All original-index Qwen3.5 controls, using scored shared IDs. Signed cost changes are SemPointer relative to the named baseline; negative means lower cost. Baselines have different coverage. These historical controls are distinct from the bounded architecture.",
          "tab:original-controls", "lrrrr", ["Baseline", "$n$", "W/T/L", r"$\Delta$ input (\%)", r"$\Delta$ gen. (\%)"], original)
    pilot = []
    for budget in (1024, 2048, 4096):
        data = read_stage(RUNS / f"bounded_pointer_pilot_v1/results/test_{budget}/results.sqlite")
        n, w, t, loss, tokens, timing = paired(data, "bounded_pointer", "bm25_budget")
        _, nw, nt, nl, _, _ = paired(data, "bounded_pointer", "adaptive_no_index")
        pilot.append([f"{budget:,}", n, f"{w}/{t}/{loss}", f"{-tokens:.2f}", f"{-timing:.2f}", f"{nw}/{nt}/{nl}"])
    table("revision_pilot.tex",
          r"Retained 100-question pilot test, separate from the 1,000-question full study. W/T/L are bounded relative to the named baseline; savings use paired BM25 totals. At 1K/2K the no-index comparison need not hold payload fixed. Pilot descriptor gains are not an independent replication of an overall full-study benefit.",
          "tab:pilot", "rrrrrr", ["Input", "$n$", "vs BM25 W/T/L", r"Tokens saved (\%)", r"Time saved (\%)", "vs off W/T/L"], pilot, font=r"\scriptsize")
    original_main = []
    for family in ("ruler", "longbench", "locomo"):
        data = read_json(RUNS / f"paper_{family}/metrics.json")
        names = {"sempointer": "V1 SemPointer", "sempointer__selector-hybrid": "V1 hybrid selector",
                 "full": "Full context", "rag_m1": "Dense RAG-1", "rag_m4": "Dense RAG-4",
                 "bm25_m4": "BM25-4", "hybrid_m4": "Hybrid-4"}
        for system, r in data["systems"].items():
            if r["n_scored"] != data["n_examples"] or r["n_errors"]:
                raise ValueError(f"Incomplete V1 main-study arm: {family}/{system}")
            original_main.append([LABELS[family], names[system], r["n_scored"],
                                  f"{r['score_mean']*100:.2f}", f"{r['active_tokens_mean']:,.0f}",
                                  f"{r['latency_ms_mean']:,.0f}"])
    table("revision_v1_main.tex",
          r"Complete retained V1 main-study controls under Qwen2.5. Each collection has identical scored IDs across its arms; metrics differ across collections. These original text-path results have different architecture/protocol conditions from the bounded main study.",
          "tab:v1-main", "llrrrr", ["Collection", "Arm", "$n$", "Score", "Input", "Answer ms"], original_main, font=r"\scriptsize")
    for run, filename, label, caption in [
        ("paper_ablation", "revision_legacy_ablation.tex", "tab:legacy-ablation",
         r"Complete retained V1 ablation: 120 Qwen2.5 questions per arm. Scores use the historical RULER metric. These variant comparisons are exploratory, can alter selection/payload, and are not the newer fixed-payload descriptor intervention."),
        ("paper_kv", "revision_legacy_kv.tex", "tab:legacy-kv",
         r"Complete retained V1 text/KV comparison: 50 Qwen2.5 questions per arm. The three KVpress arms use compression setting 0.3. This is a small historical configuration comparison, not validation of KV serving for the current bounded design.")]:
        data = read_json(RUNS / run / "metrics.json")
        names = {
            "sempointer": "V1 default text", "sempointer__ptype-random_id": "Random IDs",
            "sempointer__ptype-none": "No pointers", "sempointer__ptype-lead": "Lead descriptors",
            "sempointer__k-8": "8-token descriptors", "sempointer__k-32": "32-token descriptors",
            "sempointer__m-1": "One selected block", "sempointer__m-adaptive": "Adaptive block count",
            "sempointer__hops-2": "Two selection hops", "sempointer__selector-dense": "Dense selector",
            "sempointer__selector-bm25": "BM25 selector", "sempointer__selector-attention": "Attention selector",
            "sempointer__substrate-kv": "Cached KV substrate", "full": "Full context",
            "kvpress_snapkv_0.3": "SnapKV (0.3)", "kvpress_tova_0.3": "TOVA (0.3)",
            "kvpress_knorm_0.3": "K-norm (0.3)"}
        body = []
        for name, r in data["systems"].items():
            if r["n_scored"] != data["n_examples"] or r["n_errors"]:
                raise ValueError(f"Incomplete legacy arm: {run}/{name}")
            body.append([names[name], f"{r['score_mean']*100:.2f}", f"{r['active_tokens_mean']:,.0f}",
                         f"{r['latency_ms_mean']:,.0f}", f"{r['ttft_ms_mean']:,.0f}" if "ttft_ms_mean" in r else "--"])
        table(filename, caption, label, "lrrrr", ["Arm", "Score", "Input", "Answer ms", "TTFT ms"], body, font=r"\scriptsize")
    task_inventory(control)


def submission_tables(control, audit, history):
    """Compact views for the ten-page paper; extended tables remain unchanged."""
    budgets, families = [], []
    for budget in (1024, 2048, 4096):
        data = read_stage(FULL / f"results/test_{budget}/results.sqlite")
        n, w, t, loss, tok, timing = paired(data, "bounded_pointer", "bm25_budget")
        original_n = sum(system == "original_sempointer" for system, _ in data)
        budgets.append([f"{budget:,}", n, f"{w}/{t}/{loss}", f"{-tok:.2f}", f"{-timing:.2f}", original_n])
    table("submission_budgets.tex",
          r"Bounded versus budget-filling BM25: paired W/T/L and cost savings. $n$ is bounded/BM25 scored coverage out of 1,000; last column is full-index coverage. Input ceilings are equal, realized tokens need not be.",
          "tab:sub-budgets", "rrrrrr", ["Limit", "$n$", "W/T/L", r"Tokens saved (\%)", r"Time saved (\%)", "Index $n$"], budgets, font=r"\footnotesize")
    for family, name in LABELS.items():
        data = read_stage(FULL / "results/test_4096/results.sqlite", family)
        n, w, t, loss, tok, timing = paired(data, "bounded_pointer", "bm25_budget")
        if n != 100:
            raise ValueError(f"Submission family coverage changed: {family}")
        r = control["families"][family]
        families.append([name, f"{-tok:.2f}", f"{-timing:.2f}", f"{w}/{t}/{loss}", f"{r['wins']}/{r['ties']}/{r['losses']}"])
    table("submission_families.tex",
          r"All ten families at 4K, 100 pairs each. Savings compare bounded with budget-filling BM25. W/T/L columns distinguish that comparison from pointer-on versus fixed-payload pointer-off; counts are descriptive, including null interventions.",
          "tab:sub-families", "lrrrr", ["Family", r"Tokens saved (\%)", r"Time saved (\%)", "vs BM25", "vs off"], families, font=r"\footnotesize")
    pairs = []
    for key, name in [("all_pairs", "All verified"), ("active_descriptor_pairs", "Descriptors present"), ("null_intervention_pairs", "Identical inputs")]:
        r = control[key]
        pairs.append([name, r["n"], r["wins"], r["ties"], r["losses"]])
    table("submission_control.tex", r"Verified fixed-payload control at 4K: pointer-on versus pointer-off. The last two rows partition the first.",
          "tab:sub-control", "lrrrr", ["Pairs", "$n$", "Wins", "Ties", "Losses"], pairs, font=r"\footnotesize")
    table("submission_history.tex",
          r"Retained full-index history versus same-run BM25-4. Signed cost changes favor SemPointer when negative. Time is answer-call time for Qwen2.5 and generation time for Qwen3.5; rows differ in model, sample, and protocol.",
          "tab:sub-history", "llrrrr", ["Model", "Collection", "$n$", "W/T/L", r"$\Delta$ tokens (\%)", r"$\Delta$ time (\%)"], history, font=r"\footnotesize")
    d = audit["diagnoses"]
    evidence = [["Fine: omissions retained in full by BM25", d["annotated_units_omitted_bm25_retains"]],
                ["Fine: all annotated units present in bounded", d["annotated_units_present_cause_unresolved"]],
                ["Fine: incomplete annotated sets in both", d["annotated_units_incomplete_both"]],
                ["Coarse: support sessions only", d["support_session_coverage_requires_review"]]]
    evidence += [["No usable aligned fine annotation", audit["losses"] - sum(r[1] for r in evidence)], [r"\textbf{Total}", audit["losses"]]]
    table("submission_audit.tex", r"Evidence partition of all 229 lower-scoring 4K cases. Annotation coverage is not a causal failure label.",
          "tab:sub-audit", "lr", ["Evidence status", "Cases"], evidence, font=r"\footnotesize")
    path = FULL / "fixed_payload_pointer_control_v1/per_task.csv"
    SOURCES.add(path)
    with path.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if float(r["mean_score_difference"]) > 1e-12]
    if len(rows) != 25:
        raise ValueError("Positive task inventory changed")
    aliases = {"gov_report_summ": "Gov. report", "meeting_summ": "Meeting summary", "multidoc_qa": "Multi-doc QA",
               "narrative_qa": "NarrativeQA", "natural_question": "Natural questions", "news_summ": "News summary",
               "paper_assistant": "Paper assistant", "scientific_qa": "Scientific QA", "topic_retrieval_longchat": "Topic retrieval",
               "qa": "QA", "hotpotqa": "HotpotQA", "data": "Choice QA", "longmemeval_s_cleaned": "Memory QA",
               "shortdep_cloze": "Short cloze", "shortdep_qa": "Short QA", "vt": "Variable tracking",
               "gov_report": "Gov. report", "qmsum": "QMSum", "quality": "QuALITY"}
    positive = []
    for r in rows:
        task = r["task"].removeprefix("data/") if r["family"] == "babilong" else aliases[r["task"]]
        p = f"{float(r['holm_adjusted_p']):.3f}" if r["holm_adjusted_p"] else "--"
        positive.append([LABELS[r["family"]], task, f"{r['n']}/{r['source_contexts']}",
                         f"{100*float(r['off_mean_score']):.2f}", f"{100*float(r['on_mean_score']):.2f}",
                         f"{100*float(r['mean_score_difference']):+.2f}", p])
    table("submission_positive_tasks.tex",
          r"All 25 positive mean differences among 123 active task/metric groups (22 negative, 76 zero). Scores are native metric $\times100$, not a pooled accuracy; $n/c$ is questions/exact contexts. Holm uses 53 eligible groups; dashes are descriptive groups with fewer than five contexts. Variable tracking is 0.060 under all 123 hypotheses.",
          "tab:sub-positive", "llrrrrr", ["Family", "Task", "$n/c$", "Off", "On", "$\Delta$", "Holm $p$"], positive, font=r"\footnotesize")


def main():
    OUT.mkdir(exist_ok=True)
    summary = read_json(FULL / "results/summary.json")
    control = read_json(FULL / "fixed_payload_pointer_control_v1/summary.json")
    audit = read_json(FULL / "loss_audit_4096_v1/summary.json")
    if summary["status"] != "complete" or control["source_run_id"] != audit["run_id"]:
        raise ValueError("Incomplete or mismatched source runs")
    history = []
    for family in ("ruler", "longbench", "locomo"):
        path = RUNS / f"v2_{family}/predictions.jsonl"
        SOURCES.add(path)
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        data = {(r["system"], r["example_id"]): (r["score"], r["active_tokens"], r["latency_ms"])
                for r in rows if r.get("score") is not None}
        n, w, t, loss, tok, timing = paired(data, "sempointer_v2", "bm25_m4")
        history.append(["Qwen2.5", LABELS[family], n, f"{w}/{t}/{loss}", f"{tok:+.1f}", f"{timing:+.1f}"])
    old = read_stage(RUNS / "local_ten_qwen35_nf4_100_v1/results/test/results.sqlite")
    n, w, t, loss, tok, timing = paired(old, "sempointer", "bm25_4")
    if (n, w, t, loss) != (936, 137, 624, 175):
        raise ValueError("Historical Qwen3.5 comparison changed")
    history.append(["Qwen3.5", "Ten families", n, f"{w}/{t}/{loss}", f"{tok:+.1f}", f"{timing:+.1f}"])
    table("revision_history.tex",
          r"Retained full-index history versus same-run BM25-4. W/T/L are SemPointer's question-wise score wins/ties/losses. Signed cost changes are relative to BM25: negative means lower cost. Qwen2.5 reports answer-call time; Qwen3.5 reports generation time. Models, samples, and protocols differ, so rows are not a controlled cross-model test.",
          "tab:history", "llrrrr", ["Model", "Collection", "$n$", "W/T/L", r"$\Delta$ tokens (\%)", r"$\Delta$ time (\%)"], history)
    budgets = []
    for budget in (1024, 2048, 4096):
        data = read_stage(FULL / f"results/test_{budget}/results.sqlite")
        n, w, t, loss, tok, timing = paired(data, "bounded_pointer", "bm25_budget")
        expected = summary["comparisons"][f"test_{budget}"]["bm25_budget"]
        if (n, w, t, loss) != tuple(expected[k] for k in ("n_paired", "wins", "ties", "losses")):
            raise ValueError(f"Budget {budget}: paired counts disagree")
        budgets.append([f"{budget:,}", n, w, t, loss, f"{-tok:.2f}", f"{-timing:.2f}"])
    table("revision_budgets.tex",
          r"Bounded SemPointer versus BM25 filling the same maximum input allowance. Counts use scored shared IDs; input limits are not equal realized token counts. Savings use paired total input tokens and generation time (prefill plus decode), excluding ingestion and shared prompt preparation.",
          "tab:budgets", "rrrrrrr", ["Input limit", "$n$", "Wins", "Ties", "Losses", r"Token savings (\%)", r"Time savings (\%)"], budgets)
    controls = []
    for key, name in [("all_pairs", "All verified pairs"), ("active_descriptor_pairs", "Descriptors present"),
                      ("null_intervention_pairs", "Identical-input checks")]:
        r = control[key]
        if r["wins"] + r["ties"] + r["losses"] != r["n"]:
            raise ValueError(f"Control {key}: counts do not sum")
        controls.append([name, r["n"], r["wins"], r["ties"], r["losses"], f"{r['extra_input_pct']:+.2f}"])
    table("revision_control.tex",
          r"Verified 4K fixed-payload control: pointer-on relative to pointer-off. Expanded text and all prompt content outside the descriptor section match exactly. The last two rows partition all pairs; they are not independent replications. Input overhead is relative to pointer-off.",
          "tab:control", "lrrrrr", ["Pairs", "$n$", "Wins", "Ties", "Losses", r"Extra input (\%)"], controls)
    families = []
    for family, name in LABELS.items():
        r = audit["family_details"][family]
        families.append([name, r["losses"], r["summaries"],
                         r["diagnoses"].get("annotated_units_omitted_bm25_retains", 0), r["pointer_off_rescues"]])
    if sum(r[1] for r in families) != audit["losses"] or audit["losses"] != 229:
        raise ValueError("Loss-family counts disagree")
    families.append([r"\textbf{Total}", 229, audit["summary_task_losses"],
                     audit["diagnoses"]["annotated_units_omitted_bm25_retains"], audit["pointer_off_rescues"]])
    table("revision_loss_families.tex",
          r"All 229 lower-scoring bounded-pointer cases versus budget-filling BM25 at 4K. Omission counts require aligned fine-grained annotations: bounded omits annotated units while BM25 retains all. Zero does not establish complete evidence. The last column counts cases where pointer-off reaches or exceeds BM25; columns overlap and are not causal labels.",
          "tab:loss-families", "lrrrr", ["Family", "Losses", "Summary tasks", "Omissions", "Off reaches BM25"], families)
    d = audit["diagnoses"]
    evidence = [
        ["Fine: bounded omits units; BM25 retains all", d["annotated_units_omitted_bm25_retains"]],
        ["Fine: all annotated units present in bounded", d["annotated_units_present_cause_unresolved"]],
        ["Fine: incomplete annotated sets in both arms", d["annotated_units_incomplete_both"]],
        ["Coarse: LongMemEval support-session labels only", d["support_session_coverage_requires_review"]],
    ]
    evidence.append(["No usable aligned fine-grained annotation", audit["losses"] - sum(r[1] for r in evidence)])
    evidence.append([r"\textbf{Total}", audit["losses"]])
    table("revision_evidence.tex",
          r"Evidence coverage partition of the loss audit. Fine-grained checks align in 43 cases; 11 more have coarse session labels. The remaining 175 include two unmapped annotations. Annotation coverage does not establish a unique cause of the score difference.",
          "tab:evidence", "lr", ["Evidence status", "Cases"], evidence)
    expanded_benefit_tables(control, old)
    submission_tables(control, audit, history)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)}
    (OUT / "revision_sources.json").write_text(json.dumps({"new_generations": 0, "sources_sha256": sources}, indent=2) + "\n")
    print(f"Built 14 extended and 6 submission tables from {len(SOURCES)} retained sources; no inference.")


if __name__ == "__main__":
    main()
