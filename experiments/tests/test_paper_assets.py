"""Paper artifacts must use only new real-run rows and keep missing data visible."""

import json
from pathlib import Path

from analysis.make_paper_assets import load_runs, paired_stats, write_macros, write_tables


def _write_run(results: Path, name: str, rows: list[dict], benchmark: str = "ruler") -> None:
    folder = results / name
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"run_name": name, "benchmark": benchmark}))
    scored = {}
    for row in rows:
        if "score" in row:
            scored.setdefault(row["system"], set()).add(row["example_id"])
    expected = max((len(ids) for ids in scored.values()), default=0)
    (folder / "metrics.json").write_text(json.dumps({
        "benchmark": benchmark, "n_examples": expected,
        "systems": {name: {"n_scored": len(ids), "n_errors": 0} for name, ids in scored.items()}
    }))
    (folder / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n")


def test_assets_exclude_legacy_retry_errors_and_do_not_invent_missing_data(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write_run(results, "final_ruler", [
        {"system": "sempointer", "example_id": "x", "score": 1.0}])
    _write_run(results, "paper_ruler", [
        {"system": "sempointer", "example_id": "niah_single_1-4096-0",
         "error": "first attempt failed"},
        {"system": "sempointer", "example_id": "niah_single_1-4096-0",
         "task": "niah_single_1", "score": 0.0, "active_tokens": 100},
        {"system": "sempointer", "example_id": "niah_single_1-4096-0",
         "task": "niah_single_1", "score": 0.5, "active_tokens": 120,
         "retrieval_hit": 1.0,
         "retrieval_basis": "gold string present in selected RULER NIAH block"},
        {"system": "full", "example_id": "niah_single_1-4096-0",
         "task": "niah_single_1", "score": 1.0, "active_tokens": 500},
    ])
    out = tmp_path / "generated"
    out.mkdir()
    placeholder = Path(__file__).resolve().parents[1] / "analysis/macros_placeholder.tex"
    (out / "macros_placeholder.tex").write_text(placeholder.read_text())

    runs, _ = load_runs(results)
    assert set(runs) == {"paper_ruler"}
    assert len(runs["paper_ruler"].system("sempointer")) == 1
    write_macros(runs, out)
    write_tables(runs, out)

    macros = (out / "macros.tex").read_text()
    assert r"\providecommand{\SPRulerScore}{50.0}" in macros
    assert r"\providecommand{\FullRulerScore}{100.0}" in macros
    assert r"\providecommand{\SPLongBenchScore}{\textbf{TBD}}" in macros
    assert r"\providecommand{\SPvsFullDelta}{-50.0}" in macros
    assert r"\providecommand{\SPTokenReduction}{76.0}" in macros
    assert r"\providecommand{\SPRecallRuler}{100.0}" in macros
    assert r"\label{tab:main}" in (out / "tab_main.tex").read_text()
    assert r"\label{tab:ablation-pointer}" in (out / "tab_ablation_pointer.tex").read_text()
    assert r"\textbf{TBD}" in (out / "tab_substrate.tex").read_text()
    assert "gold-string block coverage" in (out / "tab_retrieval.tex").read_text()
    assert (out / "tab_main.tex").read_text().count("LongBench &") == 1
    assert paired_stats(runs["paper_ruler"], "sempointer", "full")["n"] == 1


def test_partial_stage_preserves_scored_rows_and_marks_missing_examples(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write_run(results, "paper_ruler", [
        {"system": "sempointer", "example_id": "x", "score": 1.0},
        {"system": "full", "example_id": "x", "score": 0.0},
    ])
    metric_path = results / "paper_ruler" / "metrics.json"
    metrics = json.loads(metric_path.read_text())
    metrics["n_examples"] = 2
    metrics["systems"]["full"]["n_errors"] = 1
    metric_path.write_text(json.dumps(metrics))
    runs, notes = load_runs(results)
    assert "paper_ruler" in runs
    assert "partial" in notes[0]
    assert "1/2 scored" in notes[0]
    out = tmp_path / "generated"
    out.mkdir()
    write_tables(runs, out)
    main = (out / "tab_main.tex").read_text()
    assert "1/2 (err 1)" in main
    assert r"100.0\textsuperscript{*}" in main
    assert paired_stats(runs["paper_ruler"], "sempointer", "full")["n"] == 1


def test_zero_scored_system_has_count_but_no_numeric_claim(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write_run(results, "paper_kv", [
        {"system": "sempointer", "example_id": "x", "score": 0.5},
        {"system": "full", "example_id": "x", "score": 1.0},
    ], benchmark="kv_compare")
    metric_path = results / "paper_kv" / "metrics.json"
    metrics = json.loads(metric_path.read_text())
    metrics["systems"]["kvpress_snapkv_0.3"] = {"n_scored": 0, "n_errors": 1}
    metric_path.write_text(json.dumps(metrics))
    runs, notes = load_runs(results)
    assert "paper_kv" in runs
    assert "kvpress_snapkv_0.3: 0/1 scored, 1 errors" in notes[0]
    out = tmp_path / "generated"
    out.mkdir()
    write_tables(runs, out)
    kv = (out / "tab_kvcompare.tex").read_text()
    assert r"kvpress\_snapkv\_0.3 & 0/1 (err 1) & \textbf{TBD}" in kv


def test_paired_stats_only_use_shared_scored_ids(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write_run(results, "paper_ruler", [
        {"system": "sempointer", "example_id": "x", "score": 0.0},
        {"system": "sempointer", "example_id": "y", "score": 1.0},
        {"system": "full", "example_id": "x", "score": 1.0},
        {"system": "full", "example_id": "y", "error": "OOM"},
    ])
    metric_path = results / "paper_ruler" / "metrics.json"
    metrics = json.loads(metric_path.read_text())
    metrics["systems"]["full"]["n_errors"] = 1
    metric_path.write_text(json.dumps(metrics))
    runs, _ = load_runs(results)
    paired = paired_stats(runs["paper_ruler"], "sempointer", "full")
    assert paired["n"] == 1
    assert paired["delta"] == -100.0


def test_complete_longbench_expands_to_scheduled_system_rows(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write_run(results, "paper_longbench", [
        {"system": name, "example_id": "x", "score": score}
        for name, score in (("sempointer", 0.4), ("full", 0.6),
                            ("rag_m4", 0.5), ("hybrid_m4", 0.5))
    ], benchmark="longbench")
    out = tmp_path / "generated"
    out.mkdir()
    runs, _ = load_runs(results)
    write_tables(runs, out)
    main = (out / "tab_main.tex").read_text()
    assert main.count("LongBench &") == 4
    assert "BM25" not in main
    significance = (out / "tab_significance.tex").read_text()
    assert significance.count("LongBench &") == 3
    assert "95\\% CI" in significance
    assert "$p_{\\rm H}$" in significance
