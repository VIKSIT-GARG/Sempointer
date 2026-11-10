"""Deadline orchestration contracts; CPU fixtures are not empirical results."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

from evaluation.controls import prepare
from evaluation.data import Task, iter_examples
from evaluation.runner import RunStore, run_tasks


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def test_router_uses_the_explicit_development_thresholds():
    prepared = prepare("mars rover mission " * 100, Tokenizer(),
                       router_options={"expand_coverage": .6, "ambiguity_coverage": .7,
                                       "expand_score_ratio": .99})
    assert prepared.router.expand_coverage == .6
    assert prepared.router.ambiguity_coverage == .7
    assert prepared.router.expand_score_ratio == .99


def test_exclusions_and_group_cap_are_applied_before_question_limit(tmp_path):
    source = tmp_path / "questions.jsonl"
    source.write_text("".join(json.dumps({"_id": str(i), "context": context,
        "input": "where?", "answers": ["mars"]}) + "\n"
        for i, context in enumerate(["old source", "mars source", "mars source", "new source"])))
    task = Task("longbench", "qasper", source, adapter="longbench")
    excluded = "longbench:" + hashlib.sha256(b"old source").hexdigest()
    tok = Tokenizer()
    parameter = SimpleNamespace(requires_grad_=lambda value: None)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768),
                            parameters=lambda: [parameter])
    engine = SimpleNamespace(model=model, tokenizer=tok, generate=lambda prompt, **kw:
        SimpleNamespace(text="mars", prompt_tokens=len(tok.encode(prompt)), latency_ms=2,
                        completion_tokens=1, peak_vram_mb=0, ttft_ms=1, stop_reason="eos"))
    args = SimpleNamespace(systems=["sempointer"], partition="all", limit=2,
        max_source_chars=10000, max_source_tokens=10000, max_new_tokens=16,
        max_prompt_tokens=1024, seed=143, excluded_group_ids=[excluded],
        examples_per_group=1)
    with RunStore(tmp_path / "run", {"fixture": "CPU-only"}) as store:
        result = run_tasks([task], store, engine, args)
        rows = [json.loads(x) for x in (store.directory / "predictions.jsonl").read_text().splitlines()]
    assert result["status_counts"] == {"scored": 2}
    assert {r["group_id"] for r in rows} == {
        "longbench:" + hashlib.sha256(b"mars source").hexdigest(),
        "longbench:" + hashlib.sha256(b"new source").hexdigest()}


def test_ruler_seeds_cannot_share_example_or_partition_ids(tmp_path, monkeypatch):
    from benchmarks import ruler
    monkeypatch.setattr(ruler, "build_ruler", lambda **kw: [SimpleNamespace(
        id="niah_single_1-4096-0", context="source", question="q", gold=["a"], meta={})])
    a = Task("ruler", "niah_single_1", tmp_path, adapter="ruler",
             extra={"context_length": 4096, "n_samples": 1, "seed": 101})
    b = Task("ruler", "niah_single_1", tmp_path, adapter="ruler",
             extra={"context_length": 4096, "n_samples": 1, "seed": 102})
    x, y = next(iter_examples(a)), next(iter_examples(b))
    assert x.id != y.id
    assert x.group_id != y.group_id


def development_run(path, scores, tokens=100, missing_baseline=False):
    with RunStore(path, {"fixture": "CPU-only development selection"}) as store:
        for i, value in enumerate(scores):
            for system, score_value in [("sempointer", value), ("bm25_4", .8)]:
                if missing_baseline and system == "bm25_4":
                    continue
                store.record({"example_id": f"dev-{i}", "system": system, "status": "scored",
                    "family": "ruler", "task": "niah_single_1", "split": "test",
                    "metric": "all_containment", "protocol": "fixture", "score": score_value,
                    "active_tokens": tokens})


def test_development_selection_prefers_cost_only_within_quality_constraint(tmp_path):
    from reviewer_suite import select_thresholds
    development_run(tmp_path / "cheap", [.6, .6], tokens=50)
    development_run(tmp_path / "balanced", [.8, .8], tokens=100)
    development_run(tmp_path / "costly", [.9, .9], tokens=200)
    candidates = [{"name": name, "path": str(tmp_path / name), "thresholds": {"expand_coverage": value}}
                  for name, value in [("cheap", .7), ("balanced", .8), ("costly", .9)]]
    selected = select_thresholds(candidates)
    assert selected["selected"]["name"] == "balanced"
    assert selected["selected"]["thresholds"] == {"expand_coverage": .8}
    assert selected["source"] == "development_only"


def test_development_selection_rejects_partial_or_unpaired_results(tmp_path):
    from reviewer_suite import select_thresholds
    development_run(tmp_path / "a", [.8, .8])
    development_run(tmp_path / "b", [.9], missing_baseline=True)
    with pytest.raises(ValueError, match="paired|incomplete|same"):
        select_thresholds([{"name": name, "path": str(tmp_path / name), "thresholds": {}}
                           for name in ["a", "b"]])


def test_reading_development_results_closes_database_before_volume_refresh(tmp_path, monkeypatch):
    import sqlite3
    import reviewer_suite
    development_run(tmp_path / "saved", [.8])
    connections = []
    connect = sqlite3.connect

    def track_connection(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(reviewer_suite.sqlite3, "connect", track_connection)
    rows = reviewer_suite.records(tmp_path / "saved")
    assert len(rows) == 2
    try:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")
    finally:
        connections[0].close()


def test_frozen_run_settings_reject_changed_resume(tmp_path):
    from reviewer_suite import freeze_json
    path = tmp_path / "settings.json"
    freeze_json(path, {"seed": 1})
    freeze_json(path, {"seed": 1})
    with pytest.raises(ValueError, match="changed|different"):
        freeze_json(path, {"seed": 2})
    assert json.loads(path.read_text()) == {"seed": 1}


def test_subprocess_timeout_returns_and_preserves_child_output(tmp_path):
    from reviewer_suite import run_command
    log = tmp_path / "child.log"
    started = time.monotonic()
    code = run_command([sys.executable, "-c",
        "import time; print('committed output', flush=True); time.sleep(60)"], log, .3)
    assert code == 124
    assert time.monotonic() - started < 4
    assert "committed output" in log.read_text()


def test_dry_run_prints_bounded_scope_without_starting_inference(tmp_path):
    import subprocess
    result = subprocess.run([sys.executable, "reviewer_suite.py", "--dry-run", "--output", str(tmp_path / "out")],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "pointer_off" in result.stdout and "bm25_matched_budget" in result.stdout
    assert "cwe_cap32" in result.stdout and "ruler_test" in result.stdout
    assert not (tmp_path / "out").exists()


def test_report_recovers_committed_rows_without_completed_exports(tmp_path):
    from reviewer_suite import write_report
    directory = tmp_path / "ruler_test"
    with RunStore(directory, {"fixture": "CPU-only partial run"}) as store:
        base = {"example_id": "test-1", "family": "ruler", "task": "niah_single_1",
                "split": "test", "metric": "all_containment", "protocol": "fixture",
                "status": "scored", "score": 1., "selected_ids": [2], "active_tokens": 100}
        store.record({**base, "system": "sempointer"})
        store.record({**base, "system": "pointer_off", "active_tokens": 80})
        store.record({**base, "system": "bm25_matched_budget", "active_tokens": 99})
    assert not (directory / "predictions.jsonl").exists()
    audit = write_report(tmp_path, {"ruler_test": {"status": "time_limit", "seconds": 1}})
    assert audit["ruler_test"]["scored"] == 3
    assert audit["ruler_test"]["pointer_selected_id_mismatches"] == 0
    assert audit["ruler_test"]["budget_exceeded_pairs"] == 0
    assert len((directory / "predictions.jsonl").read_text().splitlines()) == 3
    assert "time_limit" in (tmp_path / "REPORT.md").read_text()


def test_exposure_manifest_removes_entire_old_conversations(tmp_path, monkeypatch):
    from evaluation import data
    import reviewer_suite
    from reviewer_suite import build_exclusions
    monkeypatch.setattr(reviewer_suite, "HERE", tmp_path)
    source = tmp_path / "locomo.json"
    source.write_text(json.dumps([{"sample_id": f"conv-{i}", "conversation": {
        "session_1": [{"speaker": "A", "text": "hello"}]}, "qa": [
            {"question": "q", "answer": "hello", "category": 4}]} for i in range(4)]))
    monkeypatch.setattr(data, "discover_tasks", lambda root: ([Task("locomo", "qa", source, adapter="locomo")], []))
    groups, manifest = build_exclusions(tmp_path, tmp_path / "new")
    assert {"locomo:conv-0", "locomo:conv-1", "locomo:conv-2"} <= set(groups)
    assert "locomo:conv-3" not in groups
    assert manifest["excluded_groups"] >= 3


def test_launcher_freezes_development_choice_before_every_test_command(tmp_path, monkeypatch):
    """Exercise the real orchestrator; replace only the external model process."""
    import reviewer_suite as suite
    from evaluation import data, provenance
    output = tmp_path / "suite"
    monkeypatch.setattr(sys, "argv", ["reviewer_suite.py", "--output", str(output),
        "--dev-samples", "1", "--ruler-samples", "1", "--longbench-limit", "1",
        "--locomo-per-conversation", "1"])
    monkeypatch.setattr(provenance, "model_provenance", lambda model: {"fixture": "no weights loaded"})
    monkeypatch.setattr(data, "discover_tasks", lambda root: ([], []))
    monkeypatch.setattr(suite, "build_exclusions", lambda root, out: ([], {"fixture": "CPU-only"}))
    test_commands = []

    def fake_model_process(command, log_path, timeout_seconds):
        directory = Path(command[command.index("--output") + 1])
        if "model-check" in command:
            directory.mkdir(parents=True)
            (directory / "model_preflight.json").write_text('{"fixture":"CPU-only"}')
            return 0
        name = directory.name
        if name in {"ruler_test", "cwe_cap32", "longbench_test", "locomo_test"}:
            frozen = json.loads((output / "threshold_selection.json").read_text())
            assert frozen["selected"]["name"] == "loose"
            test_commands.append(command)
        # Counts independently derived from the requested 1-example cells.
        pairs = {"smoke": 12, "dev_loose": 24, "dev_paper": 24, "dev_cautious": 24,
                 "ruler_test": 108, "cwe_cap32": 18, "longbench_test": 30, "locomo_test": 35}[name]
        systems = command[command.index("--systems") + 1].split(",")
        with RunStore(directory, {"fixture": "CPU-only external-process substitute"}) as store:
            for i in range(pairs // len(systems)):
                identity = ("dev" if name.startswith("dev_") else "ruler_test" if name == "cwe_cap32" else name) + f":{i}"
                for system in systems:
                    store.record({"example_id": identity, "context_sha256": identity,
                        "system": system, "status": "scored", "family": "ruler", "task": "cwe",
                        "split": "test", "metric": "all_containment", "protocol": "CPU-only fixture",
                        "score": .8, "selected_ids": [0], "generation_ms": 1,
                        "active_tokens": 80 if name == "dev_loose" else 100})
        return 0

    monkeypatch.setattr(suite, "run_command", fake_model_process)
    suite.main()
    assert len(test_commands) == 4
    assert all(command[command.index("--expand-coverage") + 1] == "0.7" for command in test_commands)
    statuses = json.loads((output / "stages.json").read_text())
    assert len(statuses) == 9 and all(x["status"] == "complete" for x in statuses.values())
    assert (output / "REPORT.md").exists()
    # A completed resume needs no remaining compute allowance and must retain
    # its completion status, rather than relabeling finished tests as unstarted.
    monkeypatch.setattr(sys, "argv", sys.argv + ["--hours", "0.0000001"])
    monkeypatch.setattr(suite, "run_command", lambda *a, **k: pytest.fail("Completed stages were rerun"))
    suite.main()
    assert all(x["status"] == "complete" for x in json.loads((output / "stages.json").read_text()).values())
