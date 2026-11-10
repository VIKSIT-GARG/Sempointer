#!/usr/bin/env python3
"""Bounded, resumable reviewer experiments; no training or implicit downloads."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
TASKS = "niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,vt,cwe"
QA_TASKS = "hotpotqa,2wikimqa,musique,multifieldqa_en,narrativeqa,qasper"
CONTROLS = "sempointer,pointer_off,bm25_2,bm25_4,bm25_matched_budget"
CANDIDATES = [
    {"name": "loose", "thresholds": {"expand_coverage": .7, "ambiguity_coverage": .9, "expand_score_ratio": .8}},
    {"name": "paper", "thresholds": {"expand_coverage": .8, "ambiguity_coverage": .95, "expand_score_ratio": .85}},
    {"name": "cautious", "thresholds": {"expand_coverage": .9, "ambiguity_coverage": .98, "expand_score_ratio": .9}},
]


def freeze_json(path, value):
    """Never replace settings or selected thresholds on resume."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Settings changed at {path}; use a different --output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def run_command(command, log_path, timeout_seconds):
    """Tee logs and stop the entire subprocess group, preserving SQLite commits."""
    if timeout_seconds <= 0:
        return 124
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + shlex.join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, start_new_session=True)

        def copy_output():
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()

        def stop():
            try:
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass

        try:
            code = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stop()
            code = 124
        except BaseException:
            stop()
            raise
        finally:
            reader.join(timeout=3)
            proc.stdout.close()
    return code


def records(directory):
    path = Path(directory) / "results.sqlite"
    if not path.exists():
        return []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        return [json.loads(x[0]) for x in db.execute("SELECT payload FROM latest")]


def select_thresholds(candidates, max_loss=.02):
    """Development heuristic, not a test-set non-inferiority claim."""
    stats = []
    reference_ids = None
    for candidate in candidates:
        rows = records(candidate["path"])
        if not rows or any(x["status"] != "scored" for x in rows):
            raise ValueError("Development results are incomplete; resume before selecting thresholds")
        sem = {x["example_id"]: x for x in rows if x["system"] == "sempointer"}
        bm = {x["example_id"]: x for x in rows if x["system"] == "bm25_4"}
        if not sem or sem.keys() != bm.keys():
            raise ValueError("Development candidates require complete paired results")
        if reference_ids is not None and sem.keys() != reference_ids:
            raise ValueError("Development candidates must use the same examples")
        reference_ids = sem.keys()
        delta = sum(sem[i]["score"] - bm[i]["score"] for i in sem) / len(sem)
        stats.append({**candidate, "n_paired": len(sem), "difference_from_bm25_4": delta,
            "mean_score": sum(x["score"] for x in sem.values()) / len(sem),
            "mean_tokens": sum(x["active_tokens"] for x in sem.values()) / len(sem)})
    eligible = [x for x in stats if x["difference_from_bm25_4"] >= -max_loss - 1e-12]
    selected = min(eligible, key=lambda x: (x["mean_tokens"], -x["mean_score"], x["name"])) if eligible else min(
        stats, key=lambda x: (-x["mean_score"], x["mean_tokens"], x["name"]))
    return {"source": "development_only", "max_development_loss": max_loss,
            "rule": "Fewest mean tokens within 0.02 of development BM25-4; otherwise highest development score.",
            "selected": selected, "candidates": stats,
            "caution": "A small synthetic development heuristic, not evidence of test-set equivalence."}


def build_exclusions(data_root, output):
    """Exclude paper-exposed groups and groups in retained newer prediction logs."""
    from evaluation.data import discover_tasks, iter_examples
    groups = set()
    sources = []
    tasks, _ = discover_tasks(data_root)
    for task in tasks:
        if task.family == "longbench" and task.task in QA_TASKS.split(","):
            groups.update(x.group_id for x in islice(iter_examples(task), 100))
            sources.append(f"paper: first 100 records of {task.key}")
        elif task.family == "locomo":
            first = []
            for example in iter_examples(task):
                if example.group_id not in first:
                    first.append(example.group_id)
                if len(first) > 3:
                    break
                groups.add(example.group_id)
            sources.append("paper: first three LoCoMo conversations")
    logs = set((HERE / "results").rglob("predictions.jsonl")) | set((HERE / "results").rglob("events.jsonl"))
    for path in sorted(logs):
        if path.is_relative_to(output):
            continue
        found = set()
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if row.get("group_id") and row.get("status") == "scored":
                        found.add(row["group_id"])
        if found:
            groups.update(found)
            sources.append(str(path))
    return sorted(groups), {"sources": sources, "excluded_groups": len(groups),
        "scope": "Known paper exposure and retained exported predictions; cannot certify all prior human exposure."}


def stage_plan(args, thresholds=None):
    selected = thresholds or CANDIDATES[1]["thresholds"]
    common = [sys.executable, str(HERE / "run_all_datasets.py"), "run", "--model", args.model,
              "--data-root", str(args.data_root), "--partition", "all"]

    def stage(name, flags, pairs, router=None):
        command = common + flags
        for key, value in (router or selected).items():
            command += ["--" + key.replace("_", "-"), str(value)]
        command += ["--output", str(args.output / name)]
        return {"name": name, "command": command, "max_planned_pairs": pairs}

    ruler = ["--families", "ruler", "--context-lengths", "4096,8192,16384",
             "--max-prompt-tokens", "20000", "--limit", "0"]
    arms = CONTROLS + ",full_context"
    stages = [stage("smoke", ["--families", "ruler", "--tasks", "niah_single_1,cwe",
        "--context-lengths", "4096", "--ruler-samples", "1", "--limit", "1",
        "--seed", str(args.seed), "--systems", arms], 12)]
    for candidate in CANDIDATES:
        stages.append(stage("dev_" + candidate["name"], ["--families", "ruler", "--tasks", TASKS,
            "--context-lengths", "4096,8192", "--ruler-samples", str(args.dev_samples),
            "--limit", "0", "--seed", str(args.seed + 1), "--systems", "sempointer,bm25_4"],
            6 * 2 * args.dev_samples * 2, candidate["thresholds"]))
    stages += [
        stage("ruler_test", ruler + ["--tasks", TASKS, "--ruler-samples", str(args.ruler_samples),
            "--seed", str(args.seed + 2), "--systems", arms], 6 * 3 * args.ruler_samples * 6),
        stage("cwe_cap32", ruler + ["--tasks", "cwe", "--ruler-samples", str(args.ruler_samples),
            "--seed", str(args.seed + 2), "--systems", arms, "--max-new-tokens", "32"],
            3 * args.ruler_samples * 6),
        stage("longbench_test", ["--families", "longbench", "--tasks", QA_TASKS,
            "--seed", str(args.seed + 2), "--systems", CONTROLS,
            "--limit", str(args.longbench_limit), "--exclude-groups", str(args.output / "excluded_groups.json")],
            6 * args.longbench_limit * 5),
        stage("locomo_test", ["--families", "locomo", "--tasks", "qa", "--seed", str(args.seed + 2),
            "--systems", CONTROLS, "--limit", "0", "--examples-per-group", str(args.locomo_per_conversation),
            "--exclude-groups", str(args.output / "excluded_groups.json")], 7 * args.locomo_per_conversation * 5),
    ]
    return stages


def write_report(root, statuses):
    """Export even interrupted SQLite runs and report only real paired results."""
    from evaluation.runner import RunStore, summarize, interval
    lines = ["# Reviewer experiment evidence", "",
        "Only completed, recorded predictions are scored. Partial runs are explicitly identified.", "",
        "## Stage status", "", "| Stage | Status | Seconds |", "| --- | --- | ---: |"]
    for name, status in statuses.items():
        lines.append(f"| {name} | {status['status']} | {status.get('seconds', 0):.1f} |")
    comparisons = []
    audit = {}
    stage_rows = {}
    for db in sorted(root.glob("*/results.sqlite")):
        directory = db.parent
        with RunStore(directory, json.loads((directory / "config.json").read_text())) as store:
            store.export()
            result = summarize(store)
        rows = records(directory)
        stage_rows[directory.name] = rows
        good = {(x["example_id"], x["system"]): x for x in rows if x["status"] == "scored"}
        pointer_pairs = budget_pairs = mismatched = exceeded = 0
        for (ident, system), sem in good.items():
            if system != "sempointer":
                continue
            off = good.get((ident, "pointer_off"))
            budget = good.get((ident, "bm25_matched_budget"))
            if off:
                pointer_pairs += 1
                mismatched += int(off["selected_ids"] != sem["selected_ids"])
            if budget:
                budget_pairs += 1
                exceeded += int(budget["active_tokens"] > sem["active_tokens"])
        audit[directory.name] = {"statuses": result["status_counts"], "pointer_pairs": pointer_pairs,
            "pointer_selected_id_mismatches": mismatched, "budget_pairs": budget_pairs,
            "budget_exceeded_pairs": exceeded,
            "output_cap_hits": sum(x.get("stop_reason") == "max_new_tokens" for x in good.values()),
            "scored": len(good), "unique_examples": len({x["example_id"] for x in good.values()}),
            "document_groups": len({x["group_id"] for x in good.values() if x.get("group_id")})}
        lines += ["", f"## {directory.name}", "", f"[Per-task results]({directory.name}/REPORT.md). Status counts: `{result['status_counts']}`."]
        if result["paired"]:
            lines += ["", "Differences below are SemPointer minus comparator; scores are on a 0–1 scale.", "",
                "| Task | Comparator | Paired n | Score difference [95% CI] | Token difference |",
                "| --- | --- | ---: | --- | ---: |"]
        for pair in result["paired"]:
            comparisons.append({"stage": directory.name, **pair})
            lo, hi = pair["score_difference_ci95"]
            lines.append(f"| {pair['task']} | {pair['comparison']} | {pair['n_paired']} | "
                         f"{pair['score_difference']:.4f} [{lo:.4f}, {hi:.4f}] | "
                         f"{pair.get('active_tokens_difference', 0):.1f} |")
    long = {(x["example_id"], x["system"]): x for x in stage_rows.get("ruler_test", [])
            if x["status"] == "scored" and x["task"] == "cwe"}
    short = {(x["example_id"], x["system"]): x for x in stage_rows.get("cwe_cap32", []) if x["status"] == "scored"}
    lines += ["", "## Paired CWE output-cap audit", "",
              "Both runs use the same fresh seed, questions, evidence policy, and model. Only the output cap changes.", "",
              "| System | Paired n | Score difference (512 minus 32) [95% CI] | Cap hits at 32 / 512 |",
              "| --- | ---: | --- | --- |"]
    cap_audit = {}
    for system in sorted({s for _, s in short}):
        ids = sorted(k for k in short.keys() & long.keys() if k[1] == system)
        if not ids:
            continue
        if any(short[k]["context_sha256"] != long[k]["context_sha256"] for k in ids):
            raise ValueError("CWE contexts differ across caps")
        diff = [long[k]["score"] - short[k]["score"] for k in ids]
        ci = interval(diff)
        hits = [sum(source[k].get("stop_reason") == "max_new_tokens" for k in ids) for source in [short, long]]
        cap_audit[system] = {"n_paired": len(ids), "score_difference": sum(diff) / len(diff), "ci95": ci,
                             "cap32_hits": hits[0], "cap512_hits": hits[1]}
        lines.append(f"| {system} | {len(ids)} | {sum(diff)/len(diff):.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | {hits[0]} / {hits[1]} |")
    lines += ["", "## Interpretation limits", "",
        "Development selection uses only the separate synthetic development seed. Test labels never select thresholds.",
        "Real-data exclusions cover known retained exposure; these are additional benchmark subsets, not new corpora.",
        "LoCoMo uses an adapted token-F1 metric. These are not official benchmark submissions.",
        "Intervals resample examples, not independent conversation clusters; small/correlated samples limit inference.",
        "An interval crossing zero does not establish equivalence. Null or negative pointer results must be reported.",
        "Latency includes generation and selector time; shared ingestion/prompt preparation are logged separately.",
        "New runs use strict block boundaries and longer task-specific output caps. Compare new arms together, not against historical scores.",
        "CWE cap hits measure truncation risk; a longer cap does not guarantee good task performance.",
        "The suite produces evidence and a response worksheet; manuscript revision and PDF inspection remain manual."]
    development_contexts = {x["context_sha256"] for name, rows in stage_rows.items() if name.startswith("dev_")
                            for x in rows if x.get("context_sha256") and x["status"] == "scored"}
    test_contexts = {x["context_sha256"] for name, rows in stage_rows.items() if name in {"ruler_test", "longbench_test", "locomo_test"}
                    for x in rows if x.get("context_sha256") and x["status"] == "scored"}
    overlap = sorted(development_contexts & test_contexts)
    if overlap:
        lines += ["", f"WARNING: {len(overlap)} identical development/test contexts were found. Investigate before claiming independence."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    (root / "checks.json").write_text(json.dumps({"runs": audit, "cwe_cap_audit": cap_audit,
        "development_test_context_overlap": overlap}, indent=2) + "\n")
    (root / "comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    with (root / "comparisons.csv").open("w", newline="") as stream:
        fields = ["stage", "family", "task", "comparison", "n_paired", "score_difference",
                  "score_difference_ci95", "active_tokens_difference", "latency_ms_difference"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)
    response = ["# Reviewer response worksheet", "",
        "Use the actual paired results in REPORT.md; do not describe incomplete stages as completed.", "",
        "1. Routing thresholds: threshold_selection.json records the development-only selection rule, candidates, and frozen winner. Report fresh-seed RULER test results separately.",
        "2. Pointer contribution: use the pointer_off contrasts and selected-ID checks in checks.json. Report negative/null findings directly.",
        "3. Matched budget: use bm25_matched_budget contrasts; describe an equal upper token allowance and report slack, rather than claiming exact token equality.",
        "4. CWE: use the paired 32-versus-512 cap audit. State observed truncation and remaining performance limitations.",
        "5. Generality: identify evaluated examples/groups and exclusions in raw predictions. Broader cross-model robustness remains unestablished.",
        "6. Low natural-task scores: inspect failures; similar low scores alone do not establish preserved useful accuracy."]
    (root / "REVIEWER_RESPONSE.md").write_text("\n".join(response) + "\n")
    return audit


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE / "results/reviewer_revision")
    p.add_argument("--data-root", type=Path, default=HERE.parent / "datasets")
    p.add_argument("--model", default=str(HERE.parent / "models/Qwen2.5-3B-Instruct"))
    p.add_argument("--hours", type=float, default=8, help="Total invocation budget, including a 10-minute report reserve")
    p.add_argument("--seed", type=int, default=20261109, help="Smoke seed; development uses +1, test uses +2")
    p.add_argument("--dev-samples", type=int, default=3, help="Per task/length, shared across development candidates")
    p.add_argument("--ruler-samples", type=int, default=10, help="Per task/length on the fresh test seed")
    p.add_argument("--longbench-limit", type=int, default=50, help="Per task, after exclusion of known exposed documents")
    p.add_argument("--locomo-per-conversation", type=int, default=30)
    p.add_argument("--dry-run", action="store_true", help="Print scope and commands; creates no files and runs no inference")
    p.add_argument("--report-only", action="store_true", help="Export committed results after an interrupted run")
    return p


def main():
    start = time.monotonic()
    p = parser()
    args = p.parse_args()
    args.output = args.output.resolve()
    args.data_root = args.data_root.resolve()
    if args.hours <= 0 or min(args.dev_samples, args.ruler_samples, args.longbench_limit, args.locomo_per_conversation) < 1:
        p.error("Hours and sample counts must be positive")
    stages = stage_plan(args)
    planned = sum(s["max_planned_pairs"] for s in stages)
    print(f"At most about {planned} planned generations (real-data availability/exclusions may reduce this).", flush=True)
    print("Planning estimate: 5–8 hours on the existing RTX 5060/3B NF4 setup; the runtime budget is a cutoff, not a completion guarantee.", flush=True)
    if args.dry_run:
        for stage in stages:
            print(f"\n{stage['name']} (up to {stage['max_planned_pairs']} pairs):\n{shlex.join(stage['command'])}")
        print("\nTest commands above show the paper thresholds as placeholders; execution freezes the development-selected thresholds first.")
        return
    if args.report_only and not args.output.exists():
        p.error("No result directory exists to report")
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / "suite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("Another reviewer suite owns this output directory")
        status_path = args.output / "stages.json"
        statuses = json.loads(status_path.read_text()) if status_path.exists() else {}
        if args.report_only:
            write_report(args.output, statuses)
            print("Report:", args.output / "REPORT.md")
            return
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DEACTIVATE_ASYNC_LOAD="1",
                          HF_HUB_DISABLE_PROGRESS_BARS="1")
        from evaluation.provenance import code_fingerprint, model_provenance, file_hash
        from evaluation.data import discover_tasks
        sources, _ = discover_tasks(args.data_root)
        source_paths = set()
        for source in sources:
            if source.family not in {"ruler", "longbench", "locomo"}:
                continue
            if source.path.is_dir():
                source_paths.update(x for x in source.path.rglob("*") if x.is_file())
            else:
                source_paths.add(source.path)
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                    if k not in {"hours", "dry_run", "report_only"}}
        settings.update(code_sha256=code_fingerprint(), launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        model_provenance=model_provenance(args.model),
                        source_sha256={str(x): file_hash(x) for x in sorted(source_paths)},
                        development_candidates=CANDIDATES, max_development_loss=.02)
        freeze_json(args.output / "suite_config.json", settings)
        exclusion_path = args.output / "excluded_groups.json"
        if not exclusion_path.exists():
            groups, exposure = build_exclusions(args.data_root, args.output)
            freeze_json(exclusion_path, groups)
            freeze_json(args.output / "exposure_manifest.json", exposure)
        deadline = start + args.hours * 3600 - min(600, args.hours * 3600 * .1)
        failed = False

        def execute(stage):
            name = stage["name"]
            previous = statuses.get(name, {})
            if previous.get("status") == "complete" and previous.get("command") != stage["command"]:
                raise ValueError(f"Command changed for completed stage {name}; choose a different output directory")
            existing = (args.output / "preflight/model_preflight.json").exists() if name == "model_check" else bool(records(args.output / name))
            if previous.get("status") == "complete" and existing:
                print(f"\n{name}: already complete; keeping its recorded predictions.", flush=True)
                return 0
            remaining = deadline - time.monotonic()
            print(f"\n{name}: starting; {max(0, remaining)/3600:.2f} compute hours remain. Log: {args.output / 'logs' / (name+'.log')}", flush=True)
            statuses[name] = {"status": "running", "command": stage["command"], "started_at": time.time()}
            status_path.write_text(json.dumps(statuses, indent=2) + "\n")
            began = time.monotonic()
            code = run_command(stage["command"], args.output / "logs" / (name + ".log"), remaining)
            statuses[name].update(status="complete" if code == 0 else "time_limit" if code == 124 else "error",
                                   returncode=code, seconds=time.monotonic() - began)
            status_path.write_text(json.dumps(statuses, indent=2) + "\n")
            return code

        try:
            check = {"name": "model_check", "command": [sys.executable, str(HERE / "run_all_datasets.py"),
                "model-check", "--model", args.model, "--output", str(args.output / "preflight")]}
            if execute(check):
                raise RuntimeError("Model preflight failed; see logs/model_check.log")
            for stage in stages[:4]:
                if execute(stage):
                    raise RuntimeError(f"{stage['name']} failed or timed out; test thresholds were not selected")
                observed = records(args.output / stage["name"])
                if len(observed) != stage["max_planned_pairs"] or any(x["status"] != "scored" for x in observed):
                    raise RuntimeError(f"{stage['name']} is incomplete; refusing development selection")
                if stage["name"] == "smoke":
                    seconds = sum(x.get("generation_ms", 0) for x in observed) / 1000 / len(observed)
                    estimate = {"smoke_seconds_per_pair": seconds, "planned_pairs": planned,
                        "straight_line_hours": seconds * planned / 3600,
                        "caution": "Excludes startup/ingestion; task mix and longer contexts can change throughput substantially."}
                    (args.output / "runtime_estimate.json").write_text(json.dumps(estimate, indent=2) + "\n")
                    print(f"Smoke measured {seconds:.2f} generation seconds/pair; straight-line projection {estimate['straight_line_hours']:.2f}h before overhead (not a guarantee).", flush=True)
            candidates = [{**c, "path": str(args.output / ("dev_" + c["name"]))} for c in CANDIDATES]
            selection = select_thresholds(candidates)
            freeze_json(args.output / "threshold_selection.json", selection)
            print("\nFrozen development choice:", selection["selected"]["name"], selection["selected"]["thresholds"], flush=True)
            for stage in stage_plan(args, selection["selected"]["thresholds"])[4:]:
                already_complete = statuses.get(stage["name"], {}).get("status") == "complete" and bool(records(args.output / stage["name"]))
                if time.monotonic() >= deadline and not already_complete:
                    failed = True
                    statuses[stage["name"]] = {"status": "not_started_time_limit"}
                    continue
                code = execute(stage)
                failed |= code != 0
                if code == 124:
                    print("Compute allowance reached. Exporting committed partial evidence.", flush=True)
        except KeyboardInterrupt:
            failed = True
            print("\nInterrupted; exporting committed evidence. Repeat the same command to resume.", flush=True)
        except Exception as exc:
            failed = True
            print(f"\nSuite stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        finally:
            for status in statuses.values():
                if status["status"] == "running":
                    status["status"] = "interrupted"
            status_path.write_text(json.dumps(statuses, indent=2) + "\n")
            audit = write_report(args.output, statuses)
            failed |= any(x["statuses"].get("error", 0) or x["pointer_selected_id_mismatches"] or
                          x["budget_exceeded_pairs"] for x in audit.values())
            failed |= bool(json.loads((args.output / "checks.json").read_text())["development_test_context_overlap"])
            for stage in stages[4:]:
                if not audit.get(stage["name"], {}).get("scored", 0):
                    failed = True
            print("\nEvidence report:", args.output / "REPORT.md", flush=True)
            print("Response worksheet:", args.output / "REVIEWER_RESPONSE.md", flush=True)
            print(f"Elapsed this invocation: {(time.monotonic()-start)/3600:.2f} hours.", flush=True)
        if failed:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
