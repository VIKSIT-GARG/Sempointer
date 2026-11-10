"""Verify and analyze the already-executed fixed-payload pointer intervention.

No model weights or inference are used. Scores are paired within task/metric;
the overall summary uses descriptive win/tie/loss counts, not pooled accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from .audit_bounded_losses import expanded_blocks, table

HERE = Path(__file__).resolve().parents[1]
DEFAULT_RUN = HERE / "results/bounded_pointer_full_v1"
SEED = 20261120


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def verify_pair(on, off):
    for key in ["example_id", "question", "gold", "selected_ids", "context_sha256", "max_new_tokens",
                "metric", "family", "task", "status"]:
        if on[key] != off[key]:
            raise ValueError(f"Control changes {key}")
    if on["status"] != "scored":
        raise ValueError("Both control arms must be scored")
    left, right = expanded_blocks(on), expanded_blocks(off)
    if left != right:
        raise ValueError("Expanded payload text differs")
    prompt = on["prompt"]
    context = prompt.index("Memory Context:\n")
    start = prompt.find("Memory Pointers (unexpanded blocks):\n", 0, context)
    stripped = prompt if start < 0 else prompt[:start] + prompt[context:]
    if stripped != off["prompt"]:
        raise ValueError("Prompt differs outside the descriptor section")
    if bool(on.get("pointer_tokens", 0)) != (start >= 0):
        raise ValueError("Descriptor accounting disagrees with rendered prompt")
    return {"fixed_payload": True, "descriptors_present": start >= 0,
            "identical_prompts": on["prompt"] == off["prompt"],
            "payload_sha256": digest(left), "on_prompt_sha256": digest(on["prompt"]),
            "off_prompt_sha256": digest(off["prompt"])}


def sign_test(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n)


def holm_adjust(p_values):
    adjusted = [None] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda i: p_values[i])):
        previous = max(previous, min(1.0, p_values[index] * (len(p_values) - rank)))
        adjusted[index] = previous
    return adjusted


def clustered_net_win_interval(rows, repeats=10000, seed=SEED):
    """Resample whole same-context groups within each family; keep family weights."""
    import numpy as np
    families = defaultdict(lambda: defaultdict(list))
    for row in rows:
        families[row["family"]][row["context_sha256"]].append(row["sign"])
    rng = np.random.default_rng(seed)
    distribution = np.zeros(repeats)
    for groups in families.values():
        sums = np.array([sum(values) for values in groups.values()], dtype=float)
        sizes = np.array([len(values) for values in groups.values()])
        sampled = rng.integers(0, len(groups), size=(repeats, len(groups)))
        means = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
        distribution += means * sizes.sum() / len(rows)
    return [float(x) for x in np.percentile(distribution, [2.5, 97.5])]


def task_uncertainty(rows, repeats=10000, seed=SEED):
    """Native-score paired differences within one task; small groups descriptive."""
    import numpy as np
    groups = defaultdict(list)
    for row in rows:
        groups[row["context_sha256"]].append(row["score_difference"])
    if len(groups) < 5:
        return None, None
    sums = np.array([sum(values) for values in groups.values()], dtype=float)
    sizes = np.array([len(values) for values in groups.values()])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), size=(repeats, len(groups)))
    means = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    interval = [float(x) for x in np.percentile(means, [2.5, 97.5])]
    nonzero = sums[abs(sums) > 1e-12]
    if not len(nonzero):
        return interval, 1.0
    if len(nonzero) <= 12:
        signs = 1 - 2 * ((np.arange(2 ** len(nonzero))[:, None] >> np.arange(len(nonzero))) & 1)
        p = float(np.mean(abs(signs @ nonzero) >= abs(nonzero.sum()) - 1e-12))
    else:
        signs = rng.integers(0, 2, size=(repeats, len(nonzero))) * 2 - 1
        p = float((1 + np.count_nonzero(abs(signs @ nonzero) >= abs(nonzero.sum()) - 1e-12)) / (repeats + 1))
    return interval, p


def aggregate(rows):
    n = len(rows)
    counts = Counter("wins" if r["sign"] > 0 else "losses" if r["sign"] < 0 else "ties" for r in rows)
    on_tokens = sum(r["on_input_tokens"] for r in rows)
    off_tokens = sum(r["off_input_tokens"] for r in rows)
    on_ms = sum(r["on_generation_ms"] for r in rows)
    off_ms = sum(r["off_generation_ms"] for r in rows)
    return {"n": n, **{k: counts[k] for k in ["wins", "ties", "losses"]},
            "same_answer": sum(r["same_answer"] for r in rows),
            "source_contexts": len({r["context_sha256"] for r in rows}),
            "mean_extra_input_tokens": (on_tokens - off_tokens) / n,
            "extra_input_pct": 100 * (on_tokens / off_tokens - 1),
            "generation_time_reduction_pct": 100 * (1 - on_ms / off_ms),
            "net_win_fraction": (counts["wins"] - counts["losses"]) / n}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def run_analysis(run, output):
    if output.exists():
        raise ValueError("Choose a new analysis output directory")
    paths = [run / "input/manifest.json", run / "results/test_4096/results.sqlite",
             run / "results/test_4096/config.json", run / "environment.json", Path(__file__),
             HERE / "analysis/audit_bounded_losses.py"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest = json.loads(paths[0].read_text())
    if manifest["run_id"] != "c73abaf595dc68e55696":
        raise ValueError("This secondary analysis is pinned to the completed full evaluation")
    with closing(sqlite3.connect(f"file:{paths[1]}?mode=ro", uri=True)) as db:
        records = [json.loads(r[0]) for r in db.execute(
            "SELECT payload FROM latest WHERE system IN ('bounded_pointer','adaptive_no_index')")]
    by = {(r["example_id"], r["system"]): r for r in records}
    rows, vt_probe = [], []
    for item in manifest["rows"]:
        eid = item["id"]
        on, off = by[eid, "bounded_pointer"], by[eid, "adaptive_no_index"]
        verified = verify_pair(on, off)
        difference = on["score"] - off["score"]
        rows.append({"example_id": eid, "family": on["family"], "task": on["task"],
            "metric": on["metric"], "group_id": on["group_id"], "context_sha256": on["context_sha256"],
            **verified, "on_score": on["score"], "off_score": off["score"], "score_difference": difference,
            "sign": 1 if difference > 1e-12 else -1 if difference < -1e-12 else 0,
            "same_answer": on["answer"] == off["answer"], "on_answer": on["answer"], "off_answer": off["answer"],
            "on_input_tokens": on["active_tokens"], "off_input_tokens": off["active_tokens"],
            "on_generation_ms": on["generation_ms"], "off_generation_ms": off["generation_ms"],
            "on_generation_order": on["generation_index_in_example"], "off_generation_order": off["generation_index_in_example"]})
        if on["family"] == "ruler" and on["task"] == "vt":
            descriptor = on["prompt"].split("Memory Pointers (unexpanded blocks):\n", 1)[1].split("Memory Context:\n", 1)[0]
            payload = "\n".join(expanded_blocks(on).values())
            present = lambda text, gold: re.search(r"(?<!\w)" + re.escape(gold) + r"(?!\w)", text) is not None
            path = run / "input" / item["path"]
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            context = json.loads(path.read_text())["example"]["context"]
            assert hashlib.sha256(context.encode()).hexdigest() == on["context_sha256"]
            assigned = set(re.findall(r"(?m)^VAR\s+(\w+)\s*=", context))
            vt_probe.append({"example_id": eid, "on_score": on["score"], "off_score": off["score"],
                "pointer_only_gold_names": [g for g in on["gold"] if not present(payload, g) and present(descriptor, g)],
                "payload_gold_names": [g for g in on["gold"] if present(payload, g)],
                "num_chains": on["meta"].get("num_chains"), "num_hops": on["meta"].get("num_hops"),
                "all_assigned_names_are_gold": assigned == set(on["gold"]), "descriptor_section": descriptor})
    if len(rows) != 1000:
        raise ValueError("Expected all 1,000 questions")
    active = [r for r in rows if r["descriptors_present"]]
    inactive = [r for r in rows if not r["descriptors_present"]]
    summary = {"source_run_id": manifest["run_id"], "analysis_kind": "retrospective_verified_existing_control",
        "new_generations": 0, "input_budget": 4096, "all_pairs": aggregate(rows),
        "active_descriptor_pairs": aggregate(active), "null_intervention_pairs": aggregate(inactive),
        "net_win_fraction_ci95_clustered_active": clustered_net_win_interval(active),
        "net_win_fraction_ci95_clustered_all": clustered_net_win_interval(rows),
        "question_independent_sign_test_p": sign_test(sum(r["sign"] > 0 for r in active), sum(r["sign"] < 0 for r in active)),
        "sign_test_caution": "Assumes independent questions; same-source clustering violates that assumption.",
        "on_generated_before_off": sum(r["on_generation_order"] < r["off_generation_order"] for r in rows),
        "families": {}, "task_inference": {"minimum_source_contexts": 5, "resamples": 10000,
            "seed": SEED, "correction": "Holm over all eligible per-task cluster sign-flip p-values"}}
    for family in manifest["families"]:
        subset = [r for r in rows if r["family"] == family]
        summary["families"][family] = {**aggregate(subset),
            "active_descriptor_pairs": sum(r["descriptors_present"] for r in subset)}
    task_groups = defaultdict(list)
    for row in active:
        task_groups[row["family"], row["task"], row["metric"]].append(row)
    tasks = []
    for key, group in sorted(task_groups.items()):
        interval, p = task_uncertainty(group)
        tasks.append({"family": key[0], "task": key[1], "metric": key[2], **aggregate(group),
            "on_mean_score": sum(r["on_score"] for r in group) / len(group),
            "off_mean_score": sum(r["off_score"] for r in group) / len(group),
            "mean_score_difference": sum(r["score_difference"] for r in group) / len(group),
            "difference_ci95_same_context_clustered": interval, "cluster_sign_flip_p": p,
            "holm_adjusted_p": None,
            "inference_note": "Exploratory" if p is not None else "Fewer than five source contexts; descriptive only"})
    eligible = sorted((i for i, row in enumerate(tasks) if row["cluster_sign_flip_p"] is not None),
                      key=lambda i: tasks[i]["cluster_sign_flip_p"])
    previous = 0.0
    for rank, index in enumerate(eligible):
        previous = max(previous, min(1.0, tasks[index]["cluster_sign_flip_p"] * (len(eligible) - rank)))
        tasks[index]["holm_adjusted_p"] = previous
    # Sensitivity: retain all 123 inspected groups as the hypothesis family,
    # assigning p=1 (unrejected) to groups too small for the planned inference.
    sensitivity = holm_adjust([r["cluster_sign_flip_p"] if r["cluster_sign_flip_p"] is not None else 1.0 for r in tasks])
    for row, adjusted in zip(tasks, sensitivity):
        row["holm_all_inspected_groups_sensitivity"] = adjusted
    summary["task_inference"].update(task_groups=len(tasks), eligible_groups=len(eligible),
        holm_positive=[r for r in tasks if r["holm_adjusted_p"] is not None and r["holm_adjusted_p"] < .05 and r["mean_score_difference"] > 0],
        holm_negative=[r for r in tasks if r["holm_adjusted_p"] is not None and r["holm_adjusted_p"] < .05 and r["mean_score_difference"] < 0])
    summary["vt_mechanism"] = {"n": len(vt_probe),
        "cases_with_gold_names_only_in_descriptors": sum(bool(r["pointer_only_gold_names"]) for r in vt_probe),
        "wins_with_gold_names_only_in_descriptors": sum(bool(r["pointer_only_gold_names"]) and r["on_score"] > r["off_score"] + 1e-12 for r in vt_probe),
        "single_chain_cases": sum(r["num_chains"] == 1 for r in vt_probe),
        "cases_where_all_assigned_names_are_gold": sum(r["all_assigned_names_are_gold"] for r in vt_probe)}
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(output / "verified_pairs.csv", rows)
    write_csv(output / "per_task.csv", tasks)
    write_csv(output / "vt_mechanism.csv", vt_probe)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_files_sha256": hashes,
        "new_model_inference": False, "new_generations": 0,
        "note": "Exact saved-prompt verification and secondary analysis; not a new independent experiment."}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_report(output, summary, tasks)
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    print(json.dumps(summary, indent=2))


def write_report(output, s, tasks):
    a, active, null = s["all_pairs"], s["active_descriptor_pairs"], s["null_intervention_pairs"]
    ci = s["net_win_fraction_ci95_clustered_active"]
    lines = ["# Fixed-payload pointer-on versus pointer-off control", "",
        "## Status and scope", "",
        "Completed verification and secondary analysis of the already-executed 4K control in run "
        f"`{s['source_run_id']}`. **No new inference was launched.** This report is not a fresh independent replication.", "",
        "All 1,000 pairs have byte-identical expanded payload strings, selected IDs, question, gold references, "
        "native framing outside the descriptor section, and output caps. Removing only the entire descriptor "
        "section from the pointer-on prompt produces the pointer-off prompt exactly. Both arms share the same "
        "frozen checkpoint, NF4 precision, greedy non-thinking decoding, routing settings and stage configuration.", "",
        "The 1K and 2K experiments often trade pointer space against payload space; they are not used as this control. "
        "The 4K budget was selected for the exact intervention match, not by filtering favorable score outcomes. "
        "All 1,000 questions, not only the 229 BM25 losses, are included.", "",
        "## Overall descriptive results", "", table(
            ["Pairs", "n", "Pointer-on wins", "Ties", "Losses", "Identical answer texts"],
            [["All fixed-payload pairs", a['n'], a['wins'], a['ties'], a['losses'], a['same_answer']],
             ["Descriptor section actually present", active['n'], active['wins'], active['ties'], active['losses'], active['same_answer']],
             ["No descriptors; identical input prompts", null['n'], null['wins'], null['ties'], null['losses'], null['same_answer']]]), "",
        f"On the {active['n']} actual interventions, the net win fraction is "
        f"{100 * active['net_win_fraction']:.2f} percentage points (wins minus losses divided by question count). "
        f"A family-stratified same-context cluster bootstrap gives a 95% interval of [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}] points. "
        "This measures question-wise win balance, not mean accuracy or score equivalence. "
        "The interval is exploratory; shared documents may remain correlated beyond exact-context hashes, "
        "and LoCoMo has only four source conversations. A question-independent sign test is retained in summary.json with its invalid-independence caveat.", "",
        "## Per-family results", "", table(
            ["Family", "n", "Actual descriptor pairs", "Source contexts", "Wins", "Ties", "Losses", "Extra input tokens, mean"],
            [[f, r['n'], r['active_descriptor_pairs'], r['source_contexts'], r['wins'], r['ties'], r['losses'],
              f"{r['mean_extra_input_tokens']:.1f}"] for f, r in s['families'].items()]), "",
        "Counts are within-question comparisons under each task's own metric; heterogeneous score means are not pooled.", "",
        "## Per-task inference and multiplicity", "",
        f"All {len(tasks)} active task/metric groups are exported in [per_task.csv](per_task.csv). "
        "Within each group, report native-score means/differences. Groups with at least five exact source contexts "
        "receive paired context-cluster bootstrap intervals and cluster sign-flip p-values. Smaller groups are descriptive only. "
        "Holm correction covers all eligible groups; no task is selected for reporting because its p-value is favorable.", "",
        f"Eligible groups: {s['task_inference']['eligible_groups']}; Holm-adjusted p < 0.05: "
        f"{len(s['task_inference']['holm_positive'])} positive, {len(s['task_inference']['holm_negative'])} negative. "
        "This exploratory analysis does not establish a preregistered confirmatory result.", "",
        "### Task-specific positive result: RULER variable tracking", "",
        "The 17 variable-tracking questions score 74.12 with pointers and 37.65 without pointers on a 0–100 display scale: "
        "a +36.47-point paired mean difference, exploratory 95% context-cluster bootstrap interval [+22.35, +51.76]. "
        "There are 12 wins, five ties and no losses. The raw cluster sign-flip p-value is 0.000488; "
        "Holm adjustment across the 53 inference-eligible task groups gives 0.0259. "
        "A more conservative sensitivity treating all 123 inspected groups as the hypothesis family gives 0.0601. "
        "Thus the multiplicity conclusion depends on the justified hypothesis family; do not omit this qualification.", "",
        "A source-based mechanism probe finds 14/17 cases where descriptors expose gold variable names absent from the expanded payload; "
        "11 of the 12 pointer-on wins have such added names. All 17 examples have one chain, and every source assignment variable is a gold name. "
        "This is evidence that compact descriptors can recover otherwise unseen answer entities in this synthetic setup. "
        "It does not establish general multi-chain reasoning, learned addressing, or a model-issued dereference mechanism. "
        "Gold names were used only for retrospective analysis, never to construct descriptors. Raw probe rows are in [vt_mechanism.csv](vt_mechanism.csv).", "",
        "## Cost and mechanism", "",
        f"Across all 1,000 pairs, pointer-on adds {a['mean_extra_input_tokens']:.3f} input tokens on average "
        f"({a['extra_input_pct']:.2f}% more input). Recorded generation time is {a['generation_time_reduction_pct']:.2f}% lower. "
        f"Pointer-on was generated before pointer-off in {s['on_generated_before_off']}/1,000 cases under the existing four-arm rotation; "
        "the pair is not perfectly counterbalanced. The small timing difference is descriptive, not a reliable speed advantage.", "",
        "This intervention tests adding the whole descriptor section, including its extra tokens and positional effects. "
        "It does not isolate descriptor semantics from length/placement; that would require additional matched-length or shuffled controls. "
        "Saved greedy outputs are single observations, not repeated trials. The ten identical-input checks agree here, "
        "but that small check does not prove deterministic behavior for every prompt.", "",
        "## Claim supported by this control", "",
        "The existing experiment contains a verified fixed-payload comparison. Descriptor text changes some answers "
        "and has dataset-dependent benefits and harms, but the overall win balance is small and its clustered interval "
        "includes no effect. **No clear overall quality benefit from displaying pointers is demonstrated.** "
        "The variable-tracking result supports a narrower synthetic-task benefit with the stated multiplicity and task-design caveats. "
        "This does not prove that the effect is zero elsewhere or that pointers never help. Efficiency claims should be attributed "
        "to the adaptive evidence policy unless a stronger descriptor-specific effect is established. "
        "Do not infer cross-model robustness, official leaderboard standing, or guaranteed publication readiness.", "",
        "## Files and reproduction", "",
        "- [verified_pairs.csv](verified_pairs.csv): all pairs, intervention/payload hashes, answer texts, scores, timing and null-intervention flags.",
        "- [per_task.csv](per_task.csv): every homogeneous task group, including negative and inconclusive outcomes.",
        "- [summary.json](summary.json): aggregate results, uncertainty settings and eligible task findings.",
        "- [provenance.json](provenance.json): preserved source/config hashes and zero-new-inference declaration.", "",
        "From experiments/, create a new analysis directory when reproducing:", "", "```bash",
        "CUDA_VISIBLE_DEVICES='' .venv-local-suite/bin/python -m analysis.fixed_payload_control --output /absolute/new/analysis-directory",
        "```", "",
        "No model is imported or loaded by this analysis. It reads the existing SQLite records in read-only mode. "
        "Neither completed test data nor frozen inputs are changed."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    run_analysis(run, args.output.resolve() if args.output else run / "fixed_payload_pointer_control_v1")


if __name__ == "__main__":
    main()
