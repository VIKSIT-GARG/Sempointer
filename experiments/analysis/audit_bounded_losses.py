"""CPU-only evidence audit of the frozen 4K bounded-pointer losses.

Reads recorded generations; never loads model weights or invokes generation.
Evidence annotations, coarse support sessions and answer mentions remain distinct.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from contextlib import closing
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tarfile
import unicodedata


HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
DEFAULT_RUN = HERE / "results/bounded_pointer_full_v1"
ABSTENTIONS = {"not mentioned", "unanswerable", "yes", "no", "unknown", "none",
               "not enough information", "notmentioned", "entailment", "contradiction"}


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c
                            for c in text).split())


def mention(text, answer):
    text, answer = normalize(text), normalize(answer)
    if not answer:
        return False
    if any("\u3400" <= c <= "\u9fff" for c in answer):
        return answer in text
    return re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", text) is not None


def expanded_blocks(row):
    """Use the exact recorded question suffix; never search the entire prompt."""
    prompt = row["prompt"]
    header = "Memory Context:\n"
    start = prompt.find(header)
    suffix = "\n\nQuestion: " + row["question"].strip() + "\nAnswer:"
    end = prompt.rfind(suffix)
    if start < 0 or end < start:
        raise ValueError("Recorded prompt framing cannot be located")
    body = prompt[start + len(header):end]
    matches = list(re.finditer(r"(?:\A|\n\n)\[MEMORY (\d+)\]:\n", body))
    blocks = {}
    for i, match in enumerate(matches):
        block_id = int(match.group(1))
        if block_id in blocks:
            raise ValueError("Duplicate expanded block label")
        blocks[block_id] = body[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(body)].strip()
    if set(blocks) != set(row["selected_ids"]):
        raise ValueError("Expanded labels disagree with recorded selected IDs")
    return blocks


def payload_for_mentions(blocks):
    # A separator prevents an answer phrase being fabricated across omitted blocks.
    return "\n<block boundary>\n".join(blocks[i] for i in sorted(blocks))


def answer_probe(row, payload, source):
    metric = row["metric"]
    gold = row["gold"]
    reason = None
    if metric not in {"qa_f1", "char_f1", "exact", "all_containment"}:
        reason = "reference_is_label_summary_code_or_computed_value"
    elif row.get("task") == "samsum":
        reason = "summary_reference_even_though_adapter_uses_qa_f1"
    elif any(normalize(g) in ABSTENTIONS for g in gold):
        reason = "abstention_boolean_or_entailment_label"
    elif row.get("task", "").startswith("passage_retrieval"):
        reason = "reference_is_passage_identifier_not_evidence"
    elif not gold or any(len(str(g)) > 160 or len(str(g).split()) > 24 for g in gold):
        reason = "long_reference_not_an_extractive_span"
    if reason:
        return {"applicable": False, "reason": reason, "is_evidence_annotation": False}
    payload_hits = [mention(payload, g) for g in gold]
    source_hits = [mention(source, g) for g in gold]
    return {"applicable": True, "is_evidence_annotation": False,
            "payload_any": any(payload_hits), "payload_all": all(payload_hits),
            "source_any": any(source_hits), "source_all": all(source_hits),
            "per_reference_payload": payload_hits, "per_reference_source": source_hits,
            "warning": "Answer occurrence is not sufficient supporting evidence."}


def span_coverage(spans, offsets, visible):
    starts, ends = [s for s, e in offsets], [e for s, e in offsets]
    indices = set()
    for start, end in spans:
        lo, hi = bisect_right(ends, start), bisect_left(starts, end)
        indices.update(i for i in range(lo, hi) if offsets[i][1] > offsets[i][0])
    covered = indices & visible
    status = "unmapped" if not indices else "present" if covered == indices else "partial" if covered else "absent"
    return {"status": status, "covered_tokens": len(covered), "total_tokens": len(indices),
            "fraction": len(covered) / len(indices) if indices else None,
            "source_block_ids": sorted({i // 512 for i in indices}),
            "missing_block_ids": sorted({i // 512 for i in indices - visible})}


def classify_evidence(grade, bounded_status, bm25_status, probe_b, probe_m):
    if grade == "fine" and bounded_status != "unmapped":
        if bounded_status == "present":
            return "annotated_units_present_cause_unresolved"
        if bm25_status == "present":
            return "annotated_units_omitted_bm25_retains"
        return "annotated_units_incomplete_both"
    if grade == "coarse":
        return "support_session_coverage_requires_review"
    if probe_b and probe_b.get("applicable"):
        if probe_b["payload_any"]:
            return "answer_mention_present_not_evidence_proof"
        if probe_m and probe_m.get("payload_any"):
            return "answer_mention_omitted_bm25_retains_proxy_only"
        return "answer_mention_absent_both_cause_unresolved"
    return "reference_not_evidence_manual_review_required"


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    def clean(value):
        return str(value).replace("|", "&#124;").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(clean(v) for v in row) + " |" for row in rows])


def locomo_context_and_units(sessions):
    history, units = "", {}
    keys = sorted((k for k, v in sessions.items() if re.fullmatch(r"session_\d+", k) and isinstance(v, list)),
                  key=lambda k: int(k.split("_")[1]))
    for session_index, key in enumerate(keys):
        if session_index:
            history += "\n\n"
        header = f"{key}; date: {sessions.get(key + '_date_time', '')}"
        header_span = (len(history), len(history) + len(header))
        history += header + "\n"
        messages = [message for message in sessions[key] if message.get("text")]
        for index, message in enumerate(messages):
            # The source adapter joins messages even if their own text ends
            # with a newline. Checking history.endswith would drop a separator.
            if index:
                history += "\n"
            text = f"{message.get('speaker', '')}: {message['text']}"
            start = len(history)
            history += text
            units[message["dia_id"]] = {"id": message["dia_id"], "text": text,
                "spans": [(start, len(history))], "session_header_span": header_span}
    return history, units


def load_annotations(losses, consumed):
    """Preserve publisher support annotations rather than invent gold spans."""
    result = {}
    locomo = PROJECT / "datasets/originals/locomo/locomo10.json"
    consumed.add(locomo)
    conversations = {x["sample_id"]: x for x in json.loads(locomo.read_text())}
    for row in losses:
        if row["family"] != "locomo":
            continue
        tail = row["example_id"].split("::", 1)[1]
        conv_id, index = tail.rsplit(":", 1)
        conv = conversations[conv_id]
        qa = conv["qa"][int(index)]
        assert qa["question"] == row["question"]
        if int(qa["category"]) == 5:
            result[row["example_id"]] = {"kind": "abstention", "grade": None,
                "reason": "Category 5 requires absence/abstention; no positive gold span.", "sets": []}
            continue
        history, units = locomo_context_and_units(conv["conversation"])
        evidence = qa.get("evidence") or []
        annotated = [units.get(ref, {"id": ref, "text": "", "spans": []}) for ref in evidence]
        result[row["example_id"]] = {"kind": "locomo_dialogue_annotations", "grade": "fine" if annotated else None,
            "sets": [annotated] if annotated else [], "expected_context_hash": hashlib.sha256(history.encode()).hexdigest(),
            "reason": "Annotated turns; all annotations need not be individually necessary, and dates may add support."}

    wanted = {r["example_id"].split("::")[-1]: r for r in losses if r["family"] == "longmem"}
    if wanted:
        import ijson
        path = PROJECT / "datasets/huggingface/xiaowu0162/longmemeval-cleaned/longmemeval_s_cleaned.json"
        consumed.add(path)
        with path.open("rb") as stream:
            for item in ijson.items(stream, "item", use_float=True):
                if item["question_id"] not in wanted:
                    continue
                row = wanted[item["question_id"]]
                history, units = "", {}
                for j, (sid, date, session) in enumerate(zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"])):
                    if j:
                        history += "\n\n"
                    text = f"Session {j}; date: {date}\n" + "\n".join(f"{m['role']}: {m['content']}" for m in session)
                    start = len(history)
                    history += text
                    units[sid] = {"id": sid, "text": text, "spans": [(start, len(history))]}
                annotated = [units.get(ref, {"id": ref, "text": "", "spans": []}) for ref in item["answer_session_ids"]]
                result[row["example_id"]] = {"kind": "longmem_support_sessions", "grade": "coarse",
                    "sets": [annotated], "expected_context_hash": hashlib.sha256(history.encode()).hexdigest(),
                    "reason": "Annotated sessions are broad; their complete inclusion is not necessary to answer."}

    # QASPER publishes answer-evidence paragraphs. Map only exact source spans;
    # SCROLLS citation rewriting can prevent alignment, which remains unresolved.
    wanted = defaultdict(list)
    for row in losses:
        if row["family"] == "scrolls" and row["task"] == "qasper":
            wanted[normalize(row["question"])].append(row)
    if wanted:
        for path in (PROJECT / "datasets/originals/qasper").glob("*.tgz"):
            consumed.add(path)
            with tarfile.open(path) as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".json"):
                        continue
                    with archive.extractfile(member) as stream:
                        documents = json.load(stream)
                    for doc in documents.values():
                        for qa in doc["qas"]:
                            matching = wanted.get(normalize(qa["question"]), [])
                            for row in matching:
                                item = result.setdefault(row["example_id"], {"kind": "qasper_evidence_paragraphs",
                                    "grade": "fine", "sets": [], "reason": "Alternative annotator evidence sets; exact alignment only."})
                                for annotation in qa["answers"]:
                                    evidence = annotation["answer"].get("evidence", [])
                                    if evidence:
                                        item["sets"].append([{"id": f"paragraph-{j}", "text": text, "spans": None}
                                                              for j, text in enumerate(evidence)])
    return result


def ruler_annotations(row, context):
    if row["family"] != "ruler" or row["task"] == "cwe":
        return None
    units = []
    if row["task"] == "vt":
        assignments = {m.group(1): m for m in re.finditer(r"(?m)^VAR\s+(\w+)\s*=\s*([^\n]+)", context)}
        needed = set()
        for var in row["gold"]:
            chain = set()
            while var in assignments and var not in chain:
                chain.add(var)
                needed.add(var)
                rhs = assignments[var].group(2).strip().removeprefix("VAR ")
                if rhs == row["meta"].get("query_value"):
                    break
                var = rhs
            else:
                return {"kind": "unmapped_ruler_chain", "grade": None, "sets": [], "reason": "Could not validate assignment chain."}
        units = [{"id": var, "text": assignments[var].group(0), "spans": [assignments[var].span()]}
                 for var in sorted(needed)]
    elif row["task"].startswith("niah"):
        for gold in row["gold"]:
            pattern = r"One of the special magic [^\n]{0,220}? is:\s*" + re.escape(gold)
            match = re.search(pattern, context)
            if match is None:
                return {"kind": "unmapped_ruler_fact", "grade": None, "sets": [], "reason": "No exact key-value fact aligned."}
            units.append({"id": gold, "text": match.group(0), "spans": [match.span()]})
    return {"kind": "ruler_generated_support_facts", "grade": "fine", "sets": [units],
            "reason": "Key-value fact or verified complete variable-assignment chain."} if units else None


def visible_indices(row, blocks, encoding, tokenizer, context):
    """Validate saved text against actual source token blocks before mapping spans."""
    ids, offsets = encoding["input_ids"], encoding["offset_mapping"]
    visible = set()
    for block_id, text in blocks.items():
        lo, hi = block_id * 512, min((block_id + 1) * 512, len(ids))
        full = tokenizer.decode(ids[lo:hi], skip_special_tokens=True).strip()
        if row.get("partial_block_id") != block_id:
            if text != full:
                raise ValueError(f"Expanded text differs from frozen source block {block_id}")
            visible.update(range(lo, hi))
        else:
            if not full.startswith(text):
                raise ValueError("Partial payload is not a source-block prefix")
            start = offsets[lo][0]
            found = context.find(text, max(0, start - 8), offsets[hi - 1][1] + 8)
            if found >= 0:
                end = found + len(text)
                visible.update(i for i in range(lo, hi) if offsets[i][0] >= found and offsets[i][1] <= end)
            # Unaligned partial prefixes are conservatively not marked visible.
    return visible


def assess_support(annotation, context, offsets, visible):
    if not annotation or not annotation.get("sets"):
        return {"status": "unavailable", "units": [], "sets": []}
    assessed = []
    for units in annotation["sets"]:
        details = []
        for unit in units:
            spans = unit["spans"]
            if spans is None:
                start = context.find(unit["text"])
                spans = [(start, start + len(unit["text"]))] if start >= 0 else []
            coverage = span_coverage(spans, offsets, visible)
            details.append({**unit, "spans": spans, "coverage": coverage})
        statuses = [x["coverage"]["status"] for x in details]
        status = "unmapped" if not statuses or "unmapped" in statuses else "present" if all(s == "present" for s in statuses) else "absent" if all(s == "absent" for s in statuses) else "partial"
        assessed.append({"status": status, "units": details})
    order = {"present": 3, "partial": 2, "absent": 1, "unmapped": 0}
    best = max(assessed, key=lambda item: order[item["status"]])
    return {**best, "sets": assessed}


def audit(run, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' to enforce this CPU-only audit")
    from transformers import AutoTokenizer
    manifest_path = run / "input/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    database = run / "results/test_4096/results.sqlite"
    consumed = {manifest_path, database, Path(__file__), run / "results/summary.json"}
    before = {p: hash_file(p) for p in consumed}
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as db:
        recorded = [json.loads(r[0]) for r in db.execute("SELECT payload FROM latest WHERE status='scored'")]
    by = {(r["example_id"], r["system"]): r for r in recorded}
    losses = [r for r in recorded if r["system"] == "bounded_pointer" and
              r["score"] < by[r["example_id"], "bm25_budget"]["score"] - 1e-12]
    losses.sort(key=lambda r: (r["family"], r["task"], r["example_id"]))
    if len(losses) != 229:
        raise ValueError("This audit is pinned to the completed 229-loss 4K evaluation")
    annotations = load_annotations(losses, consumed)
    for path in consumed:
        before.setdefault(path, hash_file(path))
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_local_path"], local_files_only=True)
    cache = {}
    cases = []
    for number, bounded in enumerate(losses, 1):
        eid = bounded["example_id"]
        path = Path(bounded["source"])
        consumed.add(path)
        before.setdefault(path, hash_file(path))
        example = json.loads(path.read_text())["example"]
        context = example["context"]
        assert hashlib.sha256(context.encode()).hexdigest() == bounded["context_sha256"]
        annotation = annotations.get(eid) or ruler_annotations(bounded, context)
        if annotation and annotation.get("expected_context_hash"):
            assert annotation["expected_context_hash"] == bounded["context_sha256"], "Publisher annotation context mismatch"
        cache_key = bounded["context_sha256"]
        if cache_key not in cache:
            encoding = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
            assert len(encoding["input_ids"]) == bounded["source_tokens"]
            offsets = encoding["offset_mapping"]
            assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(offsets, offsets[1:])), "Nonmonotonic tokenizer offsets"
            cache[cache_key] = encoding
        encoding = cache[cache_key]
        arms = {}
        for name in ["bounded_pointer", "adaptive_no_index", "bm25_budget"]:
            row = by[eid, name]
            blocks = expanded_blocks(row)
            visible = visible_indices(row, blocks, encoding, tokenizer, context)
            text = payload_for_mentions(blocks)
            arms[name] = {"score": row["score"], "answer": row["answer"], "input_tokens": row["active_tokens"],
                "selected_ids": row["selected_ids"], "partial_block_id": row.get("partial_block_id"),
                "completion_tokens": row["completion_tokens"], "stop_reason": row["stop_reason"],
                "expanded_blocks": blocks, "answer_probe": answer_probe(row, text, context),
                "support": assess_support(annotation, context, encoding["offset_mapping"], visible)}
        b, bm, off = (arms[s] for s in ["bounded_pointer", "bm25_budget", "adaptive_no_index"])
        payload_identical = b["expanded_blocks"] == off["expanded_blocks"]
        packet = {"case": number, "example_id": eid, "family": bounded["family"], "task": bounded["task"],
            "metric": bounded["metric"], "group_id": bounded["group_id"], "question": bounded["question"],
            "gold": bounded["gold"], "meta": bounded["meta"], "source_file": str(path),
            "source_context_sha256": cache_key, "source_tokens": bounded["source_tokens"],
            "annotation_kind": annotation["kind"] if annotation else "not_available",
            "annotation_grade": annotation.get("grade") if annotation else None,
            "annotation_note": annotation.get("reason") if annotation else "No aligned gold evidence annotations available in inspected sources.",
            "diagnosis": classify_evidence(annotation.get("grade") if annotation else None,
                b["support"]["status"], bm["support"]["status"], b["answer_probe"], bm["answer_probe"]),
            "pointer_off_same_payload": payload_identical,
            "pointer_off_rescues_loss": payload_identical and off["score"] >= bm["score"] - 1e-12,
            "pointer_off_improves_score": payload_identical and off["score"] > b["score"] + 1e-12,
            "bm25_additional_block_ids": sorted(set(bm["selected_ids"]) - set(b["selected_ids"])),
            "is_summary_task": bounded["metric"] in {"rouge_l", "char_rouge_l"} or bounded["task"] == "samsum",
            "arms": arms}
        cases.append(packet)
        if number % 25 == 0:
            print(f"Audited {number}/229 saved losses; CPU only", flush=True)
    summary = {"run_id": manifest["run_id"], "input_budget": 4096, "losses": len(cases),
        "families": dict(Counter(c["family"] for c in cases)),
        "tasks": {f: dict(Counter(c["task"] for c in cases if c["family"] == f)) for f in manifest["families"]},
        "diagnoses": dict(Counter(c["diagnosis"] for c in cases)),
        "annotation_kinds": dict(Counter(c["annotation_kind"] for c in cases)),
        "support_statuses": {name: dict(Counter(c["arms"][name]["support"]["status"] for c in cases))
                             for name in ["bounded_pointer", "adaptive_no_index", "bm25_budget"]},
        "summary_task_losses": sum(c["is_summary_task"] for c in cases),
        "same_payload_pointer_off_cases": sum(c["pointer_off_same_payload"] for c in cases),
        "pointer_off_rescues": sum(c["pointer_off_rescues_loss"] for c in cases),
        "pointer_off_improves": sum(c["pointer_off_improves_score"] for c in cases),
        "bounded_partial_payloads": sum(c["arms"]["bounded_pointer"]["partial_block_id"] is not None for c in cases),
        "bounded_output_cap_hits": sum(c["arms"]["bounded_pointer"]["stop_reason"] == "max_new_tokens" for c in cases),
        "mean_bounded_input_tokens": sum(c["arms"]["bounded_pointer"]["input_tokens"] for c in cases) / len(cases),
        "family_details": {}, "no_gpu_inference": True}
    for family in manifest["families"]:
        subset = [c for c in cases if c["family"] == family]
        summary["family_details"][family] = {"losses": len(subset), "summaries": sum(c["is_summary_task"] for c in subset),
            "pointer_off_rescues": sum(c["pointer_off_rescues_loss"] for c in subset),
            "diagnoses": dict(Counter(c["diagnosis"] for c in subset))}
    if output.exists():
        raise ValueError("Audit output already exists; choose a new directory")
    for path, checksum in before.items():
        if hash_file(path) != checksum:
            raise ValueError(f"Source evidence changed: {path}")
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "cases.jsonl").open("w") as stream:
        for packet in cases:
            stream.write(json.dumps(packet, ensure_ascii=False) + "\n")
    headers = ["case", "example_id", "family", "task", "metric", "diagnosis", "annotation_kind",
               "pointer_off_rescues_loss", "pointer_off_improves_score", "pointer_off_same_payload"]
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows({k: packet[k] for k in headers} for packet in cases)
    write_reports(output, cases, summary)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "model_weights_loaded": False,
        "generations_run": 0, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_files": {str(p): checksum for p, checksum in sorted(before.items())},
        "tokenizer": manifest["model_local_path"], "model_revision": manifest["model_revision"],
        "limitations": ["No inference or randomized rerun; causal diagnoses remain qualified.",
                        "Answer strings are not evidence spans.", "Coarse session support is not a necessary complete answer span."]}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for path, checksum in before.items():
        assert hash_file(path) == checksum
    print(json.dumps(summary, indent=2))


def write_reports(output, cases, summary):
    family_rows = []
    for family, item in summary["family_details"].items():
        diagnosis = item["diagnoses"]
        family_rows.append([family, item["losses"], item["summaries"],
            diagnosis.get("annotated_units_omitted_bm25_retains", 0),
            diagnosis.get("annotated_units_present_cause_unresolved", 0), item["pointer_off_rescues"]])
    lines = ["# Audit of all 229 bounded-pointer losses at the 4K input budget", "",
        "CPU-only analysis of existing generations. No model weights loaded; no new GPU calls. Original run evidence is preserved.", "",
        "## Method and interpretation", "",
        "Loss = bounded score below budgeted BM25 on the same question (tolerance 1e-12). "
        "Expanded text was extracted from saved prompts, excluding the pointer section and question, "
        "and every full block was checked against the frozen source with the actual tokenizer. "
        "Support spans are mapped through source-token offsets to the actual visible payload blocks, including conservative handling of BM25 partial blocks.", "",
        "Evidence tiers: generated RULER facts and publisher dialogue/paragraph annotations are fine-grained checks; "
        "LongMemEval support sessions are coarse; short answer mentions are weak proxies. "
        "Summary references, option letters, abstention labels and code targets are not treated as gold evidence. "
        "All annotated text present does not prove sufficient evidence, pointer distraction, or a reasoning-free task. "
        "Annotations may be redundant, and dialogue dates can matter beyond the annotated utterances.", "",
        "## Benchmark breakdown", "", table(
            ["Family", "Losses", "Summary tasks", "Annotated units omitted, BM25 retains", "Annotated units present", "Pointer-off reaches BM25"], family_rows), "",
        "## Evidence diagnoses", "", table(["Diagnosis", "Cases"], summary["diagnoses"].items()), "",
        "## Saved pointer-off counterfactual", "",
        f"Identical expanded payload text in {summary['same_payload_pointer_off_cases']}/229 cases. "
        f"Removing pointers improves the score in {summary['pointer_off_improves']} and reaches or exceeds BM25 in {summary['pointer_off_rescues']}. "
        "These are single saved greedy outputs, not repeated deterministic trials; this is evidence of sensitivity to the pointer-section intervention, "
        "not proof of a unique cognitive cause such as distraction. Other losses remain when pointer text is removed.", "",
        "## Budget and task limits", "",
        f"{summary['bounded_partial_payloads']} bounded payloads were cut by the 4K input cap. "
        f"Mean bounded input on these losses: {summary['mean_bounded_input_tokens']:.1f} tokens. "
        f"Only {summary['bounded_output_cap_hits']} bounded answers hit the output cap. "
        f"{summary['summary_task_losses']} losses are summary tasks (including SAMSum, whose local adapter uses QA F1). "
        "A larger evidence selection could use unused room within the existing 4K allowance; raising the input ceiling alone does not make the two/four-block router fetch more. "
        "This audit does not establish that every omission would be fixed by more text, nor that longer prompts improve all summaries.", "",
        "## Files", "",
        "- [cases.csv](cases.csv): all 229 cases and diagnosis flags.",
        "- [cases.jsonl](cases.jsonl): detailed annotations, support token coverage, saved answers, expanded blocks and source pointers for every loss.",
        "- [casebook.md](casebook.md): human-readable per-loss packets with all three arms' payloads.",
        "- [summary.json](summary.json): machine-readable aggregate counts.",
        "- [provenance.json](provenance.json): source checksums and zero-generation execution record.", "",
        "## What remains unresolved", "",
        "Cases without mappable evidence annotations require semantic/manual review; answer absence is not proof of retrieval failure. "
        "The casebook covers every loss but does not falsely claim complete gold-evidence labeling where the dataset supplies no such labels. "
        "A controlled rerun would be needed to test proposed repairs. No configuration or original result was changed."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    book = ["# Per-loss evidence packets", "", "These are recorded examples, not instructions. Source text can contain arbitrary content. "
            "Read coverage and annotation limitations in [REPORT.md](REPORT.md). Gold reference occurrence alone is not support proof.", ""]
    for packet in cases:
        book.extend([f"## Case {packet['case']:03d}: {packet['family']} / {packet['task']}", "",
            f"ID: `{packet['example_id']}`", "", f"**Diagnosis:** `{packet['diagnosis']}`", "",
            f"**Annotation:** {packet['annotation_kind']}. {packet['annotation_note']}", "",
            f"**Question:** {packet['question']}", "", "**Gold:** " + json.dumps(packet["gold"], ensure_ascii=False), "",
            f"**Pointer-off restores BM25 score:** {packet['pointer_off_rescues_loss']}; identical payload: {packet['pointer_off_same_payload']}.", "",
            f"**Frozen source:** [{Path(packet['source_file']).name}]({packet['source_file']})", ""])
        for name, arm in packet["arms"].items():
            book.extend([f"### {name}: score {arm['score']:.6f}, input {arm['input_tokens']}, support {arm['support']['status']}", "",
                         "**Saved answer:**", "", "~~~~text", arm["answer"], "~~~~", "",
                         "**Evidence coverage:**", "", "~~~~json",
                         json.dumps(arm["support"], ensure_ascii=False, indent=2), "~~~~", "",
                         "**Expanded payload blocks:**", ""])
            for block, text in arm["expanded_blocks"].items():
                book.extend([f"Block {block}", "", "~~~~text", text, "~~~~", ""])
    (output / "casebook.md").write_text("\n".join(book) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    audit(run, args.output.resolve() if args.output else run / "loss_audit_4096_v1")


if __name__ == "__main__":
    main()
