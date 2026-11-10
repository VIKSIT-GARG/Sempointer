"""Lazy, explicit adapters. No gold annotations are included in model contexts."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import io
from itertools import groupby
import json
from pathlib import Path
import re
import tarfile
import zipfile

import ijson


@dataclass
class Task:
    family: str
    task: str
    path: Path
    adapter: str = ""
    split: str = "test"
    member: str | None = None
    max_new_tokens: int = 128
    extra: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.family}/{self.task}/{self.split}"

    def to_dict(self):
        return {**vars(self), "path": str(self.path), "key": self.key}


@dataclass
class Example:
    id: str
    context: str
    question: str
    gold: list[str]
    metric: str
    family: str
    task: str
    split: str
    group_id: str
    protocol: str = "local_adaptation"
    meta: dict = field(default_factory=dict)
    skip_reason: str | None = None


@contextmanager
def open_member(path, member=None):
    path = Path(path)
    if member is None:
        with path.open("rb") as stream:
            yield stream
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
            yield stream
    else:
        with tarfile.open(path) as archive:
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Not a regular archive member: {member}")
            with stream:
                yield stream


def read_records(path, member=None):
    """At most one JSON-array record / JSONL line / Parquet batch in memory."""
    name = member or str(path)
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            yield from batch.to_pylist()
        return
    with open_member(path, member) as stream:
        if name.endswith(".jsonl"):
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            yield from ijson.items(stream, "item", use_float=True)


def as_answers(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None and str(x).strip()]
    return [str(value)] if str(value).strip() else []


def paragraphs(value):
    if isinstance(value, dict):
        return "\n\n".join(str(t) + "\n" + " ".join(s) for t, s in zip(value["title"], value["sentences"]))
    return "\n\n".join(str(t) + "\n" + " ".join(s) for t, s in value)


def make_example(t, i, context, question, gold, metric="qa_f1", *, group=None, protocol="local_adaptation", meta=None):
    answers = as_answers(gold)
    return Example(f"{t.key}:{t.path.name}:{t.member or ''}:{i}", str(context or ""), str(question or ""), answers, metric,
                   t.family, t.task, t.split, f"{t.family}:{group if group is not None else i}",
                   protocol, meta or {}, "missing_gold" if not answers else "empty_context" if not context else None)


def longbench_metric(task):
    task = task.removesuffix("_e")
    if task == "vcsum":
        return "char_rouge_l", "Chinese character-token ROUGE-L adaptation"
    if task in {"gov_report", "qmsum", "multi_news"}:
        return "rouge_l", "ROUGE-L token-LCS adaptation"
    if task in {"lcc", "repobench-p"}:
        return "code_similarity", "first-line edit-similarity adaptation; no code execution"
    if task in {"trec", "lsht"}:
        return "exact", "normalized label exact-match adaptation"
    if task == "passage_count":
        return "numeric_sequence", "numeric exact-match adaptation"
    if task.startswith("passage_retrieval"):
        return "all_containment", "reference containment adaptation"
    if task in {"multifieldqa_zh", "dureader"}:
        return "char_f1", "Chinese character-F1 adaptation"
    return "qa_f1", "LongBench normalized QA token-F1; retrieval prompt adaptation"


def grouped_scrolls(records):
    """Official reference rows are contiguous; reject a non-contiguous repeat
    rather than silently weighting the same question multiple times."""
    seen=set()
    for ident,group in groupby(records,key=lambda x:x["id"]):
        if ident in seen:
            raise ValueError(f"Non-contiguous SCROLLS question ID: {ident}")
        seen.add(ident)
        first=None;answers=[]
        for row in group:
            if first is None:first=row
            elif row["input"]!=first["input"]:raise ValueError("SCROLLS references have different inputs")
            if row.get("output") not in (None,""):answers.append(row["output"])
        yield {**first,"_references":list(dict.fromkeys(answers))}


def iter_examples(t,tokenizer=None):
    if t.adapter == "prepared":
        yield Example(**json.loads(t.path.read_text())["example"])
        return
    if t.adapter == "ruler":
        from benchmarks.ruler import build_ruler
        # One finite, explicitly seeded generated sample set per task/length.
        for ex in build_ruler(task=t.task, context_length=t.extra["context_length"],
                              n_samples=t.extra["n_samples"], seed=t.extra["seed"],
                              tokenizer_fn=(lambda text:len(tokenizer.encode(text,add_special_tokens=False))) if tokenizer is not None else None):
            yield make_example(t, f"seed{t.extra['seed']}:{ex.id}", ex.context, ex.question, ex.gold, "all_containment",
                               protocol="project RULER generator; documented upstream deviations", meta=ex.meta)
        return
    if t.adapter == "qasper":
        with open_member(t.path, t.member) as stream:
            for doc_id, doc in ijson.kvitems(stream, "", use_float=True):
                context = (doc.get("title") or "") + "\n" + (doc.get("abstract") or "") + "\n" + "\n".join(
                    (section.get("section_name") or "") + "\n" + "\n".join(str(p) for p in (section.get("paragraphs") or []) if p is not None)
                    for section in (doc.get("full_text") or []))
                for qa in doc["qas"]:
                    answers = []
                    for annotation in qa.get("answers", []):
                        a = annotation["answer"]
                        if a.get("unanswerable"):
                            answers.append("unanswerable")
                        elif a.get("yes_no") is not None:
                            answers.append("yes" if a["yes_no"] else "no")
                        elif a.get("free_form_answer"):
                            answers.append(a["free_form_answer"])
                        else:
                            answers.append(" ".join(a.get("extractive_spans", [])))
                    yield make_example(t, qa["question_id"], context, qa["question"], answers, group=doc_id)
        return
    records=read_records(t.path,t.member)
    if t.adapter=="scrolls":records=grouped_scrolls(records)
    for index, x in enumerate(records):
        ident = x.get("_id", x.get("id", x.get("question_id", index)))
        group = x.get("doc_id", x.get("document", {}).get("id") if isinstance(x.get("document"), dict) else None)
        if t.adapter == "longbench":
            metric, protocol = longbench_metric(t.task)
            question=x["input"]
            if t.task.removesuffix("_e") in {"trec","lsht"}:
                question="Classify the following question/text using one label from: "+", ".join(x["all_classes"])+"\n"+question
            # Group paraphrased/repeated questions on the same source document.
            group = hashlib.sha256(x["context"].encode()).hexdigest()
            yield make_example(t, ident, x["context"], question, x.get("answers"), metric, group=group, protocol=protocol)
        elif t.adapter == "longbench_v2":
            question = x["question"] + "\n" + "\n".join(f"{k}. {x['choice_' + k]}" for k in "ABCD") + "\nReturn only the option letter."
            yield make_example(t, ident, x["context"], question, x.get("answer"), "choice", group=hashlib.sha256(x["context"].encode()).hexdigest(),
                               protocol="choice accuracy; SemPointer retrieval prompt adaptation", meta={k:x.get(k) for k in ["domain", "sub_domain", "difficulty", "length"]})
        elif t.adapter == "babilong":
            yield make_example(t, ident, x["input"], x["question"], x.get("target"), "exact", protocol="normalized answer exact match; retrieval adaptation")
        elif t.adapter == "longmem":
            if len(x["haystack_dates"]) != len(x["haystack_sessions"]):
                raise ValueError("LongMemEval dates/session count mismatch")
            context = "\n\n".join(f"Session {j}; date: {date}\n" + "\n".join(f"{m['role']}: {m['content']}" for m in session)
                for j, (date, session) in enumerate(zip(x["haystack_dates"], x["haystack_sessions"])))
            yield make_example(t, ident, context, x["question"] + "\nQuestion date: " + x.get("question_date", ""), x.get("answer"),
                               protocol="token-F1 surrogate; official model-based judge NOT run", group=ident,
                               meta={"question_type":x.get("question_type"), "oracle": "oracle" in t.task})
        elif t.adapter == "leval":
            if len(x["instructions"]) != len(x["outputs"]):
                raise ValueError("L-Eval instruction/reference count mismatch")
            for j, (question, answer) in enumerate(zip(x["instructions"], x["outputs"])):
                metric = "rouge_l" if x.get("evaluation") == "rouge" else "qa_f1"
                if t.task in {"quality", "tpo"}:
                    metric = "choice"
                    answer = re.search(r"\b([A-D])\b", answer).group(1)
                elif t.task == "coursera":
                    metric = "choice_set"
                elif t.task == "sci_fi":
                    metric, answer = "exact", answer.split()[0]
                elif t.task in {"gsm100", "codeU"}:
                    metric = "numeric_sequence"
                yield make_example(t, f"{ident}:{j}", x["input"], question, answer, metric, group=f"{t.task}:{index}", protocol="local L-Eval task metric adaptation")
        elif t.adapter == "loogle":
            answer = x.get("answer")
            metric = "rouge_l" if t.task == "summarization" else "qa_f1"
            if t.task == "shortdep_cloze":
                if isinstance(answer, str):
                    answer = ast.literal_eval(answer)
                metric = "all_containment"
            elif t.task == "longdep_qa" and str(answer).strip() in {"1", "2", "3", "4"}:
                metric = "numeric_sequence"
            yield make_example(t, ident, x["context"], x.get("question") or "Summarize the document.", answer, metric, group=group,
                               protocol="local LooGLE metric adaptation; no model-based judge")
        elif t.adapter == "infinite":
            answer = x.get("answer")
            metric = "rouge_l" if "sum" in t.task else "char_f1" if t.task.endswith("chn") else "qa_f1"
            question = x.get("input", "")
            if x.get("options"):
                question += "\n" + "\n".join(f"{chr(65+j)}. {v}" for j,v in enumerate(x["options"]))
                correct = next((j for j,v in enumerate(x["options"]) if str(v) in as_answers(answer)), None)
                if correct is None:
                    raise ValueError("Choice gold absent from options")
                answer, metric = chr(65+correct), "choice"
            elif t.task in {"math_calc", "math_find", "code_run"}:
                answer, metric = json.dumps(answer), "numeric_sequence"
            elif t.task in {"passkey", "number_string", "kv_retrieval"}:
                metric = "all_containment"
            yield make_example(t, ident, x["context"], question, answer, metric, protocol="local InfiniteBench metric adaptation")
        elif t.adapter == "scrolls":
            text = x["input"]
            if t.task=="quality":
                last_option=re.search(r"\(D\)",text)
                boundary=re.search(r"\n{3,}",text[last_option.end():]) if last_option else None
                if boundary is None:raise ValueError("SCROLLS QuALITY option/story boundary missing")
                start=last_option.end()+boundary.start();end=last_option.end()+boundary.end()
                question,context=text[:start],text[end:]
                if not all(f"({option})" in question for option in "ABCD"):
                    raise ValueError("SCROLLS QuALITY option missing from question")
            elif t.task in {"qasper", "narrative_qa", "qmsum", "contract_nli"}:
                question, sep, context = text.partition("\n\n")
                if not sep:
                    question, sep, context = text.partition("\n")
            else:
                question, context = "Summarize the document.", text
            metric = "rouge_l" if t.task in {"qmsum", "gov_report", "summ_screen_fd"} else "exact" if t.task == "contract_nli" else "qa_f1"
            if t.task=="contract_nli":
                question="Classify this claim as Entailment, Contradiction, or NotMentioned using the document: "+question
            yield make_example(t, ident, context, question, x["_references"], metric,
                               group=hashlib.sha256(context.encode()).hexdigest(), protocol="SCROLLS merged-reference local metric adaptation; not official aggregate")
        elif t.adapter == "hotpot":
            yield make_example(t, ident, paragraphs(x["context"]), x["question"], x.get("answer"), group=ident)
        elif t.adapter == "musique":
            context = "\n\n".join(p["title"] + "\n" + p["paragraph_text"] for p in x["paragraphs"])
            gold = [x["answer"], *x.get("answer_aliases", [])] if "answer" in x else []
            if x.get("answerable") is False:
                gold = ["unanswerable"]
            yield make_example(t, ident, context, x["question"], gold)
        elif t.adapter == "narrative":
            yield make_example(t, ident, x["document"]["text"], x["question"]["text"], [a["text"] for a in x["answers"]], group=group)
        elif t.adapter == "locomo":
            conv = x["conversation"]
            keys = sorted((k for k,v in conv.items() if re.fullmatch(r"session_\d+", k) and isinstance(v,list)), key=lambda k:int(k.split("_")[1]))
            context = "\n\n".join(f"{key}; date: {conv.get(key+'_date_time','')}\n" + "\n".join(f"{m.get('speaker','')}: {m.get('text','')}" for m in conv[key] if m.get("text")) for key in keys)
            for j,qa in enumerate(x["qa"]):
                category = int(qa["category"])
                answer = "not mentioned" if category == 5 else qa.get("answer")
                if category == 3 and isinstance(answer,str):
                    answer = answer.split(";")[0].strip()
                yield make_example(t, f"{x.get('sample_id',index)}:{j}", context, qa["question"], answer, group=x.get("sample_id",index),
                                   protocol="LoCoMo token-F1 adaptation; session dates included", meta={"category":category})
        else:
            raise ValueError(f"Unknown adapter: {t.adapter}")


def discover_tasks(root, include_train=False, seed=143, ruler_samples=10, lengths=(4096,8192,16384)):
    root = Path(root)
    tasks, coverage = [], []
    def add(family, task, path, adapter, split="test", member=None, cap=128, extra=None):
        if split != "train" or include_train:
            tasks.append(Task(family, task, path, adapter, split, member, cap, extra or {}))
    hf = root / "huggingface"
    p = hf / "zai-org/LongBench/data.zip"
    if p.exists():
        with zipfile.ZipFile(p) as z:
            for member in sorted(z.namelist()):
                if member.endswith(".jsonl"):
                    task = Path(member).stem
                    metric, _ = longbench_metric(task)
                    add("longbench", task, p, "longbench", member=member, cap=512 if metric in {"rouge_l","char_rouge_l"} else 128)
    patterns = [
        ("longbench_v2", "zai-org/LongBench-v2", "data.json", "longbench_v2", 32),
        ("longmem", "xiaowu0162/longmemeval-cleaned", "*.json", "longmem", 128),
        ("babilong", "RMT-team/babilong", "data/*/*.json", "babilong", 128),
        ("infinite", "xinrongzhang2022/InfiniteBench", "*.jsonl", "infinite", 256),
        ("leval", "L4NLP/LEval", "LEval/*/*.jsonl", "leval", 512),
        ("loogle", "bigai-nlco/LooGLE", "data/*.jsonl", "loogle", 256),
        ("hotpot_hf", "hotpotqa/hotpot_qa", "*/*.parquet", "hotpot", 128),
        ("narrative_original", "deepmind/narrativeqa", "data/*.parquet", "narrative", 128),
    ]
    for family,repo,pattern,adapter,cap in patterns:
        for p in sorted((hf/repo).glob(pattern)):
            split = "train" if p.name.startswith("train-") else "validation" if p.name.startswith("validation-") else "test"
            task = str(p.parent.relative_to(hf/repo)) if adapter=="hotpot" else "narrativeqa" if adapter=="narrative" else str(p.relative_to(hf/repo).with_suffix(""))
            if adapter in {"leval", "loogle", "infinite"}:
                task = p.stem
            add(family,task,p,adapter,split,cap=cap)
    for p in sorted((hf/"tau/scrolls").glob("*.zip")):
        with zipfile.ZipFile(p) as z:
            for member in sorted(z.namelist()):
                if member.endswith(".jsonl") and "__MACOSX" not in member:
                    add("scrolls",p.stem,p,"scrolls",Path(member).stem,member,512 if p.stem in {"qmsum","gov_report","summ_screen_fd"} else 128)
    p = root/"originals/locomo/locomo10.json"
    if p.exists():
        add("locomo","qa",p,"locomo")
    # Original corpora are retained as separate protocols, never pooled with
    # their overlapping LongBench/SCROLLS subsets.
    for folder,adapter in [("2wikimultihop","hotpot"),("musique","musique")]:
        for p in sorted((root/"originals"/folder).glob("*.zip")):
            with zipfile.ZipFile(p) as z:
                for member in sorted(z.namelist()):
                    name=Path(member).name.lower()
                    if adapter=="musique" and not name.startswith(("musique_ans_","musique_full_")):continue
                    split=next((s for s in ("train","dev","test") if re.search(r"(?:^|_)(?:"+s+r")(?:\.|_)",name)),None)
                    if split and "__MACOSX" not in member and name.endswith((".json",".jsonl")):
                        add(folder,p.stem+"/"+Path(member).stem,p,adapter,"validation" if split=="dev" else split,member)
    for p in sorted((root/"originals/hotpotqa").glob("*.json")):
        split="train" if "train" in p.name else "validation" if "dev" in p.name else "test"
        add("hotpot_original",p.stem,p,"hotpot",split)
    for p in sorted((root/"originals/qasper").glob("*.tgz")):
        with tarfile.open(p) as z:
            for m in z.getmembers():
                if m.name.endswith(".json") and Path(m.name).name.startswith("qasper-"):
                    split="train" if "-train-" in m.name else "validation" if "-dev-" in m.name else "test"
                    add("qasper_original","qasper",p,"qasper",split,m.name)
    asset_root = root.parent/"experiments/benchmarks/data/ruler_cache"
    if asset_root.exists():
        for task in ["niah_single_1","niah_single_2","niah_single_3","niah_multikey_1","niah_multikey_2","vt","cwe","fwe"]:
            for length in lengths:
                add("ruler",task,asset_root,"ruler",cap=512 if task=="cwe" else 128,
                    extra={"context_length":length,"n_samples":ruler_samples,"seed":seed})
    # Account for every queued download, including missing/partial files and
    # non-evaluation assets, independently of the set of executable tasks.
    manifest=root/"status.json"
    if manifest.exists():
        for item in json.loads(manifest.read_text())["files"]:
            path=root/item["path"]
            used=any(t.path==path for t in tasks)
            role="generation_source" if item["path"].startswith("sources/") else "retrieval_corpus" if item["collection"]=="beir" else "data_or_metadata_not_scored"
            coverage.append({"path":str(path),"collection":item["collection"],"download_status":item["status"],"exists":path.is_file(),
                             "role":"task_source" if used else role,"revision":item.get("revision"),"sha256":item.get("sha256") or item.get("expected_sha256")})
    return tasks, coverage


def partition_for(ex):
    """Keep every question from a document/conversation in the same partition."""
    return "dev" if int(hashlib.sha256(ex.group_id.encode()).hexdigest()[:8],16)%5==0 else "test"
