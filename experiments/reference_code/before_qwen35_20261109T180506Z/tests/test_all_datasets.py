"""CPU contract tests; no downloaded model, inference, or fabricated evidence."""
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.data import Task, Example, read_records, iter_examples, discover_tasks
from evaluation.controls import prepare, prompts_for_example
from evaluation.metrics import score
from evaluation.runner import RunStore, summarize


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def task(tmp_path, adapter, rows, suffix=".jsonl", **kwargs):
    path = tmp_path / (adapter + suffix)
    if suffix == ".jsonl":
        path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    else:
        path.write_text(json.dumps(rows))
    return Task(adapter, adapter, path, adapter=adapter, **kwargs)


def test_array_reader_handles_chunks_and_rejects_truncation(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps([{"x": "é" * 10000}, {"x": 2}]))
    assert [x["x"] for x in read_records(p)][1] == 2
    p.write_text('[{"x":1},')
    with pytest.raises(Exception):
        list(read_records(p))


def test_leval_keeps_each_question_reference_pair(tmp_path):
    t = task(tmp_path, "leval", [{"input": "document", "instructions": ["q1", "q2"],
                                "outputs": ["a1", "a2"], "evaluation": "f1"}])
    examples = list(iter_examples(t))
    assert [(x.question, x.gold) for x in examples] == [("q1", ["a1"]), ("q2", ["a2"])]
    assert examples[0].group_id == examples[1].group_id
    assert examples[0].id != examples[1].id


def test_longmem_dates_are_preserved_but_answer_flags_are_not(tmp_path):
    t = task(tmp_path, "longmem", [{"question_id": "q", "question": "when?", "answer": "2023",
        "question_date": "2024", "haystack_dates": ["2023"], "haystack_session_ids": ["s"],
        "haystack_sessions": [[{"role": "user", "content": "hello", "has_answer": True}]]}], suffix=".json")
    ex = next(iter_examples(t))
    assert "2023" in ex.context and "hello" in ex.context
    assert "has_answer" not in ex.context and "true" not in ex.context.lower()
    assert "2024" in ex.question
    assert "surrogate" in ex.protocol


def test_choice_adapter_and_hidden_gold(tmp_path):
    rows = [{"_id": "x", "context": "a document", "question": "which?", "answer": "B",
             **{f"choice_{c}": c.lower() for c in "ABCD"}}]
    t = task(tmp_path, "longbench_v2", rows, suffix=".json")
    ex = next(iter_examples(t))
    assert "B. b" in ex.question and ex.metric == "choice"
    assert score("choice", "The answer is B.", ex.gold) == 1
    rows[0].pop("answer")
    t.path.write_text(json.dumps(rows))
    assert next(iter_examples(t)).skip_reason == "missing_gold"


def test_babilong_uses_question_not_target_in_prompt(tmp_path):
    t = task(tmp_path, "babilong", [{"input": "Mary went home", "question": "where?", "target": "home"}], suffix=".json")
    ex = next(iter_examples(t))
    assert ex.context == "Mary went home" and ex.question == "where?" and ex.gold == ["home"]


def test_qasper_null_document_metadata_does_not_drop_questions(tmp_path):
    p=tmp_path/"qasper.json"
    p.write_text(json.dumps({"doc":{"title":None,"abstract":None,
        "full_text":[{"section_name":None,"paragraphs":["Some evidence."]}],
        "qas":[{"question_id":"q","question":"What?","answers":[{"answer":{"free_form_answer":"evidence"}}]}]}}))
    ex=next(iter_examples(Task("qasper_original","qasper",p,adapter="qasper")))
    assert "Some evidence." in ex.context
    assert ex.gold==["evidence"]


def test_discovery_includes_all_longbench_tasks_and_excludes_training(tmp_path):
    p = tmp_path / "huggingface/zai-org/LongBench/data.zip"
    p.parent.mkdir(parents=True)
    with zipfile.ZipFile(p, "w") as z:
        for name in ["qasper", "gov_report", "lcc"]:
            z.writestr(f"data/{name}.jsonl", json.dumps({"input": "q", "context": "c", "answers": ["a"]}) + "\n")
    tasks, coverage = discover_tasks(tmp_path)
    assert {x.task for x in tasks if x.family == "longbench"} == {"qasper", "gov_report", "lcc"}
    assert all(x.split != "train" for x in tasks)


def test_short_context_router_and_pointer_off_share_selected_ids():
    tok = Tokenizer()
    context = "mars rover landed in 2021"
    prepared = prepare(context, tok, block_size=512)
    variants = prompts_for_example(prepared, "when did the rover land?", tok)
    assert variants["sempointer"]["selected_ids"] == variants["pointer_off"]["selected_ids"] == [0]


def test_pointer_off_and_budget_control_are_real_controls():
    tok = Tokenizer()
    context = " ".join([f"planet{i} mars rover mission landing evidence " * 30 for i in range(8)])
    prepared = prepare(context, tok, block_size=64)
    variants = prompts_for_example(prepared, "mars rover", tok)
    sem = variants["sempointer"]
    off = variants["pointer_off"]
    budget = variants["bm25_matched_budget"]
    assert sem["selected_ids"] == off["selected_ids"]
    assert "Memory Pointers" in sem["prompt"] and "Memory Pointers" not in off["prompt"]
    assert budget["prompt_tokens"] <= sem["prompt_tokens"]
    assert budget["budget_tokens"] == sem["prompt_tokens"]
    assert all(len(tok.encode(x)) <= 64 for x in prepared.blocks)


def test_metrics_do_not_count_partial_multitarget_as_perfect():
    assert score("all_containment", "Mary", ["Mary", "John"]) == .5
    assert score("numeric_sequence", "[1,2]", ["[1,3]"]) < 1
    assert score("choice", "A or B", ["B"]) == 0
    assert score("qa_f1", "Paris", ["Paris", "London"]) == 1
    assert score("rouge_l", "hello world", ["hello world"]) == 1
    assert score("char_rouge_l", "北京天气", ["北京晴天"]) > 0


def test_resume_rejects_settings_changes_and_retries_errors(tmp_path):
    with RunStore(tmp_path, {"model": "m", "sources": "abc"}) as store:
        store.record({"example_id": "e", "system": "sempointer", "status": "error"})
        assert not store.done("e", "sempointer")
        store.record({"example_id": "e", "system": "sempointer", "status": "scored", "score": 1,
                      "family": "f", "task": "t", "metric": "qa_f1", "protocol": "adapted"})
        assert store.done("e", "sempointer")
    with pytest.raises(ValueError, match="settings"):
        RunStore(tmp_path, {"model": "different", "sources": "abc"})


def test_summary_separates_tasks_and_reports_errors(tmp_path):
    with RunStore(tmp_path, {"model": "m"}) as store:
        for system, value in [("sempointer", .5), ("bm25_4", 1.)]:
            store.record({"example_id": "x", "family": "f", "task": "t", "metric": "qa_f1",
                          "protocol": "adapted", "system": system, "status": "scored", "score": value,
                          "active_tokens": 20, "latency_ms": 1})
        store.record({"example_id": "y", "family": "f", "task": "other", "metric": "choice",
                      "system": "sempointer", "status": "error", "error": "OOM"})
        result = summarize(store)
        assert len(result["tasks"]) == 2
        assert any(x.get("n_paired") == 1 for x in result["paired"])
        assert result["status_counts"]["error"] == 1


def test_beir_index_retrieves_without_reading_gold_into_query(tmp_path):
    from evaluation.beir import build_index, retrieve, ir_metrics
    archive=tmp_path/"toy.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("toy/corpus.jsonl",'\n'.join(json.dumps(x) for x in [
            {"_id":"a","title":"Mars","text":"rover landed on Mars"},
            {"_id":"b","title":"Venus","text":"thick atmosphere"}]))
        z.writestr("toy/queries.jsonl",json.dumps({"_id":"q","text":"Mars rover"})+'\n')
        z.writestr("toy/qrels/test.tsv","query-id\tcorpus-id\tscore\nq\ta\t1\n")
    db=build_index(archive,"toy/",tmp_path/"index.sqlite")
    assert retrieve(db,"Mars rover",20)[0][0]=="a"
    assert ir_metrics(["a"],{"a":1})["ndcg_at_10"]==1
    assert ir_metrics(["b"],{"a":1})["recall_at_10"]==0
    assert not retrieve(db,'" OR *',20)
    db.close()


@pytest.mark.parametrize("separator",["\n\n","\n\n\n"])
def test_scrolls_merges_references_and_keeps_options_in_question(tmp_path,separator):
    context="What happened?"+separator+" (A) one\n (B) two\n (C) three\n (D) four\n\n\nThe same story."
    t=task(tmp_path,"scrolls",[{"id":"q1","pid":"p1","input":context,"output":"one"},
                               {"id":"q1","pid":"p2","input":context,"output":"ONE"},
                               {"id":"q2","pid":"p3","input":context.replace("What happened?","Why?"),"output":"two"}])
    t.task="quality"
    a,b=list(iter_examples(t))
    assert a.gold==["one","ONE"]
    assert "(A) one" in a.question and "(A) one" not in a.context
    assert a.context==b.context and a.group_id==b.group_id


def test_parquet_shards_cannot_collide_in_example_ids(tmp_path):
    a=task(tmp_path,"babilong",[{"input":"memory","question":"q","target":"a"}],suffix=".json")
    b=Task(a.family,a.task,tmp_path/"other.json",adapter=a.adapter)
    b.path.write_text(a.path.read_text())
    assert next(iter_examples(a)).id!=next(iter_examples(b)).id


def test_runner_records_every_control_without_training(tmp_path):
    from evaluation.runner import run_tasks
    t=task(tmp_path,"babilong",[{"input":"Mary went home","question":"where?","target":"home"}],suffix=".json")
    frozen=[]
    parameter=SimpleNamespace(requires_grad_=lambda value:frozen.append(value))
    model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768),parameters=lambda:[parameter])
    tok=Tokenizer()
    engine=SimpleNamespace(model=model,tokenizer=tok,generate=lambda prompt,**kwargs:SimpleNamespace(
        text="home",prompt_tokens=len(tok.encode(prompt)),latency_ms=2,completion_tokens=1,peak_vram_mb=0,ttft_ms=1,stop_reason="eos"))
    from evaluation.controls import SYSTEMS
    args=SimpleNamespace(systems=list(SYSTEMS),partition="all",limit=0,max_source_chars=10000,
        max_source_tokens=10000,max_new_tokens=16,max_prompt_tokens=1024,seed=143)
    with RunStore(tmp_path/"run",{"fixture":"CPU-only fake generator, not empirical evidence"}) as store:
        result=run_tasks([t],store,engine,args)
        assert result["status_counts"]=={"scored":7}
        assert len((store.directory/"predictions.jsonl").read_text().splitlines())==7
    assert frozen==[False]


def test_model_preflight_requires_every_weight_shard(tmp_path):
    from evaluation.provenance import weight_files
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"a":"one.safetensors","b":"two.safetensors"}}))
    (tmp_path/"one.safetensors").write_bytes(b"fixture")
    assert not weight_files(str(tmp_path),None)["available"]
    (tmp_path/"two.safetensors").write_bytes(b"fixture")
    assert weight_files(str(tmp_path),None)["available"]


def test_beir_end_to_end_keeps_results_and_releases_generated_index(tmp_path,monkeypatch):
    from evaluation.beir import run_beir
    from transformers import AutoTokenizer
    tok=Tokenizer();tok.init_kwargs={};tok.backend_tokenizer=SimpleNamespace(to_str=lambda:"CPU toy fixture")
    monkeypatch.setattr(AutoTokenizer,"from_pretrained",lambda *a,**k:tok)
    folder=tmp_path/"data/originals/beir";folder.mkdir(parents=True)
    with zipfile.ZipFile(folder/"toy.zip","w") as z:
        z.writestr("toy/corpus.jsonl",json.dumps({"_id":"a","title":"Mars","text":"Mars rover"})+'\n')
        z.writestr("toy/queries.jsonl",json.dumps({"_id":"q","text":"Mars rover"})+'\n')
        z.writestr("toy/qrels/test.tsv","query-id\tcorpus-id\tscore\nq\ta\t1\n")
    args=SimpleNamespace(model="CPU-toy-fixture",data_root=tmp_path/"data",tasks="all",output=tmp_path/"out",seed=143,
        beir_candidates=20,limit=1,include_train=False,max_source_chars=10000,max_source_tokens=10000,keep_beir_indices=False)
    run_beir(args)
    before=(args.output/"attempts.jsonl").read_text()
    assert len(before.splitlines())==5
    assert (folder/"toy.zip").exists()
    assert not list((args.output/"indices").glob("*.sqlite"))
    run_beir(args)
    assert (args.output/"attempts.jsonl").read_text()==before
