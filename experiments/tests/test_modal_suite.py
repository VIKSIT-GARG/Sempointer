"""Offline contracts for cloud sampling, interruption and allowance accounting."""
import json
from pathlib import Path
import subprocess
import time
import sys
from types import SimpleNamespace

import pytest


def test_balanced_quota_redistributes_small_strata_without_exceeding_family_cap():
    from evaluation.cloud_policy import balanced_counts
    assert balanced_counts({"a": 2, "b": 100, "c": 100}, 10) == {"a": 2, "b": 4, "c": 4}
    assert sum(balanced_counts({str(i): 100 for i in range(21)}, 300).values()) == 300
    assert balanced_counts({"only": 17}, 300) == {"only": 17}


def test_sampling_is_order_independent_and_deduplicates_questions():
    from evaluation.cloud_policy import CandidatePool
    rows = [{"id": str(i), "family": "ruler", "stratum": str(i % 3),
             "semantic_id": str(i), "source": "fixture", "source_tokens": 100} for i in range(100)]
    results = []
    for sequence in [rows + rows, list(reversed(rows))]:
        pool = CandidatePool(17, 12)
        for row in sequence:pool.add(row)
        results.append(sorted(x["id"] for x in pool.selected()))
    assert results[0] == results[1]
    assert len(results[0]) == 12


def test_pending_reservations_remain_charged_after_interruption(tmp_path):
    from evaluation.cloud_policy import Allowance
    path = tmp_path / "allowance.json"
    ledger = Allowance(path, 2)
    ledger.reserve("attempt-1", 1.25)
    resumed = Allowance(path, 2)
    assert resumed.remaining == pytest.approx(.75)
    with pytest.raises(ValueError, match="allowance"):
        resumed.reserve("attempt-2", 1)
    resumed.finish("attempt-1", .5)
    assert Allowance(path, 2).remaining == pytest.approx(1.5)
    with pytest.raises(ValueError):resumed.reserve("attempt-1", .1)


def test_allowance_rejects_invalid_money_and_actual_cost_above_reservation(tmp_path):
    from evaluation.cloud_policy import Allowance
    ledger = Allowance(tmp_path / "allowance.json", 2)
    for invalid in [float("nan"), -1, float("inf")]:
        with pytest.raises(ValueError):ledger.reserve("bad", invalid)
    ledger.reserve("one", 1)
    with pytest.raises(ValueError):ledger.finish("one", 1.1)
    assert ledger.remaining == 1


def test_cost_gate_uses_each_family_and_does_not_change_the_sample_set():
    from evaluation.cloud_policy import project_seconds
    rows = [{"family": "short", "id": "a"}, {"family": "long", "id": "b"}]
    assert project_seconds(rows, {"short": [1, 2], "long": [10, 20]}, safety=1.5) == 33
    with pytest.raises(ValueError, match="pilot"):
        project_seconds(rows, {"short": [1]})


def test_prepared_adapter_preserves_ids_groups_and_protocol(tmp_path):
    from evaluation.data import Example, Task, iter_examples
    row = dict(id="original-id", context="source", question="question", gold=["answer"],
        metric="qa_f1", family="longmem", task="longmemeval_s_cleaned", split="test",
        group_id="original-group", protocol="explicit adaptation", meta={"output_cap": 128})
    path = tmp_path / "example.json"
    path.write_text(json.dumps({"example": row}))
    actual = next(iter_examples(Task("longmem", "sample", path, adapter="prepared")))
    assert actual == Example(**row)


def test_runner_can_checkpoint_and_stop_between_questions_without_scoring_fake_rows(tmp_path):
    from evaluation.data import Task
    from evaluation.runner import RunStore, run_tasks
    from tests.test_all_datasets import Tokenizer
    calls = []
    tok = Tokenizer()
    tasks = []
    for i in range(3):
        row = dict(id=str(i), context="mars rover mission", question="where?", gold=["mars"],
            metric="qa_f1", family="ruler", task="fixture", split="test", group_id=str(i))
        path = tmp_path / f"{i}.json";path.write_text(json.dumps({"example": row}))
        tasks.append(Task("ruler", "fixture", path, adapter="prepared", max_new_tokens=16))
    engine = SimpleNamespace(tokenizer=tok,model=SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=1000),parameters=lambda:[]),
        generate=lambda prompt,**kw:SimpleNamespace(text="mars",prompt_tokens=len(tok.encode(prompt)),
            latency_ms=1,completion_tokens=1,peak_vram_mb=0,ttft_ms=0,stop_reason="eos"))
    args = SimpleNamespace(systems=["sempointer"],partition="all",limit=0,seed=1,
        max_source_chars=10000,max_source_tokens=10000,max_prompt_tokens=1000,max_new_tokens=0,
        should_stop=lambda:bool(calls),on_checkpoint=lambda store,task:calls.append(task.path),
        defer_summary=True)
    with RunStore(tmp_path/"run",{"fixture":"not empirical evidence"}) as store:
        result=run_tasks(tasks,store,engine,args)
        assert store.db.execute("SELECT COUNT(*) FROM latest").fetchone()[0] == 1
    assert result["status_counts"] == {"scored":1}
    assert len(calls)==1


def test_launcher_blocks_unconfirmed_cloud_start_before_preparation_or_upload():
    result=subprocess.run(["bash","run_modal_suite.sh","start"],capture_output=True,text=True,timeout=10)
    assert result.returncode==2
    assert "Launch blocked" in result.stderr
    assert "Sampling scan" not in result.stdout


def test_selection_excludes_overlapping_and_privileged_variants():
    from prepare_modal import selected_source
    from evaluation.data import Task
    assert not selected_source(Task("longbench","qasper_e",Path("data.zip")))
    assert not selected_source(Task("longmem","oracle",Path("longmemeval_oracle.json")))
    assert selected_source(Task("longmem","S",Path("longmemeval_s_cleaned.json")))
    assert not selected_source(Task("babilong","qa1",Path("qa1/1M.json")))
    assert selected_source(Task("babilong","qa1",Path("qa1/128k.json")))


def test_pilot_includes_the_longest_source_and_largest_output_allowance():
    from evaluation.cloud_policy import pilot_rows
    rows=[{"id":str(i),"family":"longbench","source_tokens":i+1,"output_cap":512 if i==2 else 128} for i in range(10)]
    selected=pilot_rows(rows)
    assert {"0","9","2"} <= {r["id"] for r in selected}
    assert len({r["id"] for r in selected})==len(selected)


def test_all_cloud_phases_resume_and_share_pilot_predictions(tmp_path,monkeypatch):
    """Real orchestration/storage with a CPU-only fake external model boundary."""
    from evaluation import cloud_worker,qwen35
    from evaluation.cloud_policy import atomic_json
    from tests.test_all_datasets import Tokenizer
    class FixtureEngine:
        def __init__(self,**kwargs):
            self.tokenizer=Tokenizer()
            self.model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768),parameters=lambda:[])
        def info(self):return {"fixture":"CPU software test; no model inference"}
        def format_prompt(self,prompt):return "CHAT "+prompt
        def generate(self,prompt,**kwargs):
            return SimpleNamespace(text="mars",prompt_tokens=len(self.tokenizer.encode(prompt)),latency_ms=1,
                completion_tokens=1,peak_vram_mb=0,ttft_ms=0,stop_reason="eos")
    monkeypatch.setattr(qwen35,"Qwen35Engine",FixtureEngine)
    monkeypatch.setattr(cloud_worker,"_ENGINE",None)
    rows=[]
    for ident,family,task in [("a","ruler","niah_single_1"),("b","longbench","qasper"),("c","ruler","cwe")]:
        path=f"examples/{ident}.json"
        example=dict(id=ident,family=family,task=task,context="mars rover mission "*50,
            question="where?",gold=["mars"],metric="qa_f1",split="test",group_id=ident)
        atomic_json(tmp_path/"input"/path,{"example":example})
        rows.append(dict(id=ident,family=family,task=task,path=path,source_tokens=150,output_cap=512 if task=="cwe" else 128))
    development=[]
    for ident in ["d","e"]:
        path=f"examples/{ident}.json"
        atomic_json(tmp_path/"input"/path,{"example":dict(id=ident,family="ruler",task="fixture",
            context="mars rover mission "*50,question="where?",gold=["mars"],metric="qa_f1",split="test",group_id=ident)})
        development.append(dict(id=ident,family="ruler",task="fixture",path=path,source_tokens=150,output_cap=128))
    atomic_json(tmp_path/"input/manifest.json",dict(run_id="fixture",code_files={},model_id="fixture",model_revision="fixture",
        dtype="nf4",seed=1,rows=rows,development=development,pilot_ids=["a","b"],cwe_audit_ids=["c"],
        limits={"source_chars":10000,"source_tokens":10000,"prompt_tokens":10000},sampling_report={}))
    for index in range(6):
        state=cloud_worker.run_phase(tmp_path,"not-a-real-checkpoint",time.monotonic()+120,lambda:None,lambda:False)
        assert state["phase"]==index+1
        if index==2:assert (tmp_path/"results/threshold_selection.json").exists()
    assert state["status"]=="complete"
    import sqlite3
    with sqlite3.connect(tmp_path/"results/test/results.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM latest").fetchone()[0]==18
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]==18  # no pilot re-generation
    with sqlite3.connect(tmp_path/"results/cwe_cap32/results.sqlite") as db:
        assert {json.loads(x[0])["max_new_tokens"] for x in db.execute("SELECT payload FROM latest")}=={32}
    cloud_worker.report(tmp_path)
    assert (tmp_path/"results/cwe_cap_audit.json").exists()


def test_detached_handoff_does_not_overwrite_a_fast_successor():
    sdk=Path(__file__).resolve().parents[1]/".venv-modal/bin/python"
    if not sdk.exists():pytest.skip("Isolated Modal SDK is not installed")
    program='''
from types import SimpleNamespace
import modal_suite as m
class Store:
    def __init__(self):self.d={"active":{"run_id":"r","owner":"o","sequence":0}}
    def get(self,k,default=None):return self.d.get(k,default)
    def put(self,k,v,skip_if_exists=False):
        if skip_if_exists and k in self.d:return False
        self.d[k]=v;return True
store=Store();m.leases=store
def spawn(*args):
    assert store.d["active"]["sequence"]==1
    store.d["active"]={"run_id":"r","owner":"o","sequence":2}
    return SimpleNamespace(object_id="child-1")
m.work=SimpleNamespace(spawn=spawn)
assert m._dispatch("r","o")=="child-1"
assert store.d["active"]["sequence"]==2
assert store.d["ticket:r:1"]=="child-1"
print("handoff contract passed without cloud execution")
'''
    result=subprocess.run([str(sdk),"-c",program],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr


def test_preparation_caps_each_family_and_freezes_a_reusable_bundle(tmp_path,monkeypatch):
    import hashlib
    import prepare_modal as prep
    import reviewer_suite
    from transformers import AutoTokenizer
    from evaluation.data import Task,Example
    from evaluation.cloud_policy import FAMILIES
    from tests.test_all_datasets import Tokenizer
    root=tmp_path/"project/experiments";root.mkdir(parents=True)
    (root/"fixture.py").write_text("# CPU-only source snapshot fixture\n")
    source_hash=hashlib.sha256((root/"fixture.py").read_bytes()).hexdigest()
    model=root.parent/"models/Qwen3.5-4B"
    (model/".cache/huggingface/download").mkdir(parents=True)
    (model/".cache/huggingface/download/config.json.metadata").write_text("1"*40+"\n")
    (model/"tokenizer.json").write_text("{}")
    monkeypatch.setattr(prep,"HERE",root)
    monkeypatch.setattr(prep,"code_files",lambda *a,**kw:{"fixture.py":source_hash})
    from transformers import PreTrainedTokenizerFast
    from tests.test_preparation import word_tokenizer
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=word_tokenizer().backend_tokenizer)
    monkeypatch.setattr(AutoTokenizer,"from_pretrained",lambda *a,**kw:tokenizer)
    monkeypatch.setattr(reviewer_suite,"build_exclusions",lambda *a:([],{"fixture":"CPU-only"}))
    monkeypatch.setattr(prep,"partition_for",lambda ex:"test")
    names={"ruler":("niah_single_1","assets"),"longbench":("hotpotqa","data.zip"),
        "longmem":("longmemeval_s_cleaned","longmemeval_s_cleaned.json"),
        "babilong":("data/qa1/4k","qa1/4k.json"),"scrolls":("qasper","qasper.zip")}
    def discover(data_root,seed=1,**kwargs):
        return [Task(f,*names.get(f,("fixture","fixture.json")),extra={"context_length":4096,"seed":seed},
                     split="validation" if f=="scrolls" else "test") for f in FAMILIES],[]
    # Paths must be Path instances, as in the real catalog.
    def sources(*a,**kw):
        tasks,coverage=discover(*a,**kw)
        for task in tasks:
            task.path=root/"sources"/task.family/Path(task.path)
            task.path.parent.mkdir(parents=True,exist_ok=True)
            if not task.path.exists():task.path.write_text("CPU-only source fixture")
        return tasks,coverage
    monkeypatch.setattr(prep,"discover_tasks",sources)
    def examples(task,tokenizer=None):
        for i in range(5):
            yield Example(f"{task.extra['seed']}:{task.family}:{i}",f"mars rover mission {i}","where?",["mars"],
                "qa_f1",task.family,task.task,"test",f"group-{i}",meta={"category":i%2,"domain":f"domain-{i%2}","question_type":f"type-{i%2}"})
    monkeypatch.setattr(prep,"iter_examples",examples)
    output=tmp_path/"bundle"
    monkeypatch.setattr(sys,"argv",["prepare_modal.py","--output",str(output),"--per-family","2"])
    prep.main()
    manifest=prep.verify_bundle(output)
    assert len(manifest["rows"])==20
    assert {r["family"] for r in manifest["rows"]}==set(FAMILIES)
    assert all(v["selected"]==2 for v in manifest["sampling_report"].values())
    assert not {r["id"] for r in manifest["rows"]}&{r["id"] for r in manifest["development"]}
    before=(output/"manifest.json").read_bytes()
    prep.main()
    assert (output/"manifest.json").read_bytes()==before
