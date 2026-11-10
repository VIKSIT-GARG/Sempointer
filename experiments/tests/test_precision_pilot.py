"""Hardware decisions require completed paired measurements, never invented scores."""
from types import SimpleNamespace
import json
import shutil

import pytest


def test_precision_choice_requires_complete_repeatable_equal_work_advantage():
    from evaluation.precision import choose_precision
    rows=[dict(dtype=dtype,case_id=str(i),round=round,decode_tokens=32,
               elapsed_ms=elapsed,status="ok")
          for dtype,elapsed in [("nf4",120),("bf16",90)] for i in range(6) for round in range(2)]
    decision=choose_precision(rows,[str(i) for i in range(6)])
    assert decision["selected_dtype"]=="bf16"
    assert decision["speedup"]==pytest.approx(120/90)
    assert choose_precision(rows[:-1],[str(i) for i in range(6)])["selected_dtype"]=="nf4"
    equal=[{**row,"elapsed_ms":100} for row in rows]
    assert choose_precision(equal,[str(i) for i in range(6)])["selected_dtype"]=="nf4"
    broken=[{**row,"status":"oom"} if row["dtype"]=="bf16" and row["case_id"]=="0" else row for row in rows]
    assert choose_precision(broken,[str(i) for i in range(6)])["selected_dtype"]=="nf4"


def test_batch_choice_requires_bf16_advantage_matching_outputs_and_measured_speedup():
    from evaluation.precision import choose_batch
    rows=[dict(batch_size=2,elapsed_ms=700,serial_ms=1000,elapsed_repeats_ms=[700,700],outputs_match=True,status="ok"),
          dict(batch_size=4,elapsed_ms=400,serial_ms=1000,outputs_match=False,status="ok")]
    assert choose_batch("bf16",rows)["batch_size"]==2
    assert choose_batch("nf4",rows)["batch_size"]==1
    assert choose_batch("bf16",[{**rows[0],"status":"oom"}])["batch_size"]==1


def test_batch_choice_accepts_a_small_repeatable_gain_but_rejects_an_unstable_one():
    from evaluation.precision import choose_batch
    row=dict(batch_size=2,elapsed_ms=94,serial_ms=100,elapsed_repeats_ms=[93.5,94.5],outputs_match=True,status="ok")
    assert choose_batch("bf16",[row])["batch_size"]==2
    assert choose_batch("bf16",[{**row,"elapsed_repeats_ms":[85,103]}])["batch_size"]==1


def test_derived_configuration_keeps_precision_charges_in_the_shared_allowance(tmp_path):
    from evaluation.inference_policy import shared_allowance_path,development_spend
    from evaluation.cloud_policy import Allowance
    account="a"*20
    manifest={"run_id":"b"*20,"budget_account_run_id":account}
    path=shared_allowance_path(tmp_path,manifest)
    assert path==tmp_path/"runs"/account/"allowance.json"
    ledger=Allowance(path)
    ledger.reserve("precision-setup-pilot",1)
    ledger.reserve("precision-gpu-pilot",.6)
    ledger.reserve("setup-suite",1)
    assert ledger.remaining==pytest.approx(22.4)
    assert development_spend(ledger)==pytest.approx(1)
    with pytest.raises(ValueError):shared_allowance_path(tmp_path,{"run_id":"../escape"})


def test_configuration_application_preserves_samples_and_rejects_a_different_model(tmp_path,monkeypatch):
    import apply_precision_result as apply
    import prepare_modal as prep
    from evaluation.cloud_policy import atomic_json,digest
    from evaluation.precision import choose_precision,choose_batch
    runtime=tmp_path/"runtime";runtime.mkdir();(runtime/"fixture.py").write_text("# software fixture only\n")
    def files(root=runtime):return {"fixture.py":prep.file_hash(root/"fixture.py")}
    monkeypatch.setattr(apply,"HERE",runtime);monkeypatch.setattr(apply,"code_files",files)
    monkeypatch.setattr(prep,"code_files",files)
    source=tmp_path/"source";shutil.copytree(runtime,source/"code")
    atomic_json(source/"examples/test/a.json",{"example":"CPU-only fixture, not evidence"})
    original=dict(model_id="fixture-model",model_revision="1"*40,dtype="nf4",code_files=files(),notes=[],
        rows=[dict(path="examples/test/a.json",sha256=prep.file_hash(source/"examples/test/a.json"))],development=[])
    original["run_id"]=digest(original)[:20];atomic_json(source/"manifest.json",original)
    unchanged=(source/"manifest.json").read_bytes()
    fixed=[dict(dtype=d,case_id=str(i),round=r,decode_tokens=32,elapsed_ms=elapsed,status="ok")
           for d,elapsed in [("nf4",200),("bf16",100)] for i in range(4) for r in range(2)]
    pilot=dict(source_run_id=original["run_id"],model_id="fixture-model",model_revision="1"*40,fixed_cases=list(map(str,range(4))))
    pilot["pilot_id"]=digest(pilot)[:20]
    report=dict(status="complete",source_run_id=original["run_id"],pilot_id=pilot["pilot_id"],fixed_work=fixed,batches=[],
                precision_decision=choose_precision(fixed,pilot["fixed_cases"]),batch_decision=choose_batch("bf16",[]))
    atomic_json(tmp_path/"pilot.json",pilot);atomic_json(tmp_path/"report.json",report)
    manifest=apply.apply_result(source,tmp_path/"report.json",tmp_path/"pilot.json",tmp_path/"derived/bundle")
    assert manifest["dtype"]=="bf16" and manifest["inference_batch_size"]==1
    assert manifest["rows"]==original["rows"]
    assert manifest["budget_account_run_id"]==original["run_id"]
    assert (source/"manifest.json").read_bytes()==unchanged
    assert prep.verify_bundle(tmp_path/"derived/bundle")["run_id"]==manifest["run_id"]
    pilot["model_revision"]="2"*40;pilot.pop("pilot_id");pilot["pilot_id"]=digest(pilot)[:20]
    report["pilot_id"]=pilot["pilot_id"]
    atomic_json(tmp_path/"pilot.json",pilot);atomic_json(tmp_path/"report.json",report)
    with pytest.raises(ValueError,match="model"):
        apply.apply_result(source,tmp_path/"report.json",tmp_path/"pilot.json",tmp_path/"wrong-model/bundle")


def test_native_batch_uses_left_padding_and_keeps_completion_lengths_per_row():
    import torch
    from evaluation.qwen35 import Qwen35Engine
    engine=Qwen35Engine.__new__(Qwen35Engine)
    engine.device="cpu";engine.max_new_tokens=4
    class Batch(dict):
        def to(self,device):return self
    class Tokenizer:
        pad_token_id=0; padding_side="right"
        def __call__(self,prompts,**kwargs):
            assert self.padding_side=="left"
            assert kwargs["padding"] is True
            return Batch(input_ids=torch.tensor([[0,1,7],[1,7,8]]),
                         attention_mask=torch.tensor([[0,1,1],[1,1,1]]))
        def decode(self,ids,**kwargs):return " ".join(str(x) for x in ids.tolist() if x not in {0,2})
    class Model:
        device="cpu";generation_config=SimpleNamespace(eos_token_id=[2])
        def generate(self,**kwargs):
            assert kwargs["max_new_tokens"]==4
            assert kwargs["use_cache"] is True
            return torch.tensor([[0,1,7,9,2,0],[1,7,8,8,9,2]])
    engine.tokenizer=Tokenizer();engine.model=Model()
    result=engine.generate_batch(["short","longer"],max_new_tokens=4)
    assert [r.prompt_tokens for r in result.results]==[2,3]
    assert [r.completion_tokens for r in result.results]==[2,3]
    assert [r.text for r in result.results]==["9","8 9"]
    assert all(r.stop_reason=="eos" for r in result.results)
    assert result.batch_size==2
    assert result.latency_ms>0
    assert engine.tokenizer.padding_side=="right"


@pytest.mark.parametrize("requested",[2,4])
def test_runner_batches_without_claiming_isolated_latency_and_recovers_from_batch_oom(tmp_path,requested):
    import torch
    from evaluation.runner import RunStore,run_tasks,summarize
    from evaluation.data import Task
    from evaluation.cloud_policy import SYSTEMS,atomic_json
    from tests.test_all_datasets import Tokenizer
    from engine.llm import GenResult
    from evaluation.qwen35 import BatchResult
    path=tmp_path/"example.json"
    atomic_json(path,{"example":dict(id="fixture",context="mars rover mission "*20,question="where?",gold=["mars"],
        metric="qa_f1",family="ruler",task="fixture",split="test",group_id="fixture")})
    tok=Tokenizer();calls=[]
    class Engine:
        device="cpu";tokenizer=tok
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1000),parameters=lambda:[])
        def generate_batch(self,prompts,max_new_tokens):
            calls.append(len(prompts))
            if len(prompts)>2:raise torch.cuda.OutOfMemoryError("CPU software fixture simulating a large-batch OOM")
            results=[GenResult("mars",p,len(tok.encode(p)),1,10,0,None,"eos") for p in prompts]
            return BatchResult(results,10,len(prompts),0)
        def generate(self,*a,**kw):raise AssertionError("All six fixture prompts fit batches of two")
    args=SimpleNamespace(systems=list(SYSTEMS),partition="all",limit=0,seed=1,max_source_chars=10000,
        max_source_tokens=1000,max_prompt_tokens=1000,max_new_tokens=8,inference_batch_size=requested,defer_summary=True)
    with RunStore(tmp_path/"results",{"fixture":"software test only"}) as store:
        run_tasks([Task("ruler","fixture",path,adapter="prepared")],store,Engine(),args)
        rows=[json.loads(r[0]) for r in store.db.execute("SELECT payload FROM latest")]
        assert len(rows)==6 and all(r["status"]=="scored" for r in rows)
        assert all(r["generation_ms"] is None and r["latency_ms"] is None and r["ttft_ms"] is None for r in rows)
        assert all(r["latency_mode"]=="shared_batch_wall" for r in rows)
        assert all(r["batch_size"]==2 for r in rows)
        result=summarize(store)
        assert result["batch_throughput"]["generated_requests"]==6
        assert result["batch_throughput"]["batches"]==3
    if requested==4:assert calls==[4,2,2,2]
    else:assert calls==[2,2,2]
