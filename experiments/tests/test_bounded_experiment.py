"""Fresh split and development-only configuration selection contracts."""
from collections import Counter
import pytest


def test_split_selection_excludes_old_ids_and_separates_document_groups():
    from evaluation.bounded_experiment import fresh_splits
    rows=[dict(id=f"{family}-{i}",semantic_id=f"{family}-{i}",family=family,
               group_id=f"{family}-group-{i//4}",stratum=str(i%3))
          for family in ["ruler","locomo"] for i in range(60)]
    excluded={"ruler-0","locomo-0"}
    dev,test=fresh_splits(rows,excluded,dev_cap=3,test_cap=10,seed=7)
    assert Counter(r["family"] for r in dev)=={"ruler":3,"locomo":3}
    assert Counter(r["family"] for r in test)=={"ruler":10,"locomo":10}
    assert not {r["id"] for r in dev+test}&excluded
    assert not {r["group_id"] for r in dev}&{r["group_id"] for r in test}
    again=fresh_splits(list(reversed(rows)),excluded,dev_cap=3,test_cap=10,seed=7)
    assert [[r["id"] for r in part] for part in (dev,test)]==[[r["id"] for r in part] for part in again]


def test_development_selection_penalizes_skips_and_never_accepts_test_results():
    from evaluation.bounded_experiment import choose_cap
    rows=[]
    for i in range(3):
        rows.extend([dict(example_id=str(i),system="bm25_budget",status="scored",score=.5,active_tokens=100),
            dict(example_id=str(i),system="bounded_128",status="scored",score=.5,active_tokens=80),
            dict(example_id=str(i),system="bounded_256",status="scored",score=.5,active_tokens=90)])
    assert choose_cap(rows,"development")["pointer_cap"]==128
    rows[2].update(status="skipped")
    assert choose_cap(rows,"development")["pointer_cap"]==128
    with pytest.raises(ValueError,match="development"):
        choose_cap(rows,"test")
    with pytest.raises(ValueError,match="incomplete"):
        choose_cap(rows[:-1],"development")


def test_real_runner_uses_budgeted_builder_and_resumes_without_repeating_calls(tmp_path):
    import json
    from types import SimpleNamespace
    from evaluation.bounded_experiment import run_stage,choose_cap,_read_rows
    from evaluation.cloud_policy import atomic_json
    from evaluation.runner import RunStore,summarize
    from tests.test_all_datasets import Tokenizer
    tok=Tokenizer();calls=[]
    example=dict(id="fresh",family="ruler",task="fixture",context="mars rover mission "*50,
        question="where?",gold=["mars"],metric="qa_f1",split="test",group_id="fresh-group")
    atomic_json(tmp_path/"input/example.json",{"example":example})
    row=dict(id="fresh",family="ruler",task="fixture",path="example.json",output_cap=16)
    manifest=dict(run_id="fixture",development=[row],rows=[row],model_id="fake",model_revision="fake",
        code_files={},seed=7,shortlist=16,router_options={},
        limits=dict(source_chars=10000,source_tokens=10000))
    def generate(prompt,**kwargs):
        calls.append(prompt)
        return SimpleNamespace(text="mars",prompt_tokens=len(tok.encode(prompt)),latency_ms=1,
            completion_tokens=1,peak_vram_mb=0,ttft_ms=0,stop_reason="eos")
    engine=SimpleNamespace(tokenizer=tok,model=SimpleNamespace(parameters=lambda:[],
        config=SimpleNamespace(max_position_embeddings=4096)),generate=generate)
    assert run_stage(tmp_path,manifest,engine,"development",128,None,lambda:False)
    choice=choose_cap(_read_rows(tmp_path/"results/development_128"),"development")
    assert run_stage(tmp_path,manifest,engine,"test",128,choice["pointer_cap"],lambda:False)
    count=len(calls)
    assert count>0
    assert run_stage(tmp_path,manifest,engine,"test",128,choice["pointer_cap"],lambda:False)
    assert len(calls)==count
    stage=tmp_path/"results/test_128"
    rows=_read_rows(stage)
    assert {r["system"] for r in rows}=={"bounded_pointer","adaptive_no_index","bm25_budget","original_sempointer"}
    assert all(r["prompt_tokens"]<=128 for r in rows if r["status"]=="scored")
    with RunStore(stage,json.loads((stage/"config.json").read_text())) as store:
        summary=summarize(store,reference="bounded_pointer")
    assert any(r["comparison"]=="bounded_pointer_vs_bm25_budget" for r in summary["paired"])
