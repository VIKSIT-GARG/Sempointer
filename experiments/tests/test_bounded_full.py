"""Fresh full-set selection and test-only orchestration contracts."""
from collections import Counter
from types import SimpleNamespace
import json
import pytest


def test_full_selection_is_fresh_balanced_group_disjoint_and_order_independent():
    from evaluation.bounded_full import full_rows
    rows=[dict(id=f"{family}-{i}",semantic_id=f"{family}-{i}",family=family,
        group_id=f"{family}-g{i//2}",stratum=str(i%5))
        for family in ["ruler","locomo"] for i in range(400)]
    excluded={"ruler-0","locomo-0"};groups={"ruler-g1","locomo-g1"}
    actual=full_rows(rows,excluded,groups,cap=100,seed=19)
    assert Counter(r["family"] for r in actual)=={"ruler":100,"locomo":100}
    assert not {r["id"] for r in actual}&excluded
    assert not {r["group_id"] for r in actual}&groups
    assert Counter(r["stratum"] for r in actual if r["family"]=="locomo")==dict.fromkeys(map(str,range(5)),20)
    assert [r["id"] for r in actual]==[r["id"] for r in full_rows(list(reversed(rows)),excluded,groups,100,19)]
    with pytest.raises(ValueError,match="Insufficient"):
        full_rows(rows[:5],set(),set(),cap=100,seed=19)


def test_full_execution_reuses_choices_and_never_runs_development(tmp_path):
    from evaluation.bounded_experiment import execute_stages,report
    from evaluation.cloud_policy import atomic_json
    from tests.test_all_datasets import Tokenizer
    tok=Tokenizer();calls=[]
    example=dict(id="fresh",family="ruler",task="fixture",context="mars rover mission "*20,
        question="where?",gold=["mars"],metric="qa_f1",split="test",group_id="fresh")
    atomic_json(tmp_path/"input/example.json",{"example":example})
    rows=[dict(id="fresh",family="ruler",task="fixture",path="example.json",output_cap=16)]
    choices={str(b):dict(source="development_only",pointer_cap=128) for b in [1024,2048,4096]}
    manifest=dict(evaluation_only=True,frozen_pointer_selection=choices,run_id="fixture",development=[],rows=rows,
        model_id="fixture",model_revision="fixture",code_files={},seed=7,shortlist=16,router_options={},
        limits=dict(source_chars=10000,source_tokens=10000))
    atomic_json(tmp_path/"input/manifest.json",manifest)
    def generate(prompt,**kwargs):
        calls.append(prompt)
        return SimpleNamespace(text="mars",prompt_tokens=len(tok.encode(prompt)),latency_ms=1,
            completion_tokens=1,peak_vram_mb=0,ttft_ms=0,stop_reason="eos")
    engine=SimpleNamespace(tokenizer=tok,model=SimpleNamespace(parameters=lambda:[],
        config=SimpleNamespace(max_position_embeddings=8192)),generate=generate)
    assert execute_stages(tmp_path,manifest,engine,lambda:False)
    assert len(calls)==12
    assert not list((tmp_path/"results").glob("development_*"))
    assert json.loads((tmp_path/"results/pointer_selection.json").read_text())==choices
    assert execute_stages(tmp_path,manifest,engine,lambda:False)
    assert len(calls)==12
    report(tmp_path)
    text=(tmp_path/"results/REPORT.md").read_text()
    assert "1 test questions" in text and "100 fresh test" not in text
