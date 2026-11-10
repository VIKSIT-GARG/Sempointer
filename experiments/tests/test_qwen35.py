"""Model-switch contracts; tiny random weights are software checks only."""
import json
import subprocess
import sys

import pytest

from evaluation.controls import prepare, prompts_for_example
from tests.test_all_datasets import Tokenizer


def test_chat_framing_is_included_in_every_control_and_matched_budget():
    tok = Tokenizer()
    prepared = prepare("mars rover mission " * 600, tok)
    wrap = lambda text: "USER START " + text + " USER END ASSISTANT THINK END"
    variants = prompts_for_example(prepared, "where is the rover?", tok, prompt_formatter=wrap)
    for variant in variants.values():
        assert variant["prompt"].startswith("USER START ")
        assert variant["prompt_tokens"] == len(tok.encode(variant["prompt"]))
    assert variants["pointer_off"]["selected_ids"] == variants["sempointer"]["selected_ids"]
    assert variants["bm25_matched_budget"]["prompt_tokens"] <= variants["sempointer"]["prompt_tokens"]


def test_qwen35_disables_thinking_in_the_actual_prompt():
    from evaluation.qwen35 import Qwen35Engine
    engine = Qwen35Engine.__new__(Qwen35Engine)
    class ChatTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            assert messages == [{"role": "user", "content": "evidence and question"}]
            return "formatted non-thinking prompt"
    engine.tokenizer = ChatTokenizer()
    assert engine.format_prompt("evidence and question") == "formatted non-thinking prompt"


def test_qwen35_launcher_prescribes_a_separate_model_and_caps_without_writing_results(tmp_path):
    output = tmp_path / "new-model"
    result = subprocess.run(["bash", "run_qwen35_suite.sh", "dry-run", "--output", str(output)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    for text in ["Qwen3.5-4B", "--engine qwen35", "--dtype nf4", "--qa-output-tokens 128",
                 "--cwe-output-tokens 512", "cwe_cap32", "pointer_off"]:
        assert text in result.stdout
    assert not output.exists()


def test_precision_and_output_caps_reach_all_reviewer_stages():
    from reviewer_suite import parser, stage_plan
    args = parser().parse_args(["--engine", "qwen35", "--dtype", "bf16",
        "--qa-output-tokens", "192", "--cwe-output-tokens", "640"])
    stages = stage_plan(args)
    for stage in stages:
        cmd = stage["command"]
        for flag, value in [("--engine", "qwen35"), ("--dtype", "bf16"),
                            ("--qa-output-tokens", "192"), ("--cwe-output-tokens", "640")]:
            assert cmd[cmd.index(flag) + 1] == value
    audit = next(s for s in stages if s["name"] == "cwe_cap32")["command"]
    assert audit[audit.index("--max-new-tokens") + 1] == "32"


def test_effective_catalog_caps_match_requested_qa_and_cwe_limits(tmp_path):
    result = subprocess.run([sys.executable, "run_all_datasets.py", "plan",
        "--families", "ruler", "--tasks", "niah_single_1,cwe", "--context-lengths", "4096",
        "--qa-output-tokens", "192", "--cwe-output-tokens", "640", "--output", str(tmp_path)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    tasks = json.loads((tmp_path / "catalog.json").read_text())["tasks"]
    assert {t["task"]: t["max_new_tokens"] for t in tasks} == {"niah_single_1": 192, "cwe": 640}


def test_report_labels_the_actual_increased_output_cap(tmp_path):
    from evaluation.runner import RunStore
    from reviewer_suite import write_report
    for stage, cap in [("ruler_test", 640), ("cwe_cap32", 32)]:
        with RunStore(tmp_path / stage, {"fixture": "software test only"}) as store:
            store.record({"example_id": "cwe-test", "system": "sempointer", "family": "ruler",
                "task": "cwe", "split": "test", "metric": "all_containment", "protocol": "fixture",
                "status": "scored", "score": .5, "active_tokens": 10, "selected_ids": [0],
                "context_sha256": "same-context", "max_new_tokens": cap, "stop_reason": "eos"})
    write_report(tmp_path, {})
    assert "640 minus 32" in (tmp_path / "REPORT.md").read_text()
    audit = json.loads((tmp_path / "checks.json").read_text())["cwe_cap_audit"]["sempointer"]
    assert audit["higher_cap_tokens"] == [640]
    assert "cap512_hits" not in audit


@pytest.mark.parametrize("conditional", [False, True])
def test_tiny_hybrid_model_runs_native_cached_generation(conditional, monkeypatch):
    import torch
    from transformers import (Qwen3_5ForCausalLM, Qwen3_5TextConfig,
                              Qwen3_5Config, Qwen3_5ForConditionalGeneration)
    from evaluation.qwen35 import Qwen35Engine
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    # This fixture intentionally stays on CPU even when optional CUDA kernels
    # are installed for real local inference in the same environment.
    for name in ("FusedRMSNormGated", "causal_conv1d_fn", "causal_conv1d_update",
                 "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
        monkeypatch.setattr(native, name, None)
    torch.manual_seed(7)
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128, eos_token_id=2, pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": .5, "mrope_section": [2, 1, 1]})
    engine = Qwen35Engine.__new__(Qwen35Engine)
    if conditional:
        composite = Qwen3_5Config(text_config=config.to_dict(), vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
            "out_hidden_size": 32, "num_position_embeddings": 16, "patch_size": 2,
            "spatial_merge_size": 2, "temporal_patch_size": 1, "deepstack_visual_indexes": []},
            image_token_id=29, video_token_id=30, vision_start_token_id=31, vision_end_token_id=28)
        engine.model = Qwen3_5ForConditionalGeneration(composite).eval()
    else:
        engine.model = Qwen3_5ForCausalLM(config).eval()
    engine.device = "cpu"
    engine.max_new_tokens = 4
    class Batch(dict):
        def to(self, device):
            return self
    class TinyTokenizer:
        eos_token_id = 2
        pad_token_id = 0
        padding_side = "right"
        def __call__(self, prompt, **kwargs):
            size=len(prompt) if isinstance(prompt,list) else 1
            return Batch(input_ids=torch.tensor([[1, 7, 8]]*size), attention_mask=torch.ones(size, 3, dtype=torch.long))
        def decode(self, ids, **kwargs):
            return " ".join(map(str, ids.tolist()))
    engine.tokenizer = TinyTokenizer()
    result = engine.generate("already rendered chat prompt", max_new_tokens=4)
    assert result.prompt_tokens == 3
    assert 1 <= result.completion_tokens <= 4
    assert result.stop_reason in {"eos", "max_new_tokens"}
    assert result.latency_ms > 0
    assert result.ttft_ms is not None
    batched=engine.generate_batch(["already rendered chat prompt"]*2,max_new_tokens=4)
    assert [row.text for row in batched.results]==[result.text]*2
    assert [row.prompt_tokens for row in batched.results]==[3,3]
