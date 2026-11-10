"""The pointer control must change only descriptors, never evidence or framing."""
import pytest


def pair():
    common = dict(status="scored", example_id="q1", question="Where?", gold=["Paris"],
                  selected_ids=[0], context_sha256="source", max_new_tokens=128,
                  metric="qa_f1", family="qa", task="lookup")
    off = {**common, "prompt": "<user>System: use memory.\n\nMemory Context:\n"
           "[MEMORY 0]:\nParis is the answer.\n\nQuestion: Where?\nAnswer:<assistant>"}
    on = {**common, "prompt": off["prompt"].replace("Memory Context:\n", "Memory Pointers (unexpanded blocks):\n"
           "[P2] France\n\nMemory Context:\n"), "pointer_tokens": 8}
    return on, off


def test_control_verifies_full_prompt_after_removing_only_descriptors():
    from analysis.fixed_payload_control import verify_pair

    on, off = pair()
    result = verify_pair(on, off)
    assert result["descriptors_present"] is True
    assert result["fixed_payload"] is True


def test_same_block_ids_with_changed_evidence_are_rejected():
    from analysis.fixed_payload_control import verify_pair

    on, off = pair()
    off["prompt"] = off["prompt"].replace("Paris is the answer.", "Lyon is the answer.")
    with pytest.raises(ValueError, match="payload"):
        verify_pair(on, off)


def test_same_evidence_with_changed_instruction_is_rejected():
    from analysis.fixed_payload_control import verify_pair

    on, off = pair()
    off["prompt"] = off["prompt"].replace("use memory", "use world knowledge")
    with pytest.raises(ValueError, match="descriptor"):
        verify_pair(on, off)


def test_changed_output_cap_is_rejected():
    from analysis.fixed_payload_control import verify_pair

    on, off = pair()
    off["max_new_tokens"] = 512
    with pytest.raises(ValueError, match="max_new_tokens"):
        verify_pair(on, off)


def test_empty_descriptor_pairs_are_identified_as_null_interventions():
    from analysis.fixed_payload_control import verify_pair

    on, off = pair()
    on["prompt"], on["pointer_tokens"] = off["prompt"], 0
    result = verify_pair(on, off)
    assert result["descriptors_present"] is False
    assert result["identical_prompts"] is True


def test_cluster_bootstrap_does_not_treat_same_document_questions_as_independent():
    from analysis.fixed_payload_control import clustered_net_win_interval

    rows = [{"family": "f", "context_sha256": "one", "sign": 1}] * 50
    rows += [{"family": "f", "context_sha256": "two", "sign": -1}] * 50
    low, high = clustered_net_win_interval(rows, repeats=2000, seed=4)
    assert low == -1.0
    assert high == 1.0


def test_exact_sign_test_ignores_ties_and_is_two_sided():
    from analysis.fixed_payload_control import sign_test

    assert sign_test(6, 0) == 0.03125
    assert sign_test(6, 6) == 1.0
    assert sign_test(0, 0) == 1.0


def test_holm_sensitivity_keeps_unavailable_tasks_in_the_larger_family():
    from analysis.fixed_payload_control import holm_adjust

    assert holm_adjust([0.01, 0.04, 1.0]) == [0.03, 0.08, 1.0]
