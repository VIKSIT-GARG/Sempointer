"""Prevent evidence audits from turning answer mentions into causal proof."""
import pytest


def test_expanded_payload_excludes_question_and_pointer_answer_mentions():
    from analysis.audit_bounded_losses import expanded_blocks

    row = {
        "question": "Is the answer Paris?",
        "selected_ids": [3],
        "prompt": "<chat>Memory Pointers (unexpanded blocks):\n[P2] Paris\n\n"
        "Memory Context:\n[MEMORY 3]:\nThe city is Lyon.\n\n"
        "Question: Is the answer Paris?\nAnswer:<assistant>",
    }
    assert expanded_blocks(row) == {3: "The city is Lyon."}


def test_payload_parser_rejects_inconsistent_recorded_selected_ids():
    from analysis.audit_bounded_losses import expanded_blocks

    row = {"question": "Where?", "selected_ids": [1],
           "prompt": "Memory Context:\n[MEMORY 2]:\nHere.\n\nQuestion: Where?\nAnswer:"}
    with pytest.raises(ValueError, match="selected"):
        expanded_blocks(row)


def test_support_coverage_counts_all_required_tokens_across_missing_blocks():
    from analysis.audit_bounded_losses import span_coverage

    offsets = [(0, 3), (4, 9), (10, 14)]  # red green blue
    result = span_coverage([(0, 14)], offsets, {0, 2})
    assert result["status"] == "partial"
    assert result["covered_tokens"] == 2
    assert result["total_tokens"] == 3
    assert span_coverage([(4, 9)], offsets, {0, 2})["status"] == "absent"
    assert span_coverage([(0, 3)], offsets, {0, 2})["status"] == "present"


@pytest.mark.parametrize("metric,gold", [("choice", ["A"]), ("rouge_l", ["Paris"]),
                                         ("qa_f1", ["not mentioned"])])
def test_non_evidence_reference_types_are_not_treated_as_missing_evidence(metric, gold):
    from analysis.audit_bounded_losses import answer_probe

    result = answer_probe({"metric": metric, "gold": gold}, "London", "Paris")
    assert result["applicable"] is False


def test_answer_probe_uses_word_boundaries_and_keeps_mention_claim_weak():
    from analysis.audit_bounded_losses import answer_probe

    result = answer_probe({"metric": "qa_f1", "gold": ["York"]}, "Yorkshire", "York")
    assert result["applicable"] is True
    assert result["payload_any"] is False
    assert result["source_any"] is True
    assert result["is_evidence_annotation"] is False


def test_present_annotated_text_does_not_prove_pointer_distraction():
    from analysis.audit_bounded_losses import classify_evidence

    assert classify_evidence("fine", "present", "present", None, None) == "annotated_units_present_cause_unresolved"
    assert classify_evidence("fine", "partial", "present", None, None) == "annotated_units_omitted_bm25_retains"
    assert classify_evidence("coarse", "absent", "present", None, None) == "support_session_coverage_requires_review"


def test_answer_absent_from_both_payloads_is_not_called_confirmed_router_failure():
    from analysis.audit_bounded_losses import classify_evidence

    missing = {"applicable": True, "payload_any": False, "source_any": True}
    assert classify_evidence(None, None, None, missing, missing) == "answer_mention_absent_both_cause_unresolved"


def test_locomo_alignment_preserves_trailing_message_newlines():
    from analysis.audit_bounded_losses import locomo_context_and_units

    conversation = {"session_1_date_time": "today", "session_1": [
        {"speaker": "A", "dia_id": "D1:1", "text": "hello\n"},
        {"speaker": "B", "dia_id": "D1:2", "text": "bye"}]}
    context, units = locomo_context_and_units(conversation)
    assert context == "session_1; date: today\nA: hello\n\nB: bye"
    start, end = units["D1:2"]["spans"][0]
    assert context[start:end] == "B: bye"
