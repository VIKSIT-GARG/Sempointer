"""Official per-benchmark scoring functions.

Re-implementations, verbatim where noted, of the official benchmark metrics:

  - LongBench: normalize_answer / f1_score / qa_f1_score are re-implemented
    EXACTLY from THUDM/LongBench `LongBench/metrics.py`
    (https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/metrics.py),
    including the lowercase -> remove punctuation -> remove articles ->
    whitespace-fix normalization and the Counter-based token F1. The
    max-over-ground-truths composition mirrors `LongBench/eval.py::scorer`
    (https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py):
        for ground_truth in ground_truths: score = max(score, metric_fn(pred, gt))
    qa_em_score applies the same normalization with set-equality EM (SQuAD
    convention; LongBench itself reports F1 only -- EM is a supplement for
    this project).

  - RULER: ruler_containment implements the per-sample logic of
    NVIDIA/RULER `scripts/eval/synthetic/constants.py`:
        string_match_all: mean over refs of 1.0 if ref.lower() in pred.lower() else 0.0
        (task score = mean over samples x 100)
    The official evaluator (scripts/eval/evaluate.py::postprocess_pred) does
    NOT extract answers; it only strips non-printable characters. The older
    "The correct answer is" extraction mentioned in some RULER discussions is
    NOT part of the current official eval path.
    https://github.com/NVIDIA/RULER, paper arXiv:2404.06654.

  - LoCoMo: locomo_f1 == qa_f1_score (LongBench token F1). Documented
    adaptation: the official protocol uses an LLM judge (paper arXiv:2402.17753)
    / Porter-stemmed F1 with a binary adversarial rule (official repo
    task_eval/evaluation.py); see benchmarks/locomo.py docstring.

Run `python -m metrics.official_metrics` for the self-test.
"""

from __future__ import annotations

import re
import string
from collections import Counter


# ---------------------------------------------------------------------------
# LongBench normalization + QA F1/EM  (THUDM/LongBench metrics.py, verbatim)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Verbatim logic from THUDM/LongBench metrics.py::normalize_answer.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _token_f1(prediction_tokens: list, ground_truth_tokens: list) -> float:
    """Verbatim logic of THUDM/LongBench metrics.py::f1_score (Counter-based
    token overlap -> harmonic mean of precision and recall)."""
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def _qa_f1_single(prediction: str, ground_truth: str) -> float:
    """Verbatim logic of THUDM/LongBench metrics.py::qa_f1_score for a single
    ground-truth string."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return _token_f1(prediction_tokens, ground_truth_tokens)


def qa_f1_score(prediction: str, ground_truths) -> float:
    """LongBench official QA F1, MAXed over all acceptable gold answers.

    The max-over-golds composition is exactly LongBench/eval.py::scorer's
    application of metrics.py::qa_f1_score. Also accepts a single string for
    convenience. Returns a float in [0, 1].
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    score = 0.0
    for ground_truth in ground_truths:
        score = max(score, _qa_f1_single(prediction, ground_truth))
    return score


def qa_em_score(prediction: str, ground_truths) -> float:
    """SQuAD-style exact match on the LongBench normalization, MAXed over all
    acceptable gold answers.

    Note: LongBench's official eval reports F1 only (dataset2metric in
    eval.py has no EM entry); this EM is a supplementary metric for this
    project using the identical normalization.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    normalized_prediction = normalize_answer(prediction)
    score = 0.0
    for ground_truth in ground_truths:
        score = max(score, float(normalized_prediction == normalize_answer(ground_truth)))
    return score


# ---------------------------------------------------------------------------
# RULER containment  (NVIDIA/RULER eval/synthetic/constants.py)
# ---------------------------------------------------------------------------

def ruler_containment(prediction: str, golds) -> float:
    """Per-sample RULER string_match_all: the fraction of gold references
    contained (case-insensitively) in the prediction.

    Verbatim official logic per ref: 1.0 if ref.lower() in pred.lower() else
    0.0 (NVIDIA/RULER scripts/eval/synthetic/constants.py::string_match_all).
    The official task score aggregates this as mean over samples x 100, so
    the per-sample value returned here is in [0, 1] and a task score of 100
    corresponds to a mean of 1.0 across examples. For single-reference
    samples (all our niah/vt/cwe/fwe tasks) this is a binary containment
    check.

    No answer extraction is applied, matching the official evaluator
    (scripts/eval/evaluate.py::postprocess_pred strips control characters
    only).
    """
    if isinstance(golds, str):
        golds = [golds]
    if not golds:
        return 0.0
    pred_lower = prediction.lower()
    hits = sum(1.0 for ref in golds if ref.lower() in pred_lower)
    return hits / len(golds)


# ---------------------------------------------------------------------------
# LoCoMo
# ---------------------------------------------------------------------------

def locomo_f1(prediction: str, ground_truths) -> float:
    """LoCoMo scoring for this project: identical to the LongBench token F1.

    Documented adaptation -- the official protocol uses an LLM judge for
    open-domain/temporal categories (paper) and Porter-stemmed F1 with a
    binary adversarial rule for category 5 (official repo
    task_eval/evaluation.py). See benchmarks/locomo.py docstring; results
    must be labeled with this adaptation.
    """
    return qa_f1_score(prediction, ground_truths)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import math

    # --- normalize_answer -------------------------------------------------
    assert normalize_answer("The Answer: 7051911!") == "answer 7051911", \
        normalize_answer("The Answer: 7051911!")
    # articles removed, punctuation stripped, whitespace fixed
    assert normalize_answer("A  the an Test") == "test"
    # LongBench normalization keeps 'and' (unlike LoCoMo's normalization):
    assert "and" in normalize_answer("rock and roll").split()

    # --- qa_f1_score: exact answers -> 1.0 ---------------------------------
    # NOTE: under the OFFICIAL LongBench token F1, extra tokens in the
    # prediction cost precision, so a verbose correct answer does NOT score
    # 1.0 ("the answer is 7051911" vs "7051911" -> tokens {answer,is,7051911}
    # vs {7051911} -> precision 1/3, recall 1 -> F1 = 0.5). This is faithful
    # to THUDM/LongBench metrics.py; we do not special-case it.
    assert qa_f1_score("the answer is 7051911", ["7051911"]) == 0.5, \
        qa_f1_score("the answer is 7051911", ["7051911"])
    assert qa_f1_score("7051911", ["7051911"]) == 1.0
    assert qa_f1_score("Miller v. California", ["Miller v. California"]) == 1.0
    # max over multiple golds
    assert qa_f1_score("paris", ["london", "paris"]) == 1.0
    # near-misses get partial credit: "the" is normalized away, so pred
    # [eiffel, tower] vs gold [eiffel, tower, in, paris]: P=1.0, R=0.5 -> 2/3
    got = qa_f1_score("the Eiffel Tower", ["Eiffel Tower in Paris"])
    assert math.isclose(got, 2 / 3), got
    # completely wrong -> 0
    assert qa_f1_score("blue", ["red"]) == 0.0
    # empty prediction -> 0
    assert qa_f1_score("", ["anything"]) == 0.0
    # word-order invariance (bag of tokens)
    assert qa_f1_score("tower eiffel", ["eiffel tower"]) == 1.0
    # stemming is NOT applied (LongBench does not stem)
    assert qa_f1_score("running", ["runs"]) == 0.0

    # --- qa_em_score -------------------------------------------------------
    # EM is exact-match on the FULL normalized string: extra words -> 0.
    assert qa_em_score("The answer is 7051911", ["7051911"]) == 0.0
    assert qa_em_score("7051911", ["7051911"]) == 1.0
    assert qa_em_score("7051911.", ["7051911"]) == 1.0  # punctuation ignored
    assert qa_em_score("the 7051911", ["7051911"]) == 1.0  # article removed
    assert qa_em_score("7051911 is the number", ["7051911"]) == 0.0
    assert qa_em_score("paris", ["london", "paris"]) == 1.0

    # --- ruler_containment -------------------------------------------------
    # official string_match_all on a single ref is binary containment
    assert ruler_containment("The special magic number for harbor-tangent is: "
                             "7051911.", ["7051911"]) == 1.0
    # case-insensitive
    assert ruler_containment("7051911", ["7051911"]) == 1.0
    assert ruler_containment("I recall 7051911 was mentioned", ["7051911"]) == 1.0
    # miss -> 0
    assert ruler_containment("I don't know.", ["7051911"]) == 0.0
    # multi-ref: fraction of refs contained (vt: 5 variable names)
    assert ruler_containment("they are AA BB CC DD EE",
                             ["AA", "BB", "CC", "DD", "EE"]) == 1.0
    assert ruler_containment("they are AA BB CC DD XX",
                             ["AA", "BB", "CC", "DD", "EE"]) == 0.8
    # substring caution (official semantics: any containment counts)
    assert ruler_containment("70519112 is not it", ["7051911"]) == 1.0

    # --- locomo_f1 ----------------------------------------------------------
    assert locomo_f1("7 May 2023", ["7 May 2023"]) == 1.0
    assert locomo_f1("not mentioned in the conversation", ["not mentioned"]) > 0.0

    print("metrics.official_metrics self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
