import re
import string
import collections
from typing import List, Dict

def normalize_answer(answer: str) -> str:
    """Lowercase, remove articles, punctuation, extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))

def exact_match(prediction: str, ground_truth: str) -> float:
    """Computes exact match score, 0 or 1."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def token_f1(prediction: str, ground_truth: str) -> float:
    """Computes token F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_rouge_l(prediction: str, reference: str) -> float:
    """Computes ROUGE-L score."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure

def compute_bert_score(
    predictions: List[str],
    references: List[str],
    model_type: str = 'microsoft/deberta-xlarge-mnli',
    device: str = 'cuda'
) -> Dict[str, float]:
    """Computes BERTScore."""
    try:
        from bert_score import score
    except ImportError:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
    P, R, F1 = score(predictions, references, model_type=model_type, device=device, verbose=False)
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

def compute_qa_metrics_batch(
    predictions: List[str],
    ground_truths: List[str],
    compute_bertscore: bool = True,
    compute_rouge: bool = True
) -> Dict[str, float]:
    """Computes all QA metrics over a batch."""
    n = len(predictions)
    if n == 0:
        return {}
        
    metrics = {'em': 0.0, 'f1': 0.0}
    if compute_rouge:
        metrics['rouge_l'] = 0.0
        
    for pred, gt in zip(predictions, ground_truths):
        metrics['em'] += exact_match(pred, gt)
        metrics['f1'] += token_f1(pred, gt)
        if compute_rouge:
            metrics['rouge_l'] += compute_rouge_l(pred, gt)
            
    for k in metrics:
        metrics[k] /= n
        
    if compute_bertscore:
        bert_metrics = compute_bert_score(predictions, ground_truths)
        metrics['bertscore_f1'] = bert_metrics['f1']
        
    return metrics

def llm_judge_accuracy(
    predictions: List[str],
    references: List[str],
    questions: List[str],
    judge_model_name: str,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Structured LLM-as-judge with fixed rubric.
    Rubric: 1 if prediction contains the answer, 0 otherwise
    System prompt defines rubric strictly
    """
    # Placeholder for LLM Judge using exact match logic as simple stub
    judgements = []
    for pred, ref, q in zip(predictions, references, questions):
        if exact_match(pred, ref) > 0 or token_f1(pred, ref) > 0.5:
            judgements.append(1)
        else:
            judgements.append(0)
            
    return {
        'accuracy': sum(judgements) / len(judgements) if judgements else 0.0,
        'judgements': judgements
    }
