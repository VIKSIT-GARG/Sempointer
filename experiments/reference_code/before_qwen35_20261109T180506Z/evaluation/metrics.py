"""Explicit local metrics. Surrogates are never labeled official judge scores."""
from collections import Counter
from difflib import SequenceMatcher
import re
import unicodedata

from metrics.official_metrics import qa_f1_score, qa_em_score


def numbers(text):
    return re.findall(r"[-+]?\d+(?:\.\d+)?", str(text).replace(",", " "))


def choices(text):
    text = text.strip()
    if re.fullmatch(r"[A-D]+",text.upper()):
        return set(text.upper())
    return set(re.findall(r"\b([A-D])\b",text))


def rouge_l(prediction, reference):
    a,b=prediction.lower().split(),reference.lower().split()
    if not a or not b:
        return float(a==b)
    row=[0]*(len(b)+1)
    for x in a:
        prev=0
        for j,y in enumerate(b,1):
            old=row[j]
            row[j]=prev+1 if x==y else max(row[j],row[j-1])
            prev=old
    return 2*row[-1]/(len(a)+len(b))


def score(metric, prediction, gold):
    if not gold:
        raise ValueError("Cannot score an example with hidden/missing gold")
    if metric=="qa_f1":
        return float(qa_f1_score(prediction,gold))
    if metric=="exact":
        return float(qa_em_score(prediction,gold))
    if metric in {"choice","choice_set"}:
        found=choices(prediction)
        return float(any(found==choices(g) and bool(found) and (metric=="choice_set" or len(found)==1) for g in gold))
    if metric=="numeric_sequence":
        found=numbers(prediction)
        return float(any(found==numbers(g) and bool(found) for g in gold))
    if metric=="all_containment":
        return sum(bool(g) and g.casefold() in prediction.casefold() for g in gold)/len(gold)
    if metric=="rouge_l":
        return max(rouge_l(prediction,g) for g in gold)
    if metric=="char_rouge_l":
        return max(rouge_l(" ".join(prediction)," ".join(g)) for g in gold)
    if metric=="code_similarity":
        p=prediction.strip().splitlines()[0] if prediction.strip() else ""
        return max(SequenceMatcher(None,p,g.strip()).ratio() for g in gold)
    if metric=="char_f1":
        def chars(x):
            return [c for c in x.lower() if not c.isspace() and not unicodedata.category(c).startswith("P")]
        a=Counter(chars(prediction))
        def f1(g):
            b=Counter(chars(g));common=sum((a&b).values());den=sum(a.values())+sum(b.values())
            return 2*common/den if den else 0.
        return max(f1(g) for g in gold)
    raise ValueError(f"Unknown metric: {metric}")
