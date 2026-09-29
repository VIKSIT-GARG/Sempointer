from typing import List, Dict, Any, Optional
from datasets import load_dataset
import uuid

def load_fever(split: str = 'paper_test', n: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_dataset('fever', 'v1.0', split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    
    results = []
    for item in ds:
        results.append({
            'claim': item['claim'],
            'evidence': item.get('evidence_wiki_url', ''), # simplified evidence
            'label': item['label'],
            'id': str(item['id'])
        })
    return results

def load_narrativeqa(split: str = 'test', n: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_dataset('narrativeqa', split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
        
    results = []
    for item in ds:
        results.append({
            'document_text': item['document']['text'],
            'question': item['question']['text'],
            'answer': item['answers'][0]['text'] if item['answers'] else '',
            'id': item['document']['id']
        })
    return results

def load_hotpotqa(split: str = 'validation', n: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_dataset('hotpot_qa', 'distractor', split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
        
    results = []
    for item in ds:
        # Context is a list of [title, list_of_sentences]
        context = ""
        for title, sents in zip(item['context']['title'], item['context']['sentences']):
            context += f"Title: {title}\n" + " ".join(sents) + "\n\n"
            
        results.append({
            'question': item['question'],
            'answer': item['answer'],
            'supporting_facts': item['supporting_facts'],
            'context': context,
            'id': item['id']
        })
    return results

def load_musique(split: str = 'validation', n: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_dataset('musique', split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
        
    results = []
    for item in ds:
        results.append({
            'question': item['question'],
            'answer': item['answer'],
            'paragraphs': item['paragraphs'],
            'id': item['id']
        })
    return results

def load_ruler_niah(context_lengths: Optional[List[int]] = None, n: Optional[int] = None) -> List[Dict[str, Any]]:
    # Simplified mock implementation for RULER NIAH
    if context_lengths is None:
        context_lengths = [4096, 8192, 32768, 131072]
        
    results = []
    count = n if n is not None else 100
    for i in range(count):
        ctx_len = context_lengths[i % len(context_lengths)]
        results.append({
            'context': "This is a long context. " * (ctx_len // 10),
            'needle': "The secret password is '12345'.",
            'question': "What is the secret password?",
            'answer': "12345",
            'context_length': ctx_len,
            'id': str(uuid.uuid4())
        })
    return results

def load_longbench(tasks: Optional[List[str]] = None, n: Optional[int] = None) -> List[Dict[str, Any]]:
    if tasks is None:
        tasks = ['multifieldqa_en', 'hotpotqa', 'narrativeqa']
        
    results = []
    for task in tasks:
        try:
            ds = load_dataset('THUDM/LongBench', task, split='test')
            if n is not None:
                ds = ds.select(range(min(n, len(ds))))
            for item in ds:
                results.append({
                    'context': item['context'],
                    'question': item['input'],
                    'answer': item['answers'][0] if isinstance(item['answers'], list) else item['answers'],
                    'id': str(uuid.uuid4()),
                    'task': task
                })
        except Exception:
            continue
    return results

def chunk_document_to_memories(
    document_text: str,
    B_tokens: int,
    tokenizer,
    overlap_tokens: int = 0
) -> List[Dict[str, Any]]:
    tokens = tokenizer.encode(document_text)
    chunks = []
    
    step = B_tokens - overlap_tokens
    if step <= 0:
        step = B_tokens
        
    chunk_id = 0
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i+B_tokens]
        if len(chunk_tokens) < B_tokens // 4 and i > 0:
            break
            
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append({
            "text": chunk_text,
            "token_count": len(chunk_tokens),
            "chunk_id": chunk_id,
            "start_char": 0, # Approximation
            "end_char": len(chunk_text)
        })
        chunk_id += 1
        
    return chunks

def prepare_benchmark_for_sempointer(
    benchmark_examples: List[Dict[str, Any]],
    benchmark_name: str,
    B_tokens: int = 512,
    N: int = 64,
    tokenizer_name: str = 'Qwen/Qwen2.5-7B-Instruct'
) -> List[Dict[str, Any]]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    sp_examples = []
    for ex in benchmark_examples:
        context = ex.get('context') or ex.get('document_text') or ex.get('claim')
        if not context:
            continue
            
        memories_data = chunk_document_to_memories(context, B_tokens, tokenizer)
        
        memories = []
        for i, md in enumerate(memories_data[:N]):
            memories.append({
                "memory_id": md["chunk_id"],
                "text": md["text"],
                "token_count": md["token_count"],
                "topic": benchmark_name,
                "inserted_at": i,
                "is_target": True, # Simplification
                "is_near_paraphrase": False,
                "is_contradictory": False
            })
            
        q_text = ex.get('question') or ex.get('claim')
        query = {
            "query_id": 0,
            "text": q_text,
            "token_count": len(tokenizer.encode(q_text)),
            "target_memory_ids": [m["memory_id"] for m in memories],
            "requires_composition": False,
            "answer": ex.get('answer') or ex.get('label') or '',
            "answer_span": {"memory_id": 0, "start": 0, "end": 0}
        }
        
        sp_ex = {
            "example_id": ex.get('id', str(uuid.uuid4())),
            "difficulty": "medium",
            "N": len(memories),
            "memories": memories,
            "queries": [query],
            "metadata": {
                "domain": benchmark_name,
                "source": benchmark_name,
                "split": "test"
            }
        }
        sp_examples.append(sp_ex)
        
    return sp_examples
