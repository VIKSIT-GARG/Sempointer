"""Query-shortlisted pointer descriptors under an exact whole-input budget."""
import time

from sempointer.pipeline import CONTEXT_HEADER, POINTER_HEADER
from .controls import token_count


def budgeted_variants(prepared, question, tokenizer, budget, pointer_caps=(128,256),
                      shortlist=16, prompt_formatter=None):
    if budget < 1 or shortlist < 1 or any(c < 0 for c in pointer_caps):
        raise ValueError("Invalid prompt/pointer budget")
    router = prepared.router
    selection = router.select(question)
    selected = selection["ids"]
    started = time.perf_counter()
    ranked = [i for i,_ in router.selector.rank(question)]
    ranking_ms = (time.perf_counter()-started)*1000
    wrap = prompt_formatter or (lambda x:x)

    def render(ids, entries=(), overrides=None):
        section = (POINTER_HEADER + "\n".join(text for _,text in entries) + "\n\n") if entries else ""
        body = "\n\n".join(f"[MEMORY {i}]:\n{(overrides or {}).get(i,router.blocks[i]).strip()}" for i in sorted(ids))
        return wrap(f"System: {router.system_prompt.strip()}\n\n" + section + CONTEXT_HEADER +
                    body + "\n\n" + f"Question: {question.strip()}\nAnswer:")

    def entries_for(ids):
        return [(i,f"[P{i}] {tokenizer.decode(router.pointer_ids[i],skip_special_tokens=True).strip()}".rstrip())
                for i in ids]

    def pack(candidates, entries):
        ids, overrides = [], {}
        partial = None
        empty = render([],entries)
        if token_count(tokenizer,empty)>budget:
            return dict(prompt=empty,prompt_tokens=token_count(tokenizer,empty),selected_ids=[],
                        partial_block_id=None,skip_reason="question_and_framing_exceed_budget")
        for i in candidates:
            if token_count(tokenizer,render(ids+[i],entries,overrides))<=budget:
                ids.append(i)
                continue
            tokens = tokenizer.encode(router.blocks[i],add_special_tokens=False)
            lo,hi = 0,len(tokens)
            while lo<hi:
                mid = (lo+hi+1)//2
                text = tokenizer.decode(tokens[:mid],skip_special_tokens=True)
                if token_count(tokenizer,render(ids+[i],entries,{**overrides,i:text}))<=budget:
                    lo=mid
                else:
                    hi=mid-1
            # The final rendered count is authoritative, including any BPE
            # boundary effects at the end of a decoded partial block.
            while lo:
                text = tokenizer.decode(tokens[:lo],skip_special_tokens=True)
                if token_count(tokenizer,render(ids+[i],entries,{**overrides,i:text}))<=budget:
                    overrides[i]=text;ids.append(i);partial=i
                    break
                lo-=1
            break
        prompt = render(ids,entries,overrides)
        return dict(prompt=prompt,prompt_tokens=token_count(tokenizer,prompt),selected_ids=ids,
                    partial_block_id=partial,_overrides=overrides)

    original_entries = entries_for(i for i in range(len(router.blocks)) if i not in selected)
    original = render(selected,original_entries)
    result = {"original_sempointer":dict(prompt=original,prompt_tokens=token_count(tokenizer,original),
        selected_ids=list(selected),pointer_ids=[i for i,_ in original_entries],
        pointer_tokens=token_count(tokenizer,original)-token_count(tokenizer,render(selected)),
        partial_block_id=None,selection_ms=selection["selection_latency_ms"])}
    if result["original_sempointer"]["prompt_tokens"]>budget:
        result["original_sempointer"]["skip_reason"]="input_budget_exceeded"
    result["adaptive_no_index"] = pack(selected,[])
    result["adaptive_no_index"].update(pointer_ids=[],pointer_tokens=0,selection_ms=selection["selection_latency_ms"])
    result["bm25_budget"] = pack(ranked,[])
    result["bm25_budget"].update(pointer_ids=[],pointer_tokens=0,selection_ms=ranking_ms)
    candidates = entries_for(i for i in ranked[:shortlist] if i not in selected)
    base_count = token_count(tokenizer,render([]))
    for cap in pointer_caps:
        began = time.perf_counter()
        entries = []
        for entry in candidates:
            tentative = entries+[entry]
            count = token_count(tokenizer,render([],tentative))
            if count-base_count<=cap and count<=budget:
                entries=tentative
        row = pack(selected,entries)
        while True:
            without = render(row["selected_ids"],[],row.get("_overrides"))
            delta = row["prompt_tokens"]-token_count(tokenizer,without)
            if delta<=cap or not entries:
                break
            entries.pop()
            row = pack(selected,entries)
        row.update(pointer_ids=[i for i,_ in entries],pointer_tokens=max(0,delta),
                   pointer_cap=cap,selection_ms=selection["selection_latency_ms"]+(time.perf_counter()-began)*1000)
        result[f"bounded_{cap}"]=row
    for row in result.values():
        row.pop("_overrides",None)
        row["input_budget_tokens"]=budget
    return result
