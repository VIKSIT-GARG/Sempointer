"""Identical blocks and routes for the causal pointer-off control."""
from __future__ import annotations

import time
from dataclasses import dataclass

from sempointer.v2_pipeline import SemPointerV2Pipeline

SYSTEMS=("sempointer","pointer_off","bm25_2","bm25_4","bm25_matched_budget","full_context","no_context")


@dataclass
class Prepared:
    router: SemPointerV2Pipeline
    blocks: list[str]
    ingestion_ms: float
    source_tokens: int


def prepare(context, tokenizer, block_size=512, seed=143, router_options=None):
    start=time.perf_counter()
    router=SemPointerV2Pipeline(tokenizer=tokenizer,k=16,m=2,m_max=4,block_size=block_size,
                               substrate="text",semantic_rescue=False,device="cpu",seed=seed,
                               **(router_options or {}))
    ids=tokenizer.encode(context,add_special_tokens=False)
    # Strict block boundaries in the new suite. Historical chunk_text merged
    # tiny trailing fragments and could exceed the documented 512-token cap.
    blocks=[tokenizer.decode(ids[i:i+block_size],skip_special_tokens=True) for i in range(0,len(ids),block_size)]
    if not blocks:
        raise ValueError("Empty source history")
    router.blocks=blocks
    router.pointer_ids=router.generator.pointers(blocks,tokenizer)
    router.selector.fit(blocks)
    router._context_id+=1
    return Prepared(router,blocks,(time.perf_counter()-start)*1000,len(ids))


def token_count(tokenizer,prompt):
    # This matches LLMEngine.generate's raw-prompt tokenizer call, including
    # model-specific automatic special tokens.
    return len(tokenizer.encode(prompt,add_special_tokens=True))


def prompts_for_example(prepared, question, tokenizer):
    router=prepared.router
    selection=router.select(question)
    selected=selection["ids"]
    start=time.perf_counter()
    ranked=[i for i,_ in router.selector.rank(question)]
    ranking_ms=(time.perf_counter()-start)*1000

    def render(ids,pointers=False,overrides=None):
        original_flag=router.show_pointers
        router.show_pointers=pointers
        old={}
        try:
            for i,text in (overrides or {}).items():
                old[i]=router.blocks[i]
                router.blocks[i]=text
            return router.build_prompt(question,ids)
        finally:
            for i,text in old.items():router.blocks[i]=text
            router.show_pointers=original_flag

    def item(ids,pointers=False,**extra):
        prompt=render(ids,pointers)
        return dict(prompt=prompt,prompt_tokens=token_count(tokenizer,prompt),selected_ids=list(ids),**extra)

    out={"sempointer":item(selected,True,selection_ms=selection["selection_latency_ms"],router=selection),
         "pointer_off":item(selected,False,selection_ms=selection["selection_latency_ms"],router=selection),
         "bm25_2":item(ranked[:2],selection_ms=ranking_ms),
         "bm25_4":item(ranked[:4],selection_ms=ranking_ms),
         "full_context":item(list(range(len(router.blocks))),selection_ms=0.),
         "no_context":item([],selection_ms=0.)}
    budget=out["sempointer"]["prompt_tokens"]
    ids=[];overrides={};partial=None
    start=time.perf_counter()
    for i in ranked:
        candidate=render(ids+[i],overrides=overrides)
        if token_count(tokenizer,candidate)<=budget:
            ids.append(i)
            continue
        # A partial final payload avoids wasting the space occupied by pointers
        # in the target prompt. Recount the complete rendered prompt each time.
        tokens=tokenizer.encode(router.blocks[i],add_special_tokens=False)
        lo,hi=0,len(tokens)
        while lo<hi:
            mid=(lo+hi+1)//2
            text=tokenizer.decode(tokens[:mid],skip_special_tokens=True)
            if token_count(tokenizer,render(ids+[i],overrides={i:text}))<=budget:
                lo=mid
            else:hi=mid-1
        if lo:
            overrides[i]=tokenizer.decode(tokens[:lo],skip_special_tokens=True)
            ids.append(i);partial=i
        break
    prompt=render(ids,overrides=overrides)
    count=token_count(tokenizer,prompt)
    out["bm25_matched_budget"]=dict(prompt=prompt,prompt_tokens=count,selected_ids=ids,
        budget_tokens=budget,budget_slack_tokens=budget-count,partial_block_id=partial,
        selection_ms=ranking_ms+(time.perf_counter()-start)*1000)
    if count>budget:
        out["bm25_matched_budget"]["skip_reason"]="question_and_framing_exceed_budget"
    return out
