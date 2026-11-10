"""Offline policy for fixed sampling and conservative cloud allowances."""
from collections import defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path

FAMILIES = ("ruler", "longbench", "locomo", "longmem", "babilong", "longbench_v2",
            "infinite", "loogle", "leval", "scrolls")
SYSTEMS = ("sempointer", "pointer_off", "bm25_2", "bm25_4", "bm25_matched_budget", "full_context")
HOURLY_RATE = (.000542 + 2 * .0000131 + 16 * .00000222) * 3600
CHUNK_SECONDS = 900
OVERHEAD_SECONDS = 135  # bounded startup, idle and billing headroom
ALLOWANCE_DOLLARS = 25.


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def balanced_counts(available, cap):
    if cap<0:raise ValueError("Negative sample cap")
    counts={key:0 for key in sorted(available)}
    remaining=min(cap,sum(available.values()))
    while remaining:
        for key in counts:
            if counts[key]<available[key]:
                counts[key]+=1;remaining-=1
                if not remaining:break
    return counts


class CandidatePool:
    """Store lightweight source references, never full contexts, in bounded heaps."""
    def __init__(self, seed, cap):
        self.seed,self.cap=seed,cap
        self.heaps=defaultdict(list);self.seen=set();self.available=defaultdict(int)

    def competitive(self, row):
        key=row["semantic_id"]
        if key in self.seen:return False
        priority=int(digest([self.seed,key]),16)
        heap=self.heaps[row["stratum"]]
        return len(heap)<self.cap or priority < -heap[0][0]

    def add(self, row):
        key=row["semantic_id"]
        if key in self.seen:return
        self.seen.add(key);self.available[row["stratum"]]+=1
        priority=int(digest([self.seed,key]),16)
        item=(-priority,key,row)
        heap=self.heaps[row["stratum"]]
        if len(heap)<self.cap:heapq.heappush(heap,item)
        elif priority < -heap[0][0]:heapq.heapreplace(heap,item)

    def selected(self):
        quotas=balanced_counts({k:len(v) for k,v in self.heaps.items()},self.cap)
        return [row for key in sorted(quotas) for _,_,row in
                sorted(self.heaps[key],key=lambda x:(-x[0],x[1]))[:quotas[key]]]


def project_seconds(rows, timings, safety=1.5):
    total=0.
    for row in rows:
        values=timings.get(row["family"],[])
        if not values:raise ValueError("Missing representative pilot timing for "+row["family"])
        total+=max(values)*safety
    return total


def pilot_rows(rows):
    chosen=[]
    for family in FAMILIES:
        group=sorted((r for r in rows if r["family"]==family),key=lambda r:(r["source_tokens"],r["id"]))
        if not group:continue
        # Include the longest source and highest output allowance, as well as
        # short/median examples. Scores do not participate in this selection.
        picks=[group[0],group[len(group)//2],group[-1],max(group,key=lambda r:(r["output_cap"],r["source_tokens"]))]
        seen=set()
        for row in picks:
            if row["id"] not in seen:chosen.append(row);seen.add(row["id"])
    return chosen


def _money(value):
    if not math.isfinite(value) or value<0:raise ValueError("Invalid allowance amount")
    return math.ceil(value*1_000_000)


class Allowance:
    """Pending attempts count at their full reservation after a process dies."""
    def __init__(self,path,limit=ALLOWANCE_DOLLARS):
        self.path=Path(path)
        self.data=json.loads(self.path.read_text()) if self.path.exists() else {"limit":_money(limit),"attempts":{}}
        if self.data["limit"]!=_money(limit):raise ValueError("Run allowance changed")

    @property
    def spent(self):return sum(x["amount"] for x in self.data["attempts"].values())/1_000_000

    @property
    def remaining(self):return self.data["limit"]/1_000_000-self.spent

    def reserve(self,key,dollars):
        amount=_money(dollars)
        if key in self.data["attempts"]:raise ValueError("Duplicate reservation")
        available=self.data["limit"]-sum(x["amount"] for x in self.data["attempts"].values())
        if amount>available:raise ValueError("Insufficient run allowance")
        self.data["attempts"][key]={"amount":amount,"status":"pending"}
        atomic_json(self.path,self.data)

    def finish(self,key,dollars):
        amount=_money(dollars);entry=self.data["attempts"][key]
        if entry["status"]!="pending" or amount>entry["amount"]:raise ValueError("Invalid reservation settlement")
        entry.update(amount=amount,status="settled")
        atomic_json(self.path,self.data)
