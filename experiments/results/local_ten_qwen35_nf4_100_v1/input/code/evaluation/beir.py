"""Disk-backed, two-stage BEIR diagnostic; not official end-to-end QA.

SQLite FTS5 BM25 generates candidates without relevance labels. The paper's
lexical router selects blocks inside that candidate pool. Pointer-off shares
the route and therefore has the same retrieval ranking by construction.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import time
import zipfile

from .data import read_records


def build_index(archive,prefix,path):
    archive,path=Path(archive),Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path)
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-32768")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
    signature=json.dumps([str(archive.resolve()),archive.stat().st_size,archive.stat().st_mtime_ns,prefix])
    old=db.execute("SELECT value FROM metadata WHERE key='source'").fetchone()
    ready=db.execute("SELECT value FROM metadata WHERE key='ready'").fetchone()
    if old and old[0]==signature and ready and ready[0]=="yes":return db
    # Only the generated index is rebuilt. Source archives are never changed.
    for table in ["documents","queries","relevance"]:db.execute("DROP TABLE IF EXISTS "+table)
    db.execute("DELETE FROM metadata")
    db.execute("CREATE VIRTUAL TABLE documents USING fts5(docid UNINDEXED,title,text)")
    db.execute("CREATE TABLE queries(qid TEXT PRIMARY KEY,text TEXT)")
    db.execute("CREATE TABLE relevance(split TEXT,qid TEXT,docid TEXT,score REAL,PRIMARY KEY(split,qid,docid))")
    db.execute("INSERT INTO metadata VALUES('source',?)",(signature,));db.commit()
    for index,row in enumerate(read_records(archive,prefix+"corpus.jsonl")):
        if index%500==0:
            db.commit()
            if shutil.disk_usage(path.parent).free<10*1024**3:
                db.close();raise RuntimeError("Index stopped at the 10 GiB disk reserve; increase storage before rerunning")
        db.execute("INSERT INTO documents(docid,title,text) VALUES(?,?,?)",(str(row["_id"]),row.get("title",""),row.get("text","")))
    for row in read_records(archive,prefix+"queries.jsonl"):
        db.execute("INSERT OR REPLACE INTO queries VALUES(?,?)",(str(row["_id"]),row["text"]))
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if member.startswith(prefix+"qrels/") and member.endswith(".tsv"):
                split=Path(member).stem
                with z.open(member) as raw:
                    for row in csv.DictReader(io.TextIOWrapper(raw),delimiter="\t"):
                        db.execute("INSERT OR REPLACE INTO relevance VALUES(?,?,?,?)",(split,row["query-id"],row["corpus-id"],float(row["score"])))
    db.execute("INSERT INTO metadata VALUES('ready','yes')");db.commit()
    return db


def retrieve(db,query,k):
    terms=list(dict.fromkeys(re.findall(r"\w+",query.lower())))[:64]
    if not terms:return []
    expression=" OR ".join('"'+term+'"' for term in terms)
    return db.execute("SELECT docid,title,text FROM documents WHERE documents MATCH ? ORDER BY bm25(documents),rowid LIMIT ?",(expression,k)).fetchall()


def ir_metrics(ranking,relevance):
    ranking=list(dict.fromkeys(ranking))
    relevant={k:v for k,v in relevance.items() if v>0}
    ideal=sorted(relevant.values(),reverse=True)[:10]
    def dcg(grades):return sum(x/math.log2(i+2) for i,x in enumerate(grades))
    denominator=dcg(ideal)
    return {"ndcg_at_10":dcg([relevant.get(x,0) for x in ranking[:10]])/denominator if denominator else 0.,
            "recall_at_10":len(set(ranking[:10])&relevant.keys())/len(relevant) if relevant else 0.,
            "recall_selected":len(set(ranking)&relevant.keys())/len(relevant) if relevant else 0.}


def run_beir(args):
    from transformers import AutoTokenizer
    from .controls import prepare
    from .runner import RunStore,summarize
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    archives=sorted((args.data_root/"originals/beir").glob("*.zip"))
    if args.tasks!="all":archives=[p for p in archives if p.stem in args.tasks.split(",")]
    if not archives:raise SystemExit("No complete BEIR archives matched --tasks; inspect the download inventory")
    hashes={}
    for path in archives:
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(4*1024**2),b""):h.update(block)
        hashes[str(path)]=h.hexdigest()
    from .provenance import code_fingerprint, package_versions
    config={"protocol":"two-stage SQLite-FTS5-BM25 candidate retrieval plus SemPointer lexical routing; no LLM generation",
            "archives":hashes,"model_tokenizer":args.model,"candidates":args.beir_candidates,"limit":args.limit,"include_train":args.include_train,
            "seed":args.seed,"max_source_chars":args.max_source_chars,"max_source_tokens":args.max_source_tokens,
            "keep_indices":args.keep_beir_indices,
            "tokenizer_commit":tokenizer.init_kwargs.get("_commit_hash"),"tokenizer_sha256":hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "code_sha256":code_fingerprint(),"packages":package_versions()}
    with RunStore(args.output,config) as store:
        completed_path=args.output/"completed_corpora.json"
        completed=json.loads(completed_path.read_text()) if completed_path.exists() else {}
        missing=[]
        manifest=args.data_root/"status.json"
        access_gaps=[]
        if manifest.exists():
            state=json.loads(manifest.read_text());access_gaps=state.get("issues",[])
            for item in state["files"]:
                if item["collection"]=="beir" and not (args.data_root/item["path"]).is_file():
                    if args.tasks=="all" or Path(item["path"]).stem in args.tasks.split(","):
                        missing.append(item["path"])
                        store.record({"example_id":"missing:"+item["path"],"system":"loader","family":"beir","task":Path(item["path"]).stem,
                                      "status":"skipped","reason":"archive_not_downloaded","download_status":item["status"]})
        (args.output/"catalog.json").write_text(json.dumps({"archives":hashes,"missing_archives":missing,"access_gaps":access_gaps,"protocol":config["protocol"]},indent=2)+"\n")
        for archive in archives:
            with zipfile.ZipFile(archive) as z:
                prefixes=[name[:-len("corpus.jsonl")] for name in z.namelist() if name.endswith("corpus.jsonl")]
            if not prefixes:
                store.record({"example_id":str(archive),"system":"loader","family":"beir","task":archive.stem,"status":"error","error":"No top-level corpus.jsonl; archive layout requires an adapter"})
            for prefix in prefixes:
                label=prefix.strip("/") or archive.stem
                if completed.get(label)==hashes[str(archive)]:continue
                try:
                    index_name=hashlib.sha256((str(archive)+prefix).encode()).hexdigest()[:20]+".sqlite"
                    db=build_index(archive,prefix,args.output/"indices"/index_name)
                    splits=[x[0] for x in db.execute("SELECT DISTINCT split FROM relevance") if args.include_train or x[0]!="train"]
                    for split in splits:
                        queries=db.execute("SELECT DISTINCT q.qid,q.text FROM queries q JOIN relevance r ON q.qid=r.qid WHERE r.split=? ORDER BY q.qid",(split,))
                        for n,(qid,question) in enumerate(queries):
                            if args.limit and n>=args.limit:break
                            eid=f"beir/{label}/{split}/{qid}"
                            systems=["candidate_bm25","sempointer","pointer_off","bm25_2","bm25_4"]
                            if all(store.done(eid,s) for s in systems):continue
                            base={"example_id":eid,"family":"beir","task":label,"split":split,"metric":"ndcg_at_10","protocol":config["protocol"],"question":question}
                            start=time.perf_counter();candidates=retrieve(db,question,args.beir_candidates)
                            candidate_ms=(time.perf_counter()-start)*1000
                            if not candidates:
                                relevance=dict(db.execute("SELECT docid,score FROM relevance WHERE split=? AND qid=?",(split,qid)))
                                for system in systems:
                                    store.record({**base,"system":system,"status":"scored","score":0.,**ir_metrics([],relevance),"candidate_recall":0.,"ranking":[],"candidate_ms":candidate_ms})
                                continue
                            if args.max_source_chars and sum(len(t)+len(text) for _,t,text in candidates)>args.max_source_chars:
                                for system in systems:store.record({**base,"system":system,"status":"skipped","reason":"candidate_source_safety_limit"})
                                continue
                            prepared=prepare("initialization",tokenizer,seed=args.seed)
                            router=prepared.router;blocks=[];docids=[];source_tokens=0
                            for docid,title,text in candidates:
                                ids=tokenizer.encode(title+"\n"+text,add_special_tokens=False)
                                source_tokens+=len(ids)
                                for offset in range(0,len(ids),512):
                                    blocks.append(tokenizer.decode(ids[offset:offset+512],skip_special_tokens=True));docids.append(docid)
                            if not blocks or (args.max_source_tokens and source_tokens>args.max_source_tokens):
                                for system in systems:
                                    store.record({**base,"system":system,"status":"skipped","reason":"candidate_token_safety_limit_or_empty","source_tokens":source_tokens})
                                continue
                            router.blocks=blocks;router.pointer_ids=router.generator.pointers(blocks,tokenizer);router.selector.fit(blocks)
                            selection=router.select(question)
                            ranked=[i for i,_ in router.selector.rank(question)]
                            mappings={"candidate_bm25":[x[0] for x in candidates],"sempointer":[docids[i] for i in selection["ids"]],
                                      "pointer_off":[docids[i] for i in selection["ids"]],"bm25_2":[docids[i] for i in ranked[:2]],"bm25_4":[docids[i] for i in ranked[:4]]}
                            # Relevance is read only after candidate generation
                            # and selection, and is used solely for scoring.
                            relevance=dict(db.execute("SELECT docid,score FROM relevance WHERE split=? AND qid=?",(split,qid)))
                            candidate_recall=ir_metrics([x[0] for x in candidates],relevance)["recall_selected"]
                            for system,ranking in mappings.items():
                                if store.done(eid,system):continue
                                measured=ir_metrics(ranking,relevance)
                                store.record({**base,"system":system,"status":"scored","score":measured["ndcg_at_10"],**measured,
                                              "ranking":list(dict.fromkeys(ranking)),"candidate_ids":[x[0] for x in candidates],"candidate_recall":candidate_recall,"candidate_ms":candidate_ms})
                        queries.close()
                    db.close()
                    completed[label]=hashes[str(archive)]
                    temp=completed_path.with_suffix(".tmp");temp.write_text(json.dumps(completed,indent=2)+"\n");temp.replace(completed_path)
                    if not args.keep_beir_indices:
                        # Only this task's generated, closed index is removed.
                        # The original archive and every result are retained.
                        index_path=args.output/"indices"/index_name
                        for suffix in ["","-wal","-shm"]:
                            artifact=Path(str(index_path)+suffix)
                            if artifact.exists():artifact.unlink()
                except Exception as exc:
                    store.record({"example_id":"beir-loader:"+label,"system":"loader","family":"beir","task":label,"status":"error","error":f"{type(exc).__name__}: {exc}"})
                store.export()
        store.export();result=summarize(store)
        if result["status_counts"].get("error",0):raise SystemExit(2)
