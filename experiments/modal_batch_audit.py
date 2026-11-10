"""Small follow-up for a completed BF16 win; no model download or full evaluation."""
import json
from pathlib import Path
import time
import uuid

import modal
import modal_suite as base

HERE=Path(__file__).resolve().parent
app=modal.App("sempointer-warmed-batching-audit")
image=base.image.add_local_file(HERE/"modal_batch_audit.py",remote_path="/opt/experiments/modal_batch_audit.py")
volume=base.volume;leases=base.leases;MOUNT=base.MOUNT


@app.function(image=image,volumes={MOUNT:volume},gpu="L40S",cpu=(2,2),memory=(16384,16384),
              max_containers=1,scaledown_window=5,timeout=900,startup_timeout=120,retries=0)
def work(pilot_id,owner):
    entered=time.monotonic();volume.reload()
    from evaluation.cloud_policy import Allowance,atomic_json,HOURLY_RATE,CHUNK_SECONDS,OVERHEAD_SECONDS
    from evaluation.batch_audit import run_audit
    root=MOUNT/"precision"/pilot_id
    package=json.loads((root/"input/pilot.json").read_text());run_id=package["source_run_id"]
    if leases.get("active",{}).get("owner")!=owner:return {"status":"superseded"}
    leases.put(f"ticket:{run_id}:2",modal.current_function_call_id(),skip_if_exists=True)
    ledger=Allowance(MOUNT/"runs"/run_id/"allowance.json")
    key="precision-batch-"+pilot_id
    try:
        reservation=(CHUNK_SECONDS+OVERHEAD_SECONDS)*HOURLY_RATE/3600
        precision_spent=sum(r["amount"] for k,r in ledger.data["attempts"].items() if k.startswith("precision-"))/1_000_000
        if precision_spent+reservation>3:raise ValueError("Precision comparison's $3 reservation ceiling reached")
        ledger.reserve(key,reservation);volume.commit()
        model=json.loads((root/"model.json").read_text())["path"]
        result=run_audit(root,model,entered+840,volume.commit)
        ledger.finish(key,(time.monotonic()-entered+OVERHEAD_SECONDS)*HOURLY_RATE/3600)
        atomic_json(root/"results/allowance.json",ledger.data);volume.commit()
        return dict(status=result["status"],batch=result["batch_decision"])
    finally:base._release(run_id,owner)


@app.local_entrypoint()
def main(pilot_dir:str,workspace_budget_confirmed:bool=False):
    if not workspace_budget_confirmed:raise ValueError("Confirm the workspace usage <=$25 and net spend $0 caps")
    directory=Path(pilot_dir).resolve();package=json.loads((directory/"pilot.json").read_text())
    pilot_id=package["pilot_id"];run_id=package["source_run_id"];owner="precision-batch-"+uuid.uuid4().hex
    if not leases.put("active",dict(run_id=run_id,owner=owner,sequence=2,warmed_batch_audit=pilot_id),skip_if_exists=True):
        raise ValueError("Another workload owns the shared lease")
    call=work.spawn(pilot_id,owner)
    leases.put(f"ticket:{run_id}:2",call.object_id,skip_if_exists=True)
    print(json.dumps(dict(event="batch_audit",call_id=call.object_id,pilot_id=pilot_id)),flush=True)
    result=call.get()
    destination=directory.parent/"results";destination.mkdir(parents=True,exist_ok=True)
    for name in ["batch_audit.json","allowance.json"]:
        (destination/name).write_bytes(b"".join(volume.read_file(f"precision/{pilot_id}/results/{name}")))
    print(json.dumps(result,indent=2),flush=True)
