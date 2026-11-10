"""Capped L40S BF16/NF4 pilot, sharing the full suite's lease and allowance."""
import json
from pathlib import Path
import time
import uuid

import modal
import modal_suite as base

HERE=Path(__file__).resolve().parent
app=modal.App("sempointer-precision-pilot")
image=base.image.add_local_file(HERE/"modal_precision.py",remote_path="/opt/experiments/modal_precision.py")
volume=base.volume;leases=base.leases;MOUNT=base.MOUNT


@app.function(image=image,volumes={MOUNT:volume},cpu=(1,1),memory=(4096,4096),timeout=1800,retries=0)
def setup(pilot_id):
    volume.reload()
    from evaluation.cloud_policy import Allowance,atomic_json,digest,HOURLY_RATE,CHUNK_SECONDS,OVERHEAD_SECONDS
    from prepare_modal import file_hash
    root=MOUNT/"precision"/pilot_id
    package=json.loads((root/"input/pilot.json").read_text());run_id=package["source_run_id"]
    if digest({k:v for k,v in package.items() if k!="pilot_id"})[:20]!=pilot_id:raise ValueError("Pilot changed")
    for name,sha in package["code_files"].items():
        if file_hash(Path("/opt/experiments")/name)!=sha:raise ValueError("Pilot source mismatch: "+name)
    owner="precision-"+uuid.uuid4().hex
    if not leases.put("active",dict(run_id=run_id,owner=owner,sequence=0,setup_call=modal.current_function_call_id(),precision_pilot=pilot_id),skip_if_exists=True):
        raise ValueError("Another suite/pilot owns the lease; no duplicate inference started")
    handed_off=False
    try:
        ledger=Allowance(MOUNT/"runs"/run_id/"allowance.json")
        if any(k.startswith("precision-") for k in ledger.data["attempts"]):raise ValueError("A precision attempt already exists; inspect its report instead of charging twice")
        reservation=(CHUNK_SECONDS+OVERHEAD_SECONDS)*HOURLY_RATE/3600
        if ledger.remaining<1+reservation:raise ValueError("Insufficient shared allowance for precision pilot")
        ledger.reserve("precision-setup-"+pilot_id,1.)
        ledger.reserve("precision-gpu-"+pilot_id,reservation)
        volume.commit()
        from huggingface_hub import snapshot_download
        model=MOUNT/"models"/package["model_revision"]
        snapshot_download(package["model_id"],revision=package["model_revision"],local_dir=model,
            allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja"],token=False,max_workers=2)
        if file_hash(model/"tokenizer.json")!=package["tokenizer_sha256"]:raise ValueError("Tokenizer pin mismatch")
        atomic_json(root/"model.json",dict(path=str(model)));volume.commit()
        active=leases.get("active");leases.put("active",{**active,"sequence":1})
        handed_off=True
        child=work.spawn(pilot_id,owner)
        leases.put(f"ticket:{run_id}:1",child.object_id,skip_if_exists=True)
        return dict(pilot_id=pilot_id,call_id=child.object_id)
    except Exception as exc:
        atomic_json(root/"setup_error.json",dict(error=f"{type(exc).__name__}: {exc}"));volume.commit()
        if not handed_off:base._release(run_id,owner)
        raise


@app.function(image=image,volumes={MOUNT:volume},gpu="L40S",cpu=(2,2),memory=(16384,16384),
              max_containers=1,scaledown_window=5,timeout=900,startup_timeout=120,retries=0)
def work(pilot_id,owner):
    entered=time.monotonic();volume.reload()
    from evaluation.cloud_policy import Allowance,atomic_json,HOURLY_RATE,OVERHEAD_SECONDS
    from evaluation.precision_run import run_comparison
    root=MOUNT/"precision"/pilot_id
    package=json.loads((root/"input/pilot.json").read_text());run_id=package["source_run_id"]
    active=leases.get("active",{})
    if active.get("owner")!=owner:return {"status":"superseded"}
    leases.put(f"ticket:{run_id}:1",modal.current_function_call_id(),skip_if_exists=True)
    ledger=Allowance(MOUNT/"runs"/run_id/"allowance.json")
    try:
        model=json.loads((root/"model.json").read_text())["path"]
        result=run_comparison(root,model,entered+840,volume.commit)
        ledger.finish("precision-gpu-"+pilot_id,(time.monotonic()-entered+OVERHEAD_SECONDS)*HOURLY_RATE/3600)
        atomic_json(root/"results/allowance.json",ledger.data);volume.commit()
        return dict(status=result["status"],precision=result["precision_decision"],batch=result["batch_decision"])
    finally:base._release(run_id,owner)


@app.local_entrypoint()
def main(pilot_dir:str,workspace_budget_confirmed:bool=False):
    if not workspace_budget_confirmed:raise ValueError("Confirm the existing workspace usage <=$25 and net spend $0 caps")
    directory=Path(pilot_dir).resolve();package=json.loads((directory/"pilot.json").read_text())
    pilot_id=package["pilot_id"]
    if leases.get("active"):raise ValueError("Another cloud workload holds the shared lease")
    with volume.batch_upload() as upload:upload.put_directory(directory,"/precision/"+pilot_id+"/input")
    call=setup.spawn(pilot_id)
    print(json.dumps(dict(event="precision_setup",pilot_id=pilot_id,call_id=call.object_id)),flush=True)
    submitted=call.get()
    report=modal.FunctionCall.from_id(submitted["call_id"]).get()
    destination=directory.parent/"results";destination.mkdir(parents=True,exist_ok=True)
    for name in ["comparison.json","allowance.json"]:
        (destination/name).write_bytes(b"".join(volume.read_file(f"precision/{pilot_id}/results/{name}")))
    print(json.dumps(report,indent=2),flush=True)
