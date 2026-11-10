"""Detached Modal application; importing this file does not launch computation."""
import json
from pathlib import Path
import sys
import time
import uuid

import modal

HERE=Path(__file__).resolve().parent
VOLUME_NAME="sempointer-ten-benchmarks-v1"
LOCK_NAME="sempointer-ten-suite-leases-v1"
MOUNT=Path("/data")
app=modal.App("sempointer-ten-benchmarks")
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=True,version=2)
leases=modal.Dict.from_name(LOCK_NAME,create_if_missing=True)

# Pinned CUDA/PyTorch pair; compile the optional convolution kernel in a devel
# image without allocating a GPU. Image build costs also require the server cap.
image=(modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",add_python="3.11")
    .apt_install("git","build-essential","ninja-build")
    .env({"MAX_JOBS":"2","TORCH_CUDA_ARCH_LIST":"8.9","PYTHONPATH":"/opt/experiments",
          "HF_HUB_DISABLE_PROGRESS_BARS":"1","HF_DEACTIVATE_ASYNC_LOAD":"1","OMP_NUM_THREADS":"2"})
    .pip_install("torch==2.10.0","torchvision==0.25.0",index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==5.2.0","accelerate==1.10.1","bitsandbytes==0.50.2",
        "numpy==2.2.6","ijson==3.4.0","pyarrow==21.0.0","packaging","wheel","ninja",
        "fla-core==0.6.0")
    .pip_install("causal-conv1d==1.7.0",extra_options="--no-build-isolation"))
for folder in ["engine","evaluation","sempointer","metrics","benchmarks"]:
    image=image.add_local_dir(HERE/folder,remote_path=f"/opt/experiments/{folder}",ignore=["__pycache__","data"])
for name in ["run_all_datasets.py","reviewer_suite.py","provenance.py","prepare_modal.py","modal_suite.py","modal_client.py"]:
    image=image.add_local_file(HERE/name,remote_path=f"/opt/experiments/{name}")


def _root(run_id):
    if len(run_id)!=20 or any(c not in "0123456789abcdef" for c in run_id):raise ValueError("Invalid run ID")
    return MOUNT/"runs"/run_id


def _release(run_id,owner):
    active=leases.get("active",{})
    if active.get("run_id")==run_id and active.get("owner")==owner:leases.pop("active",None)


def _dispatch(run_id,owner,reporting=False):
    active=leases.get("active",{})
    if active.get("owner")!=owner:raise ValueError("Run lease lost")
    seq=active.get("sequence",0)+1
    leases.put("active",{**active,"sequence":seq,"updated":time.time(),"reporting":reporting})
    call=(make_report if reporting else work).spawn(run_id,owner,seq)
    leases.put(f"ticket:{run_id}:{seq}",call.object_id,skip_if_exists=True)
    return call.object_id


@app.function(image=image,volumes={MOUNT:volume},cpu=(1,1),memory=(4096,4096),timeout=1800,retries=0)
def setup(run_id):
    volume.reload()
    sys.path.insert(0,"/opt/experiments")
    from evaluation.cloud_policy import Allowance,atomic_json,digest
    from evaluation.cloud_worker import load_json
    from evaluation.inference_policy import shared_allowance_path
    from prepare_modal import file_hash,code_files
    owner=uuid.uuid4().hex
    if not leases.put("active",{"run_id":run_id,"owner":owner,"sequence":0,"updated":time.time(),
        "setup_call":modal.current_function_call_id()},skip_if_exists=True):
        return {"status":"already_running_or_needs_resume","active":leases.get("active")}
    root=_root(run_id)
    try:
        manifest=load_json(root/"input/manifest.json")
        if digest({k:v for k,v in manifest.items() if k!="run_id"})[:20]!=run_id:raise ValueError("Manifest changed")
        if manifest["code_files"]!=code_files(Path('/opt/experiments')):raise ValueError("Cloud/source code mismatch")
        state=load_json(root/"state.json",{})
        if state.get("status") in {"complete","paused_allowance","paused_pilot_allowance","paused_projection"}:
            _release(run_id,owner)
            return {"status":state["status"],"message":"No GPU restarted. Inspect the saved report and allowance."}
        for row in manifest["rows"]+manifest["development"]:
            if file_hash(root/"input"/row["path"])!=row["sha256"]:raise ValueError("Uploaded sample checksum mismatch")
        evidence=manifest.get("configuration_evidence")
        if evidence and file_hash(root/"input"/evidence["path"])!=evidence["sha256"]:raise ValueError("Configuration evidence changed")
        ledger=Allowance(shared_allowance_path(MOUNT,manifest))
        # Includes conservative CPU setup, image-build and final-report allowance.
        ledger.reserve("setup-"+owner,1.)
        volume.commit()
        from huggingface_hub import snapshot_download
        model=MOUNT/"models"/manifest["model_revision"]
        snapshot_download(manifest["model_id"],revision=manifest["model_revision"],local_dir=model,
            allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja"],token=False,max_workers=2)
        if file_hash(model/"tokenizer.json")!=manifest["tokenizer_file_sha256"]:raise ValueError("Tokenizer revision mismatch")
        atomic_json(root/"remote_model.json",{"path":str(model),"revision":manifest["model_revision"]})
        volume.commit()
        call_id=_dispatch(run_id,owner)
        return {"status":"submitted","run_id":run_id,"call_id":call_id}
    except Exception as exc:
        current=leases.get("active",{})
        if current.get("owner")==owner and current.get("sequence",0)>0:
            # A spawn RPC may have succeeded before its response was lost.
            # Leave the successor's lease intact rather than permit a duplicate.
            raise
        atomic_json(root/"setup_error.json",{"error":f"{type(exc).__name__}: {exc}"})
        volume.commit();_release(run_id,owner)
        raise


@app.function(image=image,volumes={MOUNT:volume},gpu="L40S",cpu=(2,2),memory=(16384,16384),
              max_containers=1,scaledown_window=5,timeout=900,startup_timeout=120,retries=0)
def work(run_id,owner,sequence):
    entered=time.monotonic()
    # Reused containers otherwise retain an old Volume snapshot, including an
    # obsolete allowance ledger. No result files are open at this boundary.
    volume.reload()
    sys.path.insert(0,"/opt/experiments")
    from evaluation.cloud_policy import Allowance,atomic_json,HOURLY_RATE,CHUNK_SECONDS,OVERHEAD_SECONDS
    from evaluation.cloud_worker import run_phase,affordability,load_json,PHASES
    from evaluation.inference_policy import shared_allowance_path,development_spend
    active=leases.get("active",{})
    if active.get("owner")!=owner or active.get("sequence")!=sequence:return {"status":"superseded"}
    leases.put(f"ticket:{run_id}:{sequence}",modal.current_function_call_id(),skip_if_exists=True)
    root=_root(run_id);manifest=load_json(root/"input/manifest.json")
    ledger=Allowance(shared_allowance_path(MOUNT,manifest))
    attempt=uuid.uuid4().hex
    reservation=(CHUNK_SECONDS+OVERHEAD_SECONDS)*HOURLY_RATE/3600
    state=load_json(root/"state.json",{"phase":0,"status":"running","timings":{},"partial_times":{}})
    reserved=False
    try:
        if leases.get("cancel:"+run_id,False):state["status"]="cancelled"
        elif ledger.remaining<reservation:state["status"]="paused_allowance"
        elif state["phase"]<=3 and development_spend(ledger)+reservation>3.:
            # $1 setup plus at most $2 reserved GPU pilot/development usage.
            state["status"]="paused_pilot_allowance"
        else:
            ledger.reserve(attempt,reservation);reserved=True;volume.commit()
            model=Path(load_json(root/"remote_model.json")["path"])
            state=run_phase(root,model,entered+CHUNK_SECONDS-15,volume.commit,
                            lambda:leases.get("cancel:"+run_id,False))
            ledger.finish(attempt,(time.monotonic()-entered+OVERHEAD_SECONDS)*HOURLY_RATE/3600)
            reserved=False
            if leases.get("cancel:"+run_id,False):state["status"]="cancelled"
            if state["phase"]==4 and not (root/"results/pilot_cost_projection.json").exists():
                if not affordability(root,ledger.remaining):state["status"]="paused_projection"
        if state["status"]=="running" and ledger.remaining<reservation:
            state["status"]="paused_allowance"
        if state["status"]=="running" and state["phase"]<=3 and development_spend(ledger)+reservation>3.:
            state["status"]="paused_pilot_allowance"
        atomic_json(root/"state.json",state);volume.commit()
        if state["status"]=="running":return {"status":"continued","call_id":_dispatch(run_id,owner)}
        _dispatch(run_id,owner,reporting=True)
        return {"status":state["status"],"run_id":run_id}
    except Exception as exc:
        current=leases.get("active",{})
        if current.get("owner")==owner and current.get("sequence")!=sequence:
            raise  # handoff already began; the successor may be running
        state.update(status="paused_error",error=f"{type(exc).__name__}: {exc}")
        # An uncertain/failed attempt keeps its complete reservation charged.
        atomic_json(root/"state.json",state);volume.commit()
        _dispatch(run_id,owner,reporting=True)
        raise


@app.function(image=image,volumes={MOUNT:volume},cpu=(1,1),memory=(4096,4096),timeout=900,retries=0)
def make_report(run_id,owner,sequence):
    volume.reload()
    sys.path.insert(0,"/opt/experiments")
    from evaluation.cloud_worker import report
    from evaluation.cloud_policy import atomic_json
    from evaluation.inference_policy import shared_allowance_path
    active=leases.get("active",{})
    if active.get("owner")!=owner or active.get("sequence")!=sequence:return {"status":"superseded"}
    leases.put(f"ticket:{run_id}:{sequence}",modal.current_function_call_id(),skip_if_exists=True)
    root=_root(run_id)
    try:
        counts=report(root)
        for name,path in [("state.json",root/"state.json"),
                          ("allowance.json",shared_allowance_path(MOUNT,json.loads((root/"input/manifest.json").read_text())))]:
            if path.exists():(root/"results"/name).write_bytes(path.read_bytes())
        volume.commit()
        return counts
    finally:
        _release(run_id,owner)


@app.local_entrypoint()
def main(run_id: str):
    # --detach keeps setup and its spawned continuation chain alive after the
    # local process exits. No local loop is needed to submit subsequent chunks.
    call=setup.spawn(run_id)
    print(json.dumps({"run_id":run_id,"setup_call_id":call.object_id,
                      "status":"detached_submission","volume":VOLUME_NAME}))
