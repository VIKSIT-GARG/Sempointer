"""Fingerprints that prevent mixing data, code, or model revisions on resume."""
import hashlib
from importlib.metadata import version,PackageNotFoundError
import json
from pathlib import Path


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b""):h.update(chunk)
    return h.hexdigest()


def code_fingerprint():
    here=Path(__file__).resolve().parents[1]
    paths=[here/"run_all_datasets.py",here/"engine/llm.py",here/"metrics/official_metrics.py"]
    for folder in ["evaluation","sempointer","benchmarks"]:paths.extend(sorted((here/folder).glob("*.py")))
    h=hashlib.sha256()
    for path in paths:h.update(str(path.relative_to(here)).encode()+path.read_bytes())
    return h.hexdigest()


def package_versions():
    result={}
    for name in ["torch","transformers","bitsandbytes","numpy","ijson","pyarrow","tokenizers","rank_bm25"]:
        try:result[name]=version(name)
        except PackageNotFoundError:result[name]=None
    return result


def weight_files(model_id,revision):
    from huggingface_hub import try_to_load_from_cache
    local=Path(model_id)
    def locate(name):
        if local.is_dir():return local/name
        cached=try_to_load_from_cache(model_id,name,revision=revision or "main")
        return Path(cached) if isinstance(cached,str) else None
    for name in ["model.safetensors.index.json","pytorch_model.bin.index.json","model.safetensors","pytorch_model.bin"]:
        path=locate(name)
        if path is None or not path.is_file():continue
        files=sorted(set(json.loads(path.read_text())["weight_map"].values())) if name.endswith(".index.json") else [name]
        missing=[name for name in files if (locate(name) is None or not locate(name).is_file())]
        return {"available":not missing,"files":files,"missing":missing}
    return {"available":False,"files":[],"missing":["model weights or weight index"]}


def model_provenance(model_id):
    from transformers import AutoConfig,AutoTokenizer
    config=AutoConfig.from_pretrained(model_id,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(model_id,local_files_only=True)
    data={"model_commit":getattr(config,"_commit_hash",None),
          "config_sha256":hashlib.sha256(config.to_json_string().encode()).hexdigest(),
          "tokenizer_commit":tokenizer.init_kwargs.get("_commit_hash"),
          "tokenizer_sha256":hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
    data["weights"]=weight_files(model_id,data["model_commit"])
    path=Path(model_id)
    if path.is_dir():
        data["local_weights"]={p.name:file_hash(p) for p in sorted(path.glob("*.safetensors"))}
        if not data["local_weights"]:
            data["local_weights"]={p.name:file_hash(p) for p in sorted(path.glob("pytorch_model*.bin"))}
    return data
