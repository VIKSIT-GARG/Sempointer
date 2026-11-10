"""Download the releases enumerated in docs/STEP1_DATASET_AUDIT.md.

Run with experiments/warden/bin/python datasets/download_all.py.
No dataset scripts or benchmark generators are executed. Archives stay compressed.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from urllib.parse import quote

import requests
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESERVE = 10 * 1024**3
LOCK = threading.RLock()
STATE = {"pid": os.getpid(), "started": time.time(), "phase": "inventory", "reserve_bytes": RESERVE, "files": [], "issues": []}
RESERVATIONS = {}


def save():
    with LOCK:
        STATE["updated"] = time.time()
        temp = ROOT / "status.json.tmp"
        temp.write_text(json.dumps(STATE, indent=2) + "\n")
        temp.replace(ROOT / "status.json")


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def get_text(url):
    response = requests.get(url, timeout=(15, 45))
    response.raise_for_status()
    return response.text


def entry(collection, path, url, size=None, sha256=None, md5=None, revision=None):
    return dict(collection=collection, path=path, url=url, expected_bytes=size,
                expected_sha256=sha256, expected_md5=md5, revision=revision,
                status="pending", downloaded_bytes=0)


def hf_inventory(repo):
    info = HfApi(token=False).dataset_info(repo, files_metadata=True)
    return [entry(repo, "huggingface/" + repo + "/" + x.rfilename,
                  "https://huggingface.co/datasets/" + repo + "/resolve/" + info.sha + "/" + quote(x.rfilename, safe="/"),
                  x.size, x.lfs.sha256 if x.lfs else None, revision=info.sha)
            for x in info.siblings]


def add_direct(collection, name, url, md5=None):
    size = None
    try:
        r = requests.head(url, allow_redirects=True, timeout=(10, 20))
        if r.ok and r.headers.get("Content-Length"):
            size = int(r.headers["Content-Length"])
    except requests.RequestException:
        pass
    return entry(collection, "originals/" + collection + "/" + name, url, size, md5=md5)


def inventory():
    repos = ["zai-org/LongBench", "zai-org/LongBench-v2", "xiaowu0162/longmemeval-cleaned",
             "RMT-team/babilong", "xinrongzhang2022/InfiniteBench", "L4NLP/LEval",
             "bigai-nlco/LooGLE", "tau/scrolls", "allenai/qasper", "deepmind/narrativeqa",
             "hotpotqa/hotpot_qa"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(hf_inventory, repo): repo for repo in repos}
        for future in concurrent.futures.as_completed(futures):
            try:
                STATE["files"].extend(future.result())
            except Exception as exc:
                STATE["issues"].append({"collection": futures[future], "error": str(exc)[:500]})
            save()

    github = ["NVIDIA/RULER", "snap-research/locomo", "gkamradt/needle-in-a-haystack",
              "beir-cellar/beir", "Alab-NII/2wikimultihop", "StonyBrookNLP/musique",
              "google-deepmind/narrativeqa"]
    for repo in github:
        try:
            r = requests.get("https://api.github.com/repos/" + repo + "/commits/HEAD", timeout=(10, 30))
            r.raise_for_status()
            sha = r.json()["sha"]
            STATE["files"].append(entry(repo, "sources/" + repo + "/" + sha + ".zip",
                                        "https://codeload.github.com/" + repo + "/zip/" + sha, revision=sha))
            if repo == "snap-research/locomo":
                STATE["files"].append(entry(repo, "originals/locomo/locomo10.json",
                    "https://raw.githubusercontent.com/" + repo + "/" + sha + "/data/locomo10.json"))
        except Exception as exc:
            STATE["issues"].append({"collection": repo, "error": str(exc)[:500]})
        save()

    direct = []
    for filename in ["hotpot_train_v1.1.json", "hotpot_dev_distractor_v1.json", "hotpot_dev_fullwiki_v1.json", "hotpot_test_fullwiki_v1.json"]:
        direct.append(("hotpotqa", filename, "https://curtis.ml.cmu.edu/datasets/hotpot/" + filename, None))
    for filename in ["qasper-train-dev-v0.3.tgz", "qasper-test-and-evaluator-v0.3.tgz"]:
        direct.append(("qasper", filename, "https://qasper-dataset.s3.us-west-2.amazonaws.com/" + filename, None))
    for name, key in [("data.zip", "npidmtadreo6df2"), ("data_ids.zip", "7ep3h8unu2njfxv"),
                      ("data_ids_april7.zip", "ms2m13252h6xubs"), ("para_with_hyperlink.zip", "wlhw26kik59wbh8")]:
        direct.append(("2wikimultihop", name, "https://www.dropbox.com/s/" + key + "/" + name + "?dl=1", None))
    direct.append(("musique", "musique_v1.0.zip", "https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t", None))
    try:
        readme = get_text("https://raw.githubusercontent.com/beir-cellar/beir/main/README.md")
        (ROOT / "BEIR_SOURCE_README.md").write_text(readme)
        for line in readme.splitlines():
            matches = re.findall(r"https://public\.ukp\.informatik\.tu-darmstadt\.de/thakur/BEIR/datasets/([\w-]+\.zip)", line)
            for name in dict.fromkeys(matches):
                md5 = re.search(r"\b[0-9a-f]{32}\b", line)
                direct.append(("beir", name, "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/" + name, md5.group() if md5 else None))
            if line.startswith("|") and "❌" in line:
                STATE["issues"].append({"collection": "BEIR", "status": "manual_access", "detail": line})
    except Exception as exc:
        STATE["issues"].append({"collection": "BEIR", "error": str(exc)[:500]})
    # The example URL can occur twice in the README; download each path once.
    direct = list({(x[0], x[1]): x for x in direct}.values())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        STATE["files"].extend(pool.map(lambda args: add_direct(*args), direct))
    STATE["known_total_bytes"] = sum(x["expected_bytes"] or 0 for x in STATE["files"])
    STATE["phase"] = "downloading"
    save()


def transfer(item):
    path = ROOT / item["path"]
    if not path.parent.resolve().is_relative_to(ROOT):
        raise ValueError("Destination outside datasets directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    key = item["path"]
    # Reuse the verified paper cache without copying or changing it.
    cached = PROJECT / "experiments/benchmarks/data/longbench_cache/data.zip"
    if key == "huggingface/zai-org/LongBench/data.zip" and not path.exists() and cached.exists():
        if digest(cached) == item["expected_sha256"]:
            path.symlink_to(os.path.relpath(cached, path.parent))
    if path.exists():
        actual = digest(path)
        if item["expected_sha256"] and actual != item["expected_sha256"]:
            raise ValueError("Existing file checksum mismatch; preserved without overwrite")
        if item["expected_bytes"] is not None and path.stat().st_size != item["expected_bytes"]:
            raise ValueError("Existing file size mismatch; preserved without overwrite")
        if item["expected_md5"] and digest(path, "md5") != item["expected_md5"]:
            raise ValueError("Existing file MD5 mismatch; preserved without overwrite")
        item.update(status="verified_existing", sha256=actual, downloaded_bytes=path.stat().st_size)
        save()
        return
    offset = partial.stat().st_size if partial.exists() else 0
    if offset and offset == item["expected_bytes"]:
        actual = digest(partial)
        if item["expected_sha256"] and actual != item["expected_sha256"]:
            raise ValueError("Complete partial has a checksum mismatch; preserved for inspection")
        if item["expected_md5"] and digest(partial, "md5") != item["expected_md5"]:
            raise ValueError("Complete partial has an MD5 mismatch; preserved for inspection")
        partial.replace(path)
        item.update(status="complete", sha256=actual, downloaded_bytes=offset, completed=time.time())
        save()
        return
    with LOCK:
        remaining = max(0, (item["expected_bytes"] or 0) - offset)
        if remaining > shutil.disk_usage(ROOT).free - RESERVE - sum(RESERVATIONS.values()):
            item.update(status="deferred_disk_space", error="Would cross the 10 GiB free-space reserve")
            save()
            return
        RESERVATIONS[key] = remaining
        item["status"] = "downloading"
        save()
    try:
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        timeout = (45, 180) if item["collection"] == "beir" else (20, 90)
        with requests.get(item["url"], headers=headers, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", "").lower() and not path.name.endswith(".html"):
                raise ValueError("Server returned an HTML access/confirmation page instead of dataset bytes")
            if offset and response.status_code != 206:
                offset = 0  # Server ignored Range; restart this task-owned partial.
            if response.status_code == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Unexpected Content-Range")
            length = response.headers.get("Content-Length")
            total = offset + int(length) if length else item["expected_bytes"]
            if item["expected_bytes"] is not None and total is not None and total != item["expected_bytes"]:
                raise ValueError("Remote size differs from the pinned inventory")
            with LOCK:
                needed = max(0, (total or 0) - offset)
                other = sum(RESERVATIONS.values()) - RESERVATIONS.get(key, 0)
                if needed > shutil.disk_usage(ROOT).free - RESERVE - other:
                    item.update(status="deferred_disk_space", error="Remote transfer would cross disk reserve")
                    return
                RESERVATIONS[key] = needed
            item["transfer_bytes"] = total
            item["downloaded_bytes"] = offset
            last = time.monotonic()
            with partial.open("ab" if offset else "wb") as output:
                for chunk in response.iter_content(chunk_size=1024**2):
                    if not chunk:
                        continue
                    with LOCK:
                        if shutil.disk_usage(ROOT).free - len(chunk) < RESERVE:
                            item.update(status="deferred_disk_space", error="Disk reserve reached mid-transfer; partial kept")
                            return
                        output.write(chunk)
                        RESERVATIONS[key] = max(0, RESERVATIONS[key] - len(chunk))
                    item["downloaded_bytes"] += len(chunk)
                    if time.monotonic() - last > 5:
                        save()
                        last = time.monotonic()
        if total is not None and partial.stat().st_size != total:
            raise ValueError("Incomplete transfer; partial retained")
        actual = digest(partial)
        if item["expected_sha256"] and actual != item["expected_sha256"]:
            raise ValueError("SHA-256 mismatch; partial retained")
        if item["expected_md5"] and digest(partial, "md5") != item["expected_md5"]:
            raise ValueError("Upstream MD5 mismatch; partial retained")
        partial.replace(path)
        item.update(status="complete", sha256=actual, completed=time.time())
    finally:
        with LOCK:
            RESERVATIONS.pop(key, None)
            save()


def download(item):
    attempts = 10 if item["collection"] == "beir" else 3
    for attempt in range(attempts):
        try:
            transfer(item)
            print(item["status"], item["path"], flush=True)
            return
        except Exception as exc:
            item.update(status="retrying" if attempt < attempts - 1 else "failed", error=str(exc)[:500])
            save()
            if attempt < attempts - 1:
                time.sleep(min(30, 2 ** attempt))
    print("failed", item["path"], item["error"], flush=True)


def download_group(items, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(download, items))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beir-last", action="store_true",
                        help="Finish other sources before starting BEIR transfers")
    parser.add_argument("--retry-incomplete", action="store_true",
                        help="Reuse previously verified completed files; retry only unfinished entries")
    args = parser.parse_args()
    # One active manager only; status can be read while it runs.
    import fcntl
    with (ROOT / "download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / "status.json").exists():
            previous = json.loads((ROOT / "status.json").read_text())
            if previous.get("phase") != "inventory":
                STATE["files"] = previous.get("files", [])
                STATE["issues"] = previous.get("issues", [])
                STATE["known_total_bytes"] = previous.get("known_total_bytes")
        save()
        if not STATE["files"]:
            inventory()
        else:
            STATE["phase"] = "downloading"
            # The original CMU host can be unavailable; include HotpotQA's
            # publisher-hosted HF release in existing inventories as well.
            if not any(x["collection"] == "hotpotqa/hotpot_qa" for x in STATE["files"]):
                STATE["files"].extend(hf_inventory("hotpotqa/hotpot_qa"))
            STATE["known_total_bytes"] = sum(x["expected_bytes"] or 0 for x in STATE["files"])
            save()
        # Small files first: cover all benchmark families before large archives.
        queue = sorted(STATE["files"], key=lambda x: x["expected_bytes"] or 0)
        if args.retry_incomplete:
            queue = [x for x in queue if x["status"] not in ("complete", "verified_existing")
                     or not (ROOT / x["path"]).is_file()]
        # Keep a slot for other hosts even while BEIR's large, slow archives
        # occupy its three workers. The shared disk reservations cover both.
        beir = [x for x in queue if x["collection"] == "beir"]
        other = [x for x in queue if x["collection"] != "beir"]
        if args.beir_last:
            download_group(other, 1)
            download_group(beir, 3)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as groups:
                jobs = [groups.submit(download_group, beir, 3),
                        groups.submit(download_group, other, 1)]
                for job in jobs:
                    job.result()
        STATE["phase"] = "finished_with_gaps" if STATE["issues"] or any(x["status"] in ["failed", "deferred_disk_space"] for x in STATE["files"]) else "complete"
        save()
        print(STATE["phase"], flush=True)


if __name__ == "__main__":
    main()
