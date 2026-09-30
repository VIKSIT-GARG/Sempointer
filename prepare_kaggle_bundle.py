"""Package tracked experiment code plus untracked benchmark caches for Kaggle.

Run after committing code. Results, model weights, virtual environments, and
the V1 paper are deliberately excluded.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "sempointer_v2_kaggle_bundle.tar.gz"
PREFIX = "sempointer_v2"


def main() -> None:
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode or subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=ROOT
    ).returncode:
        raise SystemExit("Commit tracked code changes before building the Kaggle bundle")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    excluded = ("results/", "results_archive/", "logs/")
    sources = [ROOT / name for name in tracked if name and not name.startswith(excluded)]
    cache_root = ROOT / "benchmarks" / "data"
    caches = sorted(path for path in cache_root.rglob("*") if path.is_file())
    required = ("ruler_cache", "longbench_cache", "locomo_cache")
    for name in required:
        if not any(path.parent.is_relative_to(cache_root / name) for path in caches):
            raise SystemExit(f"Missing benchmark cache: {cache_root / name}")
    with tarfile.open(OUTPUT, "w:gz", compresslevel=6) as archive:
        for path in sources + caches:
            archive.add(path, arcname=f"{PREFIX}/{path.relative_to(ROOT)}", recursive=False)
        payload = (revision + "\n").encode()
        info = tarfile.TarInfo(f"{PREFIX}/BUNDLE_COMMIT")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    print(f"Bundle: {OUTPUT}")
    print(f"Revision: {revision}")
    print(f"Files: {len(sources) + len(caches) + 1}; size: {OUTPUT.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
