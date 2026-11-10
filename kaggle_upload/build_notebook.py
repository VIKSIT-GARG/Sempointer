"""Build a self-contained Kaggle notebook; no network or inference occurs here."""
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

HERE = Path(__file__).resolve().parent
source = (HERE / 'sempointer_kaggle.py').read_bytes()
encoded = base64.b64encode(zlib.compress(source, 9)).decode()
expected = hashlib.sha256(source).hexdigest()


def cell(kind, source):
    out = {'cell_type':kind, 'metadata':{}, 'source':textwrap.dedent(source).strip() + '\n'}
    if kind == 'code':
        out.update(execution_count=None, outputs=[])
    return out


intro = '''
# SemPointer: Kaggle two-T4 backup evaluation

**Attention fix:** explicitly expand grouped key/value heads and require PyTorch's
memory-efficient SDPA kernel. Both T4s passed the 16,384-token diagnostic at about
832 MiB peak allocation. Full-model pilot execution still needs validation.
Corrected results use `kaggle_results_efficient/`. If the earlier completed
development run is present, its frozen routing choice and raw records are
validated and retained with their original backend provenance; development is
not rerun. Earlier failed pilot records remain in `kaggle_results/`.

Attach the existing private Dataset containing `sempointer_qwen35_frozen.bin`.
Select **GPU T4 x2**, **Internet on**, and file persistence if available.
This notebook includes its launcher; no second code Dataset is needed.

Run all cells with `RUN_MODE = "pilot"` first. This runs the original development
phase and representative pilot, then exports results and stops. Inspect the
projection before changing the mode to `"run"` for the full suite. Pilot responses
are reused. Sources, selected questions, controls, output caps and metrics remain
frozen. Precision is NF4/FP16, separately labeled from the L40S BF16 experiment.

`HOURS` is an allowance **from launcher start**, not from Kaggle session creation.
Keep it below the session time and GPU quota remaining. A new full-run saved
session can use 9 hours; the default pilot allowance is 2 hours. Reserve time for
setup, exporting and saving results before Kaggle's 12-hour limit.

Download `sempointer_kaggle_results.tar.gz` after each invocation. For a new
session, attach the previous notebook's saved output as another Input; setup
automatically restores a single matching archive. If several exist, set
`RESTORE_ARCHIVE` to the chosen archive's absolute input path. Persistence alone
is not a durable backup. A session stop means partial coverage, not completion.

For unattended work use Kaggle **Save Version → Save & Run All** with these clean
cells. An interactive tab closing does not provide a reliable background run.
'''

settings = '''
RUN_MODE = "pilot"    # "pilot" first; "run" continues the complete evaluation
HOURS = 2.0           # e.g. 9.0 for a fresh full-run session; use less time if needed
RESTORE_ARCHIVE = ""  # optional absolute /kaggle/input/.../sempointer_kaggle_results.tar.gz
'''

setup = '''
import base64
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile
import time
import zlib
import torch

WORK = Path("/kaggle/working/sempointer")
WORK.mkdir(parents=True, exist_ok=True)
ROOT = WORK / "frozen_bundle"
PYTHON = WORK / "venv/bin/python"
RUNNER = WORK / "sempointer_kaggle.py"

assert RUN_MODE in {"pilot", "run"}
assert 0.05 <= HOURS <= 10.5
assert torch.cuda.device_count() == 2, "Select GPU T4 x2 before running."

payload = zlib.decompress(base64.b64decode(ENCODED_RUNNER))
assert hashlib.sha256(payload).hexdigest() == EXPECTED_RUNNER_SHA
RUNNER.write_bytes(payload)

if not (ROOT / "manifest.json").exists():
    matches = list(Path("/kaggle/input").rglob("sempointer_qwen35_frozen.bin"))
    assert len(matches) == 1, "Attach exactly one frozen bundle Dataset."
    with matches[0].open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    assert checksum == "ea23e4556c9fbdf8c14669fdcc078a026bf66b07e0e71728a7f7e06dbf5f22a1"
    with tarfile.open(matches[0], "r:gz") as archive:
        archive.extractall(WORK, filter="data")

helpers = runpy.run_path(str(RUNNER))
OUTPUT = WORK / helpers["OUTPUT_NAME"]
manifest = helpers["verify_bundle"](ROOT)
print("Frozen inputs verified:", len(manifest["rows"]), "main questions")

previous_archive = WORK / 'sempointer_kaggle_results.tar.gz'
previous_backup = WORK / 'sempointer_kaggle_results_before_attention_fix.tar.gz'
if (not (OUTPUT/'config.json').exists() and (WORK/'kaggle_results/config.json').exists()
        and previous_archive.exists() and not previous_backup.exists()):
    shutil.copy2(previous_archive, previous_backup)
    print('Previous result archive preserved:', previous_backup)

# Use a pip-less venv: Kaggle's ensurepip bootstrap previously failed.
healthy = False
if PYTHON.exists():
    probe = subprocess.run([
        str(PYTHON), "-c", "import pip, torch; assert torch.cuda.device_count() == 2"
    ], capture_output=True, text=True)
    healthy = probe.returncode == 0
if not healthy:
    venv = WORK / "venv"
    if venv.exists() or venv.is_symlink():
        venv.rename(WORK / f"venv.previous-{time.time_ns()}")
    subprocess.run([
        sys.executable, "-m", "venv", "--without-pip",
        "--system-site-packages", str(venv)
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "--python", str(PYTHON),
        "install", "--no-cache-dir", "pip"
    ], check=True)

constraints = WORK / "constraints.txt"
constraints.write_text(f"torch=={torch.__version__}\\n")
subprocess.run([
    str(PYTHON), "-m", "pip", "install", "--no-cache-dir",
    "--only-binary=:all:", "--constraint", str(constraints),
    "transformers==5.2.0", "accelerate==1.10.1", "bitsandbytes==0.50.2",
    "numpy==2.2.6", "ijson==3.4.0", "pyarrow==21.0.0"
], check=True)

# Download into one local directory, avoiding a duplicate weight cache.
download = r\"\"\"
import hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download
work = Path('/kaggle/working/sempointer')
manifest = json.loads((work/'frozen_bundle/manifest.json').read_text())
model = work/'models'/manifest['model_revision']
snapshot_download(manifest['model_id'], revision=manifest['model_revision'],
    local_dir=model, token=False, max_workers=2,
    allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
with (model/'tokenizer.json').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest() == manifest['tokenizer_file_sha256']
index = json.loads((model/'model.safetensors.index.json').read_text())
assert all((model/name).is_file() for name in set(index['weight_map'].values()))
print('Pinned checkpoint ready:', model, flush=True)
\"\"\"
result = subprocess.run([str(PYTHON), "-u", "-c", download], check=True)

if not (OUTPUT/"config.json").exists():
    archives = ([Path(RESTORE_ARCHIVE)] if RESTORE_ARCHIVE else
                list(Path('/kaggle/input').rglob('sempointer_kaggle_results.tar.gz')))
    assert len(archives) <= 1, "Set RESTORE_ARCHIVE to the saved run you want to resume."
    if archives:
        restored = helpers["restore_results_archive"](archives[0], WORK)
        print('Results available at:', restored)

print('Setup ready. Mode:', RUN_MODE, '| launcher allowance:', HOURS, 'hours')
'''

launch = '''
# A fresh interpreter uses the pinned environment and the frozen evaluator.
import signal

command = [str(PYTHON), '-u', str(RUNNER), '--mode', RUN_MODE, '--hours', str(HOURS)]
process = subprocess.Popen(command, start_new_session=True)
try:
    returncode = process.wait()
except KeyboardInterrupt:
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    raise

print('Launcher exit code:', returncode)
archive = WORK / 'sempointer_kaggle_results.tar.gz'
if archive.exists():
    print('Download/save this result archive:', archive)
if returncode:
    for log in sorted(OUTPUT.glob('gpu*/worker.log')):
        print('\\nLast worker log lines:', log)
        print('\\n'.join(log.read_text(errors='replace').splitlines()[-35:]))
    raise RuntimeError('Evaluation paused with an error. Raw checkpoints were retained; inspect the logs above.')
'''

finish = '''
from IPython.display import FileLink, display

projection = OUTPUT / 'pilot_projection.json'
if projection.exists():
    report = json.loads(projection.read_text())
    print('Projected remaining wall hours:', round(report['projected_wall_seconds']/3600, 2))
    print(report['caution'])

archive = WORK / 'sempointer_kaggle_results.tar.gz'
if archive.exists():
    print('Results are also available in Kaggle’s Output panel under sempoin­ter/.')
    display(FileLink(str(archive.relative_to(Path('/kaggle/working')))))
'''.replace('sempoin­ter', 'sempointer')

setup = setup.replace('ENCODED_RUNNER', repr(encoded)).replace('EXPECTED_RUNNER_SHA', repr(expected))
notebook = {
    'nbformat':4, 'nbformat_minor':5,
    'metadata':{'kernelspec':{'display_name':'Python 3', 'language':'python', 'name':'python3'},
                'language_info':{'name':'python'}},
    'cells':[cell('markdown', intro), cell('code', settings), cell('code', setup),
             cell('code', launch), cell('code', finish)]
}
for index, entry in enumerate(notebook['cells']):
    entry['id'] = f'sempointer-{index}'
    if entry['cell_type'] == 'code':
        compile(entry['source'], f'cell-{index}', 'exec')
path = HERE / 'SemPointer_Kaggle_T4x2.ipynb'
path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n')
fixed_path = HERE / 'SemPointer_Kaggle_T4x2_AttentionFix.ipynb'
fixed_path.write_bytes(path.read_bytes())
print(path)
print(fixed_path)
print('Embedded launcher SHA256:', expected)
