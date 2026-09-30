# Run SemPointer V2 on Kaggle

Kaggle's general T4 x2 accelerator has **two separate 16 GB GPUs**. This
benchmark uses one of them; it does not get a combined 32 GB address space.
A T4 has more usable memory than the local 8 GB laptop GPU, but may not be
faster per example. A notebook session has a 12-hour execution limit.

## 1. Prepare the upload locally

From `/home/viksit/Projects/tokens2/experiments`:

```bash
warden/bin/python prepare_kaggle_bundle.py
```

Upload `/home/viksit/Projects/tokens2/sempointer_v2_kaggle_bundle.bin`
as a **private Kaggle Dataset**. The archive contains code and cached RULER,
LongBench, and LoCoMo data. It excludes model weights, results, and V1 paper.
The `.bin` extension prevents Kaggle's automatic archive extraction, which
can fail with `Directory already exists: data`. The notebook extracts it.

## 2. Configure a Kaggle Notebook

Create a new private Notebook. In its settings select **GPU T4 x2** and turn
**Internet on** so the public Qwen model and Python packages can download.
Enable **Files persistence** if available. Add your private bundle Dataset
under **Input**. The notebook needs less than 20 GB in `/kaggle/working`.

Run this first cell:

```python
from pathlib import Path
import glob, os, subprocess, sys

matches = glob.glob('/kaggle/input/**/sempointer_v2_kaggle_bundle.bin', recursive=True)
assert len(matches) == 1, f'Expected one bundle, found: {matches}'
subprocess.run(['tar', '-xzf', matches[0], '-C', '/kaggle/working'], check=True)
ROOT = Path('/kaggle/working/sempointer_v2')
print('Source revision:', (ROOT / 'BUNDLE_COMMIT').read_text().strip())
```

Install the experiment dependencies while keeping Kaggle's CUDA PyTorch:

```python
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                str(ROOT / 'requirements-kaggle.txt')], check=True)
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
```

Check the GPU and model access:

```python
import torch
from huggingface_hub import snapshot_download
print('PyTorch:', torch.__version__)
print('GPUs:', torch.cuda.device_count())
print('GPU 0:', torch.cuda.get_device_name(0))
print('GPU 0 GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
assert torch.cuda.is_available()
snapshot_download('Qwen/Qwen2.5-3B-Instruct')
```

## 3. Run one stage at a time

In a new cell, first use `stage = 'smoke'`. Change that one line to `ruler`,
`ruler_holdout`, `longbench`, or `locomo` for later runs. Run each stage only
after the previous stage finishes and you have checked its metrics.

```python
stage = 'smoke'
env = os.environ.copy()
env['SEMPTR_PYTHON'] = sys.executable
env['SEMPTR_OFFLINE'] = '0'
subprocess.run(['bash', str(ROOT / 'run_v2_suite.sh'), stage],
               cwd=ROOT, env=env, check=True)
print((ROOT / 'results' / f'v2_{stage}' / 'metrics.json').read_text()[:2000])
```

The script writes `results/v2_<stage>/predictions.jsonl` after each example
and uses `--resume`, so rerun the same cell to skip scored examples. In the
Notebook **Save Version** dialog, use **Quick Save** with the advanced option
to save output files after each completed stage. Files persistence helps
between interactive sessions but is best effort; a saved Notebook output is
the durable checkpoint.

For a fresh session, reattach your previous Notebook's output via **Add
Input**, rerun setup, then restore any earlier results before rerunning a
stage:

```python
import shutil
for config in Path('/kaggle/input').rglob('v2_*/config.json'):
    source = config.parent
    target = ROOT / 'results' / source.name
    if not target.exists():
        shutil.copytree(source, target)
        print('Restored', target)
```

## 4. Export results

After each stage, make a small download archive:

```python
subprocess.run(['tar', '-czf', '/kaggle/working/v2_results.tar.gz',
                '-C', str(ROOT), 'results'], check=True)
print('/kaggle/working/v2_results.tar.gz')
```

Download the archive from the Notebook's **Output** panel or use a
`IPython.display.FileLink`. Keep `config.json`, `environment.json`,
`predictions.jsonl`, and `metrics.json` together. These files include the
source revision and the GPU/compute dtype needed to interpret score changes.
