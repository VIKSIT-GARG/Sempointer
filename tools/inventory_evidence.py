"""Inventory retained evidence without inference, source edits, or database writes."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASES = [
    'experiments/results', 'experiments/modal_runs', 'experiments/reference_code',
    'submission/camera_ready/kaggle_evidence',
    'kaggle_upload/report_review_2026-11-10', 'kaggle_upload/publication_2026-11-11',
    'kaggle_upload/reference_before_attention_fix', 'kaggle_upload/sempointer_qwen35_frozen.bin',
]
SKIP = {'__pycache__', '.pytest_cache'}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def group_for(path):
    parts = path.relative_to(ROOT).parts
    if parts[0] == 'experiments' and len(parts) > 2 and (ROOT.joinpath(*parts[:3])).is_dir():
        return '/'.join(parts[:3])
    return '/'.join(parts[:2])


def main():
    manifest = ROOT / 'docs/manifests'
    manifest.mkdir(parents=True, exist_ok=True)
    records = []
    totals = {}
    for base in BASES:
        for path in ([ROOT / base] if (ROOT / base).is_file() else sorted((ROOT / base).rglob('*'))):
            if not path.is_file() or SKIP.intersection(path.parts):
                continue
            record = {'group': group_for(path), 'path': path.relative_to(ROOT).as_posix(),
                      'bytes': path.stat().st_size, 'sha256': digest(path)}
            records.append(record)
            total = totals.setdefault(record['group'], {'files': 0, 'bytes': 0})
            total['files'] += 1
            total['bytes'] += record['bytes']
    with (manifest / 'evidence_files.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group', 'path', 'bytes', 'sha256'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    (manifest / 'evidence_totals.json').write_text(json.dumps(totals, indent=2) + '\n')
    lines = [
        '# Retained experiment inventory', '',
        'This inventory records files available in the author workspace. Presence, size,',
        'or a matching checksum does not establish that a run completed or that its',
        'results were independently validated. See [the evidence map](EVIDENCE.md) and',
        '[per-run guides](../experiments/results/README.md) for interpretation.', '',
        '| Evidence group | Files | Logical bytes |', '| --- | ---: | ---: |',
    ]
    for group, counts in sorted(totals.items()):
        lines.append(f"| `{group}` | {counts['files']:,} | {counts['bytes']:,} |")
    lines += ['', f"Total: {len(records):,} files, {sum(r['bytes'] for r in records):,} logical bytes.", '',
              'The [file manifest](manifests/evidence_files.csv) retains each relative path,',
              'size, and SHA-256 fingerprint. Large raw databases, source examples and cloud',
              'bundles remain local and are ignored by Git; a manifest is not a backup.', '',
              'To regenerate from available local files, run `python tools/inventory_evidence.py`.',
              'The command reads bytes without loading models or modifying databases.', '',
              'The separate Kaggle combined report is an aggregate snapshot. Its raw archive',
              'download is incomplete; raw-worker-database validation remains pending.', '']
    (ROOT / 'docs/EXPERIMENT_INVENTORY.md').write_text('\n'.join(lines))
    print(f'Indexed {len(records):,} files; {sum(r["bytes"] for r in records):,} logical bytes')


if __name__ == '__main__':
    main()
