#!/usr/bin/env python3
"""Two-T4 adapter for the immutable SemPointer bundle. Default: pilot only.

No inference runs on import. Uses the bundle's existing evaluator and metrics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time

WORK = Path('/kaggle/working/sempointer')
SYSTEMS = ['sempointer', 'pointer_off', 'bm25_2', 'bm25_4', 'bm25_matched_budget', 'full_context']


class EvaluationFailure(BaseException):
    """Escape the frozen runner's ordinary per-example error recovery."""


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f'Configuration changed at {path}; restore the matching runtime or use a new output directory')
    else:
        atomic(path, value)


def verify_bundle(root):
    root = Path(root)
    manifest = read(root / 'manifest.json')
    if not manifest or digest({k:v for k,v in manifest.items() if k != 'run_id'})[:20] != manifest['run_id']:
        raise ValueError('Frozen manifest checksum mismatch')
    paths = [(root / 'code' / name, expected) for name, expected in manifest['code_files'].items()]
    paths += [(root / r['path'], r['sha256']) for r in manifest['rows'] + manifest['development']]
    for path, expected in paths:
        if not path.resolve().is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f'Frozen source/example checksum mismatch: {path}')
    return manifest


def split_rows(rows, worker):
    groups = defaultdict(list)
    for row in rows:
        groups[row['family']].append(row)
    return [r for family in sorted(groups) for r in sorted(groups[family], key=lambda x:x['id'])[worker::2]]


def patch_engine(source):
    changes = {
        'self.nf4_compute_dtype = nf4_compute_dtype(device) if dtype == "nf4" else None':
        'self.nf4_compute_dtype = torch.float16 if dtype == "nf4" else None',
        'kwargs = dict(config=config, local_files_only=True, dtype=torch.bfloat16)':
        'kwargs = dict(config=config, local_files_only=True, dtype=torch.float16)',
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError('Unexpected frozen precision implementation; refuse an ambiguous patch')
        source = source.replace(before, after)
    return source


def prepare_runtime(bundle, runtime, manifest):
    runtime.mkdir(parents=True, exist_ok=True)
    for name in manifest['code_files']:
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (bundle / 'code' / name).read_bytes()
        if name == 'evaluation/qwen35.py':
            payload = patch_engine(payload.decode()).encode()
        if not target.exists() or target.read_bytes() != payload:
            target.write_bytes(payload)
    return {name:file_hash(runtime / name) for name in manifest['code_files']}


def worker_loop(root, model, deadline, stop_phase, engine_cls=None):
    """Run unchanged frozen phases; checkpoints stop between generations."""
    from evaluation import cloud_worker, qwen35
    from evaluation.runner import RunStore
    root = Path(root)
    state_path = root / 'state.json'
    original_engine = qwen35.Qwen35Engine
    original_record = RunStore.record

    if engine_cls is None:
        import torch
        from transformers import LogitsProcessor

        class FiniteLogits(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if (torch.isnan(scores).any().item() or torch.isposinf(scores).any().item()
                        or not torch.isfinite(scores).any(dim=-1).all().item()):
                    raise FloatingPointError('Non-finite generation logits; stop and inspect precision')
                return scores

        class T4Engine(original_engine):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                if self.nf4_compute_dtype != torch.float16:
                    raise ValueError('T4 runtime must use explicit FP16 compute')
                self.model.generation_config.output_scores = False
                generate = self.model.generate

                def checked_generate(*args, **options):
                    options['logits_processor'] = [FiniteLogits()]
                    return generate(*args, **options)

                self.model.generate = checked_generate

            def info(self):
                from transformers.models.qwen3_5 import modeling_qwen3_5 as native
                return {**super().info(), 'non_quantized_dtype':'torch.float16',
                        'finite_logits_guard':True, 'batch_size':1,
                        'fast_path_available':native.is_fast_path_available}

        engine_cls = T4Engine

    def guarded_record(store, row):
        original_record(store, row)
        if row.get('status') == 'error':
            raise EvaluationFailure(row.get('error', 'Evaluation error; raw attempt retained'))

    qwen35.Qwen35Engine = engine_cls
    RunStore.record = guarded_record
    cloud_worker._ENGINE = None
    state = read(state_path, {'phase':0, 'status':'running', 'timings':{}, 'partial_times':{}})
    def checkpoint():
        phase = read(state_path, {}).get('active_phase', 'starting')
        print(f'CHECKPOINT {root.name} {phase}', flush=True)

    try:
        while state['phase'] < stop_phase:
            if time.monotonic() >= deadline - 60 or (root.parent / 'STOP').exists():
                state.update(status='paused_deadline' if not (root.parent / 'STOP').exists() else 'paused_stop')
                break
            state = cloud_worker.run_phase(root, model, deadline, checkpoint,
                                          lambda:(root.parent / 'STOP').exists())
        if state['phase'] >= stop_phase:
            state['status'] = 'complete' if state['phase'] >= 6 else 'stage_complete'
        atomic(state_path, state)
        return state
    except BaseException as exc:
        state = read(state_path, state)
        state.update(status='paused_error', error=f'{type(exc).__name__}: {exc}')
        atomic(state_path, state)
        raise
    finally:
        RunStore.record = original_record
        qwen35.Qwen35Engine = original_engine
        cloud_worker._ENGINE = None


def project_remaining(manifests, states, completed=None):
    timings = defaultdict(list)
    for state in states:
        for family, values in state.get('timings', {}).items():
            timings[family].extend(values)
    totals = []
    for i, manifest in enumerate(manifests):
        done = (completed or [{}, {}])[i]
        pilot = set(manifest['pilot_ids'])
        audit = set(manifest['cwe_audit_ids'])
        rows = [r for r in manifest['rows'] if r['id'] not in pilot and r['id'] not in done.get('test', set())]
        rows += [r for r in manifest['rows'] if r['id'] in audit and r['id'] not in done.get('cwe_cap32', set())]
        total = 0.
        for row in rows:
            if not timings[row['family']]:
                raise ValueError('Missing completed pilot timing for ' + row['family'])
            total += max(timings[row['family']]) * 1.5
        totals.append(total)
    return {'projected_worker_seconds':totals, 'projected_wall_seconds':max(totals, default=0) + 300,
            'safety_factor':1.5, 'basis':'Worst completed pilot question per family across both workers, '
            '50% headroom, plus five minutes for reporting. Includes all six arms per question.',
            'caution':'Projection only; interrupted work resumes without changing selected samples.'}


def export_archive(output):
    """Snapshot committed SQLite data, including WAL, rather than copying live DBs."""
    output = Path(output)
    archive = output.parent / 'sempointer_kaggle_results.tar.gz'
    # One temporary DB at a time, outside the saved-output directory. Raw
    # attempts and predictions already live in SQLite; avoid duplicate exports.
    with tempfile.TemporaryDirectory(prefix='sempointer-export-') as scratch:
        temporary = archive.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz', compresslevel=3) as tf:
            for path in sorted(output.rglob('*')):
                relative = path.relative_to(output)
                if (not path.is_file() or path.is_symlink() or 'input' in relative.parts
                        or path.name.endswith(('-wal', '-shm', '.lock', '.tmp')) or path.name == 'STOP'
                        or (path.name in {'events.jsonl','predictions.jsonl','attempts.jsonl'}
                            and (path.parent / 'results.sqlite').exists())):
                    continue
                arcname = str(Path('kaggle_results') / relative)
                if path.suffix == '.sqlite':
                    target = Path(scratch) / 'snapshot.sqlite'
                    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as source:
                        with sqlite3.connect(target) as dest:
                            source.backup(dest)
                    tf.add(target, arcname=arcname)
                    target.unlink()
                else:
                    tf.add(path, arcname=arcname)
        temporary.replace(archive)
    return archive


def summarize_all(output):
    from evaluation import runner
    from evaluation.cloud_worker import report
    output = Path(output)
    manifests = [read(output / f'gpu{i}/input/manifest.json') for i in range(2)]
    states = [read(output / f'gpu{i}/state.json', {'phase':0}) for i in range(2)]
    original_store = runner.RunStore
    original_export = original_store.export
    # SQLite remains the raw source of truth. Summaries do not need a second
    # JSONL copy of every complete prompt and prediction.
    original_store.export = lambda self: None
    try:
        for i in range(2):
            if (output / f'gpu{i}/results').exists():
                report(output / f'gpu{i}')
    finally:
        original_store.export = original_export
    combined = output / 'combined'
    atomic(combined / 'input/manifest.json', read(output / 'source_manifest.json'))
    status = 'complete' if all(s['phase'] >= 6 for s in states) else 'partial'
    atomic(combined / 'state.json', {'status':status})
    for stage in ['test', 'cwe_cap32']:
        paths = [output / f'gpu{i}/results/{stage}/results.sqlite' for i in range(2)]
        if not any(p.exists() for p in paths):
            continue
        freeze(combined / 'results' / stage / 'config.json',
               {'source':'Read-only union of the two raw worker databases', 'run':read(output / 'config.json')})

    class AggregateStore:
        def __init__(self, directory, config):
            self.directory = Path(directory)
            self.db = sqlite3.connect(':memory:', uri=True)
            selects = []
            for i in range(2):
                path = output / f'gpu{i}/results' / self.directory.name / 'results.sqlite'
                if path.exists():
                    self.db.execute(f'ATTACH DATABASE ? AS w{i}', (f'file:{path}?mode=ro',))
                    selects.append(f'SELECT * FROM w{i}.latest')
            self.db.execute('CREATE TEMP VIEW latest AS ' + ' UNION ALL '.join(selects))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.db.close()

        def export(self):
            pass

    runner.RunStore = AggregateStore
    try:
        counts = report(combined)
    finally:
        runner.RunStore = original_store
    title = combined / 'results/REPORT.md'
    title.write_text(title.read_text().replace('# Modal ten-benchmark evaluation', '# Kaggle two-T4 evaluation'))
    return counts


def packages():
    result = {}
    for name in ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'numpy', 'ijson', 'pyarrow', 'tokenizers',
                 'huggingface_hub', 'fla-core', 'flash-linear-attention', 'causal-conv1d']:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def completed_questions(output):
    result = []
    for i in range(2):
        done = {}
        for stage in ['test', 'cwe_cap32']:
            path = output / f'gpu{i}/results/{stage}/results.sqlite'
            done[stage] = set()
            if path.exists():
                with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
                    done[stage] = {row[0] for row in db.execute(
                        "SELECT example_id FROM latest WHERE system IN (?,?,?,?,?,?) "
                        "AND status IN ('scored','skipped') GROUP BY example_id HAVING COUNT(DISTINCT system)=6", SYSTEMS)}
        result.append(done)
    return result


def recover_pilot_timings(output, manifests, states):
    """A hard kill can commit responses before their question timer checkpoint."""
    recovered = []
    for i, (manifest, state) in enumerate(zip(manifests, states)):
        timings = {f:list(values) for f, values in state.get('timings', {}).items()}
        pilot = set(manifest['pilot_ids'])
        families = {r['id']:r['family'] for r in manifest['rows']}
        records = defaultdict(list)
        path = output / f'gpu{i}/results/test/results.sqlite'
        if path.exists():
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
                for (payload,) in db.execute('SELECT payload FROM latest'):
                    row = json.loads(payload)
                    if row['example_id'] in pilot:
                        records[row['example_id']].append(row)
        for ident, rows in records.items():
            if {r['system'] for r in rows if r['status'] in {'scored','skipped'}} != set(SYSTEMS):
                continue
            floor_ms = sum(r.get('generation_ms', 0) for r in rows)
            floor_ms += max(r.get('ingestion_ms', 0) for r in rows)
            floor_ms += max(r.get('prompt_suite_preparation_ms', 0) for r in rows)
            timings.setdefault(families[ident], []).append(floor_ms / 1000)
        recovered.append({**state, 'timings':timings})
    return recovered


def prepare_workers(bundle, output, manifest, code_hashes):
    for i in range(2):
        root = output / f'gpu{i}'
        (root / 'input').mkdir(parents=True, exist_ok=True)
        examples = root / 'input/examples'
        if not examples.exists():
            examples.symlink_to(bundle / 'examples', target_is_directory=True)
        rows = split_rows(manifest['rows'], i)
        ids = {r['id'] for r in rows}
        derived = {**manifest, 'source_bundle_run_id':manifest['run_id'], 'worker':i,
                   'code_files':code_hashes, 'dtype':'nf4', 'inference_batch_size':1,
                   'kaggle_precision':'NF4 weights, explicit FP16 compute and non-quantized components',
                   'rows':rows, 'pilot_ids':[x for x in manifest['pilot_ids'] if x in ids],
                   'cwe_audit_ids':[x for x in manifest['cwe_audit_ids'] if x in ids]}
        derived['sampling_report'] = {f:{'selected':sum(r['family'] == f for r in rows)} for f in manifest['families']}
        derived.pop('run_id')
        derived['run_id'] = digest(derived)[:20]
        freeze(root / 'input/manifest.json', derived)


def progress(output):
    entries = []
    for i in range(2):
        state = read(output / f'gpu{i}/state.json', {'phase':0, 'status':'starting'})
        counts = {}
        for stage in ['test', 'cwe_cap32']:
            path = output / f'gpu{i}/results/{stage}/results.sqlite'
            if path.exists():
                with sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=5) as db:
                    counts[stage] = dict(db.execute('SELECT status,COUNT(*) FROM latest GROUP BY status'))
        entries.append(f"GPU{i} phase={state['phase']} {state.get('status')} {counts}")
    return ' | '.join(entries)


def run_children(args, workers, stop_phase, deadline):
    processes = []
    streams = []
    output = args.output
    try:
        for i in workers:
            stream = (output / f'gpu{i}/worker.log').open('a', buffering=1)
            streams.append(stream)
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(i), 'OMP_NUM_THREADS':'2',
                           'TOKENIZERS_PARALLELISM':'false', 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'}
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(i),
                       '--output', str(output), '--runtime', str(args.runtime), '--model', str(args.model),
                       '--deadline', str(deadline), '--stop-phase', str(stop_phase)]
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment))
        last = 0.
        stop_since = None
        while any(p.poll() is None for p in processes):
            now = time.monotonic()
            if now-last >= 30:
                print(progress(output), flush=True)
                last = now
            if any(p.poll() not in (None, 0) for p in processes):
                (output / 'STOP').touch()
            if (output / 'STOP').exists():
                stop_since = stop_since or now
            if now >= deadline+10 or (stop_since is not None and now-stop_since > 30):
                for proc in processes:
                    if proc.poll() is None:
                        proc.terminate()
                break
            time.sleep(2)
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('A worker stopped with an error or reached the hard deadline. '
                               'Committed predictions are retained; inspect gpu*/worker.log before resuming.')
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for stream in streams:
            stream.close()


def controller(args):
    start = time.monotonic()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_bundle(args.bundle)
        if manifest['systems'] != SYSTEMS:
            raise ValueError('Unexpected system definitions in frozen manifest')
        import torch
        if torch.cuda.device_count() != 2 or any(torch.cuda.get_device_capability(i) != (7,5) for i in range(2)):
            raise ValueError('This adapter requires exactly two Turing T4 GPUs')
        args.model = args.model or WORK / 'models' / manifest['model_revision']
        if file_hash(args.model / 'tokenizer.json') != manifest['tokenizer_file_sha256']:
            raise ValueError('Pinned checkpoint/tokenizer is missing or changed; rerun setup')
        index = read(args.model / 'model.safetensors.index.json')
        weights = {name:file_hash(args.model / name) for name in sorted(set(index['weight_map'].values()))}
        hashes = prepare_runtime(args.bundle, args.runtime, manifest)
        config = {'source_bundle_run_id':manifest['run_id'], 'launcher_sha256':file_hash(__file__),
                  'model_revision':manifest['model_revision'], 'weight_sha256':weights, 'runtime_code_files':hashes,
                  'precision':'nf4_fp16', 'batch_size':1, 'systems':SYSTEMS, 'packages':packages(),
                  'gpus':[torch.cuda.get_device_name(i) for i in range(2)],
                  'latency_scope':'Serial per GPU; both GPU workers may run concurrently; finite-logits guard enabled'}
        freeze(output / 'config.json', config)
        freeze(output / 'source_manifest.json', manifest)
        prepare_workers(args.bundle, output, manifest, hashes)
        shutil.copy2(__file__, output / 'sempointer_kaggle.py')
        (output / 'STOP').unlink(missing_ok=True)
        sys.path.insert(0, str(args.runtime))
        deadline = start + args.hours*3600 - 120
        previous_handlers = {}
        def request_stop(signum, frame):
            (output / 'STOP').touch()
        for signum in [signal.SIGINT, signal.SIGTERM]:
            previous_handlers[signum] = signal.signal(signum, request_stop)
        try:
            selection = output / 'gpu0/results/threshold_selection.json'
            if not selection.exists():
                print('Development: 36 frozen synthetic questions, three candidates, two systems.', flush=True)
                run_children(args, [0], 3, deadline)
            if not selection.exists():
                print('Development paused at the session allowance; resume the same output directory.', flush=True)
                return
            freeze(output / 'gpu1/results/threshold_selection.json', read(selection))
            second_state = output / 'gpu1/state.json'
            if not second_state.exists():
                atomic(second_state, {'phase':3, 'status':'running', 'timings':{}, 'partial_times':{}})
            print('Pilot: both GPUs, original selected pilot questions and unchanged output caps.', flush=True)
            run_children(args, [0,1], 4, deadline)
            manifests = [read(output / f'gpu{i}/input/manifest.json') for i in range(2)]
            states = [read(output / f'gpu{i}/state.json', {'phase':0}) for i in range(2)]
            if not all(s['phase'] >= 4 for s in states):
                print('Pilot is partial; no completion projection or full evaluation started.', flush=True)
                return
            measured = recover_pilot_timings(output, manifests, states)
            projection = project_remaining(manifests, measured, completed_questions(output))
            projection['remaining_invocation_seconds'] = max(0, deadline-time.monotonic())
            atomic(output / 'pilot_projection.json', projection)
            print(json.dumps(projection, indent=2), flush=True)
            if args.mode == 'pilot':
                print('PILOT COMPLETE. Inspect projection; select run mode to continue without repeating pilot predictions.', flush=True)
                return
            print('FULL EVALUATION STARTING. A deadline stop preserves all unfinished questions for the next session.', flush=True)
            run_children(args, [0,1], 6, deadline)
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
            try:
                counts = summarize_all(output)
                print('Combined recorded statuses:', counts, flush=True)
            finally:
                archive = export_archive(output)
                print(f'RESULT ARCHIVE: {archive}', flush=True)
                print('Download/save this archive before ending the Kaggle session.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['pilot', 'run'], default='pilot')
    parser.add_argument('--hours', type=float, default=2.)
    parser.add_argument('--bundle', type=Path, default=WORK / 'frozen_bundle')
    parser.add_argument('--output', type=Path, default=WORK / 'kaggle_results')
    parser.add_argument('--runtime', type=Path, default=WORK / 'kaggle_runtime')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--worker', type=int, choices=[0,1], help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--stop-phase', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not .05 <= args.hours <= 10.5:
        parser.error('--hours must be between 0.05 and 10.5, and below your remaining Kaggle session time')
    if args.worker is not None:
        sys.path.insert(0, str(args.runtime))
        worker_loop(args.output / f'gpu{args.worker}', args.model, args.deadline, args.stop_phase)
    else:
        controller(args)


if __name__ == '__main__':
    main()
