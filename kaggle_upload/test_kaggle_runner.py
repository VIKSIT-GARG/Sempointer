"""Software checks only: the model boundary below is a CPU fixture."""
import hashlib
import importlib.util
import json
import sqlite3
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "experiments/modal_runs/ten_qwen35_v1/bundle"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FROZEN / "code"))


def launcher():
    assert (HERE / "sempointer_kaggle.py").exists(), "Kaggle launcher has not been implemented"
    import sempointer_kaggle
    return sempointer_kaggle


def test_shards_are_disjoint_exhaustive_balanced_and_order_independent():
    m = launcher()
    rows = [{"id": f"{f}-{i}", "family": f} for f in ["a", "b"] for i in range(7)]
    shards = [m.split_rows(rows, i) for i in range(2)]
    ids = [{r["id"] for r in s} for s in shards]
    assert not ids[0] & ids[1]
    assert ids[0] | ids[1] == {r["id"] for r in rows}
    assert m.split_rows(list(reversed(rows)), 0) == shards[0]
    assert all(abs(sum(r["family"] == f for r in shards[0]) -
                   sum(r["family"] == f for r in shards[1])) <= 1 for f in ["a", "b"])


def test_runtime_changes_only_explicit_t4_precision_and_rejects_unknown_source():
    m = launcher()
    original = (FROZEN / "code/evaluation/qwen35.py").read_text()
    patched = m.patch_engine(original)
    assert "dtype=torch.bfloat16" not in patched
    assert "self.nf4_compute_dtype = torch.float16 if dtype == \"nf4\" else None" in patched
    assert patched.replace("dtype=torch.float16)", "dtype=torch.bfloat16)").replace(
        'self.nf4_compute_dtype = torch.float16 if dtype == "nf4" else None',
        'self.nf4_compute_dtype = nf4_compute_dtype(device) if dtype == "nf4" else None') == original
    with pytest.raises(ValueError, match="precision"):
        m.patch_engine(original.replace("dtype=torch.bfloat16", "dtype=unexpected"))


def test_resume_refuses_configuration_changes(tmp_path):
    m = launcher()
    path = tmp_path / "config.json"
    m.freeze(path, {"revision": "a", "precision": "nf4_fp16"})
    m.freeze(path, {"revision": "a", "precision": "nf4_fp16"})
    with pytest.raises(ValueError, match="changed"):
        m.freeze(path, {"revision": "b", "precision": "nf4_fp16"})


def test_bundle_verification_rejects_changed_source_and_examples(tmp_path):
    m = launcher()
    root = tmp_path / "bundle"
    (root / "code").mkdir(parents=True)
    (root / "code/a.py").write_text("source")
    (root / "sample.json").write_text("sample")
    manifest = {"code_files": {"a.py": m.file_hash(root / "code/a.py")},
                "rows": [{"path": "sample.json", "sha256": m.file_hash(root / "sample.json")}],
                "development": []}
    manifest["run_id"] = m.digest(manifest)[:20]
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert m.verify_bundle(root)["run_id"] == manifest["run_id"]
    (root / "sample.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        m.verify_bundle(root)
    (root / "sample.json").write_text("sample")
    (root / "code/a.py").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        m.verify_bundle(root)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


class FixtureEngine:
    """Not a model and not benchmark evidence."""
    calls = 0
    fail = False

    def __init__(self, **kwargs):
        self.tokenizer = Tokenizer()
        self.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768), parameters=lambda: [])

    def info(self):
        return {"fixture": "CPU-only software check"}

    def format_prompt(self, text):
        return "CHAT " + text

    def generate(self, prompt, **kwargs):
        type(self).calls += 1
        if type(self).fail:
            raise FloatingPointError("non-finite fixture logits")
        return SimpleNamespace(text="mars", prompt_tokens=len(self.tokenizer.encode(prompt)),
            latency_ms=1, completion_tokens=1, peak_vram_mb=0, ttft_ms=0, stop_reason="eos")


def fixture_root(tmp_path):
    root = tmp_path / "gpu0"
    (root / "input/examples").mkdir(parents=True)
    rows = []
    for ident in ["a", "b", "dev"]:
        path = f"examples/{ident}.json"
        example = dict(id=ident, family="ruler", task="cwe" if ident == "b" else "fixture",
            context="mars rover mission " * 50, question="where?", gold=["mars"], metric="qa_f1",
            split="test", group_id=ident)
        (root / "input" / path).write_text(json.dumps({"example": example}))
        rows.append(dict(id=ident, family="ruler", task=example["task"], path=path,
                         source_tokens=150, output_cap=512 if ident == "b" else 128))
    manifest = dict(run_id="fixture", code_files={}, model_id="fixture", model_revision="fixture",
        dtype="nf4", seed=1, rows=rows[:2], development=rows[2:], pilot_ids=["a"], cwe_audit_ids=["b"],
        limits={"source_chars":10000, "source_tokens":10000, "prompt_tokens":10000}, sampling_report={})
    (root / "input/manifest.json").write_text(json.dumps(manifest))
    return root


def test_real_evaluator_reuses_pilot_predictions_and_preserves_cwe_caps(tmp_path):
    m = launcher()
    root = fixture_root(tmp_path)
    FixtureEngine.calls = 0
    FixtureEngine.fail = False
    state = m.worker_loop(root, "fixture", time.monotonic()+120, 4, engine_cls=FixtureEngine)
    assert state["phase"] == 4
    pilot_calls = FixtureEngine.calls
    assert pilot_calls == 12  # three dev candidates x two systems, plus six pilot arms
    state = m.worker_loop(root, "fixture", time.monotonic()+120, 6, engine_cls=FixtureEngine)
    assert state["phase"] == 6
    assert FixtureEngine.calls == pilot_calls + 12  # only remaining main question + cap32 audit
    with sqlite3.connect(root / "results/test/results.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 12
        caps = {json.loads(r[0])["max_new_tokens"] for r in db.execute("SELECT payload FROM latest WHERE example_id='b'")}
        assert caps == {512}
    with sqlite3.connect(root / "results/cwe_cap32/results.sqlite") as db:
        assert {json.loads(r[0])["max_new_tokens"] for r in db.execute("SELECT payload FROM latest")} == {32}


def test_deadline_stops_without_inference_and_error_is_recorded_without_fake_score(tmp_path):
    m = launcher()
    root = fixture_root(tmp_path)
    FixtureEngine.calls = 0
    state = m.worker_loop(root, "fixture", time.monotonic()-1, 4, engine_cls=FixtureEngine)
    assert FixtureEngine.calls == 0
    assert state["status"] == "paused_deadline"
    FixtureEngine.fail = True
    try:
        with pytest.raises(m.EvaluationFailure):
            m.worker_loop(root, "fixture", time.monotonic()+120, 4, engine_cls=FixtureEngine)
    finally:
        FixtureEngine.fail = False
    with sqlite3.connect(root / "results/dev_loose/results.sqlite") as db:
        row = json.loads(db.execute("SELECT payload FROM latest").fetchone()[0])
        assert row["status"] == "error" and "score" not in row


def test_archive_captures_committed_wal_and_excludes_inputs_weights_and_environments(tmp_path):
    m = launcher()
    out = tmp_path / "kaggle_results"
    db_path = out / "gpu0/results/test/results.sqlite"
    db_path.parent.mkdir(parents=True)
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE evidence(value TEXT)")
    db.execute("INSERT INTO evidence VALUES ('committed')")
    db.commit()
    (out / "input").mkdir()
    (out / "input/secret-not-for-export").write_text("fixture")
    archive = m.export_archive(out)
    with tarfile.open(archive) as tf:
        names = tf.getnames()
        assert not any("secret-not-for-export" in name or name.endswith("-wal") for name in names)
        tf.extractall(tmp_path / "restored", filter="data")
    with sqlite3.connect(tmp_path / "restored/kaggle_results/gpu0/results/test/results.sqlite") as restored:
        assert restored.execute("SELECT value FROM evidence").fetchone()[0] == "committed"
    db.close()


def test_projection_is_conservative_and_rejects_incomplete_family_timings():
    m = launcher()
    manifests = [{"rows":[{"id":"a", "family":"ruler"}], "pilot_ids":[], "cwe_audit_ids":[]},
                 {"rows":[{"id":"b", "family":"ruler"}], "pilot_ids":[], "cwe_audit_ids":["b"]}]
    result = m.project_remaining(manifests, [{"timings":{"ruler":[10]}}, {"timings":{"ruler":[20]}}])
    assert result["projected_worker_seconds"] == [30,60]
    assert result["projected_wall_seconds"] >= 60
    with pytest.raises(ValueError, match="timing"):
        m.project_remaining(manifests, [{},{}])


def test_projection_excludes_completed_main_and_audit_questions():
    m = launcher()
    manifests = [{"rows":[{"id":"a", "family":"ruler"}], "pilot_ids":[], "cwe_audit_ids":[]},
                 {"rows":[{"id":"b", "family":"ruler"}], "pilot_ids":[], "cwe_audit_ids":["b"]}]
    result = m.project_remaining(manifests, [{"timings":{"ruler":[10]}}, {}],
                                 [{"test":{"a"}}, {"test":{"b"}, "cwe_cap32":{"b"}}])
    assert result["projected_worker_seconds"] == [0,0]


def test_combined_report_keeps_original_predictions_and_paired_cwe_audit(tmp_path):
    m = launcher()
    out = tmp_path / "kaggle_results"
    root = fixture_root(out)
    FixtureEngine.fail = False
    m.worker_loop(root, "fixture", time.monotonic()+120, 6, engine_cls=FixtureEngine)
    second = out / 'gpu1'
    (second / 'input').mkdir(parents=True)
    manifest = json.loads((root / 'input/manifest.json').read_text())
    (second / 'input/manifest.json').write_text(json.dumps({**manifest, 'rows':[]}))
    (second / 'state.json').write_text(json.dumps({'phase':6, 'status':'complete'}))
    m.atomic(out / 'source_manifest.json', manifest)
    m.atomic(out / 'config.json', {'fixture':'software check'})
    counts = m.summarize_all(out)
    assert counts['test'] == {'scored':12}
    assert counts['cwe_cap32'] == {'scored':6}
    audit = m.read(out / 'combined/results/cwe_cap_audit.json')
    assert all(audit[s]['paired_questions'] == 1 for s in m.SYSTEMS)
    m.summarize_all(out)
    assert not (out / 'combined/results/test/results.sqlite').exists(), 'Do not duplicate the raw database'
    assert not (out / 'gpu0/results/test/predictions.jsonl').exists(), 'SQLite retains the complete raw evidence'
    with sqlite3.connect(root / 'results/test/results.sqlite') as db:
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 12


def test_real_child_processes_exit_without_loading_a_model_when_stage_is_complete(tmp_path):
    m = launcher()
    for i in range(2):
        (tmp_path / f'gpu{i}').mkdir()
        m.atomic(tmp_path / f'gpu{i}/state.json', {'phase':6, 'status':'complete'})
    args = SimpleNamespace(output=tmp_path, runtime=FROZEN / 'code', model=Path('unused'))
    m.run_children(args, [0,1], 6, time.monotonic()+60)
    assert all(m.read(tmp_path / f'gpu{i}/state.json')['status'] == 'complete' for i in range(2))


def test_reporting_before_any_main_predictions_still_exports_partial_status(tmp_path):
    m = launcher()
    root = fixture_root(tmp_path)
    manifest = m.read(root / 'input/manifest.json')
    (tmp_path / 'gpu1/input').mkdir(parents=True)
    m.atomic(tmp_path / 'gpu1/input/manifest.json', manifest)
    m.atomic(tmp_path / 'source_manifest.json', manifest)
    m.atomic(tmp_path / 'config.json', {'fixture':'not model evidence'})
    assert m.summarize_all(tmp_path) == {}
    assert m.read(tmp_path / 'combined/state.json')['status'] == 'partial'


def test_notebook_contains_the_exact_verified_launcher_and_valid_python_cells():
    import ast
    import base64
    import zlib
    notebook = json.loads((HERE / 'SemPointer_Kaggle_T4x2.ipynb').read_text())
    for entry in notebook['cells']:
        if entry['cell_type'] == 'code':
            compile(entry['source'], '<notebook-cell>', 'exec')
    setup = ast.parse(notebook['cells'][2]['source'])
    decode = next(n for n in ast.walk(setup) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'b64decode')
    embedded = zlib.decompress(base64.b64decode(ast.literal_eval(decode.args[0])))
    assert embedded == (HERE / 'sempointer_kaggle.py').read_bytes()


def test_pilot_projection_recovers_recorded_work_after_an_interrupted_checkpoint(tmp_path):
    m = launcher()
    path = tmp_path / 'gpu0/results/test/results.sqlite'
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE latest(payload TEXT)')
        for system in m.SYSTEMS:
            db.execute('INSERT INTO latest VALUES (?)', (json.dumps({
                'example_id':'pilot', 'system':system, 'status':'scored',
                'generation_ms':1000, 'ingestion_ms':50, 'prompt_suite_preparation_ms':20}),))
    manifests = [{'pilot_ids':['pilot'], 'rows':[{'id':'pilot', 'family':'ruler'}]},
                 {'pilot_ids':[], 'rows':[]}]
    states = [{'timings':{'ruler':[1]}}, {}]
    result = m.recover_pilot_timings(tmp_path, manifests, states)
    assert max(result[0]['timings']['ruler']) == pytest.approx(6.07)
    assert states[0]['timings']['ruler'] == [1]  # preserve original checkpoint evidence


def test_efficient_attention_context_expands_kv_preserves_result_and_restores_dispatch(monkeypatch):
    from contextlib import nullcontext
    import torch
    from transformers.integrations import sdpa_attention
    from torch.nn.attention import SDPBackend
    m = launcher()
    selected = []
    def cpu_context(backend):
        selected.append(backend)
        return nullcontext()  # Only the external CUDA kernel boundary is replaced.
    monkeypatch.setattr(torch.nn.attention, 'sdpa_kernel', cpu_context)
    original = sdpa_attention.use_gqa_in_sdpa
    module = SimpleNamespace(num_key_value_groups=4, is_causal=True)
    torch.manual_seed(4)
    q = torch.randn(1, 8, 16, 16)
    k = torch.randn(1, 2, 16, 16)
    v = torch.randn_like(k)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    with m.t4_attention_context():
        assert sdpa_attention.use_gqa_in_sdpa(None, k) is False
        actual, _ = sdpa_attention.sdpa_attention_forward(module, q, k, v, None)
    torch.testing.assert_close(actual.transpose(1,2), expected, rtol=1e-5, atol=1e-6)
    assert selected == [SDPBackend.EFFICIENT_ATTENTION]
    assert sdpa_attention.use_gqa_in_sdpa is original
    with pytest.raises(RuntimeError):
        with m.t4_attention_context():
            raise RuntimeError('fixture')
    assert sdpa_attention.use_gqa_in_sdpa is original


def completed_legacy_development(tmp_path):
    m = launcher()
    source = tmp_path / 'legacy'
    root = fixture_root(source)
    FixtureEngine.fail = False
    m.worker_loop(root, 'fixture', time.monotonic()+120, 3, engine_cls=FixtureEngine)
    manifest = m.read(root / 'input/manifest.json')
    identity = dict(source_bundle_run_id=manifest['run_id'], model_revision=manifest['model_revision'],
        runtime_code_files={}, weight_sha256={'fixture':'not-real-weights'},
        precision='nf4_fp16', batch_size=1, systems=m.SYSTEMS, packages={'torch':'fixture'})
    m.atomic(source / 'config.json', {**identity, 'launcher_sha256':
        'd57df846362f26b2bc1e8747bbd6056132b2b3b54892848eaa91564349839eb1'})
    return source, manifest, identity


def test_backend_upgrade_reuses_frozen_development_only_with_separate_provenance(tmp_path):
    m = launcher()
    source, manifest, identity = completed_legacy_development(tmp_path)
    before = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    target = tmp_path / 'efficient'
    assert m.import_completed_development(source, target, manifest, identity)
    selection = m.read(target / 'gpu0/results/threshold_selection.json')
    original = m.read(source / 'gpu0/results/threshold_selection.json')
    assert selection['selected']['thresholds'] == original['selected']['thresholds']
    assert m.read(target / 'gpu0/state.json')['phase'] == 3
    assert (target / 'development_reference/dev_loose/results.sqlite').exists()
    assert not (target / 'gpu0/results/test/results.sqlite').exists()
    reference = m.read(target / 'development_reference/manifest.json')
    assert reference['source_attention_backend'] == 'automatic_sdpa'
    assert reference['target_attention_backend'] == m.ATTENTION_BACKEND
    for name, data in before.items():
        assert (source/name).read_bytes() == data


def test_backend_upgrade_rejects_changed_model_and_incomplete_development(tmp_path):
    m = launcher()
    source, manifest, identity = completed_legacy_development(tmp_path)
    with pytest.raises(ValueError, match='identity'):
        m.import_completed_development(source, tmp_path/'bad-model', manifest,
                                       {**identity, 'model_revision':'different'})
    path = source/'gpu0/results/dev_loose/results.sqlite'
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM latest WHERE system='bm25_4'")
        db.commit()
    with pytest.raises(ValueError, match='complete'):
        m.import_completed_development(source, tmp_path/'incomplete', manifest, identity)


def test_result_restore_routes_backend_versions_to_separate_directories(tmp_path):
    m = launcher()
    source = tmp_path / 'source'
    source.mkdir()
    m.atomic(source/'config.json', {'attention_backend':m.ATTENTION_BACKEND})
    archive = m.export_archive(source)
    work = tmp_path / 'work'
    restored = m.restore_results_archive(archive, work)
    assert restored == work/m.OUTPUT_NAME
    assert m.read(restored/'config.json')['attention_backend'] == m.ATTENTION_BACKEND
    assert m.restore_results_archive(archive, work) == restored
    m.atomic(source/'config.json', {'launcher_sha256':m.LEGACY_LAUNCHER_SHA})
    legacy_archive = m.export_archive(source)
    assert m.restore_results_archive(legacy_archive, work) == work/'kaggle_results'
