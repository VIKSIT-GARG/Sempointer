"""CPU regressions for the two model residency paths in the KV comparison."""

from types import SimpleNamespace
import weakref

def test_kvpress_nf4_uses_device_map_without_moving_quantized_model(monkeypatch):
    from baselines import kvpress_baseline as mod

    class FakeModel:
        def to(self, device):
            raise AssertionError("quantized model must not be moved with .to()")

        def eval(self):
            return self

    seen = {}
    fake_model = FakeModel()

    def load(model_id, **kwargs):
        seen.update(kwargs)
        return fake_model

    monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(
        pad_token="pad", eos_token="eos"
    ))
    monkeypatch.setattr(mod.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(mod, "KVPressTextGenerationPipeline", lambda **kwargs: SimpleNamespace(**kwargs))
    baseline = mod.KVPressBaseline("knorm", dtype="nf4", device="cuda")
    baseline._ensure_loaded()

    assert seen["device_map"] == {"": "cuda"}
    assert seen["attn_implementation"] == "eager"
    assert baseline.model is fake_model


def test_pipeline_cleanup_releases_kv_model_reference():
    from sempointer.pipeline import SemPointerPipeline

    class Model:
        pass

    model = Model()
    ref = weakref.ref(model)
    kv = SimpleNamespace(model=model)
    pipeline = object.__new__(SemPointerPipeline)
    pipeline._kv = kv
    pipeline._kv_context_id = 1
    pipeline._kv_encode = {"stored_bytes": 8}
    del model, kv

    pipeline.cleanup()
    assert pipeline._kv is None
    assert pipeline._kv_context_id is None
    assert pipeline._kv_encode == {}
    assert ref() is None


def test_release_shared_models_cleans_systems_before_engine():
    from run_benchmark import release_shared_models

    calls = []
    systems = {"sempointer__substrate-kv": SimpleNamespace(cleanup=lambda: calls.append("kv")),
               "full": SimpleNamespace()}
    engine = SimpleNamespace(cleanup=lambda: calls.append("engine"))
    release_shared_models(systems, engine)
    assert calls == ["kv", "engine"]
