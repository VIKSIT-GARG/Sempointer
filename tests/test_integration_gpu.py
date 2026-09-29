"""GPU integration tests — REAL inference, small scale.

Requires: CUDA, Qwen/Qwen2.5-1.5B-Instruct cached locally.
Run:  pytest tests/test_integration_gpu.py -v -m gpu
These download nothing if the model is cached; they load the real model and
generate real text. Skipped automatically when CUDA is unavailable.
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

CONTEXT = (
    "The library on Fifth Avenue opened in 1921 and houses a collection of "
    "nineteenth-century cartography. Its reading room accommodates two hundred "
    "visitors. The adjacent museum exhibits porcelain from the Ming dynasty and "
    "hosts lectures on art history every spring. A café on the ground floor "
    "serves pastries to museum members. The city archive stores municipal "
    "records dating back to 1848 in climate-controlled vaults. "
    "The Mars rover Perseverance landed in Jezero Crater in February 2021. "
    "Its companion helicopter Ingenuity flew the first powered flight on another "
    "planet in April 2021. The rover's primary mission lasted until 2023, when it "
    "entered an extended phase. Perseverance collected rock samples in the MCS "
    "region for later return to Earth. The mission is operated by NASA's Jet "
    "Propulsion Laboratory in Pasadena, California."
)
QUERY = "Where is NASA's Mars mission operated from?"
GOLD_HINT = "Jet Propulsion Laboratory"


@pytest.fixture(scope="module")
def llm():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from engine.llm import LLMEngine

    eng = LLMEngine(model_id=MODEL, dtype="bf16", device="cuda", max_new_tokens=48)
    yield eng
    eng.cleanup()


def test_engine_generates_real_text(llm):
    res = llm.generate("What is 2+3? Answer with only the number.")
    assert res.text and res.prompt_tokens > 3 and res.completion_tokens > 0
    assert res.latency_ms > 0
    assert "5" in res.text


def test_sempointer_end_to_end(llm):
    """query -> pointer selection -> resolution -> active prompt -> REAL
    generation; the selected block must contain the answer."""
    from sempointer.pipeline import SemPointerPipeline

    p = SemPointerPipeline(tokenizer=llm.tokenizer, k=8, m=1, block_size=48, device="cuda")
    p.ingest(CONTEXT)
    out = p.answer(QUERY, llm)
    sel_text = p.registry.get_text(out["selected_ids"][0])
    assert GOLD_HINT in sel_text
    assert out["answer"], "LLM must produce a non-empty answer"
    assert out["active_tokens"] < len(llm.tokenizer.encode(CONTEXT))  # compression


def test_full_context_baseline_end_to_end(llm):
    from baselines.full_context import FullContextBaseline

    b = FullContextBaseline(tokenizer=llm.tokenizer, block_size=48)
    b.ingest(CONTEXT)
    out = b.answer(QUERY, llm)
    assert out["answer"]
    assert out["active_tokens"] > 0


def test_bm25_baseline_end_to_end(llm):
    from baselines.bm25_baseline import BM25Baseline

    b = BM25Baseline(tokenizer=llm.tokenizer, m=1, block_size=48)
    b.ingest(CONTEXT)
    out = b.answer(QUERY, llm)
    assert out["answer"]


def test_dense_rag_baseline_end_to_end(llm):
    from baselines.dense_rag import DenseRAGBaseline

    b = DenseRAGBaseline(tokenizer=llm.tokenizer, m=1, block_size=48, device="cuda")
    b.ingest(CONTEXT)
    out = b.answer(QUERY, llm)
    assert out["answer"]


def test_sempointer_active_context_is_smaller_than_full(llm):
    """The core comparison, on one example, with real measurements."""
    from baselines.full_context import FullContextBaseline
    from sempointer.pipeline import SemPointerPipeline

    sp = SemPointerPipeline(tokenizer=llm.tokenizer, k=8, m=1, block_size=48, device="cuda")
    sp.ingest(CONTEXT)
    sp_out = sp.answer(QUERY, llm)

    fc = FullContextBaseline(tokenizer=llm.tokenizer, block_size=48)
    fc.ingest(CONTEXT)
    fc_out = fc.answer(QUERY, llm)

    assert sp_out["active_tokens"] < fc_out["active_tokens"]
