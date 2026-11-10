"""Local sampling and checkpoint control; fixtures are not model evidence."""
import json
from collections import Counter

import pytest


def test_local_subset_caps_whole_families_and_balances_categories():
    from evaluation.local_suite import select_subset
    rows = [dict(id=f"{family}-{i}", semantic_id=f"{family}-{i}", family=family,
                 stratum=str(i % 5))
            for family, size in [("locomo", 300), ("longmem", 37)] for i in range(size)]
    first = select_subset(rows, 100, 17)
    second = select_subset(list(reversed(rows)), 100, 17)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert Counter(r["family"] for r in first) == {"locomo": 100, "longmem": 37}
    assert Counter(r["stratum"] for r in first if r["family"] == "locomo") == dict.fromkeys(map(str, range(5)), 20)
    with pytest.raises(ValueError):
        select_subset(rows, 0, 17)


def test_local_controller_skips_financial_pilot_and_resumes_saved_phase(tmp_path):
    from evaluation.local_suite import advance_local
    from evaluation.cloud_policy import atomic_json
    atomic_json(tmp_path / "state.json", {"phase": 3, "status": "running"})
    calls = []

    def run_phase(root, model, deadline, commit, cancelled):
        state = json.loads((root / "state.json").read_text())
        calls.append(state["phase"])
        state["phase"] += 1
        state["status"] = "complete" if state["phase"] == 6 else "running"
        atomic_json(root / "state.json", state)
        return state

    result = advance_local(tmp_path, "fixture-model", run_phase, lambda: False)
    assert calls == [4, 5]
    assert result["status"] == "complete"
    advance_local(tmp_path, "fixture-model", run_phase, lambda: False)
    assert calls == [4, 5]


def test_local_controller_preserves_checkpoint_when_stop_requested(tmp_path):
    from evaluation.local_suite import advance_local
    from evaluation.cloud_policy import atomic_json
    atomic_json(tmp_path / "state.json", {"phase": 4, "status": "running"})

    def no_work(*args):
        pytest.fail("A stopped local run must not launch another phase")

    state = advance_local(tmp_path, "fixture", no_work, lambda: True)
    assert state["phase"] == 4
    assert state["status"] == "paused_local"
