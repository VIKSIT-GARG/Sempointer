# SemPointer experiments

This directory contains implementation, task adapters, runners, tests, and retained evidence for the current [camera-ready paper](../submission/camera_ready/main.tex). The Git repository now starts at the project root; run the commands below from `experiments/`.

## Evidence map

- [Run guide](results/README.md): all 18 immediate result groups, with a README in each.
- [Claim-to-evidence map](../docs/EVIDENCE.md): historical Qwen2.5, intermediate full-index, bounded pilot/full, descriptor controls, and separate Kaggle snapshot.
- [Inventory](../docs/EXPERIMENT_INVENTORY.md): file counts and SHA-256 provenance without upgrading partial runs to completed benchmarks.

`sempointer/v2_pipeline.py` implements lexical selection and exact text resolution. `evaluation/bounded.py` supplies full-prompt budget contracts; the saved bounded study uses 1K/2K/4K input ceilings. `analysis/fixed_payload_control.py` and `analysis/audit_bounded_losses.py` inspect saved outputs without new inference. Older pipeline and baseline code remain for historical comparisons.

## Software verification without inference

Using the retained local environment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 warden/bin/python -m pytest -m 'not gpu' -q
```

Some legacy non-GPU tests still request cached model/tokenizer assets. The cleanup record distinguishes those environment-dependent checks from fixture-only tests. A clean checkout needs Python 3.11 and the dependencies listed in `requirements.txt` / `requirements-evaluation.txt`; `warden/` is a local environment and is not uploaded.

## Experiment runners

The runner scripts are retained for reproducibility. Running them can load models and launch inference; it is separate from compiling the paper or checking saved records.

| Script | Role |
| --- | --- |
| `run_v2_suite.sh` | Original Qwen2.5 RULER, LongBench, LoCoMo evaluation |
| `run_reviewer_suite.sh` | Reviewer-control evaluation |
| `run_qwen35_suite.sh` | Qwen3.5 configuration and evaluation |
| `run_local_suite.sh` | Local ten-family full-index evaluation |
| `run_bounded_suite.sh` | Bounded development/pilot comparison |
| `run_bounded_full.sh` | Full bounded 1,000-question comparison |
| `run_modal_suite.sh` | Cloud preparation/evaluation workflow |
| `run_downloaded_suite.sh` | Downloaded-dataset validation and staged evaluation |

No experiment was rerun during cleanup. Raw SQLite, predictions, configurations, environments, state, source manifests, and analyses stay under their original run paths. Partial/failed attempts are retained and labeled rather than treated as finished evidence. `reference_code/` preserves immutable source snapshots used for provenance. `modal_runs/` retains cloud artifacts locally; its [guide](modal_runs/README.md) explains their status.

The removed E01–E17 archive and obsolete final/smoke paths were already absent before this cleanup; their history remains in Git and the earlier recovery location documented in [the 2026-11-08 record](../docs/CLEANUP.md).
