# Verification of the downloaded-data evaluation runner

These are software/schema checks, not model benchmark results.

## Passed

- **27 focused CPU tests** across `tests/test_all_datasets.py` and
  `tests/test_v2_router.py`: streaming/truncation, IDs, hidden gold, dates and
  annotation leakage, reference merging, choice parsing, short documents,
  strict block bounds, pointer-off identity, budget accounting, metrics,
  freezing, seven-arm execution with a clearly identified fake test generator,
  config mismatch/resume, BEIR indexing/ranking/scoring/resume, and generated
  index cleanup.
- **276 local source/task configurations sampled**: 269 valid labeled sources,
  seven hidden-label sources explicitly identified, zero loader errors.
- Full SCROLLS QuALITY and QASPER reference scans: **6,597 adapted questions**,
  zero loader errors; QuALITY's hidden-label test split remains unscored.
- Full generation-source count: **276 configurations**, zero loader errors after fixing nullable original QASPER document metadata; a regression test covers that case. Counts are in `results/dataset_counts/REPORT.md`.
- Python compilation, shell syntax, CLI help, and the changed V2 line's
  whitespace check.
- Independent read-only review identified issues in SCROLLS splitting/grouping,
  model provenance, and BEIR metrics; the reported issues were corrected and
  covered by targeted checks.

Commands:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 warden/bin/python -m pytest \
  tests/test_all_datasets.py tests/test_v2_router.py -q --tb=short
warden/bin/python run_all_datasets.py validate --output results/suite_preflight
warden/bin/python run_all_datasets.py validate --families scrolls \
  --tasks quality,qasper --validate-all --output results/suite_reference_checks
```

See `results/suite_preflight/{catalog,validation,model_preflight}.json` and
`results/suite_reference_checks/validation.json` for diagnostic evidence.

## Not yet established

- No GPU smoke or full model evaluation was performed. The configured HF cache
  lacks the paper model's tokenizer/config. Run the explicit model-setup stage
  before inference. A real-tokenizer control check was attempted and could not
  proceed for this reason; no claim is made that it passed.
- Sample validation does not prove every record of every multi-gigabyte source
  is valid. `validate --validate-all` supplies the exhaustive schema scan.
- The broader legacy non-GPU suite is **not green**: the check reported
  56 passed, seven failed, 21 setup errors, and seven deselected. Most failures
  require uncached GPT-2/Qwen/embedding assets; one unrelated existing paper-table
  assertion also fails. Those tests and their assets were not rewritten to hide
  the failures.
- Official judge-based scores, large-context hardware feasibility, and the
  method's quality/cost advantage remain empirical questions for the actual runs.
