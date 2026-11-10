# Kaggle manuscript results package

- `results_manuscript.md`: results narrative, paired resource tables, illustrative quality examples, and complete all-system score appendix.
- `results_section.tex`: manuscript fragment requiring booktabs. The complete appendix must accompany the selected examples.
- `results_appendix.tex`: complete score and resource tables; requires longtable and pdflscape.
- `results_document.tex`: standalone wrapper that includes the results section and complete appendix.
- PDF: `/home/viksit/Projects/tokens2/output/pdf/sempointer_kaggle_results_draft.pdf`.
- `all_task_scores.csv`: all 828 system/task cells, original 0-1 score means, skips, and unrecorded counts.
- `paired_comparisons.csv`: all 605 paired task/control comparisons.
- `report_data.json`: preserved report snapshot and frozen identifiers.

Supported headline: lower active input and reported request cost than eligible full context, plus lower recorded latency at a matched retrieval budget, for this checkpoint/hardware configuration.

The main quality table consists of explicitly post-hoc illustrative examples. It does not establish score equivalence, statistical superiority, or an isolated causal pointer effect. The complete appendix remains part of this package.

Draft evidence status: source archive download incomplete; full raw-database validation pending. These T4/FP16 measurements are separate from the existing bounded-interface experiment. The matched full-context estimates are reconstructed from saved per-task paired differences and the full-context means, with identical total n=622.
