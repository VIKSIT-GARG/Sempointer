"""Paper-artifact generation for the SemPointer suite (tables, macros, figures).

Entry point:  python -m analysis.make_paper_artifacts --results results --out ../paper/generated
Every number is computed from results/<run>/predictions.jsonl or
results/efficiency/*.json; missing results render as \\textbf{--}.
"""
