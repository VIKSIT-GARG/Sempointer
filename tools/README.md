# Project maintenance tools

- `inventory_evidence.py`: read existing evidence files, compute SHA-256 fingerprints and per-run totals, and write the repository inventory. No inference or database mutation.
- `verify_submission.py`: verify the current built PDF, citations, numbered callouts, native tables, fonts, page count, template identity and original reference hashes. Run from the project root after compiling `submission/camera_ready/main.tex`. Requires `pdfplumber` plus Poppler's `pdffonts`.
- `submission/camera_ready/tools/check_kaggle_snapshot.py` (inside the source package): check resource-table values and all five illustrative quality rows against retained aggregates, including sample sizes and paired intervals. Uses the standard library only.

```bash
python tools/inventory_evidence.py
python tools/verify_submission.py
python submission/camera_ready/tools/check_kaggle_snapshot.py
```

The inventory includes locally retained raw evidence. A clean Git checkout does not contain ignored multi-gigabyte databases or frozen source contexts, so regenerating its local inventory will list only the files available in that checkout. Consult the committed manifest for the author's retained records.
