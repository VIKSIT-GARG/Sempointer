SemPointer camera-ready source package

Main manuscript: main.tex
Bibliography: references.bib and main.bbl
Vector figures: architecture.pdf and family_costs.pdf
Figure data: family_costs.csv (paired cost reductions, 100 questions per family)
Kaggle evidence: kaggle_evidence/ (aggregate snapshot, CSVs, runtime provenance)
Template: sn-jnl.cls and sn-mathphys-num.bst, copied unchanged from the
user-supplied sn-article-template directory.

Build with a current TeX installation:
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

Alternatively, with Tectonic:
  tectonic main.tex

All table and equation content is editable LaTeX in main.tex. Both figures
are supplied separately as vector PDFs with selectable text. Table bodies
use 10-point type, captions 9-point type, and figure labels approximately
9-10-point type. Body text and margins retain the supplied class defaults. No external manuscript fragments or local experiment directories are needed
to compile this package. The manuscript is eleven pages including references.

Verify Kaggle resource headlines and all five quality-table rows without model calls:
  python tools/check_kaggle_snapshot.py

Optional figure regeneration:
  python tools/draw_architecture.py  (requires reportlab)
  python tools/draw_family_costs.py  (requires matplotlib and numpy; reads CSV)

This package follows the attached Springer Nature sn-jnl template. The
attached package describes itself as a journal template, while the venue
instructions name the Springer ICT Series. Confirm that the organizers
intend this template before uploading. No ICT-specific certification is
claimed.

The Turnitin similarity and AI-writing reports are not included. The
author will provide them and check the venue's 15% requirement. No
detector score or acceptance outcome is inferred from the prose.

Viksit Garg is retained as the sole author from the manuscript supplied
for revision. The author is marked with an asterisk as corresponding
author. Dr. is part of the institute's official name, not an author title.
No funding or competing-interest declaration has been invented.
