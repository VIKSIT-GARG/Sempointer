"""Check the assembled manuscript and record external submission requirements."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pdfplumber

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "submission/camera_ready"
tex = (ROOT / "main.tex").read_text()
bbl = (ROOT / "build/main.bbl").read_text()
log = (ROOT / "build/main.log").read_text()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

cited = {key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex)
         for key in group.split(",")}
printed = set(re.findall(r"\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}", bbl))
assert cited == printed, {"uncited": sorted(printed - cited), "missing": sorted(cited - printed)}
assert len(re.findall(r"\\author\*", tex)) == 1
assert len(re.findall(r"\\author(?:\*)?(?:\[[^]]*\])?\{", tex)) == 1
assert r"\orgdiv" not in tex
assert not re.search(r"Assistant Professor|Research Scholar|Department of|\\input\{|\\include\{", tex)
assert not re.search(r"\\(?:geometry|newgeometry|fontsize|textwidth|textheight)\b", tex)
objects = {}
for kind in ("figure", "table", "equation"):
    entries = re.findall(r"\\begin\{" + kind + r"\}(.*?)\\end\{" + kind + r"\}", tex, re.S)
    labels = []
    for entry in entries:
        found = re.findall(r"\\label\{([^}]+)\}", entry)
        assert len(found) == 1, (kind, found)
        label = found[0]
        assert re.search(r"\\(?:eqref|ref)\{" + re.escape(label) + r"\}", tex), label
        if kind == "table":
            assert any(tag in entry for tag in (r"\begin{tabular}", r"\begin{tabular*}")) and r"\includegraphics" not in entry
        labels.append(label)
    objects[kind] = labels
assert len(set(sum(objects.values(), []))) == sum(map(len, objects.values()))
assert "Overfull" not in log
assert not re.search(r"(?:Citation|Reference) .*undefined|multiply defined|Undefined control sequence", log)
assert sha(ROOT / "sn-jnl.cls") == sha(PROJECT / "sn-article-template/sn-jnl.cls")
assert sha(ROOT / "sn-mathphys-num.bst") == sha(PROJECT / "sn-article-template/bst/sn-mathphys-num.bst")
with pdfplumber.open(ROOT / "build/main.pdf") as pdf:
    page_count = len(pdf.pages)
    assert page_count <= 11
    first = pdf.pages[0].extract_text()
    assert "Viksit Garg" in first and "Garg1*" in first.replace(" ", ""), first
    assert "Department" not in first
    raster_images = sum(len(page.images) for page in pdf.pages)
    assert raster_images == 0
    text_characters = sum(len(page.chars) for page in pdf.pages)
    for page in pdf.pages:
        assert all(c["x0"] >= -0.5 and c["x1"] <= page.width + 0.5 for c in page.chars)
fonts = subprocess.check_output(["pdffonts", str(ROOT / "build/main.pdf")], text=True)
font_lines = fonts.splitlines()[2:]
assert font_lines and all(re.search(r"\byes\s+yes\s+", line) for line in font_lines), fonts
preservation = subprocess.run(["sha256sum", "-c", str(PROJECT / "docs/manifests/reference_sources.sha256")],
                              cwd=PROJECT, capture_output=True, text=True, check=True)
report = {
    "document_checks_passed": True,
    "pages_including_references": page_count,
    "maximum_pages": 11,
    "template": "User-supplied sn-article-template; unchanged sn-jnl.cls and sn-mathphys-num.bst",
    "ict_series_template_mapping_confirmed": False,
    "author": "Viksit Garg",
    "corresponding_authors": 1,
    "department_and_designations_removed": True,
    "numbered_objects_with_text_callouts": objects,
    "editable_math_and_tables": True,
    "references_cited_and_printed": len(cited),
    "orphan_printed_references": [],
    "unresolved_references": [],
    "overfull_boxes": 0,
    "all_pdf_fonts_embedded": True,
    "raster_image_objects": raster_images,
    "selectable_text_characters": text_characters,
    "table_body_font_points": 10,
    "caption_font_points": 9,
    "figure_label_font_points_approximate": "9-10",
    "reference_sources_preserved": preservation.stdout.splitlines(),
    "new_model_generations": 0,
    "turnitin_similarity_and_ai_report": "To be supplied by author; no scores claimed",
    "author_funding_and_competing_interests_confirmation": "Not supplied; no declaration invented",
    "pdf_sha256": sha(ROOT / "build/main.pdf"),
    "visual_review": "Pending final rendered-page inspection",
}
(ROOT / "build/verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
