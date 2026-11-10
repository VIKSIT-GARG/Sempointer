# SemPointer submission files

The revised manuscript is eleven pages including references. The author details are retained from the existing manuscript. The document was built with the supplied Springer Nature template and checked after rendering; the source ZIP was extracted into a separate directory and rebuilt successfully with matching text and pagination.

## Files to upload

1. `../pdf/SemPointer_camera_ready.pdf` — camera-ready manuscript prepared in the attached format.
2. `SemPointer_source.zip` — one manuscript `.tex`, cited bibliography and generated `.bbl`, unchanged supplied class/style, two vector figures, cost-data CSV, figure-generation sources, Kaggle aggregate evidence with a headline-check script, build instructions, and source checksums.
3. Your Turnitin similarity and AI-writing report — to be added by you. No report or detector percentage has been generated or inferred here.

`REVIEWER_RESPONSE.md` is an optional response draft. Its internal provenance section is for author review and should be removed before sending it to the conference. `verification.json` records the document checks; it is not a Springer acceptance or Turnitin certificate.

## Checks against the venue instructions

| Requirement | Result |
| --- | --- |
| Springer ICT Series template | The attached `sn-article-template` was used unchanged. It identifies itself as a Springer Nature journal template, so its applicability to the ICT Series remains to be confirmed with the organizers. |
| Plagiarism and AI content at or below the venue's 15% limit | Pending your Turnitin report. Prose was revised for clarity and specificity; no score is claimed. |
| Author names without titles | Pass: Viksit Garg. |
| Organization, city, country only | Pass: Dr. Akhilesh Das Gupta Institute of Professional Studies, New Delhi, India. “Dr.” is part of the institute's proper name; the author has no title. |
| No designation or department | Pass. |
| One corresponding author marked with an asterisk | Pass: Viksit Garg is the sole starred author. |
| Numbered figures, tables, and equations cited in text | Pass: two figures, six tables, and two equations, each cited. |
| References cited in the main text | Pass: all 21 printed references are cited; no orphan or unresolved references. |
| Editable math and tables | Pass: native LaTeX equations and tabular tables. The diagram is a separate vector PDF. |
| Springer final review | Remains an external publication decision. No acceptance outcome is implied by these checks. |

The existing sole-author name and email were retained. No funding or competing-interest declaration was invented; supply any declaration the organizers require before the final upload. The newer bounded-study records have not been represented as publicly archived. The public code link is described as the earlier implementation.

## Kaggle integration

Section 5.2 and Tables 3–4 incorporate the separate Kaggle report into the main paper. The matched-budget comparison covers 2,400 shared questions, with essentially equal mean input and 8.9% lower reported latency. Quality is higher/tied/lower in 33/54/38 task groups. The 622-pair full-context subset has 76.7% fewer input tokens and 61.3% lower reported latency, with mixed quality outcomes (14/20/34 task groups). The manuscript distinguishes task-group directions from question W/T/L, and T4 request latency from local generation-only time.

Table 4 additionally reports all five requested illustrative quality comparisons: LoCoMo QA, LongBench HotpotQA, Qasper, RepoBench-P, and InfiniteBench KV retrieval. The native table includes paired sample sizes, both system means, paired differences, and 95% intervals. Task-specific adapted metrics and post-selection are disclosed. Differences use unrounded scores; every example-level bootstrap interval includes zero, is unadjusted for multiplicity, and may understate uncertainty from shared sources. These examples do not establish superiority or equivalence.

The raw Kaggle archive is incomplete and not independently audited. Included aggregates reconcile with the retained report snapshot; this does not validate the raw worker databases. The standalone Kaggle PDF was preserved unchanged.

## Revision highlights

- Separated the original three-collection evaluation from the newer bounded study.
- Explained threshold development and descriptor-budget selection independently.
- Added the CWE output-cap audit and historical matched-token quality result.
- Preserved the aggregate quality tradeoff and inconclusive overall descriptor effect.
- Replaced the positive-only task inventory with complete family-level outcomes and a qualified variable-tracking analysis.
- Rebuilt the architecture diagram with readable labels and added a vector cost chart.
- Enlarged native table bodies to 10 pt and captions to 9 pt.
- Restored research questions, related-work scope, development details, and source-grounded failure examples.
- Added explicit equation callouts.

Original manuscripts, the submitted reference PDF, and experiment evidence were preserved. No new model generations were run. Three GPT-6.1 Terra subagents at low effort handled template auditing, evidence auditing, and manuscript drafting/review.
