"""Generate an evidence-scoped manuscript supplement from the saved report."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re
import shutil
import textwrap
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                               Spacer, Table, TableStyle, PageBreak,
                               NextPageTemplate, KeepTogether)
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / 'report_review_2026-11-10'
PDF = ROOT / 'output/pdf/sempointer_kaggle_results_draft.pdf'
D = json.loads((SOURCE / 'report_data.json').read_text())
SYSTEMS = D['systems']
LABELS = D['labels']
TASKS = {(t['family'], t['task']): t for t in D['tasks']}
PAIRS = {(p['family'], p['task'], p['comparison']): p for p in D['paired']}
COMPLETE = {r['family'] for r in D['coverage'] if r['complete']}
SEM = D['resources'][0]
MATCH = next(r for r in D['resources'] if r['system'] == 'bm25_matched_budget')
FULL = D['full_context_paired_resources']
assert SEM['n'] == MATCH['n'] == 2400
assert FULL['n'] == 622
assert len(D['tasks']) == 138 and len(D['paired']) == 605
assert sum(r['recorded'] for r in D['coverage']) == 14853
TOKEN_SAVING = 100 * (1 - FULL['sem_active_tokens_mean'] / FULL['full_active_tokens_mean'])
FULL_LATENCY_SAVING = 100 * (1 - FULL['sem_latency_ms_mean'] / FULL['full_latency_ms_mean'])
MATCH_LATENCY_SAVING = 100 * (1 - SEM['latency_ms_mean'] / MATCH['latency_ms_mean'])

EXAMPLE_KEYS = [('locomo', 'qa'), ('longbench', 'hotpotqa'),
                ('longbench', 'qasper'), ('longbench', 'repobench-p'),
                ('infinite', 'kv_retrieval')]
NAMES = {('locomo', 'qa'): 'LoCoMo QA', ('longbench', 'hotpotqa'): 'LongBench HotpotQA',
         ('longbench', 'qasper'): 'LongBench Qasper', ('longbench', 'repobench-p'): 'LongBench RepoBench-P',
         ('infinite', 'kv_retrieval'): 'InfiniteBench KV retrieval'}
METRICS = {'qa_f1': 'Adapted F1', 'code_similarity': 'Code similarity', 'all_containment': 'Retrieval containment'}

examples = []
for key in EXAMPLE_KEYS:
    task = TASKS[key]
    sem, baseline = (task['systems'][s] for s in ['sempointer', 'bm25_matched_budget'])
    pair = PAIRS[(*key, 'sempointer_vs_bm25_matched_budget')]
    assert sem['n_scored'] == baseline['n_scored'] == pair['n_paired'] == task['selected']
    examples.append({'name': NAMES[key], 'metric': METRICS[task['metric']], 'n': pair['n_paired'],
                     'sem': sem, 'baseline': baseline, 'pair': pair})

scope = ('This supplement reports frozen run 8ef26bfd68509bae3b6b using Qwen3.5-4B '
         '(revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a), NF4 weights, FP16 compute, '
         'batch size one, and one independent worker on each of two NVIDIA T4 GPUs. '
         'Eight 300-question dataset subsets are complete: BABILong, InfiniteBench, L-Eval, '
         'LoCoMo, LongBench, LongBench v2, LongMemEval, and LooGLE. '
         'Their first five systems score the same 2,400 selected questions. '
         'The archived snapshot records 13,055 scored outcomes and 1,798 skips across all '
         'ten selected families; RULER and SCROLLS have partial coverage. '
         'The planned 3,000 questions produce up to 18,000 system-question outcomes, '
         'rather than 18,000 independent questions.')

headline = (f'On 622 questions scored by both SemPointer and full context, SemPointer used '
            f'{TOKEN_SAVING:.1f}% fewer active input tokens and {FULL_LATENCY_SAVING:.1f}% '
            f'lower reported per-query latency. Across the same 2,400 questions as '
            f'matched-budget BM25, it used essentially the same mean active input '
            f'budget and recorded {MATCH_LATENCY_SAVING:.1f}% lower reported latency. '
            'These measurements characterize an efficient evidence-access operating point '
            'for the evaluated checkpoint and hardware.')

measurement = ('Active tokens count the model input reported by the frozen evaluator. '
               'Reported latency includes generation and measured selector cost; shared '
               'ingestion and outer prompt preparation are accounted for separately. '
               'Means below are question-weighted, and savings are ratios of paired totals. '
               'Both workers operate concurrently, but the latency column measures individual '
               'requests rather than serving throughput. Timing percentages describe this '
               'recorded execution; no repeat-run timing interval has been estimated.')

full_text = (f'Within the 622-question full-context-eligible subset, mean active input falls '
             f'from {FULL["full_active_tokens_mean"]:,.1f} to '
             f'{FULL["sem_active_tokens_mean"]:,.1f} tokens. Mean reported latency falls '
             f'from {FULL["full_latency_ms_mean"]/1000:.3f} to '
             f'{FULL["sem_latency_ms_mean"]/1000:.3f} s, corresponding to '
             f'{TOKEN_SAVING:.1f}% and {FULL_LATENCY_SAVING:.1f}% reductions, respectively. '
             'Eligibility is defined by the original evaluator and its 16,384-token prompt '
             'limit; these resource estimates apply to the scored shared subset.')

matched_text = (f'Over all 2,400 questions in the eight complete subsets, SemPointer and '
                f'matched-budget BM25 use {SEM["active_tokens_mean"]:,.1f} and '
                f'{MATCH["active_tokens_mean"]:,.1f} active input tokens on average. '
                f'Reported latency is {SEM["latency_ms_mean"]/1000:.3f} versus '
                f'{MATCH["latency_ms_mean"]/1000:.3f} s, a '
                f'{MATCH_LATENCY_SAVING:.1f}% reduction at an effectively matched input budget.')

example_text = ('Table 2 presents illustrative matched-budget operating points selected after '
                'inspection of the recorded task inventory. On LoCoMo QA, SemPointer scores '
                '24.73 versus 24.69 with approximately 2,197 active tokens in each system. '
                'On HotpotQA it records 73.28 versus 68.13; on RepoBench-P it records '
                '24.69 versus 20.94. InfiniteBench key-value retrieval scores 80.00 in both '
                'systems. These examples span conversational memory, multi-hop QA, code '
                'completion, and retrieval. Each comparison uses the same scored question '
                'IDs and the task-specific adapted metric.')

interpretation = ('Together, the measurements support lower active-input and recorded request '
                  'cost relative to eligible full context, and lower recorded latency at '
                  'a matched retrieval budget. The quality examples document task-specific '
                  'operating points; their score intervals are reported directly, without '
                  'an equivalence or universal-improvement claim. The complete all-system '
                  'inventory accompanies these examples in Appendix A.')

inference = ('Quality scores are shown on a 0-100 display scale while retaining each task metric. '
             'Confidence intervals are the frozen evaluator\'s paired example-level bootstrap '
             'intervals, with no multiple-comparison adjustment. Shared source documents and '
             'conversations can introduce dependence between questions. Missing scores and '
             'recorded skips are distinguished from zero scores. These are adapted local '
             'benchmark metrics; code similarity is not executable-code correctness. '
             'The selected examples are illustrative rather than a prespecified confirmatory '
             'test. No heterogeneous overall accuracy is computed.')

provenance = ('Draft evidence status: complete combined-report members were read from an '
              'incomplete archive download. All displayed counts reconcile with the saved '
              'metrics, but full raw-database validation remains pending. This T4/FP16 '
              'run is separate from the bounded-interface study in the existing manuscript.')

resource_rows = [
    ['Matched retrieval budget', 2400, 'SemPointer', SEM['active_tokens_mean'], SEM['latency_ms_mean']/1000],
    ['Matched retrieval budget', 2400, 'Matched-budget BM25', MATCH['active_tokens_mean'], MATCH['latency_ms_mean']/1000],
    ['Eligible full context', 622, 'SemPointer', FULL['sem_active_tokens_mean'], FULL['sem_latency_ms_mean']/1000],
    ['Eligible full context', 622, 'Full context', FULL['full_active_tokens_mean'], FULL['full_latency_ms_mean']/1000],
]

def interval(p):
    low, high = p['score_difference_ci95']
    return f'[{100*low:+.2f}, {100*high:+.2f}]'

def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])

quality_rows = [[r['name'], r['metric'], r['n'], f'{100*r["sem"]["score_mean"]:.2f}',
                 f'{100*r["baseline"]["score_mean"]:.2f}', f'{100*r["pair"]["score_difference"]:+.2f}',
                 interval(r['pair'])] for r in examples]
resource_display = [[r[0], r[1], r[2], f'{r[3]:,.1f}', f'{r[4]:.3f}'] for r in resource_rows]

markdown = ['# Resource-Efficient Evidence Access: Kaggle Evaluation', '',
            '> ' + provenance, '', '## Supported empirical claim', '', headline, '',
            '## Evaluation setting', '', scope, '', measurement, '',
            '## Resource use on paired question sets', '', full_text, '', matched_text, '',
            '**Table 1. Resource measurements on shared scored question IDs.**', '',
            md_table(['Comparison', 'Questions', 'System', 'Mean active tokens', 'Reported latency (s)'], resource_display), '',
            'The two comparison blocks use different eligible question sets; each within-block comparison is paired.', '',
            '## Illustrative matched-budget quality results', '', example_text, '',
            '**Table 2. Selected task-specific operating points. Scores and paired differences use a 0-100 scale.**', '',
            md_table(['Task', 'Metric', 'Paired n', 'SemPointer', 'Matched BM25', 'Difference', 'Paired 95% interval'], quality_rows), '',
            inference, '', '## Empirical interpretation', '', interpretation, '',
            '## Appendix A. Complete results', '',
            'The complete task scores below retain every baseline and all selected task groups. Counts in parentheses are scored-question counts. A dash means no scored result. Skips and unrecorded work are given in the accompanying CSV.', '']
for family in sorted(r['family'] for r in D['coverage']):
    markdown += [f'### {family}', '', md_table(['Task', 'Metric', 'Selected n'] + LABELS, [
        [t['task'], t['metric'], t['selected']] + [
            f'{100*t["systems"][s]["score_mean"]:.2f} ({t["systems"][s]["n_scored"]})'
            if t['systems'].get(s, {}).get('score_mean') is not None else '-'
            for s in SYSTEMS]
        for t in D['tasks'] if t['family'] == family]), '']
markdown += ['## Appendix B. Resource accounting for every system', '',
             'The first five systems cover the same 2,400 questions. Full context covers 622 and its all-question raw means are not a matched comparison to the other rows.', '',
             md_table(['System', 'Scored n', 'Active tokens', 'Reported latency (s)', 'Generation (s)', 'TTFT (s)', 'Peak memory (reported MB)'], [
                 [r['label'], r['n'], f'{r["active_tokens_mean"]:.1f}', f'{r["latency_ms_mean"]/1000:.3f}',
                  f'{r["generation_ms_mean"]/1000:.3f}', f'{r["ttft_ms_mean"]/1000:.3f}', f'{r["peak_vram_mb_mean"]:.1f}']
                 for r in D['resources']]), '',
             '## Reproducibility', '',
             f'Run ID: `{D["run_id"]}`. Source metrics SHA256: `{D["source_metrics_sha256"]}`.', '',
             'The accompanying `all_task_scores.csv` retains 828 system/task rows, including all unrecorded cells. '
             '`paired_comparisons.csv` retains all 605 paired task/control comparisons. '
             'CSV score means remain on the original 0-1 scale. '
             'The source snapshot is preserved in `report_data.json`.']
(HERE / 'results_manuscript.md').write_text('\n'.join(markdown) + '\n')

def tex_escape(s):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                    '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(s))

def tex_paragraph(text):
    escaped = tex_escape(text)
    escaped = re.sub(r'\b[0-9a-f]{40,64}\b', lambda m: r'\texttt{' + r'\allowbreak{}'.join(textwrap.wrap(m.group(),8)) + '}', escaped)
    return escaped + '\n\n'

tex = [r'% Insert with \input; requires booktabs. All values derive from the saved Kaggle report.',
       r'% Draft evidence status: raw archive validation pending. See manuscript notes.',
       r'\section{Kaggle Evaluation: Resource Use and Matched-Budget Operating Points}',
       tex_paragraph(scope), tex_paragraph(measurement),
       r'\paragraph{Resource use.}', tex_paragraph(full_text), tex_paragraph(matched_text),
       r'\begin{table}[t]\centering\small\setlength{\tabcolsep}{3pt}',
       r'\caption{Resource measurements on shared scored question sets. The two comparison blocks use different eligible sets.}',
       r'\label{tab:kaggle_resources}', r'\begin{tabular}{lrrr}\toprule',
       r'System & $n$ & Active tokens & Latency (s) \\ \midrule']
for i, r in enumerate(resource_rows):
    if i == 2:
        tex.append(r'\midrule')
    tex.append(f'{tex_escape(r[2])} & {r[1]} & {r[3]:,.1f} & {r[4]:.3f}' + r' \\')
tex += [r'\bottomrule\end{tabular}\end{table}', r'\paragraph{Illustrative matched-budget comparisons.}',
        tex_paragraph(example_text), r'\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3pt}',
        r'\caption{Illustrative matched-budget quality results selected after inspection. Scores are task-specific and displayed on a 0--100 scale. The interval is the paired difference interval.}',
        r'\label{tab:kaggle_examples}', r'\begin{tabular}{lrrrrl}\toprule',
        r'Task & $n$ & SemPointer & Matched BM25 & $\Delta$ & 95\% interval \\ \midrule']
for r in quality_rows:
    tex.append(' & '.join(tex_escape(x) for x in [r[0], r[2], r[3], r[4], r[5], r[6]]) + r' \\')
tex += [r'\bottomrule\end{tabular}\end{table}', tex_paragraph(inference),
        r'\paragraph{Interpretation.}', tex_paragraph(interpretation),
        r'% Include the complete result inventory and all controls as an appendix or supplement.',
        r'% This fragment concerns the T4 run only; do not merge its percentages with other runs.']
(HERE / 'results_section.tex').write_text('\n'.join(tex) + '\n')

appendix_tex = [r'\clearpage\appendix\begin{landscape}',
                r'\section{Complete Kaggle Score Inventory}',
                tex_paragraph('All 138 selected task groups and all six systems are retained. Scores use a 0-100 display scale; scored n follows in parentheses. A dash indicates no scored result, rather than a zero score. Full-context coverage may differ. Per-cell skips and unrecorded counts are retained in the CSV.'),
                r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}',
                r'\begin{longtable}{@{}p{19mm}p{52mm}p{25mm}rrrrrrr@{}}',
                r'\toprule Dataset & Task & Metric & Selected $n$ & SemPointer & Pointer off & BM25-2 & BM25-4 & Matched BM25 & Full context \\ \midrule\endfirsthead',
                r'\toprule Dataset & Task & Metric & Selected $n$ & SemPointer & Pointer off & BM25-2 & BM25-4 & Matched BM25 & Full context \\ \midrule\endhead',
                r'\bottomrule\endfoot']
for task in D['tasks']:
    cells = [task['family'],task['task'],task['metric'],task['selected']]
    for system in SYSTEMS:
        value=task['systems'].get(system,{})
        cells.append(f'{100*value["score_mean"]:.2f} ({value["n_scored"]})' if value.get('score_mean') is not None else '-')
    appendix_tex.append(' & '.join(tex_escape(c) for c in cells)+r' \\')
appendix_tex += [r'\end{longtable}\endgroup',r'\section{All-System Resource Accounting}',
                 tex_paragraph('The first five systems score the same 2,400 questions. Full context scores 622 and its raw averages describe that eligible subset. Table 1 supplies the paired full-context comparison.'),
                 r'\begin{tabular}{lrrrrrr}\toprule',
                 r'System & $n$ & Active tokens & Latency (s) & Generation (s) & TTFT (s) & Peak memory \\ \midrule']
for row in D['resources']:
    cells=[row['label'],row['n'],f'{row["active_tokens_mean"]:.1f}',f'{row["latency_ms_mean"]/1000:.3f}',f'{row["generation_ms_mean"]/1000:.3f}',f'{row["ttft_ms_mean"]/1000:.3f}',f'{row["peak_vram_mb_mean"]:.1f}']
    appendix_tex.append(' & '.join(tex_escape(c) for c in cells)+r' \\')
appendix_tex += [r'\bottomrule\end{tabular}',tex_paragraph('Memory retains the source report\'s MB units. All 605 paired task/control comparisons accompany the complete score table as a CSV.'),r'\end{landscape}']
(HERE/'results_appendix.tex').write_text('\n'.join(appendix_tex)+'\n')
(HERE/'results_document.tex').write_text(r'''\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{booktabs,longtable,pdflscape}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{3em}
\begin{document}
\begin{center}\Large\bfseries SemPointer: Kaggle Evaluation\end{center}
\noindent\textit{Draft evidence status: combined report snapshot; full raw archive validation pending.}
\input{results_section.tex}
\input{results_appendix.tex}
\end{document}
''')

# The PDF contains the main section and the full score inventory in one document.
PDF.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitlePaper', fontName='Times-Bold', fontSize=19, leading=23, spaceAfter=12))
styles.add(ParagraphStyle(name='BodyPaper', fontName='Times-Roman', fontSize=10, leading=13.4, spaceAfter=8))
styles.add(ParagraphStyle(name='HeadingPaper', fontName='Times-Bold', fontSize=12, leading=15, spaceBefore=10, spaceAfter=7))
styles.add(ParagraphStyle(name='CaptionPaper', fontName='Times-Roman', fontSize=8.5, leading=11, spaceAfter=8))
styles.add(ParagraphStyle(name='SmallPaper', fontName='Times-Roman', fontSize=8, leading=10.5, spaceAfter=8, textColor=colors.HexColor('#46505b')))
styles.add(ParagraphStyle(name='TablePaper', fontName='Helvetica', fontSize=8, leading=10))
styles.add(ParagraphStyle(name='AppendixCell', fontName='Helvetica', fontSize=7.5, leading=9.5))
styles.add(ParagraphStyle(name='AppendixHead', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5))
for name, filename in [('PaperSerif','LiberationSerif-Regular.ttf'),('PaperSerifBold','LiberationSerif-Bold.ttf'),
                       ('PaperSans','LiberationSans-Regular.ttf'),('PaperSansBold','LiberationSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, '/usr/share/fonts/liberation/' + filename))
font_map={'Times-Roman':'PaperSerif','Times-Bold':'PaperSerifBold','Helvetica':'PaperSans','Helvetica-Bold':'PaperSansBold'}
for name in ['TitlePaper','BodyPaper','HeadingPaper','CaptionPaper','SmallPaper','TablePaper','AppendixCell','AppendixHead']:
    styles[name].fontName=font_map[styles[name].fontName]

def P(text, style='BodyPaper'):
    return Paragraph(escape(str(text)), styles[style])

def table_pdf(headers, rows, widths, appendix=False):
    cell = 'AppendixCell' if appendix else 'TablePaper'
    head = 'AppendixHead' if appendix else 'TablePaper'
    matrix = [[P(h, head) for h in headers]] + [[P(c, cell) for c in row] for row in rows]
    table = Table(matrix, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), .7, colors.HexColor('#233342')),
        ('LINEBELOW', (0,0), (-1,0), .5, colors.HexColor('#687582')),
        ('LINEBELOW', (0,-1), (-1,-1), .7, colors.HexColor('#233342')),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eef2f5')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4 if not appendix else 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4 if not appendix else 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    return table

def footer(canvas, doc):
    width, height = canvas._pagesize
    canvas.saveState()
    canvas.setFont('PaperSans', 7)
    canvas.setFillColor(colors.HexColor('#55616c'))
    canvas.drawString(18*mm, 12*mm, 'SemPointer | Kaggle report snapshot | Draft - raw archive validation pending')
    canvas.drawRightString(width-18*mm, 12*mm, str(doc.page))
    canvas.restoreState()

portrait = A4
wide = landscape(A4)
doc = BaseDocTemplate(str(PDF), pagesize=portrait, leftMargin=18*mm, rightMargin=18*mm,
                      topMargin=17*mm, bottomMargin=20*mm,
                      title='SemPointer: Resource-Efficient Evidence Access - Kaggle Evaluation', author='SemPointer research project')
doc.addPageTemplates([
    PageTemplate(id='main', pagesize=portrait, frames=[Frame(18*mm,20*mm,174*mm,260*mm,id='mainframe',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer),
    PageTemplate(id='appendix', pagesize=wide, frames=[Frame(18*mm,20*mm,261*mm,173*mm,id='wideframe',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer),
])
story = [P('Resource-Efficient Evidence Access: Kaggle Evaluation', 'TitlePaper'),
         P(provenance, 'SmallPaper'), P(headline), P('Evaluation setting', 'HeadingPaper'),
         P(scope), P(measurement), P('Resource measurements', 'HeadingPaper'), P(full_text), P(matched_text),
         table_pdf(['Comparison','Paired n','System','Active tokens','Latency (s)'],resource_display,[42*mm,16*mm,48*mm,35*mm,33*mm]),
         Spacer(1,5*mm),P('Table 1. Question-weighted resource measurements. Comparisons are paired within each block; active tokens and measured per-query latency retain the frozen evaluator definitions.','CaptionPaper'),
         PageBreak(),P('Illustrative matched-budget quality results','HeadingPaper'),P(example_text)]
compact_quality = [[r[0],r[2],r[3],r[4],r[5],r[6]] for r in quality_rows]
story += [table_pdf(['Task','Paired n','SemPointer','Matched BM25','Difference','Paired 95% interval'],compact_quality,[47*mm,14*mm,23*mm,25*mm,19*mm,46*mm]),
          Spacer(1,5*mm),P('Table 2. Adapted F1 for LoCoMo, HotpotQA, and Qasper; code similarity for RepoBench-P; containment for KV retrieval. Scores and differences use a 0-100 display scale. Examples were selected after inspection.','CaptionPaper'),
          P(inference),P('Empirical interpretation','HeadingPaper'),P(interpretation),
          NextPageTemplate('appendix'),PageBreak(),P('Appendix A. Complete all-system score inventory','HeadingPaper'),
          P('All 138 selected task groups are retained, including five groups without a recorded score. Each score is followed by its scored n in parentheses. A dash means no scored result; it does not mean zero. Skip and unrecorded counts are retained in the CSV files. Full-context cells often describe smaller subsets.','CaptionPaper')]

full_rows = []
for t in D['tasks']:
    cells = [t['family'],t['task'],t['metric'],t['selected']]
    for s in SYSTEMS:
        value=t['systems'].get(s,{})
        cells.append(f'{100*value["score_mean"]:.2f} ({value["n_scored"]})' if value.get('score_mean') is not None else '-')
    full_rows.append(cells)
story.append(table_pdf(['Dataset','Task','Metric','Selected n','SemPointer','Pointer off','BM25-2','BM25-4','Matched BM25','Full context'],
                       full_rows,[22*mm,58*mm,29*mm,14*mm,23*mm,23*mm,23*mm,23*mm,23*mm,23*mm],True))
story += [PageBreak(),P('Appendix B. Coverage and resource accounting','HeadingPaper'),
          table_pdf(['Dataset','Selected questions','Scored outcomes','Skipped outcomes','Unrecorded outcomes','Status'],[
              [r['family'],r['selected'],r.get('scored',0),r.get('skipped',0),r['selected']*6-r['recorded'],'Complete' if r['complete'] else 'Partial']
              for r in D['coverage']],[48*mm,34*mm,42*mm,42*mm,43*mm,36*mm]),Spacer(1,6*mm),
          P('Resource averages. The first five systems score the same 2,400 questions. Full context scores 622; its raw means below describe that eligible subset. Table 1 supplies the matched full-context comparison.','CaptionPaper'),
          table_pdf(['System','Scored n','Active tokens','Latency (s)','Generation (s)','TTFT (s)','Peak memory*'],[
              [r['label'],r['n'],f'{r["active_tokens_mean"]:.1f}',f'{r["latency_ms_mean"]/1000:.3f}',f'{r["generation_ms_mean"]/1000:.3f}',f'{r["ttft_ms_mean"]/1000:.3f}',f'{r["peak_vram_mb_mean"]:.1f}']
              for r in D['resources']],[58*mm,23*mm,38*mm,34*mm,34*mm,31*mm,27*mm]),Spacer(1,5*mm),
          P('* Memory retains the source report\'s MB units and is the mean of recorded per-question peak measurements. CSV files preserve all 828 system/task cells and all 605 paired task/control comparisons. Source metrics SHA256: '+D['source_metrics_sha256'],'CaptionPaper')]
doc.build(story)

for name in ['all_task_scores.csv','paired_comparisons.csv','report_data.json']:
    shutil.copy2(SOURCE/name,HERE/name)
shutil.copy2(SOURCE/'original_metrics.json',HERE/'original_metrics.json')
report_configuration=Path('/tmp/sempointer_latest_report_review/test/config.json')
if report_configuration.exists():
    shutil.copy2(report_configuration,HERE/'source_report_configuration.json')

readme = f'''# Kaggle manuscript results package

- `results_manuscript.md`: results narrative, paired resource tables, illustrative quality examples, and complete all-system score appendix.
- `results_section.tex`: manuscript fragment requiring booktabs. The complete appendix must accompany the selected examples.
- `results_appendix.tex`: complete score and resource tables; requires longtable and pdflscape.
- `results_document.tex`: standalone wrapper that includes the results section and complete appendix.
- PDF: `{PDF}`.
- `all_task_scores.csv`: all 828 system/task cells, original 0-1 score means, skips, and unrecorded counts.
- `paired_comparisons.csv`: all 605 paired task/control comparisons.
- `report_data.json`: preserved report snapshot and frozen identifiers.

Supported headline: lower active input and reported request cost than eligible full context, plus lower recorded latency at a matched retrieval budget, for this checkpoint/hardware configuration.

The main quality table consists of explicitly post-hoc illustrative examples. It does not establish score equivalence, statistical superiority, or an isolated causal pointer effect. The complete appendix remains part of this package.

Draft evidence status: source archive download incomplete; full raw-database validation pending. These T4/FP16 measurements are separate from the existing bounded-interface experiment. The matched full-context estimates are reconstructed from saved per-task paired differences and the full-context means, with identical total n={FULL['n']}.
'''
(HERE/'README.md').write_text(readme)

reader=PdfReader(str(PDF))
all_text='\n'.join(page.extract_text() or '' for page in reader.pages)
assert len(reader.pages)>=4
assert '76.7%' in all_text and '61.3%' in all_text and '8.9%' in all_text
assert 'Appendix A.' in all_text and 'Appendix B.' in all_text
assert 'Code similarity' in all_text or 'code similarity' in all_text
for family in COMPLETE:
    assert family in all_text
for t in D['tasks']:
    assert t['task'] in all_text, t['task']
assert len(list(csv.DictReader((HERE/'all_task_scores.csv').open())))==828

verification = {'run_id':D['run_id'],'source_metrics_sha256':D['source_metrics_sha256'],
                'selected_task_groups':len(D['tasks']),'system_task_cells':828,'paired_comparisons':len(D['paired']),
                'matched_budget_questions':SEM['n'],'full_context_paired_questions':FULL['n'],
                'active_token_reduction_vs_full_percent':TOKEN_SAVING,'reported_latency_reduction_vs_full_percent':FULL_LATENCY_SAVING,
                'reported_latency_reduction_vs_matched_percent':MATCH_LATENCY_SAVING,
                'quality_examples':len(examples),'pdf_pages':len(reader.pages),
                'raw_archive_validation':'pending; combined report snapshot only'}
(HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps({'pdf':str(PDF),**verification},indent=2))
