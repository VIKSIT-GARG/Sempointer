"""Present the downloaded report snapshot; do not run or alter evaluation."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import shutil

HERE = Path(__file__).resolve().parent
SOURCE = Path('/tmp/sempointer_latest_report_review')
BUNDLE = HERE.parents[1] / 'experiments/modal_runs/ten_qwen35_v1/bundle'
SYSTEMS = ['sempointer', 'pointer_off', 'bm25_2', 'bm25_4', 'bm25_matched_budget', 'full_context']
LABELS = ['SemPointer', 'Pointer off', 'BM25-2', 'BM25-4', 'Matched-budget BM25', 'Full context']
data = json.loads((SOURCE / 'test/metrics.json').read_text())
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
config = json.loads((SOURCE / 'test/config.json').read_text())
assert config['run']['source_bundle_run_id'] == manifest['run_id']
selected = Counter((r['family'], r['task']) for r in manifest['rows'])
tasks = {(t['family'], t['task']): t for t in data['tasks']}
assert len(tasks) == len(data['tasks'])
assert set(tasks) <= set(selected)
rows = []
families = defaultdict(lambda: defaultdict(Counter))
for (family, task), n in sorted(selected.items()):
    row = {**tasks.get((family, task), {'family': family, 'task': task, 'metric': 'not started', 'systems': {}}), 'selected': n}
    row['recorded'] = sum(sum(s.get('status_counts', {}).values()) for s in row['systems'].values())
    assert row['recorded'] <= n * len(SYSTEMS)
    for system, value in row['systems'].items():
        assert sum(value['status_counts'].values()) <= n
        families[family][system].update(value['status_counts'])
    rows.append(row)

coverage = []
for family in sorted(families):
    by_system = families[family]
    count = Counter()
    for values in by_system.values():
        count.update(values)
    n_selected = sum(n for (f, _), n in selected.items() if f == family)
    complete = all(sum(by_system[s].values()) == n_selected for s in SYSTEMS)
    coverage.append({'family': family, 'selected': n_selected, **dict(count),
                     'recorded': sum(count.values()), 'complete': complete})
complete_families = {x['family'] for x in coverage if x['complete']}
complete_tasks = [t for t in rows if t['family'] in complete_families]
resources = []
resource_keys = ['active_tokens_mean', 'latency_ms_mean', 'generation_ms_mean',
                 'ttft_ms_mean', 'ingestion_ms_mean', 'peak_vram_mb_mean']
for system, label in zip(SYSTEMS, LABELS):
    values = [t['systems'].get(system, {}) for t in complete_tasks]
    n = sum(v.get('n_scored', 0) for v in values)
    averages = {}
    for key in resource_keys:
        observed = [v for v in values if v.get('n_scored', 0) and v.get(key) is not None]
        denominator = sum(v['n_scored'] for v in observed)
        averages[key] = sum(v[key] * v['n_scored'] for v in observed) / denominator if denominator else None
    resources.append({'system': system, 'label': label, 'n': n, **averages})

comparison_counts = []
for system, label in zip(SYSTEMS[1:], LABELS[1:]):
    pairs = [p for p in data['paired'] if p['family'] in complete_families and p['comparison'] == 'sempointer_vs_' + system]
    sign = Counter('higher' if p['score_difference'] > 1e-12 else 'lower' if p['score_difference'] < -1e-12 else 'tie' for p in pairs)
    interval = Counter('above_zero' if p['score_difference_ci95'][0] > 0 else 'below_zero' if p['score_difference_ci95'][1] < 0 else 'includes_zero' for p in pairs)
    comparison_counts.append({'system': system, 'label': label, 'tasks': len(pairs), **dict(sign), **dict(interval)})

full_pairs = [p for p in data['paired'] if p['family'] in complete_families and p['comparison'] == 'sempointer_vs_full_context']
full_n = sum(p['n_paired'] for p in full_pairs)
full_resources = resources[-1]
assert full_n == full_resources['n']
full_paired_resources = {'n': full_n}
for mean, difference in [('active_tokens_mean', 'active_tokens_difference'), ('latency_ms_mean', 'latency_ms_difference')]:
    delta = sum(p[difference] * p['n_paired'] for p in full_pairs) / full_n
    full_paired_resources['full_' + mean] = full_resources[mean]
    full_paired_resources['sem_' + mean] = full_resources[mean] + delta

observed_counts = Counter()
for row in coverage:
    observed_counts.update({k: row.get(k, 0) for k in ['scored', 'skipped', 'error']})
assert {k: v for k, v in observed_counts.items() if v} == data['status_counts']
assert sum(x['recorded'] for x in coverage) == 14853
assert len(complete_families) == 8
assert len(rows) == 138 and len(tasks) == 133
assert all(r['n'] == 2400 for r in resources[:5])

snapshot = {
    'run_id': manifest['run_id'], 'model_revision': manifest['model_revision'],
    'source_metrics_sha256': hashlib.sha256((SOURCE / 'test/metrics.json').read_bytes()).hexdigest(),
    'source_archive': '/home/viksit/Downloads/sempointer_kaggle_results(1).WQENSREL.tar.gz.part',
    'source_archive_state': 'Incomplete archive. Complete combined report members were readable; raw databases were not audited.',
    'status_counts': data['status_counts'], 'systems': SYSTEMS, 'labels': LABELS,
    'coverage': coverage, 'resources': resources, 'comparisons': comparison_counts,
    'full_context_paired_resources': full_paired_resources,
    'tasks': rows, 'paired': data['paired'], 'notes': data['notes'],
}
(HERE / 'report_data.json').write_text(json.dumps(snapshot, indent=2))
shutil.copy2(SOURCE / 'test/metrics.json', HERE / 'original_metrics.json')
shutil.copy2(SOURCE / 'test/REPORT.md', HERE / 'original_task_report.md')

score_fields = ['family', 'task', 'split', 'metric', 'protocol', 'selected', 'system', 'n_scored',
                'n_skipped', 'n_error', 'not_recorded', 'score_mean', 'score_ci95_low', 'score_ci95_high'] + resource_keys
with (HERE / 'all_task_scores.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=score_fields)
    writer.writeheader()
    for row in rows:
        for system in SYSTEMS:
            value = row['systems'].get(system, {})
            status = value.get('status_counts', {})
            out = {k: row.get(k, '') for k in ['family', 'task', 'split', 'metric', 'protocol', 'selected']}
            out.update({k: value.get(k, '') for k in resource_keys + ['n_scored', 'score_mean']})
            ci = value.get('score_ci95', ['', ''])
            out.update(system=system, n_scored=value.get('n_scored', 0), n_skipped=status.get('skipped', 0),
                       n_error=status.get('error', 0), not_recorded=row['selected'] - sum(status.values()),
                       score_ci95_low=ci[0], score_ci95_high=ci[1])
            writer.writerow(out)
with (HERE / 'paired_comparisons.csv').open('w', newline='') as f:
    fields = list(dict.fromkeys(k for row in data['paired'] for k in row))
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(data['paired'])

def score_cell(value):
    n = value.get('n_scored', 0)
    score = value.get('score_mean')
    skips = value.get('status_counts', {}).get('skipped', 0)
    return (f'{100 * score:.2f} (n={n})' if score is not None else '—') + (f'; skipped={skips}' if skips else '')

lines = [
    '# SemPointer: complete score and resource comparison', '',
    f"Frozen run `{manifest['run_id']}`; Qwen3.5-4B, NF4/FP16, two T4 GPUs.", '',
    '**Snapshot:** 13,055 scored outcomes, 1,798 skips, zero recorded errors. Eight datasets complete; RULER and SCROLLS partial. Development: 216 scored outcomes across three candidates. CWE cap audit is unfinished.', '',
    '**Source limitation:** the latest download is an incomplete `.part` archive. Its complete combined report members were readable, but the full archive and raw prediction databases have not been audited.', '',
    '## Reading the comparisons', '',
    '- All quality scores below are multiplied by 100. They are the reported adapted task metrics, not a universal accuracy measure.',
    '- Compare systems within the same task. Do not average different metrics into an overall quality score.',
    '- Every score includes its scored-question count. Full context often covers a smaller subset; compare its paired results rather than unrelated raw means.',
    '- A missing score is not zero. Recorded skips and unrecorded work are separate.',
    '- Task higher/lower/tie counts weight every task group equally; they are descriptive and not question-weighted benchmark wins. There are 69 BABILong task/length groups.',
    '- Confidence intervals are the evaluator’s existing example-level intervals, without multiple-comparison correction. Source/document clustering and tiny sample counts limit inference.', '',
    '## Dataset coverage', '',
    '| Dataset | Selected questions | Scored | Skipped | Unrecorded outcomes | Status |',
    '|---|---:|---:|---:|---:|---|',
]
for row in coverage:
    lines.append(f"| {row['family']} | {row['selected']} | {row.get('scored', 0)} | {row.get('skipped', 0)} | {row['selected'] * 6 - row['recorded']} | {'Complete' if row['complete'] else 'Partial'} |")
lines += ['', '## Task comparisons on the eight complete datasets', '',
          '| Baseline | SemPointer higher | Lower | Tied | Paired CI above zero | Below zero | Includes zero |',
          '|---|---:|---:|---:|---:|---:|---:|']
for row in comparison_counts:
    lines.append(f"| {row['label']} | {row.get('higher',0)} | {row.get('lower',0)} | {row.get('tie',0)} | {row.get('above_zero',0)} | {row.get('below_zero',0)} | {row.get('includes_zero',0)} |")
lines += ['', '## Resource use on the eight complete datasets', '',
          'Means are weighted by scored-question count. The first five systems cover the same 2,400 selected questions; full context covers only 622. Reported latency is generation plus measured selector cost. Shared ingestion and outer preparation are separate; this is not end-to-end wall time.', '',
          '| System | Scored questions | Mean active tokens | Mean latency (s) | Generation (s) | TTFT (s) | Mean per-question peak GPU memory (reported MB) |',
          '|---|---:|---:|---:|---:|---:|---:|']
for row in resources:
    lines.append(f"| {row['label']} | {row['n']} | {row['active_tokens_mean']:.1f} | {row['latency_ms_mean']/1000:.3f} | {row['generation_ms_mean']/1000:.3f} | {row['ttft_ms_mean']/1000:.3f} | {row['peak_vram_mb_mean']:.1f} |")
lines += ['', f"On the same {full_n} full-context-eligible questions, SemPointer averaged {full_paired_resources['sem_active_tokens_mean']:.1f} active tokens and {full_paired_resources['sem_latency_ms_mean']/1000:.3f} s reported latency; full context averaged {full_paired_resources['full_active_tokens_mean']:.1f} tokens and {full_paired_resources['full_latency_ms_mean']/1000:.3f} s. This demonstrates resource savings on that subset, but does not establish preserved quality: the task comparisons favor full context more often."]
lines += ['', '## Every selected task and every system', '',
          'Scores ×100; n is the scored-question count for that cell. All five unstarted task groups are included.', '']
for family in sorted(families):
    lines += [f'### {family}', '', '| Task | Metric | Selected questions | ' + ' | '.join(LABELS) + ' |',
              '|---|---|---:|' + '---:|' * len(SYSTEMS)]
    for row in rows:
        if row['family'] == family:
            lines.append(f"| {row['task']} | {row['metric']} | {row['selected']} | " + ' | '.join(score_cell(row['systems'].get(s, {})) for s in SYSTEMS) + ' |')
    lines.append('')
lines += ['## Every paired comparison', '',
          'Positive differences favor SemPointer. Score differences and their 95% intervals are on the 0–100 score scale. Negative token or latency differences mean SemPointer used less. Intervals are uncorrected, and all incomplete-dataset comparisons remain preliminary.', '',
          '| Dataset | Task | Comparison | Paired n | Score difference | 95% interval | Active token difference | Latency difference (s) |',
          '|---|---|---|---:|---:|---|---:|---:|']
for row in sorted(data['paired'], key=lambda p: (p['family'], p['task'], p['comparison'])):
    low, high = row['score_difference_ci95']
    lines.append(f"| {row['family']} | {row['task']} | {row['comparison']} | {row['n_paired']} | {100*row['score_difference']:+.2f} | [{100*low:+.2f}, {100*high:+.2f}] | {row['active_tokens_difference']:+.2f} | {row['latency_ms_difference']/1000:+.3f} |")
(HERE / 'complete_comparison.md').write_text('\n'.join(lines) + '\n')

template = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">
<title>SemPointer — all scores and comparisons</title>
<style>
:root{color-scheme:dark;font:15px/1.5 system-ui,sans-serif;background:#0e141e;color:#e5ecf5}*{box-sizing:border-box}body{margin:0;padding:28px;max-width:1800px;margin-inline:auto}h1{font-size:30px;margin:0 0 8px}h2{margin:28px 0 10px}p{max-width:1100px;color:#becbdb}a{color:#88c7ff}.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.card{background:#172232;border:1px solid #314258;border-radius:12px;padding:14px 20px;min-width:180px}.card strong{font-size:28px;display:block}.card span{color:#b3c3d8}.notice{border-left:3px solid #d5a84f;padding:8px 14px;background:#251f16;color:#efdbb6}.controls{position:relative;z-index:2;display:flex;gap:12px;align-items:end;flex-wrap:wrap;background:#0e141ef2;padding:14px 0;border-bottom:1px solid #334155}label{display:flex;flex-direction:column;gap:4px;font-size:13px;color:#becbdb}input,select,button{font:inherit;color:inherit;background:#192536;border:1px solid #41516b;border-radius:6px;padding:9px}input[type=search]{width:250px}button{cursor:pointer}button:hover{background:#263a54}.check{flex-direction:row;align-items:center;padding-bottom:10px}.scroll{overflow:auto;max-height:74vh;border:1px solid #334155;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;border-bottom:1px solid #29364a;text-align:right;vertical-align:top;white-space:nowrap}th{background:#1c2a3d;color:#e3efff;position:sticky;top:0;z-index:1}th:first-child,td:first-child{text-align:left}td:nth-child(2),th:nth-child(2){text-align:left}.sem{background:#12332e}.best{box-shadow:inset 0 -3px #71c8a9}.small{font-size:11px;color:#9eb2c9}.partial{color:#f2c379}.muted{color:#95a9be}.table-note{font-size:13px}.downloads{display:flex;gap:10px;flex-wrap:wrap}.details{margin:16px 0}.chart{background:#172232;border:1px solid #314258;border-radius:10px;padding:20px;margin:18px 0;max-width:1000px}.barrow{display:grid;grid-template-columns:180px 1fr 145px;align-items:center;gap:12px;margin:9px 0;font-size:13px}.track{height:18px;background:#27374c;border-radius:4px;overflow:hidden}.bar{height:100%;background:#8297b5}.bar.sembar{background:#6ad0aa}.rowname{overflow-wrap:anywhere}.legend{font-size:12px;color:#9eb2c9}summary{cursor:pointer;color:#9fd5ff}.metric-key{max-width:1100px;display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:8px}.metric-key div{padding:10px;background:#172232;border-radius:6px}@media(max-width:700px){body{padding:14px}.barrow{grid-template-columns:110px 1fr 95px;font-size:11px}.controls{position:static}input[type=search]{width:200px}.card{min-width:140px}}
</style>
<h1>SemPointer: every score, every comparison</h1>
<p>Qwen3.5-4B · NF4 weights / FP16 compute · two T4 GPUs · frozen run <code id="runid"></code>. Downloaded report snapshot from 10 November 2026.</p>
<div class="notice">Eight datasets are complete; RULER and SCROLLS are partial. The combined report was read from an incomplete archive download. Raw prediction databases and full archive integrity have not been audited.</div>
<div class="kpis"><div class="card"><strong>13,055</strong><span>scored outcomes</span></div><div class="card"><strong>1,798</strong><span>recorded skips</span></div><div class="card"><strong>0</strong><span>recorded errors</span></div><div class="card"><strong>8 / 10</strong><span>complete datasets</span></div><div class="card"><strong>133 + 5</strong><span>recorded / unstarted task groups</span></div></div>
<p>Quality scores are shown on a 0–100 scale. Compare systems within each task; different metrics are not pooled. A dash means no score, never zero. Full-context scores often cover fewer questions. Green underlining identifies the highest observed mean among the five retrieval systems; it does not establish a statistical win.</p>
<div class="downloads"><button data-download="scores">Download all task scores (CSV)</button><button data-download="pairs">Download paired comparisons (CSV)</button><button data-download="data">Download source data (JSON)</button></div>
<details class="details"><summary>Metrics, uncertainty, and measurement limits</summary><div class="metric-key"><div><b>qa_f1 / char_f1</b><br>Adapted token / character overlap F1.</div><div><b>rouge_l / char_rouge_l</b><br>Adapted longest-subsequence overlap.</div><div><b>choice / choice_set / exact</b><br>Adapted answer-choice or exact-answer matching.</div><div><b>all_containment / numeric_sequence</b><br>Task-specific retrieval or numeric matching; consult the frozen evaluator for details.</div><div><b>code_similarity</b><br>Adapted code similarity, not executable-code correctness.</div><div><b>Confidence intervals</b><br>Existing example-level intervals; no adjustment for many comparisons or source/document clustering. Tiny n can produce misleadingly narrow intervals.</div></div><p>Reported latency is generation plus selector cost. Shared ingestion, outer preparation, and parallel wall time are separate. GPU-memory values retain the source report's units. Full-context raw means should not be compared to all-question means when coverage differs.</p></details>
<h2>Completion by dataset</h2><div class="scroll" style="max-height:none"><table id="coverage"></table></div>
<h2>Resource use on the eight complete datasets</h2><p class="table-note">Question-weighted means. The first five systems use the same 2,400 questions; full context uses only 622 and is shown separately for context.</p><div class="scroll" style="max-height:none"><table id="resources"></table></div>
<p id="fullpaired" class="table-note"></p>
<h2>How often is SemPointer higher?</h2><p class="table-note">Descriptive counts across 125 task groups from the eight complete datasets. Every group gets one vote regardless of sample count; BABILong contributes 69 groups. These are not question-weighted dataset wins. Interval columns are exploratory and uncorrected.</p><div class="scroll" style="max-height:none"><table id="comparisons"></table></div>
<h2>Explore all tasks</h2>
<div class="controls"><label>Dataset<select id="family"><option value="">All datasets</option></select></label><label>Task search<input id="query" type="search" placeholder="e.g. hotpotqa, qa5/256k"></label><label>Display<select id="display"><option value="score_mean">Quality score ×100</option><option value="active_tokens_mean">Active tokens</option><option value="latency_ms_mean">Reported latency (s)</option><option value="generation_ms_mean">Generation time (s)</option><option value="ttft_ms_mean">Time to first token (s)</option><option value="ingestion_ms_mean">Shared ingestion (s)</option><option value="peak_vram_mb_mean">Peak GPU memory (reported MB)</option><option value="paired">Paired score differences</option></select></label><label>Paired baseline<select id="baseline"></select></label><label class="check"><input id="completeOnly" type="checkbox">Complete datasets only</label><button id="reset">Reset filters</button></div>
<p id="count" class="table-note"></p><div id="tasktable" class="scroll"></div>
<h2>Inspect one task</h2><label>Selected task<select id="charttask"></select></label><div id="chart" class="chart"></div>
<p class="legend">Each bar is the reported quality mean ×100. Sample counts are shown. For full context, a smaller sample count means the bar covers a different subset. Hover score-table cells for score confidence intervals. Paired differences use only common scored questions.</p>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('payload').textContent);
const $=id=>document.getElementById(id), esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=2)=>v==null?'—':Number(v).toLocaleString('en-US',{minimumFractionDigits:n,maximumFractionDigits:n});
const state={family:'',query:'',display:'score_mean',baseline:'bm25_matched_budget',completeOnly:false};
const complete=new Set(D.coverage.filter(x=>x.complete).map(x=>x.family));
function table(headers,body){return '<table><thead><tr>'+headers.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+body.join('')+'</tbody></table>'}
function tr(cells){return '<tr>'+cells.map(c=>'<td>'+c+'</td>').join('')+'</tr>'}
$('runid').textContent=D.run_id;
$('coverage').innerHTML=table(['Dataset','Selected questions','Scored','Skipped','Unrecorded outcomes','Status'],D.coverage.map(x=>tr([esc(x.family),x.selected,fmt(x.scored||0,0),fmt(x.skipped||0,0),fmt(x.selected*6-x.recorded,0),x.complete?'Complete':'<span class="partial">Partial</span>']))).replace(/^<table>|<\/table>$/g,'');
$('resources').innerHTML=table(['System','Scored n','Active tokens','Latency (s)','Generation (s)','TTFT (s)','Peak GPU memory'],D.resources.map(x=>tr([esc(x.label),x.n,fmt(x.active_tokens_mean,1),fmt(x.latency_ms_mean/1000,3),fmt(x.generation_ms_mean/1000,3),fmt(x.ttft_ms_mean/1000,3),fmt(x.peak_vram_mb_mean,1)]))).replace(/^<table>|<\/table>$/g,'');
const F=D.full_context_paired_resources;
$('fullpaired').textContent='On the same '+F.n+' full-context-eligible questions: SemPointer averaged '+fmt(F.sem_active_tokens_mean,1)+' active tokens and '+fmt(F.sem_latency_ms_mean/1000,3)+' s reported latency; full context averaged '+fmt(F.full_active_tokens_mean,1)+' tokens and '+fmt(F.full_latency_ms_mean/1000,3)+' s. Quality preservation is not established; paired task comparisons favor full context more often.';
$('comparisons').innerHTML=table(['Baseline','SemPointer higher','Lower','Tied','CI above zero','Below zero','Includes zero'],D.comparisons.map(x=>tr([esc(x.label),x.higher||0,x.lower||0,x.tie||0,x.above_zero||0,x.below_zero||0,x.includes_zero||0]))).replace(/^<table>|<\/table>$/g,'');
for(const x of D.coverage)$('family').add(new Option(x.family,x.family));
for(let i=1;i<D.systems.length;i++)$('baseline').add(new Option(D.labels[i],D.systems[i]));
$('baseline').value=state.baseline;
function filtered(){return D.tasks.filter(t=>(!state.family||t.family===state.family)&&(!state.completeOnly||complete.has(t.family))&&(t.family+' '+t.task+' '+t.metric).toLowerCase().includes(state.query.toLowerCase()))}
function render(){
 const rows=filtered();
 if(state.display==='paired'){
  const keys=new Set(rows.map(t=>t.family+'\t'+t.task));
  const pairs=D.paired.filter(p=>keys.has(p.family+'\t'+p.task)&&p.comparison==='sempointer_vs_'+state.baseline).sort((a,b)=>a.family.localeCompare(b.family)||a.task.localeCompare(b.task));
  $('count').textContent=pairs.length+' paired comparisons. Positive score differences favor SemPointer; negative token and latency differences mean less use. Intervals are uncorrected.';
  $('tasktable').innerHTML=table(['Dataset','Task','Metric','Paired n','Score difference','95% interval','Token difference','Latency difference (s)'],pairs.map(p=>tr([esc(p.family),esc(p.task),esc(p.metric),p.n_paired,fmt(100*p.score_difference),p.score_difference_ci95.map(x=>fmt(100*x)).join(' to '),fmt(p.active_tokens_difference),fmt(p.latency_ms_difference/1000,3)])));
 }else{
  $('count').textContent=rows.length+' of '+D.tasks.length+' selected task groups. Every system cell shows its scored n; skips are listed separately.';
  const scale=state.display==='score_mean'?100:state.display.endsWith('_ms_mean')?0.001:1;
  $('tasktable').innerHTML=table(['Dataset','Task','Metric','Selected n',...D.labels],rows.map(t=>{
   const comparable=D.systems.slice(0,5).map(s=>t.systems[s]?.[state.display]).filter(v=>v!=null);
   const best=state.display==='score_mean'&&comparable.length?Math.max(...comparable):null;
   return '<tr><td>'+esc(t.family)+'</td><td>'+esc(t.task)+'</td><td>'+esc(t.metric)+'</td><td>'+t.selected+'</td>'+D.systems.map((s,i)=>{
    const r=t.systems[s]||{}, skips=r.status_counts?.skipped||0, n=r.n_scored||0, value=r[state.display], ci=r.score_ci95;
    const title=ci?'Score 95% CI: '+ci.map(x=>fmt(100*x)).join(' to '):'No score interval';
    const cls=(s==='sempointer'?'sem ':'')+(i<5&&best!=null&&Math.abs(value-best)<1e-12?'best':'');
    return '<td class="'+cls+'" title="'+esc(title)+'">'+fmt(value==null?null:value*scale,state.display==='score_mean'?2:state.display.endsWith('_ms_mean')?3:1)+'<div class="small">n='+n+(skips?' · skips='+skips:'')+'</div></td>';
   }).join('')+'</tr>';
  }));
 }
 const previous=$('charttask').value;
 $('charttask').innerHTML='';
 for(const t of rows)$('charttask').add(new Option(t.family+' / '+t.task,t.family+'\t'+t.task));
 if(rows.some(t=>t.family+'\t'+t.task===previous))$('charttask').value=previous;
 chart();
}
function chart(){
 const t=D.tasks.find(t=>t.family+'\t'+t.task===$('charttask').value);
 if(!t){$('chart').textContent='No tasks match these filters.';return}
 $('chart').innerHTML='<b>'+esc(t.family+' / '+t.task)+'</b><div class="small">'+esc(t.metric)+' · '+t.selected+' selected questions</div>'+D.systems.map((s,i)=>{
  const r=t.systems[s]||{},value=r.score_mean;
  return '<div class="barrow"><div>'+esc(D.labels[i])+'</div><div class="track"><div class="bar '+(i===0?'sembar':'')+'" style="width:'+(value==null?0:Math.max(0,Math.min(100,value*100)))+'%"></div></div><div>'+fmt(value==null?null:value*100)+' · n='+(r.n_scored||0)+'</div></div>';
 }).join('');
}
for(const id of ['family','query','display','baseline','completeOnly'])$(id).addEventListener(id==='query'?'input':'change',()=>{state[id]=$(id).type==='checkbox'?$(id).checked:$(id).value;render()});
$('charttask').addEventListener('change',chart);
$('reset').addEventListener('click',()=>{Object.assign(state,{family:'',query:'',display:'score_mean',baseline:'bm25_matched_budget',completeOnly:false});for(const k of Object.keys(state))$(k).type==='checkbox'?$(k).checked=state[k]:$(k).value=state[k];render()});
function csv(rows){if(!rows.length)return '';const keys=Array.from(new Set(rows.flatMap(Object.keys))),quote=v=>'"'+String(v==null?'':Array.isArray(v)?JSON.stringify(v):v).replace(/"/g,'""')+'"';return [keys.map(quote).join(','),...rows.map(r=>keys.map(k=>quote(r[k])).join(','))].join('\r\n')}
document.querySelectorAll('[data-download]').forEach(b=>b.addEventListener('click',()=>{
 const kind=b.dataset.download;
 let text,name,type;
 if(kind==='data'){text=JSON.stringify(D,null,2);name='sempointer_report_data.json';type='application/json'}
 else if(kind==='pairs'){text=csv(D.paired);name='sempointer_paired_comparisons.csv';type='text/csv'}
 else{const rows=[];for(const t of D.tasks)for(const s of D.systems){const r=t.systems[s]||{};rows.push({family:t.family,task:t.task,metric:t.metric,selected:t.selected,system:s,n_scored:r.n_scored||0,n_skipped:r.status_counts?.skipped||0,not_recorded:t.selected-Object.values(r.status_counts||{}).reduce((a,b)=>a+b,0),score_mean:r.score_mean,score_ci95:r.score_ci95,active_tokens_mean:r.active_tokens_mean,latency_ms_mean:r.latency_ms_mean,generation_ms_mean:r.generation_ms_mean,ttft_ms_mean:r.ttft_ms_mean,ingestion_ms_mean:r.ingestion_ms_mean,peak_vram_mb_mean:r.peak_vram_mb_mean})}text=csv(rows);name='sempointer_all_task_scores.csv';type='text/csv'}
 const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}));
render();
</script></html>'''
payload = json.dumps(snapshot, ensure_ascii=False).replace('<', '\\u003c')
(HERE / 'comparison.html').write_text(template.replace('__PAYLOAD__', payload))
print(json.dumps({'output': str(HERE), 'selected_task_groups': len(rows), 'recorded_task_groups': len(tasks),
                  'paired_comparisons': len(data['paired']), 'complete_datasets': sorted(complete_families),
                  'status_counts': data['status_counts']}, indent=2))
