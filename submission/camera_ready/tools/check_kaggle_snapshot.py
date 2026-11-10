"""Reproduce paper headline values from saved Kaggle aggregates; no inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'kaggle_evidence'
D = json.loads((SOURCE / 'report_data.json').read_text())
config = json.loads((SOURCE / 'source_report_configuration.json').read_text())
complete = {r['family'] for r in D['coverage'] if r['complete']}
assert len(complete) == 8
assert config['run']['precision'] == 'nf4_fp16'
assert config['run']['source_bundle_run_id'] == '8ef26bfd68509bae3b6b'
sem = next(r for r in D['resources'] if r['system'] == 'sempointer')
matched = next(r for r in D['resources'] if r['system'] == 'bm25_matched_budget')
full = D['full_context_paired_resources']
assert sem['n'] == matched['n'] == 2400
assert full['n'] == 622

def directions(comparison):
    pairs = [p for p in D['paired'] if p['family'] in complete and p['comparison'] == comparison]
    result = Counter('higher' if p['score_difference'] > 0 else 'lower' if p['score_difference'] < 0 else 'tied' for p in pairs)
    return {'task_groups': len(pairs), **dict(result)}

match_quality = directions('sempointer_vs_bm25_matched_budget')
full_quality = directions('sempointer_vs_full_context')
assert match_quality == {'task_groups':125, 'higher':33, 'tied':54, 'lower':38}
assert full_quality == {'task_groups':68, 'higher':14, 'tied':20, 'lower':34}
rows = [
    ['SemPointer',2400,sem['active_tokens_mean'],sem['latency_ms_mean']/1000],
    ['Matched-budget BM25',2400,matched['active_tokens_mean'],matched['latency_ms_mean']/1000],
    ['SemPointer',622,full['sem_active_tokens_mean'],full['sem_latency_ms_mean']/1000],
    ['Full context',622,full['full_active_tokens_mean'],full['full_latency_ms_mean']/1000],
]
tex = (ROOT / 'main.tex').read_text()
for label,n,tokens,latency in rows:
    assert f'{label} & {n:,} & {tokens:,.1f} & {latency:.3f}' in tex
# Selected examples retain their actual metric, paired sample, and uncertainty.
quality_rows = []
for family, task, label in [
    ('locomo', 'qa', 'LoCoMo QA'),
    ('longbench', 'hotpotqa', 'HotpotQA'),
    ('longbench', 'qasper', 'Qasper'),
    ('longbench', 'repobench-p', 'RepoBench-P'),
    ('infinite', 'kv_retrieval', 'InfiniteBench KV'),
]:
    entry = next(t for t in D['tasks'] if (t['family'], t['task']) == (family, task))
    pair = next(p for p in D['paired'] if (p['family'], p['task']) == (family, task)
                and p['comparison'] == 'sempointer_vs_bm25_matched_budget')
    left, right = [entry['systems'][system] for system in ['sempointer', 'bm25_matched_budget']]
    n = pair['n_paired']
    assert left['n_scored'] == right['n_scored'] == n == entry['selected']
    a, b = 100*left['score_mean'], 100*right['score_mean']
    delta = 100*pair['score_difference']
    assert abs((a-b)-delta) < 1e-9
    lo, hi = [100*x for x in pair['score_difference_ci95']]
    delta_str = f'{delta:+.2f}' if delta else '0.00'
    lo_str = f'{lo:.2f}'
    hi_str = f'{hi:+.2f}' if hi else '0.00'
    expected = f'{label} & {n} & {a:.2f} & {b:.2f} & ${delta_str}$ & $[{lo_str},{hi_str}]$'
    assert expected in tex, expected
    quality_rows.append({'task':label, 'metric':entry['metric'], 'pairs':n,
                         'sempointer':a, 'matched_bm25':b, 'paired_difference':delta,
                         'paired_difference_ci95':[lo,hi]})
report = {
    'illustrative_quality_examples_selected_after_inspection': quality_rows,
    'quality_intervals': 'Frozen example-level paired bootstrap; no multiplicity adjustment; raw archive validation pending',
    'run': config['run']['source_bundle_run_id'],
    'raw_archive_validation': 'pending; report-snapshot reconciliation only',
    'snapshot_sha256': hashlib.sha256((SOURCE/'report_data.json').read_bytes()).hexdigest(),
    'table_rows_system_pairs_tokens_seconds': rows,
    'matched_latency_reduction_percent': 100*(1-sem['latency_ms_mean']/matched['latency_ms_mean']),
    'full_input_reduction_percent': 100*(1-full['sem_active_tokens_mean']/full['full_active_tokens_mean']),
    'full_latency_reduction_percent': 100*(1-full['sem_latency_ms_mean']/full['full_latency_ms_mean']),
    'matched_quality_task_group_directions': match_quality,
    'full_quality_task_group_directions': full_quality,
}
print(json.dumps(report,indent=2))
