"""Record the outcome-blind disposition required by the frozen external QC gate."""
import pathlib,json,hashlib,datetime,collections
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
DATA=ROOT/'data/external'
OUT=ROOT/'research/external_validity_decision.json'
assert not OUT.exists(), 'A recorded validity decision may not be overwritten'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
quality=json.loads((DATA/'quality_audit.json').read_text())
rows=json.loads((DATA/'trial_integrity.json').read_text())
assert len(rows)==3600 and quality['invalid']==49 and quality['source_target_duplicate_group_count']==0
assert not (ROOT/'experiments/calibration/external/selected.json').exists()
assert not (ROOT/'experiments/cnn/external/selection.json').exists()
groups=collections.defaultdict(list)
for r in rows:
    if r['valid']:groups[r['normalized_content_sha256']].append(r)
duplicates=[]
for group in groups.values():
    group=sorted(group,key=lambda x:x['id'])
    duplicates.extend(r['id'] for r in group[1:])
assert duplicates==['03_35_083']
eligible=[r for r in rows if r['valid'] and r['id'] not in duplicates]
assert len(eligible)==3550
cells=collections.Counter((r['labels'][0],r['labels'][1]) for r in eligible)
assert len(cells)==36 and min(cells.values())>=93
assert len({r['normalized_content_sha256'] for r in eligible})==len(eligible)
for modulus in [3,5]:
    for fold in range(modulus):
        source=[r for r in eligible if r['labels'][1] in [30,35]]
        assert {r['labels'][0] for r in source if r['labels'][2]%modulus==fold}==set(range(12))
        assert {r['labels'][0] for r in source if r['labels'][2]%modulus!=fold}==set(range(12))
exclusions=[{'id':r['id'],'row_original':r['row_original'],'reasons':r['reasons']}
            for r in rows if not r['valid']]
exclusions += [{'id':r['id'],'row_original':r['row_original'],'reasons':['exact_duplicate_episode'],
                'retained_duplicate':'03_35_001'} for r in rows if r['id'] in duplicates]
def quantile(values):return dict(zip(['minimum','median','p95','maximum'],map(float,np.quantile(values,[0,.5,.95,1]))))
result={
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'decision':'accept_common_eligible_episodes_after_documented_QC_review_and_exact_deduplication',
 'decider':'parent research orchestrator, under the user-authorized local study',
 'stage':'before any external classifier fitting or classification outcomes',
 'rationale':'The original 1% value triggers review, not automatic relaxation of per-episode criteria. The observed 49/3600 quality exclusions (1.3611%) are retained. All 36 class-speed cells retain 93-100 episodes; no source-target duplicate is present. One same-source-speed duplicate would cross both modulo-3 and modulo-5 CV folds, so keep the lexicographically smallest ID and remove the other before model selection. This preserves all actual eligible unique observations without imputing recordings or selecting by accuracy.',
 'unchanged':['nonfinite/decreasing-time/duration rules','50ms window-gap cutoff','all filtering/window/feature settings','source30/35 target40','SVM/CNN grids and inner-fold formulas','external primary contrast and inference rule'],
 'protocol_disposition':'The frozen quality_audit.json remains needs_root_decision as an immutable record. This separate artifact supplies the required disposition. It does not rewrite the prior protocol or claim that its 1% alert was never crossed.',
 'planned_episodes':3600,'quality_excluded':49,'additional_exact_duplicates_excluded':1,'eligible_episodes':3550,
 'source_episodes':2354,'target_episodes':1196,
 'quality_exclusions_by_speed':{str(v):sum(not r['valid'] and r['labels'][1]==v for r in rows) for v in [30,35,40]},
 'eligible_by_speed':{str(v):sum(r['labels'][1]==v for r in eligible) for v in [30,35,40]},
 'eligible_by_class_speed':{f'{c:02}_{v}':n for (c,v),n in sorted(cells.items())},
 'duplicate_policy':'Among exact canonical numeric/time-content duplicates, retain the lexicographically smallest ID; same data, no choice based on quality or model outcomes beyond the common eligibility mask.',
 'additional_excluded_ids':duplicates,'exclusions':exclusions,
 'eligible_ids':[r['id'] for r in eligible],
 'eligible_original_rows':[r['row_original'] for r in eligible],
 'eligible_window_max_gap_ms':quantile([max(r['window_max_timestamp_gap_seconds'])*1000 for r in eligible]),
 'quality_excluded_window_max_gap_ms':quantile([max(r['window_max_timestamp_gap_seconds'])*1000 for r in rows if not r['valid']]),
 'limits':['Quality exclusion is uneven across speeds/classes; results condition on the eligible panel and do not establish missing-at-random sampling.',
           'One nominal dataset episode is an exact copy, not an independent acquisition. Other non-exact duplication and acquisition-session dependence are not ruled out.',
           'Extrema recurrence and repeated values are reported, but no unverified hardware clipping threshold or physical update rate is asserted.'],
 'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [DATA/'quality_audit.json',DATA/'trial_integrity.json',ROOT/'research/external_protocol.md',ROOT/'research/external_protocol.freeze.json']},
 'script_sha256':sha(__file__)
}
OUT.write_text(json.dumps(result,indent=2))
(ROOT/'research/external_validity_decision.freeze.json').write_text(json.dumps({'utc':result['utc'],'file':'external_validity_decision.json','sha256':sha(OUT)},indent=2))
print(json.dumps({k:result[k] for k in ['decision','eligible_episodes','source_episodes','target_episodes','eligible_by_speed']}))
