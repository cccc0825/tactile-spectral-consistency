"""Deterministic summaries of sealed acquisition-level predictions. No fitting."""
import pathlib,json,hashlib,datetime,zipfile,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
RAW=ROOT/'experiments/raw_results';OUT=ROOT/'analysis'
DOMAINS=['id','speed','direction','load','speed_direction_load','speed_direction','speed_load','direction_load']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 complete=json.loads((RAW/'completion.json').read_text());assert complete['status']=='complete'
 manifest=json.loads((RAW/'run_manifest.json').read_text());assert len(manifest)==30
 expected_runs={f'mode0_repeat{r}_{m}.csv' for r in [0,1] for m in ['temporal','spatial','partial','endpoint_union','dual','metadata','mel','temporal_gain','spatial_gain']}|{f'mode1_repeat{r}_{m}.csv' for r in [0,1] for m in ['temporal','spatial']}|{f'perturb_{d:+.2f}_{m}.csv' for d in [-.1,-.05,.05,.1] for m in ['spatial','partial']}
 assert {m['output_file'] for m in manifest}==expected_runs
 with zipfile.ZipFile(ROOT/'corpus/texture_list.xlsx') as z:
  ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
  ss=[''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
  meta={}
  for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row',ns)[1:]:
   cells={}
   for c in row.findall('s:c',ns):
    v=c.find('s:v',ns)
    if v is not None:cells[c.attrib['r'].rstrip('0123456789')]=ss[int(v.text)] if c.attrib.get('t')=='s' else v.text
   meta[int(cells['A'])]={'name':cells['B'].strip(),'category':cells['D'].strip()}
 assert len(meta)==118
 (OUT/'texture_metadata.json').write_text(json.dumps(meta,indent=2))
 rng=np.random.default_rng(20270912);indices=rng.integers(0,118,(10000,118))
 summaries=[];textures=[];arrays={};frames={};costs=[]
 for m in manifest:
  p=RAW/m['output_file'];assert sha(p)==m['output_sha256'];df=pd.read_csv(p);assert len(df)==17464 and not df.duplicated(['texture','direction','speed','force','repeat']).any()
  assert np.array_equal(df['correct'],(df['prediction']==df['texture']).astype(int))
  labels=np.load(ROOT/'data/processed/all_psd.npz')['labels']
  source=np.isin(labels[:,2],[30,40,50])&np.isin(labels[:,1],[0,90,180,270])&(labels[:,3]==500)&(labels[:,4]==m['training_repeat'])
  expected=set(map(tuple,labels[~source]));keys=['texture','direction','speed','force','repeat']
  assert set(map(tuple,df[keys].to_numpy()))==expected
  expected_domains=['_'.join(n for n,flag in [('speed',v not in [30,40,50]),('direction',d not in [0,90,180,270]),('load',f!=500)] if flag) or 'id' for _,d,v,f,_ in df[keys].to_numpy()]
  assert np.array_equal(df.domain.to_numpy(),expected_domains)
  frames[p.stem]=df
  for d in DOMAINS:
   sub=df[df.domain==d];x=sub.groupby('texture').correct.mean().reindex(range(118)).to_numpy();assert np.isfinite(x).all()
   ci=np.quantile(x[indices].mean(1),[.025,.975]);arrays[(p.stem,d)]=x
   summaries.append({'run':p.stem,'mode':m['mode'],'training_repeat':m['training_repeat'],'method':m['method'],'speed_error':m.get('speed_error',0),'domain':d,'n':len(sub),'specimens':118,'accuracy':x.mean(),'ci_low':ci[0],'ci_high':ci[1]})
   for t,v in enumerate(x):textures.append({'run':p.stem,'domain':d,'texture':t,'accuracy':v})
  costs.append({'run':p.stem,'predict_seconds':m['prediction_seconds'],'milliseconds_per_acquisition':1000*m['prediction_seconds']/len(df),'fit_seconds':m.get('fit_seconds',np.nan),'support_vectors':m.get('support_vectors',np.nan),'model_bytes':(ROOT/'experiments/selection'/m['model_file']).stat().st_size if 'model_file' in m else np.nan})
 pd.DataFrame(summaries).to_csv(OUT/'scores.csv',index=False);pd.DataFrame(textures).to_csv(OUT/'per_specimen_scores.csv',index=False);pd.DataFrame(costs).to_csv(OUT/'costs.csv',index=False)
 contrasts=[]
 for mode in [0,1]:
  for repeat in [0,1]:
   base=f'mode{mode}_repeat{repeat}_temporal'
   methods=['spatial'] if mode else ['spatial','partial','endpoint_union','dual','metadata','mel','temporal_gain','spatial_gain']
   for method in methods:
    run=f'mode{mode}_repeat{repeat}_{method}'
    for domain in DOMAINS:
     diff=arrays[(run,domain)]-arrays[(base,domain)];ci=np.quantile(diff[indices].mean(1),[.025,.975]);primary=mode==0 and repeat==0 and method=='spatial' and domain=='speed'
     contrasts.append({'mode':mode,'training_repeat':repeat,'method':method,'reference':'temporal','domain':domain,'difference_pp':100*diff.mean(),'ci_low_pp':100*ci[0],'ci_high_pp':100*ci[1],'positive_specimens':int(np.sum(diff>0)),'negative_specimens':int(np.sum(diff<0)),'equal_specimens':int(np.sum(diff==0)),'primary':primary})
 for repeat in [0,1]:
  for base in ['temporal','spatial']:
   for domain in DOMAINS:
    diff=arrays[(f'mode0_repeat{repeat}_{base}_gain',domain)]-arrays[(f'mode0_repeat{repeat}_{base}',domain)]
    ci=np.quantile(diff[indices].mean(1),[.025,.975])
    contrasts.append({'mode':0,'training_repeat':repeat,'method':base+'_gain','reference':base,'domain':domain,'difference_pp':100*diff.mean(),'ci_low_pp':100*ci[0],'ci_high_pp':100*ci[1],'positive_specimens':int(np.sum(diff>0)),'negative_specimens':int(np.sum(diff<0)),'equal_specimens':int(np.sum(diff==0)),'primary':False})
 pd.DataFrame(contrasts).to_csv(OUT/'contrasts.csv',index=False)
 # Additional descriptive summaries specified in the protocol, with distinct weighting.
 aggregate=[];categories=[];confusions=[]
 for name,df in frames.items():
  per=df.groupby('texture').correct.mean()
  aggregate.append({'run':name,'pooled_accuracy':df.correct.mean(),'specimen_macro_accuracy':per.mean(),'equal_domain_macro_accuracy':np.mean([arrays[(name,d)].mean() for d in DOMAINS]),'n':len(df)})
  for domain in DOMAINS:
   for cat in sorted(set(x['category'] for x in meta.values())):
    ids=[i for i,x in meta.items() if x['category']==cat];v=arrays[(name,domain)][ids]
    categories.append({'run':name,'domain':domain,'category':cat,'n_specimens':len(ids),'mean_specimen_accuracy':v.mean()})
  if name in ['mode0_repeat0_temporal','mode0_repeat0_spatial']:
   for (domain,true,pred),n in df.groupby(['domain','texture','prediction']).size().items():confusions.append({'run':name,'domain':domain,'true':int(true),'predicted':int(pred),'count':int(n)})
 pd.DataFrame(aggregate).to_csv(OUT/'aggregate_accuracy.csv',index=False);pd.DataFrame(categories).to_csv(OUT/'category_scores.csv',index=False);pd.DataFrame(confusions).to_csv(OUT/'confusion_counts.csv',index=False)
 # Every predeclared velocity, including both held-out endpoints; descriptive CIs.
 rows=[]
 for name,df in frames.items():
  for speed in [20,30,40,50,60]:
   sub=df[(df.speed==speed)&df.domain.isin(['id','speed'])]
   x=sub.groupby('texture').correct.mean().reindex(range(118)).to_numpy();ci=np.quantile(x[indices].mean(1),[.025,.975])
   rows.append({'run':name,'speed':speed,'n':len(sub),'accuracy':x.mean(),'ci_low':ci[0],'ci_high':ci[1]})
 pd.DataFrame(rows).to_csv(OUT/'speed_breakdown.csv',index=False)
 # All confusion entries and all per-specimen paired effects; deterministic extrema
 # are explicitly exploratory, not selected examples for model tuning.
 errors=[]
 for name in ['mode0_repeat0_temporal','mode0_repeat0_spatial']:
  sub=frames[name];sub=sub[(sub.domain=='speed')&(sub.correct==0)]
  for (true,pred),n in sub.groupby(['texture','prediction']).size().items():errors.append({'run':name,'true':int(true),'predicted':int(pred),'count':int(n)})
 pd.DataFrame(errors).to_csv(OUT/'speed_confusions.csv',index=False)
 d=arrays[('mode0_repeat0_spatial','speed')]-arrays[('mode0_repeat0_temporal','speed')]
 pd.DataFrame({'texture':range(118),'difference_pp':100*d}).to_csv(OUT/'all_specimen_speed_effects.csv',index=False)
 primary=next(x for x in contrasts if x['primary']);support=primary['difference_pp']>=3 and primary['ci_low_pp']>0
 report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'analysis_sha256':sha(pathlib.Path(__file__)),'raw_completion_sha256':sha(RAW/'completion.json'),'bootstrap_seed':20270912,'bootstrap_replicates':10000,'resampling_unit':'paired physical specimen, conditional on trained models and this panel; not new-material population inference','primary':primary,'prospective_support_rule_met':bool(support),'true_effect_at_least_3pp_supported_by_lower_bound':bool(primary['ci_low_pp']>=3),'primary_design_n_specimens':118,'all_negative_results_retained':True,'secondary_intervals':'descriptive, no familywise claims','source_files':{m['output_file']:m['output_sha256'] for m in manifest}}
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['primary'],indent=2));print('Prospective support rule:',support)
if __name__=='__main__':main()
