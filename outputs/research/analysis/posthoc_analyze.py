import pathlib,json,hashlib
import pandas as pd
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];D=ROOT/'experiments/posthoc_grid';OUT=ROOT/'analysis'
assert json.loads((D/'completion.json').read_text())['status']=='complete'
manifest=json.loads((D/'manifest.json').read_text());assert len(manifest)==36
rows=[];values={}
for m in manifest:
 p=D/m['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256'];df=pd.read_csv(p)
 for domain in sorted(df.domain.unique()):
  v=df[df.domain==domain].groupby('texture').correct.mean().reindex(range(118)).to_numpy();assert np.isfinite(v).all()
  values[(m['training_repeat'],m['C'],m['gamma_multiplier'],m['method'],domain)]=v
  rows.append({**{k:m[k] for k in ['training_repeat','C','gamma_multiplier','method']},'domain':domain,'accuracy':v.mean()})
pd.DataFrame(rows).to_csv(OUT/'posthoc_grid_scores.csv',index=False)
rng=np.random.default_rng(20270912);idx=rng.integers(0,118,(10000,118));diffs=[]
for repeat in [0,1]:
 for C in [1,10,100]:
  for g in [.25,1,4]:
   for domain in sorted(df.domain.unique()):
    d=values[(repeat,C,g,'spatial',domain)]-values[(repeat,C,g,'temporal',domain)];ci=np.quantile(d[idx].mean(1),[.025,.975])
    diffs.append({'training_repeat':repeat,'C':C,'gamma_multiplier':g,'domain':domain,'difference_pp':100*d.mean(),'ci_low_pp':100*ci[0],'ci_high_pp':100*ci[1],'exploratory':True})
df=pd.DataFrame(diffs);df.to_csv(OUT/'posthoc_grid_contrasts.csv',index=False);s=df[df.domain=='speed'];summary={'status':'exploratory_posthoc','n_matched_pairs':len(s),'negative_pairs':int((s.difference_pp<0).sum()),'minimum_pp':s.difference_pp.min(),'maximum_pp':s.difference_pp.max(),'not_independent_replications':True}
(OUT/'posthoc_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
