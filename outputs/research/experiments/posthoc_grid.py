"""Post hoc complete matched grid. Never selects a target-optimal method."""
import pathlib,json,datetime,time,itertools,hashlib
import numpy as np
import pandas as pd
from run_experiment import ROOT,OUT,load,representations,domains,budget_check,sha,savej,now
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
DEST=OUT/'posthoc_grid';DEST.mkdir(exist_ok=True)
def main():
 source=json.loads((OUT/'selection/freeze.json').read_text())
 for p,h in source['sha256'].items():assert sha(ROOT/p)==h
 freeze=DEST/'freeze.json'
 assert not freeze.exists(),'Do not overwrite diagnostic run'
 savej(freeze,{'timestamp_utc':now(),'status':'exploratory_posthoc_before_diagnostic_execution','primary_completion_sha256':sha(OUT/'raw_results/completion.json'),'protocol_sha256':sha(ROOT/'research/posthoc_diagnostic_01.md'),'code_sha256':sha(pathlib.Path(__file__)),'selected_target_scores_already_seen':True})
 psd,labels=load();xs=representations(psd,labels,0);records=[];start=time.monotonic()
 for repeat in [0,1]:
  train,test,d=domains(labels,repeat)
  for C,g in itertools.product([1,10,100],[.25,1,4]):
   for method,rep in [('temporal','beta_0'),('spatial','beta_1')]:
    budget_check();x=xs[rep];t=time.monotonic();model=make_pipeline(StandardScaler(),SVC(C=C,gamma=g/128,kernel='rbf',cache_size=256));model.fit(x[train],labels[train,0]);pred=model.predict(x[test]);elapsed=time.monotonic()-t;budget_check()
    name=f'repeat{repeat}_C{C}_g{g:g}_{method}.csv';p=DEST/name;assert not p.exists()
    df=pd.DataFrame(labels[test],columns=['texture','direction','speed','force','repeat']);df['domain']=d[test];df['prediction']=pred;df['correct']=(pred==labels[test,0]).astype(int);df.to_csv(p,index=False)
    records.append({'file':name,'training_repeat':repeat,'C':C,'gamma_multiplier':g,'method':method,'sha256':sha(p),'fit_and_predict_seconds':elapsed})
    savej(DEST/'manifest.json',records)
   print(json.dumps({'repeat':repeat,'C':C,'gamma':g,'completed':len(records),'elapsed_seconds':time.monotonic()-start}),flush=True)
 savej(DEST/'completion.json',{'timestamp_utc':now(),'files':len(records),'status':'complete','persistent_budget_elapsed_seconds':budget_check()})
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
