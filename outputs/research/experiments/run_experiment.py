"""Source-only selection and separately invoked frozen target evaluation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['VECLIB_MAXIMUM_THREADS']='2'
import argparse,pathlib,json,hashlib,time,datetime,math,itertools,traceback,resource,sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
import joblib
from threadpoolctl import threadpool_limits
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments';FREQ=np.fft.rfftfreq(1024,1/6000)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def savej(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def cumulative_at(psd,edges):
 """Exact antiderivative of the piecewise-linear PSD, at per-row edges."""
 psd=np.asarray(psd,dtype=float);e=np.asarray(edges,dtype=float)
 if e.ndim==1:e=np.broadcast_to(e,(len(psd),len(e)))
 assert e.min()>=0 and e.max()<=FREQ[-1]
 cum=np.c_[np.zeros(len(psd)),np.cumsum((psd[:,:-1]+psd[:,1:])*.5*np.diff(FREQ),axis=1)]
 i=np.minimum(np.searchsorted(FREQ,e,side='right')-1,len(FREQ)-2);d=e-FREQ[i]
 p=np.take_along_axis(psd,i,1);s=(np.take_along_axis(psd,i+1,1)-p)/(FREQ[i+1]-FREQ[i])
 return np.take_along_axis(cum,i,1)+p*d+.5*s*d*d

def spectral_features(psd,speeds,beta=0,gain=False):
 edges=np.geomspace(40,800,129)[None,:]*(speeds[:,None]/40.)**beta
 masses=np.diff(cumulative_at(psd,edges),axis=1);assert masses.min()>-1e-10
 masses=np.maximum(masses,0);total=masses.sum(axis=1,keepdims=True);assert np.all(total>0)
 x=np.log(np.maximum(masses/total,1e-12))
 if gain:x=np.c_[x,np.log(total[:,0])]
 return x

def mel_features(psd):
 mel=lambda f:2595*np.log10(1+f/700)
 hz=lambda m:700*(10**(m/2595)-1)
 edges=hz(np.linspace(mel(20),mel(1200),130));w=np.zeros((128,len(FREQ)))
 # Piecewise-linear triangular filters sampled on the observed frequency grid.
 for i in range(128):w[i]=np.maximum(0,np.minimum((FREQ-edges[i])/(edges[i+1]-edges[i]),(edges[i+2]-FREQ)/(edges[i+2]-edges[i+1])))
 assert np.all(w.sum(1)>0)
 masses=psd@w.T*(FREQ[1]-FREQ[0]);return np.log(np.maximum(masses/masses.sum(1,keepdims=True),1e-12))

def representations(psd,labels,mode,perturb=0.):
 p=psd[:,mode,:];speed=labels[:,2]*(1+perturb)
 x={f'beta_{b:g}':spectral_features(p,speed,b) for b in [0,.25,.5,.75,1]}
 x['dual']=np.c_[x['beta_0'],x['beta_1']]
 x['metadata']=np.c_[x['beta_0'],speed/40,np.sin(np.deg2rad(labels[:,1])),np.cos(np.deg2rad(labels[:,1])),labels[:,3]/500]
 x['mel']=mel_features(p)
 x['temporal_gain']=spectral_features(p,speed,0,True);x['spatial_gain']=spectral_features(p,speed,1,True)
 return x

def domains(labels,train_repeat):
 texture,direction,speed,force,repeat=labels.T
 a=~np.isin(speed,[30,40,50]);b=~np.isin(direction,[0,90,180,270]);c=force!=500
 mask=(~a)&(~b)&(~c);train=mask&(repeat==train_repeat)
 names=[]
 for x,y,z in zip(a,b,c):names.append('_'.join(n for n,flag in [('speed',x),('direction',y),('load',z)] if flag) or 'id')
 names=np.array(names);test=~train
 assert not np.any(train&test);assert train.sum()==1416
 return train,test,names

def load():
 p=ROOT/'data/processed/all_psd.npz';z=np.load(p);psd=z['psd'];labels=z['labels'];assert psd.shape==(18880,2,513)
 q=json.loads((ROOT/'data/processed/quality.json').read_text());valid=np.array([x['valid'] for x in q]);assert sum(~valid)<=188
 # Do not silently operate a changed balanced design. Current dataset expected complete.
 if not valid.all():raise RuntimeError('Invalid acquisitions: inspect and record common-cell handling before inference')
 assert np.isfinite(psd).all() and (psd>=0).all();assert len(set(map(tuple,labels)))==len(labels)
 expected=set(itertools.product(range(118),range(0,360,45),[20,30,40,50,60],[500,1000],[0,1]))
 assert set(map(tuple,labels))==expected
 assert [x['id'] for x in q]==['_'.join(map(str,row)) for row in labels]
 return psd,labels

def macro(y,yp):return float(recall_score(y,yp,labels=np.arange(118),average='macro',zero_division=0))
def key(r):return (-r['cv_macro_accuracy'],r['C'],abs(math.log(r['gamma_multiplier'])),r['gamma_multiplier'],r.get('beta',0.))
def candidate_cv(x,labels,train,rep,C,gamma):
 scores=[];t=time.monotonic();fit_ids=[]
 for speed in [30,40,50]:
  tr=train&(labels[:,2]!=speed);va=train&(labels[:,2]==speed)
  model=make_pipeline(StandardScaler(),SVC(C=C,gamma=gamma/x.shape[1],kernel='rbf',cache_size=256))
  model.fit(x[tr],labels[tr,0]);pred=model.predict(x[va]);scores.append(macro(labels[va,0],pred))
  fit_ids.append({'held_speed':speed,'train_n':int(tr.sum()),'validation_n':int(va.sum()),'train_ids_sha256':hashlib.sha256(labels[tr].tobytes()).hexdigest(),'validation_ids_sha256':hashlib.sha256(labels[va].tobytes()).hexdigest()})
 return {'representation':rep,'C':C,'gamma_multiplier':gamma,'beta':float(rep[5:]) if rep.startswith('beta_') else 0.,'cv_fold_scores':scores,'cv_macro_accuracy':float(np.mean(scores)),'elapsed_seconds':time.monotonic()-t,'folds':fit_ids,'feature_dimension':x.shape[1]}

def context():
 files=[pathlib.Path(__file__),ROOT/'experiments/extract_features.py',ROOT/'experiments/audit_positions.py',ROOT/'research/protocol_v1.md',ROOT/'research/protocol_addendum_01.md',ROOT/'research/protocol_addendum_02.md',ROOT/'data/processed/all_psd.npz',ROOT/'data/processed/extraction_manifest.json',ROOT/'data/processed/quality.json',ROOT/'data/processed/position_audit.json']
 files += [ROOT/p for p in ['experiments/requirements-lock.txt','experiments/environment.json','experiments/verify_data.py','data/range_plan.json','data/download_manifest.json','data/processed/integrity_check.json']]
 return {str(p.relative_to(ROOT)):sha(p) for p in files}
def budget_check():
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
 if peak>3_000_000_000:raise MemoryError('3 GB peak process memory budget exceeded')
 clock=OUT/'selection/budget_clock.json'
 if not clock.exists():savej(clock,{'start_epoch':time.time(),'start_utc':now(),'limit_wall_seconds':5400})
 d=json.loads(clock.read_text())
 if time.time()-d['start_epoch']>d['limit_wall_seconds']:raise TimeoutError('90 minute persistent selection/evaluation wall deadline exceeded')
 return time.time()-d['start_epoch']
def select():
 start=time.monotonic();psd,labels=load();sel=OUT/'selection';sel.mkdir(parents=True,exist_ok=True);all_choices=[]
 ctx=context();cp=sel/'context.json'
 if cp.exists():assert json.loads(cp.read_text())==ctx,'Source-selection cache context changed; preserve old run and start a separate version'
 else:savej(cp,ctx)
 budget_check()
 savej(sel/'split_manifest.json',{'generated_utc':now(),'acquisitions':18880,'rows_sha256':hashlib.sha256(labels.tobytes()).hexdigest(),'by_repeat':{str(r):{'training':int(domains(labels,r)[0].sum()),'test_domains':dict(pd.Series(domains(labels,r)[2][domains(labels,r)[1]]).value_counts().astype(int).items())} for r in [0,1]}})
 # Both source-only selections are done before any held-out inference.
 for mode in [0,1]:
  xs=representations(psd,labels,mode)
  reps=list(xs) if mode==0 else ['beta_0','beta_1']
  for repeat in [0,1]:
   train,_,_=domains(labels,repeat);rows=[]
   cache=sel/f'cv_mode{mode}_repeat{repeat}.json'
   if cache.exists():rows=json.loads(cache.read_text())
   completed={(r['representation'],r['C'],r['gamma_multiplier']) for r in rows}
   for rep in reps:
    for C,gamma in itertools.product([1,10,100],[.25,1,4]):
     if (rep,C,gamma) in completed:continue
     budget_check()
     row=candidate_cv(xs[rep],labels,train,rep,C,gamma);rows.append(row);savej(cache,rows);budget_check()
    print(json.dumps({'phase':'selection','mode':mode,'repeat':repeat,'representation':rep,'candidates':len(rows),'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
   methods={'temporal':['beta_0'],'spatial':['beta_1']}
   if mode==0:methods.update({'partial':[f'beta_{b:g}' for b in [0,.25,.5,.75,1]],'endpoint_union':['beta_0','beta_1'],'dual':['dual'],'metadata':['metadata'],'mel':['mel'],'temporal_gain':['temporal_gain'],'spatial_gain':['spatial_gain']})
   for method,allowed in methods.items():
    budget_check()
    best=sorted([r for r in rows if r['representation'] in allowed],key=key)[0];rep=best['representation'];x=xs[rep];t=time.monotonic()
    model=make_pipeline(StandardScaler(),SVC(C=best['C'],gamma=best['gamma_multiplier']/x.shape[1],kernel='rbf',cache_size=256));model.fit(x[train],labels[train,0]);fitseconds=time.monotonic()-t;budget_check()
    fname=f'mode{mode}_repeat{repeat}_{method}.joblib';joblib.dump(model,sel/fname)
    all_choices.append({'mode':mode,'training_repeat':repeat,'method':method,'selected':best,'fit_seconds':fitseconds,'support_vectors':int(len(model[-1].support_)),'training_ids_sha256':hashlib.sha256(labels[train].tobytes()).hexdigest(),'model_file':fname,'model_sha256':sha(sel/fname)})
 savej(sel/'selected_models.json',all_choices)
 files=[ROOT/p for p in context()]+[sel/'split_manifest.json',sel/'selected_models.json',sel/'context.json',sel/'budget_clock.json']
 files += sorted(sel.glob('cv_mode*.json'))
 savej(sel/'freeze.json',{'timestamp_utc':now(),'status':'sealed_before_target_inference','elapsed_seconds':time.monotonic()-start,'sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}})
 print('SOURCE_SELECTION_FROZEN',flush=True)

def evaluate():
 start=time.monotonic();sel=OUT/'selection';freeze=json.loads((sel/'freeze.json').read_text())
 for p,h in freeze['sha256'].items():assert sha(ROOT/p)==h,('frozen_file_changed',p)
 choices=json.loads((sel/'selected_models.json').read_text());psd,labels=load();pred_dir=OUT/'raw_results';pred_dir.mkdir(parents=True,exist_ok=True)
 manifest=[]
 for mode in [0,1]:
  xs=representations(psd,labels,mode)
  for choice in [c for c in choices if c['mode']==mode]:
   repeat=choice['training_repeat'];method=choice['method'];rep=choice['selected']['representation'];model_path=sel/choice['model_file'];assert sha(model_path)==choice['model_sha256'];model=joblib.load(model_path)
   train,test,d=domains(labels,repeat);x=xs[rep];filename=f'mode{mode}_repeat{repeat}_{method}.csv';path=pred_dir/filename
   if path.exists():raise RuntimeError('Refusing to overwrite formal predictions: '+filename)
   budget_check();t=time.monotonic();pred=model.predict(x[test]);elapsed=time.monotonic()-t
   df=pd.DataFrame(labels[test],columns=['texture','direction','speed','force','repeat']);df['domain']=d[test];df['prediction']=pred;df['correct']=(pred==labels[test,0]).astype(int);df.to_csv(path,index=False)
   rec={**choice,'output_file':filename,'output_sha256':sha(path),'prediction_seconds':elapsed,'test_n':int(test.sum()),'timestamp_utc':now(),'status':'complete'};manifest.append(rec);savej(pred_dir/'run_manifest.json',manifest)
   print(json.dumps({'phase':'evaluation','file':filename,'prediction_seconds':round(elapsed,2),'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
   budget_check()
 # Fixed metadata perturbations: selected spatial/partial models, primary repeat only.
 base_choices=[c for c in choices if c['mode']==0 and c['training_repeat']==0 and c['method'] in ['spatial','partial']]
 for delta in [-.10,-.05,.05,.10]:
  budget_check();xs=representations(psd,labels,0,delta);_,test,d=domains(labels,0)
  for c in base_choices:
   budget_check();model=joblib.load(sel/c['model_file']);t=time.monotonic();pred=model.predict(xs[c['selected']['representation']][test]);elapsed=time.monotonic()-t
   name=f'perturb_{delta:+.2f}_{c["method"]}.csv';p=pred_dir/name
   if p.exists():raise RuntimeError('Refusing overwrite '+name)
   df=pd.DataFrame(labels[test],columns=['texture','direction','speed','force','repeat']);df['domain']=d[test];df['prediction']=pred;df['correct']=(pred==labels[test,0]).astype(int);df.to_csv(p,index=False)
   manifest.append({'mode':0,'training_repeat':0,'method':c['method'],'speed_error':delta,'synthetic_metadata_perturbation':True,'output_file':name,'output_sha256':sha(p),'prediction_seconds':elapsed,'status':'complete','timestamp_utc':now()});savej(pred_dir/'run_manifest.json',manifest)
 savej(pred_dir/'completion.json',{'status':'complete','timestamp_utc':now(),'selection_freeze_sha256':sha(sel/'freeze.json'),'selection_plus_evaluation_seconds':freeze['elapsed_seconds']+time.monotonic()-start,'prediction_files':len(manifest),'persistent_budget_elapsed_seconds':budget_check()})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['select','evaluate']);a=ap.parse_args()
 try:
  with threadpool_limits(limits=2):select() if a.phase=='select' else evaluate()
 except Exception as e:
  p=OUT/'failures.jsonl'
  with p.open('a') as f:f.write(json.dumps({'timestamp_utc':now(),'phase':a.phase,'error':repr(e),'traceback':traceback.format_exc()})+'\n')
  raise
 finally:
  r=resource.getrusage(resource.RUSAGE_SELF)
  with (OUT/'resource_usage.jsonl').open('a') as f:f.write(json.dumps({'phase':a.phase,'timestamp_utc':now(),'cpu_user_seconds':r.ru_utime,'cpu_system_seconds':r.ru_stime,'peak_rss_bytes':r.ru_maxrss*(1 if sys.platform=='darwin' else 1024)})+'\n')
