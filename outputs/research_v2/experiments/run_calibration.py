"""Protocol-locked source selection, frozen evaluation, and full run ledger."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import pathlib,json,hashlib,datetime,time,itertools,argparse,sys,resource,traceback
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from response_calibration import features_for_split,algebra_test

ROOT=pathlib.Path(__file__).resolve().parents[1]
METHODS=['temporal','spatial','corrected','pooled_whitening','mel','mfcc','corrected_weak','corrected_strong','wrong_order']
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2))
def score(y,p):return float(recall_score(y,p,average='macro',zero_division=0))
def key(x):return (-x['cv_mean'],x['C'],abs(np.log(x['g'])),x['g'])

def load_cluster():
    path=ROOT.parent/'research/data/processed/all_psd.npz';z=np.load(path)
    labels=z['labels'];p=z['psd'][:,0,:];f=z['frequency']
    configurations=[]
    y,d,v,force,r=labels.T
    a=~np.isin(v,[30,40,50]);b=~np.isin(d,[0,90,180,270]);c=force!=500
    domains=np.array(['_'.join(n for n,flag in [('speed',aa),('direction',bb),('load',cc)] if flag) or 'id' for aa,bb,cc in zip(a,b,c)])
    for rep in [0,1]:
        train=(~a)&(~b)&(~c)&(r==rep);assert train.sum()==1416
        folds=[(train&(v!=s),train&(v==s),str(s)) for s in [30,40,50]]
        configurations.append({'name':f'repeat{rep}','train':train,'test':~train,'folds':folds,'domains':domains})
    settings={'lo':40,'hi':800,'ref':40,'support':[20,1200],'mel_hi':1200}
    return p,f,y,v,labels,configurations,settings,path

def load_external():
    config=json.loads((ROOT/'research/external_settings.json').read_text())
    path=ROOT/config['data_file'];z=np.load(path)
    p=z['psd'];f=z['frequency'];labels=z['labels'];y=labels[:,0];v=labels[:,1];trial=labels[:,2]
    assert p.ndim==2 and len(p)==len(labels)
    source=np.isin(v,config['source_speeds']);test=v==config['target_speed']
    # Stable trial-index groups. All derived signals stay with their episode.
    folds=[(source&(trial%5!=k),source&(trial%5==k),f'trial_mod5_{k}') for k in range(5)]
    domains=np.where(test,'speed','source')
    return p,f,y,v,labels,[{'name':'external45','train':source,'test':test,'folds':folds,'domains':domains}],config['feature_settings'],path

def inputs(dataset,data_path):
    files=[pathlib.Path(__file__),ROOT/'experiments/response_calibration.py',ROOT/'research/calibration_protocol.md',data_path]
    if dataset=='external':files += [ROOT/'research/external_settings.json',ROOT/'research/external_protocol.md']
    return {str(p):sha(p) for p in files}

def execute(dataset,stage):
    start=time.monotonic();p,f,y,v,labels,configs,settings,path=(load_cluster() if dataset=='cluster' else load_external())
    out=ROOT/'experiments/calibration'/dataset;out.mkdir(parents=True,exist_ok=True)
    env={'utc':now(),'python':sys.version,'numpy':np.__version__,'torch_used':False,'cpu_threads':2}
    save(out/f'environment_{stage}.json',env)
    hashes=inputs(dataset,path)
    if stage=='select':
        assert not (out/'freeze.json').exists(), 'Frozen runs may not be overwritten.'
        save(out/'input_hashes.json',hashes);save(out/'algebra_test.json',algebra_test());records=[]
        for config in configs:
            source=config['train'];split=config['name']
            assert not np.any(source&config['test'])
            save(out/f'{split}_split.json',{'train_indices':np.flatnonzero(source).tolist(),'test_indices':np.flatnonzero(config['test']).tolist(),'folds':[{'name':name,'train_indices':np.flatnonzero(tr).tolist(),'validation_indices':np.flatnonzero(va).tolist()} for tr,va,name in config['folds']]})
            for method in METHODS:
                if time.monotonic()-start>7200:raise TimeoutError('Two-hour active selection cap')
                t=time.monotonic();fold_data=[];cv=[]
                for tr,va,fold in config['folds']:
                    x,info=features_for_split(p,f,v,y,tr,settings,method)
                    fold_data.append((x[tr],x[va],y[tr],y[va],fold,info))
                for C,g in itertools.product([1,10,100],[.25,1,4]):
                    scores=[]
                    for xt,xv,yt,yv,fold,info in fold_data:
                        model=make_pipeline(StandardScaler(),SVC(C=C,gamma=g/xt.shape[1],cache_size=200))
                        model.fit(xt,yt);pred=model.predict(xv);scores.append(score(yv,pred))
                    cv.append({'C':C,'g':g,'cv_mean':float(np.mean(scores)),'fold_scores':scores})
                choice=sorted(cv,key=key)[0];x,diag=features_for_split(p,f,v,y,source,settings,method)
                model=make_pipeline(StandardScaler(),SVC(C=choice['C'],gamma=choice['g']/x.shape[1],cache_size=200))
                model.fit(x[source],y[source]);mp=out/f'{split}_{method}.joblib';joblib.dump(model,mp,compress=3)
                record={'split':split,'method':method,'selected':choice,'candidates':cv,'feature_dimension':x.shape[1],'source_fit_diagnostics':diag,'fold_diagnostics':[{'fold':entry[4],**entry[5]} for entry in fold_data],'fit_seconds':time.monotonic()-t,'model':mp.name,'model_sha256':sha(mp),'settings':settings,'source_acquisitions':int(source.sum()),'utc':now()}
                save(out/f'{split}_{method}_selection.json',record);records.append(record)
                print(json.dumps({k:record[k] for k in ['split','method','selected','fit_seconds']}),flush=True)
        save(out/'selected.json',records)
        frozen={str(q.relative_to(out)):sha(q) for q in out.iterdir() if q.is_file() and q.name!='freeze.json'}
        save(out/'freeze.json',{'utc':now(),'inputs':hashes,'files':frozen,'method_order':METHODS,'elapsed_seconds':time.monotonic()-start,'stage':'before extension target prediction','scope':'exploratory on Cluster; externally locked on external data'})
    else:
        freeze=json.loads((out/'freeze.json').read_text());assert freeze['inputs']==hashes
        assert all(sha(out/name)==h for name,h in freeze['files'].items())
        assert not (out/'evaluation_complete.json').exists(), 'Do not overwrite a completed evaluation'
        records=json.loads((out/'selected.json').read_text());results=[]
        for config in configs:
            target=config['test'];source=config['train']
            for method in METHODS:
                rec=next(a for a in records if a['split']==config['name'] and a['method']==method)
                model=joblib.load(out/rec['model']);x,diag=features_for_split(p,f,v,y,source,settings,method)
                assert diag==rec['source_fit_diagnostics']
                t=time.monotonic();pred=model.predict(x[target]);seconds=time.monotonic()-t
                fn=f'{config["name"]}_{method}_predictions.csv';idx=np.flatnonzero(target)
                frame=pd.DataFrame({'index':idx,'acquisition_id':['_'.join(map(str,row)) for row in labels[target]],'texture':y[target],'speed':v[target],'domain':config['domains'][target],'prediction':pred,'correct':(pred==y[target]).astype(int)})
                frame.to_csv(out/fn,index=False)
                results.append({'split':config['name'],'method':method,'file':fn,'sha256':sha(out/fn),'rows':len(frame),'prediction_seconds':seconds})
                print(json.dumps({'split':config['name'],'method':method,'predictions_saved':len(frame)}),flush=True)
        save(out/'evaluation_complete.json',{'utc':now(),'selection_freeze_sha256':sha(out/'freeze.json'),'records':results,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['cluster','external']);parser.add_argument('stage',choices=['select','evaluate']);args=parser.parse_args()
    try:execute(args.dataset,args.stage)
    except Exception:
        failure=ROOT/'experiments/calibration'/args.dataset/f'failure_{args.stage}_{int(time.time())}.json'
        save(failure,{'utc':now(),'exception':traceback.format_exc()});raise
