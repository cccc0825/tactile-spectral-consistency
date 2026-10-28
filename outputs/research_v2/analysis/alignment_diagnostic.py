"""Exploratory within-specimen speed dispersion and specimen-centroid dispersion."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json,pathlib,hashlib,datetime
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from run_calibration import load_cluster
from run_external_calibration import load_external,external_features
from response_calibration import features_for_split

def js(p):
    p=p/p.sum(axis=-1,keepdims=True)
    average=p.mean(axis=0)
    return float(np.mean(np.sum(p*np.log(np.maximum(p/average,1e-300)),axis=-1)))

rows=[];summary=[];inputs={}
for dataset in ['cluster','external']:
    if not (ROOT/f'experiments/calibration/{dataset}/evaluation_complete.json').exists():continue
    p,f,y,v,labels,configs,settings,path=load_cluster() if dataset=='cluster' else load_external()
    train=configs[0]['train']
    use=(np.isin(labels[:,1],[0,90,180,270])&(labels[:,3]==500)) if dataset=='cluster' else np.ones(len(y),bool)
    for method in ['temporal','spatial','corrected']:
        x,diag=features_for_split(p,f,v,y,train,settings,method)
        prob=np.exp(x);centroids=[];within=[]
        for c in np.unique(y):
            a=np.stack([prob[use&(y==c)&(v==speed)].mean(axis=0) for speed in np.unique(v)])
            d=js(a);within.append(d);centroids.append(a.mean(axis=0))
            rows.append({'dataset':dataset,'method':method,'texture':int(c),'within_specimen_speed_js_nats':d})
        between=js(np.stack(centroids))
        summary.append({'dataset':dataset,'method':method,'mean_within_specimen_speed_js_nats':float(np.mean(within)),
                        'between_specimen_centroid_js_nats':between,'within_between_ratio':float(np.mean(within)/between),
                        'specimens':len(within),'speeds':len(np.unique(v)),
                        'scope':'exploratory coordinate-dependent spectral descriptor; no model selection'})
    inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
pd.DataFrame(rows).to_csv(ROOT/'analysis/alignment_per_specimen.csv',index=False)
pd.DataFrame(summary).to_csv(ROOT/'analysis/alignment_summary.csv',index=False)
(ROOT/'analysis/alignment_manifest.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'inputs':inputs,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
 'protocol_sha256':hashlib.sha256((ROOT/'research/alignment_diagnostic_protocol.md').read_bytes()).hexdigest()},indent=2))
print(pd.DataFrame(summary).to_string(index=False))
