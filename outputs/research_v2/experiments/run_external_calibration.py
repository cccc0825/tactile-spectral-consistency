"""External sensor bandwidth adapter, locked before external classification."""
import pathlib,json,hashlib,sys,traceback,time
import numpy as np
from scipy.fft import dct
import run_calibration as runner
from response_calibration import features_for_split as core_features

ROOT=runner.ROOT
def external_features(p,f,v,y,train,settings,method):
    if method not in ['mel','mfcc']:return core_features(p,f,v,y,train,settings,method)
    hz=lambda m:700*(10**(m/2595)-1)
    ml=lambda x:2595*np.log10(1+x/700)
    edges=hz(np.linspace(ml(5),ml(100),34));weights=[]
    for i in range(32):weights.append(np.maximum(0,np.minimum((f-edges[i])/(edges[i+1]-edges[i]),(edges[i+2]-f)/(edges[i+2]-edges[i+1]))))
    w=np.array(weights);assert np.all(w.sum(1)>0)
    mass=p@w.T*(f[1]-f[0]);x=np.log(np.maximum(mass/mass.sum(1,keepdims=True),1e-12))
    if method=='mfcc':x=dct(x,type=2,norm='ortho',axis=1)[:,1:21]
    return x,{'external_mel_bands':32,'external_mel_limits':[5,100]}

def load_external():
    config=json.loads((ROOT/'research/external_settings.json').read_text());path=ROOT/config['data_file'];z=np.load(path)
    p=z['psd'];f=z['frequency'];labels=z['labels'];y,v,trial=labels.T
    source=np.isin(v,[30,35]);test=v==40
    assert np.isfinite(p).all() and len(set(map(tuple,labels)))==len(labels)
    folds=[(source&(trial%5!=i),source&(trial%5==i),f'trial_mod5_{i}') for i in range(5)]
    for tr,va,name in folds:
        assert not np.any(tr&va) and set(y[tr])==set(y[va])==set(range(12))
    domains=np.where(test,'speed','source')
    return p,f,y,v,labels,[{'name':'external40','train':source,'test':test,'folds':folds,'domains':domains}],config['feature_settings'],path

def inputs(dataset,data_path):
    files=[pathlib.Path(__file__),ROOT/'experiments/run_calibration.py',ROOT/'experiments/response_calibration.py',ROOT/'research/external_settings.json',ROOT/'research/external_protocol.md',data_path,
           ROOT/'research/external_validity_decision.json',ROOT/'research/external_validity_decision.freeze.json',
           ROOT/'data/external/processed/manifest.json',ROOT/'data/external/final_eligibility_audit.json',
           ROOT/'data/external/publish_under_decision.py',ROOT/'experiments/extract_external.py']
    return {str(p):runner.sha(p) for p in files}

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('select','evaluate')
    runner.features_for_split=external_features;runner.load_external=load_external;runner.inputs=inputs
    try:runner.execute('external',sys.argv[1])
    except Exception:
        runner.save(ROOT/'experiments/calibration/external'/f'failure_{sys.argv[1]}_{int(time.time())}.json',{'utc':runner.now(),'exception':traceback.format_exc()});raise
