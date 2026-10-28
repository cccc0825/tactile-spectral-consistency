"""Exploratory estimator controls. Real acquisition and synthetic outputs separated."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import pathlib,json,hashlib,datetime,time,sys,math,argparse,itertools
import numpy as np
import pandas as pd
from scipy.signal import welch,resample_poly,butter,sosfilt
from scipy.special import logsumexp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from response_calibration import integral,log_masses
from run_calibration import load_cluster,score,key,save,sha,now
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/diagnostics'
sys.path.insert(0,str(ROOT.parent/'research/experiments'))
from extract_features import decode,parse_csv

def extraction():
    dest=ROOT/'data/spatial_resolution';dest.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT.parent/'research/data/range_plan.json').read_text());start=time.monotonic()
    spectra=[];labels=[];checks=[]
    for part in plan['parts']:
        if not part['name'].startswith('accel'):continue
        p={**part,'bytes':(ROOT.parent/'research/data/raw_archive_ranges'/part['name']).read_bytes()}
        for e in sorted(part['records'],key=lambda e:e['name']):
            raw=decode(p,e);arr=parse_csv(raw,4);label=list(map(int,pathlib.Path(e['name']).stem.split('_')))
            t=arr[:,0];a=arr[:,1:];u,inv,cnt=np.unique(t,return_inverse=True,return_counts=True)
            aa=np.zeros((len(u),3));np.add.at(aa,inv,a);aa/=cnt[:,None]
            v=label[2];duration=20/v;center=(u[0]+u[-1])/2
            grid=center-duration/2+np.arange(round(duration*6000))/6000
            signal=np.stack([np.interp(grid,u,aa[:,axis]) for axis in range(3)],axis=1)
            divisor=math.gcd(v,40);spatial=resample_poly(signal,v//divisor,40//divisor,axis=0)
            assert spatial.shape==(3000,3)
            q,power=welch(spatial,fs=150,window='hann',nperseg=1024,noverlap=512,detrend='constant',scaling='density',axis=0)
            spectra.append(power.sum(1).astype('float32'));labels.append(label)
            checks.append({'id':'_'.join(map(str,label)),'raw_sha256':hashlib.sha256(raw).hexdigest(),'input_n':len(signal),'spatial_n':len(spatial)})
        print(json.dumps({'stage':'real_spatial_extraction','part':part['name'],'records':len(labels),'elapsed_seconds':time.monotonic()-start}),flush=True)
    labels=np.array(labels);assert len(labels)==18880
    np.savez_compressed(dest/'psd.npz',psd=np.stack(spectra),labels=labels,frequency=q)
    save(dest/'acquisition_audit.json',checks);save(dest/'manifest.json',{'utc':now(),'script_sha256':sha(__file__),'file_sha256':sha(dest/'psd.npz'),'acquisitions':len(labels),'elapsed_seconds':time.monotonic()-start})

def normalized_from_mass(m):
    m=np.maximum(m,0);assert (m.sum(1)>0).all();return np.log(np.maximum(m/m.sum(1,keepdims=True),1e-12))

def real():
    p,f,y,v,labels,configs,settings,data_path=load_cluster();cache=np.load(ROOT/'data/spatial_resolution/psd.npz')
    assert np.array_equal(cache['labels'],labels)
    q=cache['frequency'];sp=cache['psd'];features={}
    edges=np.geomspace(40,800,129)
    for method in ['temporal','spatial']:
        temporal_edges=edges[None,:]*(v[:,None]/40 if method=='spatial' else np.ones((len(v),1)))
        clipped=np.clip(temporal_edges,60,400)
        features['common_band_'+method]=normalized_from_mass(np.diff(integral(p,f,clipped),axis=1))
        spatial_edges=edges[None,:]/(40 if method=='spatial' else v[:,None])
        features['fixed_space_'+method]=normalized_from_mass(np.diff(integral(sp,q,spatial_edges),axis=1))
    selected=[];models={};start=time.monotonic()
    for c in configs:
        for name,x in features.items():
            cv=[]
            for C,g in itertools.product([1,10,100],[.25,1,4]):
                scores=[]
                for tr,va,fold in c['folds']:
                    model=make_pipeline(StandardScaler(),SVC(C=C,gamma=g/128,cache_size=200));model.fit(x[tr],y[tr]);scores.append(score(y[va],model.predict(x[va])))
                cv.append({'C':C,'g':g,'cv_mean':float(np.mean(scores)),'fold_scores':scores})
            chosen=sorted(cv,key=key)[0];model=make_pipeline(StandardScaler(),SVC(C=chosen['C'],gamma=chosen['g']/128,cache_size=200));model.fit(x[c['train']],y[c['train']])
            import joblib
            modelpath=OUT/f'{c["name"]}_{name}.joblib';joblib.dump(model,modelpath,compress=3)
            selected.append({'split':c['name'],'method':name,'selected':chosen,'candidates':cv,'model_sha256':sha(modelpath)})
            models[(c['name'],name)]=model
    save(OUT/'real_selection.json',selected)
    save(OUT/'real_selection_freeze.json',{'utc':now(),'script_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'research/diagnostic_protocol.md'),'selection_sha256':sha(OUT/'real_selection.json'),'before_target_predictions':True})
    results=[]
    for c in configs:
        ix=np.flatnonzero(c['test'])
        for name,x in features.items():
            pred=models[c['name'],name].predict(x[ix]);fn=f'{c["name"]}_{name}.csv'
            pd.DataFrame({'index':ix,'texture':y[ix],'speed':v[ix],'domain':c['domains'][ix],'prediction':pred,'correct':(pred==y[ix]).astype(int)}).to_csv(OUT/fn,index=False)
            results.append({'split':c['name'],'method':name,'file':fn,'sha256':sha(OUT/fn),'rows':len(ix)})
    save(OUT/'real_results_manifest.json',{'utc':now(),'status':'complete','records':results,'seconds':time.monotonic()-start})

def synthetic():
    rng=np.random.default_rng(20270913);grid=np.arange(18000)/6000;frequencies=np.linspace(1,16,128)
    psd=[[],[]];labels=[]
    for texture in range(12):
        peaks=[3+texture*.25,9+texture*.22]
        power=sum(np.exp(-.5*((frequencies-center)/.2)**2) for center in peaks)
        amp=np.sqrt(power/power.sum())
        for v,repeat in itertools.product([20,30,40,50,60],range(4)):
            phases=rng.uniform(0,2*np.pi,128)
            signal=np.cos(2*np.pi*(grid[:,None]*v)*frequencies[None,:]+phases)@amp
            filtered=sosfilt(butter(4,200,fs=6000,output='sos'),signal)
            for k,s in enumerate([signal,filtered]):
                f,p=welch(s[6000:12000],fs=6000,window='hann',nperseg=1024,noverlap=512,detrend='constant',scaling='density');psd[k].append(p)
            labels.append([texture,v,repeat])
    labels=np.array(labels);y,v,r=labels.T;train=np.isin(v,[30,40,50])&(r<2);test=~np.isin(v,[30,40,50]);results=[]
    np.savez_compressed(OUT/'synthetic_signals_psd.npz',psd=np.array(psd),labels=labels,frequency=f)
    for observation,k in [('ideal',0),('lowpass_200Hz',1)]:
        for name,beta in [('temporal',0),('spatial',1)]:
            x=log_masses(np.array(psd[k]),f,v,beta=beta);cv=[]
            for C,g in itertools.product([1,10,100],[.25,1,4]):
                scores=[]
                for held in [30,40,50]:
                    tr=train&(v!=held);va=train&(v==held);m=make_pipeline(StandardScaler(),SVC(C=C,gamma=g/128));m.fit(x[tr],y[tr]);scores.append(score(y[va],m.predict(x[va])))
                cv.append({'C':C,'g':g,'cv_mean':float(np.mean(scores)),'fold_scores':scores})
            selected=sorted(cv,key=key)[0];m=make_pipeline(StandardScaler(),SVC(C=selected['C'],gamma=selected['g']/128));m.fit(x[train],y[train]);pred=m.predict(x[test])
            # Jensen-Shannon divergence averaged over class-speed mean probability vectors.
            prob=np.exp(x);div=[]
            for c in np.unique(y):
                means=np.stack([prob[(y==c)&(v==speed)].mean(0) for speed in np.unique(v)])
                pooled=means.mean(0);div.append(float(np.mean(np.sum(means*np.log(np.maximum(means/pooled,1e-30)),axis=1))))
            fn=f'synthetic_{observation}_{name}.csv';pd.DataFrame({'texture':y[test],'speed':v[test],'repeat':r[test],'prediction':pred,'correct':(pred==y[test]).astype(int)}).to_csv(OUT/fn,index=False)
            results.append({'observation':observation,'representation':name,'synthetic':True,'target_accuracy':score(y[test],pred),'mean_speed_JS_divergence':float(np.mean(div)),'source_selection':selected,'all_source_candidates':cv,'file':fn,'sha256':sha(OUT/fn)})
    save(OUT/'synthetic_results.json',{'utc':now(),'seed':20270913,'scope':'Constructed stationary spatial signals only, no physical sensor/robot measurements','records':results})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['extract','real','synthetic']);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    {'extract':extraction,'real':real,'synthetic':synthetic}[a.stage]()
