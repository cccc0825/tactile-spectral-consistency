"""CRC-verified, outcome-blind Cluster accelerometer log-mel extraction.

Uses the same acquisition order and central one-second samples as the v1
extract_features.py. This independently implemented extractor never writes v1.
Run from the workspace root with the pinned research virtual environment.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='2'
import argparse,datetime,hashlib,itertools,json,pathlib,resource,struct,sys,time,zlib
import numpy as np
from scipy.signal import stft
import torch
import torch.nn.functional as F

BASE=pathlib.Path(__file__).resolve().parents[3]
DATA=BASE/'outputs/research_v2/data'
V1=BASE/'outputs/research'
PREFIX=DATA/'cluster_spectrograms'
FS=6000
NFFT=1024
HOP=256

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def savej(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def decode(part,entry):
    pos=entry['offset']-part['start'];b=part['bytes']
    header=struct.unpack('<4s5H3L2H',b[pos:pos+30])
    if header[0]!=b'PK\x03\x04' or header[3]!=8:
        raise ValueError('Unsupported or corrupt local ZIP entry')
    start=pos+30+header[-2]+header[-1]
    raw=zlib.decompress(b[start:start+entry['compressed']],-15)
    if len(raw)!=entry['size'] or (zlib.crc32(raw)&0xffffffff)!=entry['crc32']:
        raise ValueError('ZIP entry length or CRC mismatch: '+entry['name'])
    return raw

def central_window(arr):
    t=arr[:,0];a=arr[:,1:]
    if arr.shape[1]!=4 or not np.isfinite(arr).all() or np.any(np.diff(t)<0):
        raise ValueError('Invalid acquisition values or timestamps')
    duplicates=int(np.sum(np.diff(t)==0))
    if duplicates:
        t,idx,counts=np.unique(t,return_inverse=True,return_counts=True)
        merged=np.zeros((len(t),3));np.add.at(merged,idx,a);a=merged/counts[:,None]
    center=(t[0]+t[-1])/2
    intersect=(t[:-1]<center+.5)&(t[1:]>center-.5)
    maxgap=float(np.max(np.diff(t)[intersect])) if intersect.any() else 99.
    if t[-1]-t[0]<1.1 or maxgap>.02:raise ValueError('Short acquisition or central timestamp gap')
    grid=center-.5+np.arange(FS)/FS
    x=np.stack([np.interp(grid,t,a[:,j]) for j in range(3)])
    return x,{'duplicate_timestamps':duplicates,'central_max_gap':maxgap,'n_raw_rows':len(arr),'start':float(t[0]),'end':float(t[-1]),'center':float(center)}

def mel_filterbank(fs=FS,nfft=NFFT):
    mel=lambda f:2595*np.log10(1+np.asarray(f)/700)
    hz=lambda m:700*(10**(m/2595)-1)
    edges=hz(np.linspace(mel(20),mel(1200),130))
    freq=np.fft.rfftfreq(nfft,1/fs)
    weights=np.maximum(0,np.minimum((freq[None,:]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),(edges[2:,None]-freq[None,:])/(edges[2:,None]-edges[1:-1,None])))
    if not np.all(weights.sum(axis=1)>0):raise ValueError('Empty mel filter')
    return weights.astype(np.float32)

def logmel(x,fs=FS):
    """Three acceleration axes -> 3 x 128 x 64 float16; no learned transform."""
    if x.ndim!=2 or x.shape[0]!=3:raise ValueError('Expected three axes')
    _,_,z=stft(x,fs=fs,window='hann',nperseg=NFFT,noverlap=NFFT-HOP,nfft=NFFT,detrend='constant',return_onesided=True,boundary=None,padded=False,axis=-1)
    power=(np.abs(z)**2).astype(np.float32)
    mel=np.einsum('mf,cft->cmt',mel_filterbank(fs),power,optimize=True)
    ref=np.maximum(mel.max(axis=(1,2),keepdims=True),np.float32(1e-20))
    db=np.clip(10*np.log10(np.maximum(mel/ref,np.float32(1e-8))),-80,0)
    scaled=torch.from_numpy((db/40+1).astype(np.float32)).unsqueeze(0)
    resized=F.interpolate(scaled,size=(128,64),mode='bilinear',align_corners=False,antialias=False)[0].numpy().astype(np.float16)
    if not np.isfinite(resized).all() or resized.min()<-1 or resized.max()>1:raise ValueError('Invalid spectrogram')
    return resized

def extract():
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    started=time.monotonic();DATA.mkdir(parents=True,exist_ok=True)
    final=PREFIX.with_suffix('.npy');partial=PREFIX.with_suffix('.npy.partial')
    if final.exists():raise RuntimeError('Refusing to overwrite completed spectrograms')
    if partial.exists():raise RuntimeError('Partial extraction exists; preserve it and use documented recovery')
    planpath=V1/'data/range_plan.json';plan=json.loads(planpath.read_text())
    parts=sorted([p for p in plan['parts'] if p['name'].startswith('accel')],key=lambda p:p['name'])
    n=sum(len(p['records']) for p in parts)
    if n!=18880:raise ValueError('Changed acquisition count')
    inputs=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float16,shape=(n,3,128,64))
    labels=[];records=[];ranges=[];i=0
    for part in parts:
        path=V1/'data/raw_archive_ranges'/part['name']
        p={**part,'bytes':path.read_bytes()};ranges.append({'name':part['name'],'sha256':sha(path)})
        for entry in sorted(p['records'],key=lambda e:e['name']):
            raw=decode(p,entry);fid=pathlib.Path(entry['name']).stem;label=list(map(int,fid.split('_')))
            arr=np.fromstring(raw.split(b'\n',1)[1].replace(b',',b' '),sep=' ').reshape(-1,4)
            window,q=central_window(arr);inputs[i]=logmel(window)
            labels.append(label);records.append({'row':i,'id':fid,'archive_entry':entry['name'],'source_crc32':entry['crc32'],'source_sha256':hashlib.sha256(raw).hexdigest(),**q})
            i+=1
        inputs.flush()
        print(json.dumps({'part':part['name'],'acquisitions':i,'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if peak>3_000_000_000:raise MemoryError('Extraction exceeded 3 GB process memory budget')
    ls=np.asarray(labels,dtype=np.int64)
    expected=set(itertools.product(range(118),range(0,360,45),[20,30,40,50,60],[500,1000],[0,1]))
    if set(map(tuple,ls))!=expected or i!=18880:raise ValueError('Dataset factorial integrity failed')
    prior=np.load(V1/'data/processed/all_psd.npz')['labels']
    if not np.array_equal(ls,prior):raise ValueError('Spectrogram rows do not match v1 order')
    labelpath=PREFIX.with_suffix('.labels.npy');np.save(labelpath,ls)
    savej(PREFIX.with_suffix('.records.json'),records)
    del inputs;partial.replace(final)
    manifest={'timestamp_utc':now(),'status':'complete','target_metrics_computed':False,'source_dataset':'Cluster Haptic Texture Dataset','acquisitions':i,'shape':[i,3,128,64],'dtype':'float16','file_sha256':sha(final),'labels_sha256':sha(labelpath),'label_rows_sha256':hashlib.sha256(ls.tobytes()).hexdigest(),'records_sha256':sha(PREFIX.with_suffix('.records.json')),'extractor_sha256':sha(__file__),'source_range_plan_sha256':sha(planpath),'source_ranges':ranges,'same_order_as_v1':True,'sampling_hz':FS,'window_seconds':1,'duplicate_timestamps':'average all axes at duplicate times before interpolation','stft':{'nfft':NFFT,'hop':HOP,'window':'periodic Hann','detrend':'constant per frame','boundary':None,'padded':False},'mel':{'scale':'HTK','count':128,'fmin_hz':20,'fmax_hz':1200,'normalization':'unnormalized triangular filters'},'log':'per-axis maximum reference, 10 log10 power ratio, clipped [-80,0] dB','output_map':'db/40+1; bilinear resize 128x64, align_corners=False, antialias=False','elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
    savej(PREFIX.with_suffix('.manifest.json'),manifest)
    print(json.dumps({'status':'complete','acquisitions':i,'elapsed_seconds':manifest['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    try:extract()
    except Exception as e:
        import traceback
        with PREFIX.with_suffix('.failures.jsonl').open('a') as f:f.write(json.dumps({'timestamp_utc':now(),'error':repr(e),'traceback':traceback.format_exc()})+'\n')
        raise
