"""Independent public-source spot check. Never edits v1 artifacts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, pathlib, hashlib, datetime, subprocess, struct, zlib, io, time
import numpy as np
from scipy.signal import welch
import requests

ROOT=pathlib.Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'research'
DEST=ROOT/'data/original_spotcheck'
DEST.mkdir(parents=True,exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
plan=json.loads((OLD/'data/range_plan.json').read_text())
entries=[e for p in plan['parts'] if p['name'].startswith('accel') for e in p['records']]
entries.sort(key=lambda e:e['name'])
rng=np.random.default_rng(20270913)
chosen=[entries[i] for i in sorted(rng.choice(len(entries),12,replace=False))]
lock=DEST/'sample_plan.json'
if lock.exists():
    assert json.loads(lock.read_text())['entries']==chosen
else:
    lock.write_text(json.dumps({'created_utc':now(),'seed':20270913,'sampling':'12 of 18880 without replacement, sorted names; no result filtering','source':plan['url'],'entries':chosen},indent=2))
q={x['id']:x for x in json.loads((OLD/'data/processed/quality.json').read_text())}
cache=np.load(OLD/'data/processed/all_psd.npz')
indices={'_'.join(map(str,x)):i for i,x in enumerate(cache['labels'])}
checks=[]
for e in chosen:
    fid=pathlib.Path(e['name']).stem
    target=DEST/(fid+'.range'); headers=DEST/(fid+'.headers')
    begin=e['offset']; end=begin+e['compressed']+2047
    start=time.monotonic();response=None
    for attempt in range(1,5):
        try:
            response=requests.get(plan['url'],headers={'Range':f'bytes={begin}-{end}'},timeout=60)
            response.raise_for_status();assert response.status_code==206
            assert response.headers['Content-Range']==f'bytes {begin}-{end}/15220748922'
            target.write_bytes(response.content)
            with (DEST/'authorized_download_events.jsonl').open('a') as log:
                log.write(json.dumps({'utc':now(),'id':fid,'attempt':attempt,'success':True,'url':plan['url'],'range':[begin,end],'headers':dict(response.headers),'seconds':time.monotonic()-start})+'\n')
            break
        except Exception as ex:
            with (DEST/'authorized_download_events.jsonl').open('a') as log:
                log.write(json.dumps({'utc':now(),'id':fid,'attempt':attempt,'success':False,'error':repr(ex)})+'\n')
            if attempt==4:raise
            time.sleep(2)
    b=target.read_bytes();h=struct.unpack('<4s5H3L2H',b[:30]);assert h[0]==b'PK\x03\x04'
    off=30+h[-2]+h[-1];raw=zlib.decompress(b[off:off+e['compressed']],-15)
    assert len(raw)==e['size'] and zlib.crc32(raw)&0xffffffff==e['crc32']
    assert sha(raw)==q[fid]['source_sha256']
    # Parse independently from the original np.fromstring implementation.
    arr=np.loadtxt(io.BytesIO(raw),delimiter=',',skiprows=1)
    times, inv=np.unique(arr[:,0],return_inverse=True)
    vals=np.empty((len(times),3))
    counts=np.bincount(inv)
    for a in range(3): vals[:,a]=np.bincount(inv,weights=arr[:,a+1])/counts
    assert np.isfinite(vals).all()
    out=[];center=(times[0]+times[-1])/2;speed=int(fid.split('_')[2])
    for duration in (1.,20./speed):
        grid=center-duration/2+np.arange(round(6000*duration))/6000
        # Welch independently per axis rather than vectorized axis argument.
        powers=[]
        for a in range(3):
            signal=np.interp(grid,times,vals[:,a])
            _, power=welch(signal,fs=6000,window='hann',nperseg=1024,noverlap=512,detrend='constant',scaling='density')
            powers.append(power)
        out.append(np.sum(powers,axis=0).astype('float32'))
    actual=np.stack(out);saved=cache['psd'][indices[fid]]
    error=float(np.max(np.abs(actual-saved)))
    assert np.allclose(actual,saved,rtol=2e-6,atol=1e-12)
    check={'id':fid,'rows':len(arr),'source_sha256':sha(raw),'crc32_verified':True,'public_bytes_equal_old_input':True,'psd_max_abs_error':error,'psd_recomputed_matches':True}
    checks.append(check);print(json.dumps(check),flush=True)
(DEST/'report.json').write_text(json.dumps({'finished_utc':now(),'script_sha256':sha(pathlib.Path(__file__).read_bytes()),'sample_plan_sha256':sha(lock.read_bytes()),'scope':'Public bytes and independent parsing/PSD recomputation for 12 randomly sampled acquisitions. This does not prove every acquisition or original collection conditions.','checked':checks,'pass':len(checks)==12},indent=2))
