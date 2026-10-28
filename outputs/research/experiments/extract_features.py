"""CRC-verified accelerometer decoding and outcome-blind spectral extraction."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import pathlib,json,struct,zlib,hashlib,time,io,datetime,argparse
import numpy as np
from scipy.signal import welch
ROOT=pathlib.Path(__file__).resolve().parents[1]
def decode(part,entry):
 pos=entry['offset']-part['start'];b=part['bytes'];v=struct.unpack('<4s5H3L2H',b[pos:pos+30]);assert v[0]==b'PK\x03\x04'
 start=pos+30+v[-2]+v[-1];raw=zlib.decompress(b[start:start+entry['compressed']],-15)
 assert len(raw)==entry['size'] and (zlib.crc32(raw)&0xffffffff)==entry['crc32']
 return raw
def parse_csv(raw,cols):
 return np.fromstring(raw.split(b'\n',1)[1].replace(b',',b' '),sep=' ').reshape(-1,cols)
def spectra(arr,speed):
 t=arr[:,0];a=arr[:,1:];q={'n_rows':len(t),'finite':bool(np.isfinite(arr).all()),'negative_dt':int(np.sum(np.diff(t)<0)),'duplicate_timestamps':int(np.sum(np.diff(t)==0)),'start':float(t[0]),'end':float(t[-1]),'saturation_fraction':float(np.mean(np.abs(a)>=15.99))}
 if not q['finite'] or q['negative_dt']:return None,{**q,'valid':False,'reason':'nonfinite_or_decreasing_time'}
 if q['duplicate_timestamps']:
  u,idx,counts=np.unique(t,return_inverse=True,return_counts=True);merged=np.zeros((len(u),3));np.add.at(merged,idx,a);a=merged/counts[:,None];t=u
 center=(t[0]+t[-1])/2;sel=(t>=center-.5)&(t<=center+.5)
 q['median_dt']=float(np.median(np.diff(t)))
 intersect=(t[:-1]<center+.5)&(t[1:]>center-.5)
 q['central_max_gap']=float(np.max(np.diff(t)[intersect])) if intersect.any() else 99.
 if t[-1]-t[0]<1.1 or q['central_max_gap']>.02:return None,{**q,'valid':False,'reason':'short_or_central_gap'}
 out=[]
 for duration in [1.,20./speed]:
  n=int(round(duration*6000));grid=center-duration/2+np.arange(n)/6000
  u=np.stack([np.interp(grid,t,a[:,j]) for j in range(3)],axis=1)
  f,p=welch(u,fs=6000,window='hann',nperseg=1024,noverlap=512,detrend='constant',scaling='density',axis=0)
  out.append(p.sum(axis=1).astype(np.float32))
 q['valid']=True;q['reason']='';return np.stack(out),q
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args()
 plan=json.loads((ROOT/'data/range_plan.json').read_text());dest=ROOT/'data/processed';dest.mkdir(parents=True,exist_ok=True)
 pending={p['name']:p for p in plan['parts'] if p['name'].startswith('accel')};start=time.monotonic()
 while pending:
  progressed=False
  for name in sorted(list(pending)):
   p=pending[name];rawpath=ROOT/'data/raw_archive_ranges'/name;out=dest/name.replace('.bin','.npz')
   if not rawpath.exists():continue
   version=out.with_suffix('.version.json')
   current_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
   if out.exists() and version.exists() and json.loads(version.read_text()).get('script_sha256')==current_hash:del pending[name];continue
   p={**p,'bytes':rawpath.read_bytes()};results=[];ids=[];quality=[]
   for entry in sorted(p['records'],key=lambda x:x['name']):
    raw=decode(p,entry);fid=pathlib.Path(entry['name']).stem;labels=list(map(int,fid.split('_')));arr=parse_csv(raw,4);s,q=spectra(arr,labels[2]);quality.append({'id':fid,'source_crc32':entry['crc32'],'source_sha256':hashlib.sha256(raw).hexdigest(),**q})
    if s is None:s=np.full((2,513),np.nan,dtype=np.float32)
    results.append(s);ids.append(labels)
   with out.with_suffix('.partial').open('wb') as handle:np.savez_compressed(handle,psd=np.stack(results),labels=np.array(ids),frequency=np.fft.rfftfreq(1024,1/6000))
   out.with_suffix('.partial').replace(out);out.with_suffix('.json').write_text(json.dumps(quality,indent=2));version.write_text(json.dumps({'script_sha256':current_hash,'source_range_sha256':hashlib.sha256(p['bytes']).hexdigest()}));del pending[name];progressed=True
   print(json.dumps({'part':name,'remaining':len(pending),'valid':sum(x['valid'] for x in quality),'total':len(quality),'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
  if pending:
   if not args.wait:break
   if not progressed:time.sleep(3)
 if not pending:
  ps=[];ls=[];quality=[]
  for p in sorted(dest.glob('accel_*.npz')):
   z=np.load(p);ps.append(z['psd']);ls.append(z['labels']);quality+=json.loads(p.with_suffix('.json').read_text())
  labels=np.concatenate(ls);psd=np.concatenate(ps);assert len(labels)==18880 and len(set(map(tuple,labels)))==18880
  np.savez_compressed(dest/'all_psd.npz',psd=psd,labels=labels,frequency=np.fft.rfftfreq(1024,1/6000))
  (dest/'quality.json').write_text(json.dumps(quality,indent=2))
  (dest/'extraction_manifest.json').write_text(json.dumps({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((ROOT/'research/protocol_v1.md').read_bytes()).hexdigest(),'acquisitions':len(labels),'valid':sum(x['valid'] for x in quality),'psd_sha256':hashlib.sha256((dest/'all_psd.npz').read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-start},indent=2))
