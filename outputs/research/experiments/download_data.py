"""Download exact accelerometer/position ZIP byte ranges from the CC-BY-4.0 archive.
Run from the research directory. No extraction or execution of archive paths.
"""
import requests,json,hashlib,pathlib,time,concurrent.futures,datetime,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1]
URL='https://ndownloader.figshare.com/files/61118716'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plan():
 rows=json.loads((ROOT/'corpus/archive_index.json').read_text());total=15220748922
 parts=[]
 for texture in range(118):
  prefix=f'cluster_haptic_texture_dataset/sensor_data/accel/{texture}/'
  selected=[r for r in rows if r['name'].startswith(prefix) and r['name'].endswith('.csv')]
  assert len(selected)==160,(texture,len(selected))
  a=min(x['offset'] for x in selected);last=max(selected,key=lambda x:x['offset'])
  following=min(x['offset'] for x in rows if x['offset']>last['offset'])
  parts.append({'name':f'accel_{texture:03d}.bin','start':a,'end':following-1,'records':selected})
 selected=[r for r in rows if '/sensor_data/position/' in r['name'] and r['name'].endswith('.csv')]
 a=min(x['offset'] for x in selected);last=max(selected,key=lambda x:x['offset']);following=min(x['offset'] for x in rows if x['offset']>last['offset'])
 parts.append({'name':'position_all.bin','start':a,'end':following-1,'records':selected})
 return parts
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
 dst=ROOT/'data/raw_archive_ranges';dst.mkdir(parents=True,exist_ok=True)
 parts=plan();(ROOT/'data/range_plan.json').write_text(json.dumps({'url':URL,'license':'CC BY 4.0','parts':parts},indent=2))
 log=ROOT/'data/download_events.jsonl';start=time.monotonic()
 def one(part):
  out=dst/part['name'];expected=part['end']-part['start']+1
  side=out.with_suffix('.json')
  if out.exists() and side.exists():
   rec=json.loads(side.read_text())
   if out.stat().st_size==expected and digest(out)==rec['sha256']:return {**rec,'status':'cached'}
  # A failed streaming response leaves a contiguous, validated prefix.
  # Resume that prefix in bounded requests instead of restarting a 40 MB part.
  for attempt in range(1,21):
   t=time.monotonic();tmp=out.with_suffix('.partial')
   try:
    count=tmp.stat().st_size if tmp.exists() else 0
    assert count<=expected
    etag=None
    while count<expected:
     a=part['start']+count;b=min(a+8*1024*1024-1,part['end'])
     with requests.get(URL,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(30,90)) as r:
      r.raise_for_status();assert r.status_code==206,r.status_code
      assert r.headers['Content-Range']==f'bytes {a}-{b}/15220748922'
      etag=r.headers.get('ETag')
      with tmp.open('ab') as f:
       for chunk in r.iter_content(256*1024):f.write(chunk);count+=len(chunk)
      assert count==b-part['start']+1
    rec={'name':part['name'],'status':'complete','attempt':attempt,'bytes':count,'sha256':digest(tmp),'start':part['start'],'end':part['end'],'etag':etag,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t,'transport':'resumable_8MiB'}
    tmp.replace(out);side.write_text(json.dumps(rec,indent=2));return rec
   except Exception as e:
    with log.open('a') as f:f.write(json.dumps({'name':part['name'],'attempt':attempt,'status':'failed','error':repr(e),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')
    if attempt==20:raise
    time.sleep(min(attempt*2,10))
 done=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  for future in concurrent.futures.as_completed([pool.submit(one,p) for p in parts]):
   rec=future.result();done.append(rec)
   with log.open('a') as f:f.write(json.dumps(rec)+'\n')
   print(json.dumps({'completed':len(done),'total':len(parts),'name':rec['name'],'GB':round(sum(x['bytes'] for x in done)/1e9,3),'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
 (ROOT/'data/download_manifest.json').write_text(json.dumps({'url':URL,'license':'CC BY 4.0','parts':done,'elapsed_seconds':time.monotonic()-start},indent=2))
