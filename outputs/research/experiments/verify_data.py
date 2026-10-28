"""Verify raw/processed lineage before model selection, including cached parts."""
import pathlib,json,hashlib,datetime,itertools
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((ROOT/'data/range_plan.json').read_text());out=[]
for part in plan['parts']:
 p=ROOT/'data/raw_archive_ranges'/part['name'];h=sha(p);m=json.loads(p.with_suffix('.json').read_text())
 assert p.stat().st_size==part['end']-part['start']+1 and h==m['sha256']
 row={'name':part['name'],'bytes':p.stat().st_size,'sha256':h}
 if part['name'].startswith('accel'):
  f=ROOT/'data/processed'/p.name.replace('.bin','.npz');v=json.loads(f.with_suffix('.version.json').read_text())
  assert v['source_range_sha256']==h and v['script_sha256']==sha(ROOT/'experiments/extract_features.py')
  row['processed_sha256']=sha(f)
 out.append(row)
z=np.load(ROOT/'data/processed/all_psd.npz');labels=z['labels'];q=json.loads((ROOT/'data/processed/quality.json').read_text())
assert set(map(tuple,labels))==set(itertools.product(range(118),range(0,360,45),[20,30,40,50,60],[500,1000],[0,1]))
assert [r['id'] for r in q]==['_'.join(map(str,l)) for l in labels]
raw_hashes=[r['source_sha256'] for r in q]
assert len(set(raw_hashes))==len(raw_hashes),'Exact duplicate accelerometer CSV content'
record={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(pathlib.Path(__file__)),'status':'pass','raw_parts':out,'unique_accel_csv_sha256':len(set(raw_hashes)),'valid_acquisitions':sum(r['valid'] for r in q),'acquisitions':len(labels),'source_and_cache_hashes_verified':True,'whole_array_sha256':sha(ROOT/'data/processed/all_psd.npz')}
(ROOT/'data/processed/integrity_check.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='raw_parts'}))
