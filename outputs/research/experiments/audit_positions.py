"""Outcome-blind motion audit; flags are descriptive, never exclusion criteria."""
import pathlib,json,hashlib,datetime
import numpy as np
from extract_features import decode,parse_csv
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 plan=json.loads((ROOT/'data/range_plan.json').read_text())
 p=next(p for p in plan['parts'] if p['name']=='position_all.bin')
 p={**p,'bytes':(ROOT/'data/raw_archive_ranges'/p['name']).read_bytes()}
 quality=json.loads((ROOT/'data/processed/quality.json').read_text());qmap={q['id']:q for q in quality};rows=[]
 for e in sorted(p['records'],key=lambda x:x['name']):
  fid=pathlib.Path(e['name']).stem;speed=int(fid.split('_')[2]);q=qmap[fid]
  arr=parse_csv(decode(p,e),3);t=arr[:,0];xy=arr[:,1:];assert np.isfinite(arr).all() and np.all(np.diff(t)>=0)
  center=(q['start']+q['end'])/2;a=center-.5;b=center+.5
  # Position stream measures commanded stage path at approximately 100 Hz.
  covered=bool(t[0]<=a and t[-1]>=b)
  grid=np.linspace(a,b,101);pos=np.stack([np.interp(grid,t,xy[:,j]) for j in range(2)],axis=1)
  mean=float(np.linalg.norm(pos[-1]-pos[0]));moving=np.linalg.norm(pos[10:]-pos[:-10],axis=1)/.1
  rows.append({'id':fid,'covered':covered,'commanded_speed_mm_s':speed,'central_endpoint_speed_mm_s':mean,'speed_ratio':mean/speed,'under_80_percent':bool(mean<.8*speed),'has_100ms_window_below_10_percent':bool(np.any(moving<.1*speed)),'position_crc32':e['crc32'],'duration_s':float(t[-1]-t[0]),'whole_path_endpoint_distance_mm':float(np.linalg.norm(xy[-1]-xy[0]))})
 assert set(x['id'] for x in rows)==set(qmap) and len(rows)==18880
 ratios=np.array([r['speed_ratio'] for r in rows]);out={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'position_range_sha256':hashlib.sha256(p['bytes']).hexdigest(),'flags_only_no_exclusions':True,'interpretation':'Recorded stage positions, not an independent measurement of contact slip. Endpoints checked for the central 1 s accelerometer window.','n':len(rows),'summary':{'uncovered':sum(not r['covered'] for r in rows),'under_80_percent':sum(r['under_80_percent'] for r in rows),'slow_100ms_windows':sum(r['has_100ms_window_below_10_percent'] for r in rows),'speed_ratio_quantiles':dict(zip(['min','q01','median','q99','max'],np.quantile(ratios,[0,.01,.5,.99,1]).tolist()))},'records':rows}
 (ROOT/'data/processed/position_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['summary']))
if __name__=='__main__':main()
