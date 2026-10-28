"""Read-only acceptance audit of sealed external data; no classifier outcomes.

Full restricted-passive pickle decode and independent QC/hash reconstruction.
One eligible episode per class-speed cell gets author and independent spectral
replay. No producer main/publication/training entrypoints are invoked.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import collections,datetime,hashlib,itertools,json,pathlib,resource,time
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,resample_poly
import scipy
import torch
BASE=pathlib.Path(__file__).resolve().parents[4];ROOT=BASE/'outputs/research_v2';DATA=ROOT/'data/external';PROCESSED=DATA/'processed';OUT=ROOT/'audits/replay'
sys.path.insert(0,str(ROOT/'experiments'))
import extract_external as producer
CHECKS=[];HASHES=[];SPOTS=[];FOLDS=[];START=time.monotonic();CACHE={}
def read(path):return json.loads(pathlib.Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2*1024*1024),b''):h.update(b)
    return h.hexdigest()
def check(name,passed,**details):
    r=dict(check=name,passed=bool(passed),**details);CHECKS.append(r)
    if not passed:print(json.dumps({'FAILED':r}),flush=True)
def hc(path,expected,origin):
    path=pathlib.Path(path);key=str(path)
    if key not in CACHE:CACHE[key]=sha(path)
    actual=CACHE[key];passed=actual==expected
    HASHES.append(dict(path=str(path.relative_to(BASE)),expected=expected,actual=actual,origin=origin,passed=passed));check('sha256',passed,path=str(path.relative_to(BASE)),origin=origin)
def compare(name,a,b,atol=1e-12,rtol=1e-11,**details):
    maximum=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));check(name,np.allclose(a,b,atol=atol,rtol=rtol),maximum_absolute_difference=maximum,atol=atol,rtol=rtol,**details);return maximum
def progress(stage,**details):print(json.dumps(dict(stage=stage,elapsed_seconds=time.monotonic()-START,**details)),flush=True)

COLUMNS=['imu_ax','imu_ay','imu_az','imu_gx','imu_gy','imu_gz','imu_mx','imu_my','imu_mz']
def independent_quality(frame):
    assert list(frame.columns)==COLUMNS and isinstance(frame.index,pd.DatetimeIndex) and not frame.index.hasnans
    ns=frame.index.to_numpy(dtype='datetime64[ns]').view(np.int64);ns=ns-ns[0]
    values=np.array(frame.to_numpy(),dtype='<f8',order='C');acceleration=values[:,:3]
    delta=np.diff(ns);total=int(ns[-1]);reasons=[]
    if not np.isfinite(acceleration).all():reasons.append('nonfinite_acceleration')
    if np.any(delta<0):reasons.append('decreasing_timestamp')
    if total<4_400_000_000:reasons.append('duration_below4.4s')
    gaps=[]
    # Exact integer comparisons avoid reusing the producer's floating window tests.
    for numerator in [1,3,5,7]:
        overlap=(8*ns[:-1]<numerator*total+4_000_000_000)&(8*ns[1:]>numerator*total-4_000_000_000)
        gaps.append(int(delta[overlap].max())/1e9 if overlap.any() else None)
    if any(g is None or g>.05 for g in gaps):reasons.append('window_timestamp_gap_above50ms')
    canonical=values.copy();canonical[canonical==0]=0.0
    h=hashlib.sha256(json.dumps(COLUMNS,separators=(',',':')).encode()+np.asarray(ns,dtype='<i8').tobytes()+canonical.tobytes()).hexdigest()
    return dict(valid=not reasons,reasons=reasons,gaps=gaps,duration=total/1e9,negative_steps=int(np.sum(delta<0))),h,ns,acceleration

def independent_windows(ns,acceleration):
    unique,first,count=np.unique(ns,return_index=True,return_counts=True)
    means=np.add.reduceat(acceleration,first,axis=0)/count[:,None]
    duration=float(ns[-1])*1e-9;grid=np.arange(int(np.floor(duration*1500))+1)/1500
    interpolated=np.stack([np.interp(grid,unique*1e-9,means[:,a]) for a in range(3)],axis=1)
    filtered=sosfiltfilt(butter(6,100,btype='lowpass',fs=1500,output='sos'),interpolated,axis=0)
    down=resample_poly(filtered,1,3,axis=0,window=('kaiser',5.0),padtype='constant')
    centers=duration*np.array([1,3,5,7])/8;starts=np.rint((centers-.5)*500).astype(int)
    return np.array([down[start:start+500].T for start in starts])

def independent_features(windows):
    window=.5-.5*np.cos(2*np.pi*np.arange(256)/256)
    ps=[]
    for leg in windows:
        powers=[]
        for start in [0,128]:
            segment=leg[:,start:start+256];segment=segment-segment.mean(1)[:,None]
            power=np.abs(np.fft.rfft(segment*window,axis=1))**2/(500*np.sum(window**2));power[:,1:-1]*=2;powers.append(power)
        ps.append(np.mean(powers,axis=0).sum(0))
    psd=np.mean(ps,axis=0)
    parts=np.stack([windows[:,:,start:start+256] for start in [0,64,128,192]],axis=-1)
    parts=parts-parts.mean(2,keepdims=True);complex_stft=np.fft.rfft(parts*window[None,None,:,None],axis=2)/window.sum();power=(np.abs(complex_stft)**2).astype(np.float32)
    f=np.arange(129)*500/256;edges=700*(10**(np.linspace(2595*np.log10(1+5/700),2595*np.log10(1+100/700),34)/2595)-1);bank=np.zeros((32,129),np.float32)
    for j in range(32):
        up=(f>=edges[j])&(f<=edges[j+1]);down=(f>edges[j+1])&(f<=edges[j+2]);bank[j,up]=(f[up]-edges[j])/(edges[j+1]-edges[j]);bank[j,down]=(edges[j+2]-f[down])/(edges[j+2]-edges[j+1])
    mel=np.array([[bank@power[leg,axis] for axis in range(3)] for leg in range(4)],np.float32)
    peak=np.maximum(mel.max(axis=(2,3),keepdims=True),np.float32(1e-20));db=np.clip(10*np.log10(np.maximum(mel/peak,np.float32(1e-8))),-80,0)
    joined=np.concatenate(list(db),axis=-1);scaled=(joined/40+1).astype(np.float32)
    # Independent separable half-pixel bilinear resize 32x16 -> 128x64.
    for axis,size in [(1,128),(2,64)]:
        position=np.clip((np.arange(size,dtype=np.float32)+.5)*scaled.shape[axis]/size-.5,0,scaled.shape[axis]-1)
        left=np.floor(position).astype(int);right=np.minimum(left+1,scaled.shape[axis]-1);alpha=(position-left).astype(np.float32)
        shape=[1,1,1];shape[axis]=size;alpha=alpha.reshape(shape)
        scaled=np.take(scaled,left,axis=axis)*(1-alpha)+np.take(scaled,right,axis=axis)*alpha
    return psd,scaled.astype(np.float16)

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    manifest=read(PROCESSED/'manifest.json');decision=read(ROOT/'research/external_validity_decision.json');decision_freeze=read(ROOT/'research/external_validity_decision.freeze.json');quality=read(DATA/'quality_audit.json');records=read(DATA/'trial_integrity.json');eligibility=read(DATA/'final_eligibility_audit.json');plan=read(DATA/'acquisition_plan.json')
    hc(ROOT/'research/external_validity_decision.json',decision_freeze['sha256'],'root QC decision freeze')
    for path,h in decision['input_sha256'].items():hc(ROOT/path,h,'root QC decision inputs')
    file_bindings={'decision_sha256':manifest['decision_file'],'decision_freeze_sha256':'research/external_validity_decision.freeze.json','producer_sha256':manifest['producer_file'],'publication_helper_sha256':manifest['publication_helper_file'],'download_helper_sha256':manifest['download_helper_file'],'original_quality_audit_sha256':'data/external/quality_audit.json','trial_integrity_sha256':'data/external/trial_integrity.json','protocol_sha256':manifest['protocol_file'],'protocol_freeze_sha256':'research/external_protocol.freeze.json','acquisition_plan_sha256':'data/external/acquisition_plan.json','blocked_tensor_sha256':manifest['blocked_tensor_file'],'final_eligibility_audit_sha256':manifest['final_eligibility_audit_file']}
    for field,path in file_bindings.items():hc(ROOT/path,manifest[field],'published common-input manifest')
    for name,details in manifest['artifacts'].items():hc(PROCESSED/name,details['sha256'],'published data artifact');check('data file size', (PROCESSED/name).stat().st_size==details['bytes'],file=name)
    check('manifest disposition/settings',manifest['status']=='complete' and manifest['settings']==producer.SETTINGS==quality['settings'] and quality['status']=='needs_root_decision' and not eligibility['eligibility_criteria_changed'] and not manifest['classification_outcomes_computed'])
    check('QC decision predates published data',decision['utc']<=decision_freeze['utc']<manifest['utc'])
    check('444 raw blocks/3600 listed episodes sealed',plan['status']=='complete' and len(plan['blocks'])==444 and sum(len(b['members']) for b in plan['blocks'])==3600 and all(b['status']=='complete' for b in plan['blocks']))
    for block in plan['blocks']:hc(DATA/block['path'],block['sha256'],'complete acquisition block');check('range byte bounds',(DATA/block['path']).stat().st_size==block['bytes']==block['end']-block['start']+1,path=block['path'])
    index={a['name']:a for a in read(DATA/'zip_index.json')['members']};locations={name:(index[name],block) for block in plan['blocks'] for name in block['members']};ordered=sorted(locations)
    check('raw record inventory exact/order',len(records)==3600 and [r['member'] for r in records]==ordered and {tuple(r['labels']) for r in records}==set(itertools.product(range(12),[30,35,40],range(1,101))))
    labels=np.load(PROCESSED/'labels.npy');metadata=pd.read_csv(PROCESSED/'metadata.csv',keep_default_na=False);valid_mask=np.load(PROCESSED/'final_valid_mask.npy');psd=np.load(PROCESSED/'psd.npz');images=np.load(PROCESSED/'input.npy',mmap_mode='r');old=np.load(ROOT/manifest['blocked_tensor_file'],mmap_mode='r')
    chosen=np.asarray(decision['eligible_original_rows']);row_from_original={int(a):i for i,a in enumerate(chosen)}
    check('3550 exact eligible ordered IDs and mask',len(chosen)==len(np.unique(chosen))==3550 and np.array_equal(np.flatnonzero(valid_mask),chosen) and metadata.id.tolist()==decision['eligible_ids']==[records[i]['id'] for i in chosen] and np.array_equal(metadata.row,np.arange(3550)) and np.array_equal(metadata.row_original,chosen) and metadata.id.is_unique)
    check('label representations identical',np.array_equal(labels,psd['labels']) and np.array_equal(labels,metadata[['class_id','speed','trial']]) and np.array_equal(labels,np.array([records[i]['labels'] for i in chosen])))
    check('source2354 target1196 and fold metadata',(metadata.source==1).sum()==2354 and (metadata.source==0).sum()==1196 and np.array_equal(metadata.source,np.isin(labels[:,1],[30,35]).astype(int)) and np.array_equal(metadata.cv_fold.astype(int),labels[:,2]%3) and np.array_equal(metadata.domain,np.where(labels[:,1]==40,'speed','source')))
    check('PSD/full tensor shape/frequency/nonnegative',psd['psd'].shape==(3550,129) and np.isfinite(psd['psd']).all() and (psd['psd']>=0).all() and np.array_equal(psd['frequency'],np.arange(129)*500/256) and images.shape==(3550,3,128,64) and images.dtype==np.float16)
    for start in range(0,3550,128):
        block=np.asarray(images[start:start+128]);check('all eligible tensor values match blocked original bitwise',np.array_equal(block,old[chosen[start:start+128]]) and np.isfinite(block).all() and block.min()>=-1 and block.max()<=1,start=start,rows=len(block))
    spot_original={int(chosen[np.flatnonzero((labels[:,0]==c)&(labels[:,1]==s))[((c*13+s)%np.sum((labels[:,0]==c)&(labels[:,1]==s)))]]) for c in range(12) for s in [30,35,40]}
    original_valid=[];raw_hashes={};canonical_hashes={};qcreasons={};recomputed=[]
    for original,name in enumerate(ordered):
        if time.monotonic()-START>540:raise TimeoutError('External acceptance active execution ceiling')
        member,block=locations[name];blob=producer.member_payload(member,block);raw_sha=hashlib.sha256(blob).hexdigest();frame,inspection=producer.load_passive_dataframe(blob)
        q,content,ns,acceleration=independent_quality(frame);record=records[original]
        check('independent all-episode QC and content digests',q['valid']==record['valid'] and q['reasons']==record['reasons'] and raw_sha==record['source_pickle_sha256'] and content==record['normalized_content_sha256'] and not inspection['forbidden_globals'],id=record['id'])
        compare('independent integer window-gap diagnostic',q['gaps'],record['window_max_timestamp_gap_seconds'],atol=1e-12,rtol=1e-12,id=record['id'])
        original_valid.append(q['valid']);raw_hashes[record['id']]=raw_sha;canonical_hashes[record['id']]=content;qcreasons[record['id']]=q['reasons'];recomputed.append(q)
        if original in spot_original:
            aq,aw,ah=producer.quality_and_windows(frame);freq,ap,ai=producer.spectral_features(aw);pubrow=row_from_original[original]
            compare('author PSD from restricted original payload',ap,psd['psd'][pubrow],atol=0,rtol=0,id=record['id']);check('author tensor bitwise replay',np.array_equal(ai,images[pubrow]),id=record['id'])
            iw=independent_windows(ns,acceleration);we=compare('independent window construction',iw,aw,atol=1e-11,rtol=1e-11,id=record['id']);ip,ii=independent_features(iw)
            pe=compare('independent explicit Hann/rFFT Welch',ip,psd['psd'][pubrow],atol=1e-12,rtol=1e-10,id=record['id']);ie=compare('independent STFT/mel/resize tensor',ii.astype(float),images[pubrow].astype(float),atol=.001,rtol=0,id=record['id'])
            SPOTS.append(dict(id=record['id'],original_row=original,published_row=pubrow,window_max_abs=we,psd_max_abs=pe,tensor_max_abs=ie,author_tensor_bitwise_equal=bool(np.array_equal(ai,images[pubrow]))))
        if (original+1)%500==0:progress('raw QC independently checked',episodes=original+1)
    valid=np.array(original_valid);invalid_ids=[records[i]['id'] for i in np.flatnonzero(~valid)];by_speed={str(s):int(sum(not q['valid'] and records[i]['labels'][1]==s for i,q in enumerate(recomputed))) for s in [30,35,40]}
    check('exact49 QC exclusions unchanged',int((~valid).sum())==49 and by_speed==decision['quality_exclusions_by_speed']=={'30':31,'35':14,'40':4} and set(invalid_ids)=={r['id'] for r in decision['exclusions'] if r['reasons']!=['exact_duplicate_episode']})
    groups=collections.defaultdict(list)
    for identifier,h in canonical_hashes.items():groups[h].append(identifier)
    duplicate_groups=[sorted(ids) for ids in groups.values() if len(ids)>1]
    check('sole exact canonical duplicate and fixed retention',duplicate_groups==[['03_35_001','03_35_083']] and raw_hashes['03_35_001']==raw_hashes['03_35_083'] and '03_35_001' in decision['eligible_ids'] and '03_35_083' not in decision['eligible_ids'])
    keep=valid.copy();keep[next(i for i,r in enumerate(records) if r['id']=='03_35_083')]=False
    check('published mask equals independent QC minus fixed duplicate',np.array_equal(keep,valid_mask) and len({canonical_hashes[r] for r in decision['eligible_ids']})==3550 and len({raw_hashes[r] for r in decision['eligible_ids']})==3550)
    cells={f'{c:02d}_{s}':int(np.sum((labels[:,0]==c)&(labels[:,1]==s))) for c in range(12) for s in [30,35,40]};check('all36 class-speed cells and counts',cells==decision['eligible_by_class_speed']==eligibility['class_speed_counts'] and min(cells.values())==93 and max(cells.values())==100)
    source=labels[:,1]!=40;target=~source;preflight=read(ROOT/'experiments/cnn/external_split_preflight.json');svm_split=read(ROOT/'experiments/calibration/external/external40_split.json')
    for count in [3,5]:
        partitions=[]
        for fold in range(count):
            va=source&(labels[:,2]%count==fold);tr=source&~va;partitions.append(va)
            class_counts=lambda mask:{str(c):int(np.sum(mask&(labels[:,0]==c))) for c in range(12)}
            detail=dict(fold_count=count,fold=fold,train_rows=int(tr.sum()),validation_rows=int(va.sum()),train_by_class=class_counts(tr),validation_by_class=class_counts(va))
            check('independent episode folds disjoint/class complete',not np.any(tr&va) and not np.any(target&(tr|va)) and np.array_equal(tr|va,source) and min(detail['train_by_class'].values())>0 and min(detail['validation_by_class'].values())>0,fold_count=count,fold=fold)
            if count==3:
                saved=preflight['folds'][str(fold)];train_ids=metadata.loc[tr,'id'];val_ids=metadata.loc[va,'id']
                check('CNN preflight ID hash/class counts',hashlib.sha256('\n'.join(train_ids).encode()).hexdigest()==saved['training_ids_sha256'] and hashlib.sha256('\n'.join(val_ids).encode()).hexdigest()==saved['validation_ids_sha256'] and saved['training_by_class']==detail['train_by_class'] and saved['validation_by_class']==detail['validation_by_class'],fold=fold)
            else:
                saved=svm_split['folds'][fold];check('SVM saved split equals independent modulo5',np.array_equal(saved['train_indices'],np.flatnonzero(tr)) and np.array_equal(saved['validation_indices'],np.flatnonzero(va)),fold=fold)
            FOLDS.append(detail)
        check('validation partitions cover source exactly once',np.array_equal(np.sum(partitions,axis=0),source.astype(int)),fold_count=count)
    check('SVM source/target saved masks',np.array_equal(svm_split['train_indices'],np.flatnonzero(source)) and np.array_equal(svm_split['test_indices'],np.flatnonzero(target)))
    status='pass_data_acceptance_only' if all(c['passed'] for c in CHECKS) else 'failed_checks'
    limits=['All 3600 raw IMU episodes were reparsed only through the existing opcode-inspected allowlisted passive NumPy/Pandas reconstruction. The restricted parser and ZIP payload function are shared author code; QC and canonical-content recomputation are independent.',
        'Spectral replay covers 36 eligible episodes, one per class-speed cell. Independent window interpolation/aggregation and Hann FFT PSD were implemented; Butterworth zero-phase filtering and polyphase resampling use the same SciPy primitives. Tensor comparison includes float16 quantization differences, with exact author-path replay checked separately.',
        'All 444 acquired range hashes were verified, but no full 3.83GB archive digest is claimed: unrequested archive modalities were never downloaded.',
        'Exclusions are uneven (31/14/4 by speed), and the 1% alert remains visibly exceeded at 49/3600 = 1.3611%. The separate root decision keeps the existing per-episode QC thresholds and removes one exact duplicate. This does not establish missing-at-random sampling, non-exact deduplication, session independence or measured physical contact speed.',
        'This acceptance covers data and split integrity only. No external classifier predictions, selection outcomes, checkpoints or partial training ledgers were read. It does not authorize target inference before the separate global SVM/CNN source-model gate.']
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,elapsed_seconds=time.monotonic()-START,command='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 work/research/venv/bin/python outputs/research_v2/audits/replay/audit_external_data.py',environment={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'cpu_threads':1},check_count=len(CHECKS),failed_checks=[c for c in CHECKS if not c['passed']],hash_checks=HASHES,raw_episodes_reparsed=3600,eligible=3550,source=2354,target=1196,quality_exclusions=49,additional_exact_duplicate_exclusions=1,duplicate_groups=duplicate_groups,independent_spectral_spots=SPOTS,folds=FOLDS,checks=CHECKS,limits=limits,script_sha256=sha(__file__),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (OUT/'external_data_audit.json').write_text(json.dumps(result,indent=2))
    md=['# External sealed data acceptance audit','',f"Status: **{status}**. {len(CHECKS)} checks; {len(result['failed_checks'])} failed. All 3,600 raw episodes reparsed through the restricted producer; all 444 acquired block hashes match.",'','- 3,550 eligible unique episodes: 2,354 source (30/35 mm/s) and 1,196 target (40 mm/s).','- All 49 original QC exclusions reproduced independently; 03_35_083 is the sole additional exact duplicate removed, retaining 03_35_001.','- All 36 class-speed counts match; modulo-3 and modulo-5 folds are disjoint, cover source episodes exactly once in validation, retain every class and match saved split records.','- All published eligible CNN tensors match their original blocked-run tensors bitwise.','- Independent raw-to-window and spectral replay: one eligible episode per class-speed cell (36); author-path PSD/tensor outputs reproduce saved data exactly.',f"- Maximum independent PSD difference: {max(r['psd_max_abs'] for r in SPOTS):.6g}; maximum independent float16 tensor difference: {max(r['tensor_max_abs'] for r in SPOTS):.6g}.",'',f"Command: `{result['command']}`",'',f"Active execution: {result['elapsed_seconds']:.1f}s, one CPU thread, existing environment. Author code, data and models were not changed.",'','## Failed checks','']
    md += [f'- `{json.dumps(c)}`' for c in result['failed_checks']] or ['None.'];md+=['','## Acceptance boundaries','']+[f'- {s}' for s in limits]
    (OUT/'external_data_audit.md').write_text('\n'.join(md)+'\n');progress('external data audit complete',status=status,failed=len(result['failed_checks']))
    return 0 if not result['failed_checks'] else 1
if __name__=='__main__':raise SystemExit(main())
