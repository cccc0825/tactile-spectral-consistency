#!/usr/bin/env python3
"""Prepare and check a new source-fit tree; never run target prediction phases.

Original producers/protocols are copied byte-for-byte. Derived QC decisions and
handoffs bind the new tree. No dependency installation or large file copying.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,ast,collections,datetime,hashlib,importlib.metadata,json,pathlib,shutil,subprocess,sys,time,traceback
sys.dont_write_bytecode=True
V1=pathlib.Path('outputs/research');V2=pathlib.Path('outputs/research_v2')
FILES=[V1/'experiments/download_data.py',V1/'experiments/extract_features.py',
       V1/'corpus/archive_index.json',V1/'research/protocol_v1.md']
FILES += [V2/'experiments'/p for p in ['run_calibration.py','response_calibration.py','run_external_calibration.py',
    'estimator_diagnostics.py','download_external.py','extract_external.py','extract_cluster_spectrograms.py',
    'cnn_baseline.py','baseline_protocol.md']]
FILES += [V2/'research'/p for p in ['calibration_protocol.md','calibration_protocol.freeze.json',
    'diagnostic_protocol.md','diagnostic_protocol.freeze.json','external_protocol.md','external_protocol.freeze.json',
    'external_settings.json','record_external_validity_decision.py']]
FILES += [V2/'experiments/cnn'/p for p in ['configure_external.py','source_eval_audit.py',
    'run_external_baseline.py','external_baseline_protocol.md']]
FILES += [V2/'data/external/publish_under_decision.py',V2/'environment/requirements.txt',V2/'environment/runtime.json',
          V2/'tools/prepare_fresh_run.py']
OFFICIAL=pathlib.Path('work/revision/vendor/cluster-official')
FILES += [OFFICIAL/p for p in ['LICENSE','train/classification/ResnetClassification.py',
    'train/datasets/accel_dataset_mel.py','train/haptics_texture_classification.py']]
FILES += [pathlib.Path('work/revision')/p for p in ['mendeley_state.js','mendeley_files.json','mendeley_landing.html']]
STEPS={
 'acquire-cluster':['outputs/research/experiments/download_data.py','--workers','2'],
 'extract-cluster':['outputs/research/experiments/extract_features.py'],
 'extract-fixed-space':['outputs/research_v2/experiments/estimator_diagnostics.py','extract'],
 'select-cluster-svm':['outputs/research_v2/experiments/run_calibration.py','cluster','select'],
 'extract-cluster-cnn':['outputs/research_v2/experiments/extract_cluster_spectrograms.py'],
 'configure-cluster-cnn':['outputs/research_v2/experiments/cnn_baseline.py','configure-cluster'],
 'select-cluster-cnn':['outputs/research_v2/experiments/cnn_baseline.py','select'],
 'acquire-external':['outputs/research_v2/experiments/download_external.py','--mode','all','--workers','2','--max-block-mib','2'],
 'smoke-external':['outputs/research_v2/experiments/extract_external.py','--mode','smoke'],
 'extract-external':['outputs/research_v2/experiments/extract_external.py','--mode','extract','--protocol-file',
     'outputs/research_v2/research/external_protocol.md','--freeze-file','outputs/research_v2/research/external_protocol.freeze.json'],
 'external-qc-decision':['outputs/research_v2/research/record_external_validity_decision.py'],
 'publish-external':['outputs/research_v2/data/external/publish_under_decision.py'],
 'external-handoff':None,
 'configure-external-cnn':['outputs/research_v2/experiments/cnn/configure_external.py'],
 'select-external-svm':['outputs/research_v2/experiments/run_external_calibration.py','select'],
 'select-external-cnn':['outputs/research_v2/experiments/cnn/run_external_baseline.py','select'],
}
WEIGHT=V2/'experiments/cnn/resnet34-b627a593.pth'
WEIGHT_HASH='b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as handle:
        for b in iter(lambda:handle.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,value):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as handle:json.dump(value,handle,indent=2)
def require(condition,message):
    if not condition:raise ValueError(message)
def relative(value):
    p=pathlib.Path(value)
    require(not p.is_absolute() and '..' not in p.parts,'Unsafe relative path')
    return p
def manifest(root):
    m=read(root/'FRESH_RUN.json')
    require(m['kind']=='new_source_fit_tree' and root!=pathlib.Path(m['original_source_root']).resolve(),
            'This command requires a separately prepared tree')
    for name,digest in m['source_files_sha256'].items():
        p=root/relative(name)
        require(p.is_file() and p.resolve().is_relative_to(root) and sha(p)==digest,'Prepared source drift: '+name)
    return m

def prepare(source,root):
    require(not root.exists(),'Fresh destination must not exist')
    require(root!=source and not root.is_relative_to(source/V1) and not root.is_relative_to(source/V2),
            'Fresh tree cannot be inside a historical research output tree')
    missing=[str(p) for p in FILES if not (source/p).is_file()]
    require(not missing,'Missing preparation inputs: '+repr(missing))
    root.mkdir(parents=True)
    hashes={};sizes={}
    for rel in FILES:
        p=source/rel
        require(p.resolve().is_relative_to(source),'Preparation input resolves outside source tree')
        digest=sha(p);target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
        require(sha(p)==digest==sha(target),'Input changed while copying: '+str(rel))
        if target.suffix=='.py':compile(ast.parse(target.read_text()),str(target),'exec')
        hashes[str(rel)]=digest;sizes[str(rel)]=target.stat().st_size
    m=dict(kind='new_source_fit_tree',created_utc=now(),original_source_root=str(source),
        source_files_sha256=hashes,copied_bytes=sum(sizes.values()),copied_files=len(hashes),
        dependencies=[dict(path=name,bytes=sizes[name],sha256=digest,
                           outside_outputs=not name.startswith('outputs/')) for name,digest in hashes.items()],
        official_code_commit='e05d6b022d127e24f73583146f0aa229c6934449',
        dataset_acquisition='original public downloaders; no raw data copied by setup',
        qc_policy='unchanged frozen producer, exactly 49 invalid episodes and one canonical duplicate removed; otherwise stop',
        target_prediction_authorized=False,external_release_copied=False,packages_installed=False)
    write(root/'FRESH_RUN.json',m)
    commands={k:([sys.executable,'-B']+v if v else [sys.executable,'-B',str(V2/'tools/prepare_fresh_run.py'),
        'run','--project-root',str(root),'--step',k]) for k,v in STEPS.items()}
    write(root/'reproduction/source_steps.json',dict(created_utc=now(),commands=commands,
        note='Execute through helper run --step for receipts; all commands stop before target inference.',
        required_weight=dict(path=str(WEIGHT),sha256=WEIGHT_HASH,url='https://download.pytorch.org/models/resnet34-b627a593.pth')))
    print(json.dumps(dict(status='prepared',root=str(root),files=len(hashes),bytes=sum(sizes.values()),
          next_command=[sys.executable,'-B',str(root/V2/'tools/prepare_fresh_run.py'),'check','--project-root',str(root)]),indent=2))

def check(root):
    m=manifest(root)
    scripts={}
    for rel in FILES:
        if rel.suffix=='.py':
            tree=ast.parse((root/rel).read_text())
            flags=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
                    flags.append([a.value for a in node.args if isinstance(a,ast.Constant) and isinstance(a.value,str)])
            scripts[str(rel)]={'syntax_valid':True,'declared_arguments':flags}
    packages={}
    for name in ['numpy','scipy','pandas','scikit-learn','joblib','torch','torchvision','requests']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    required={
        'cluster_raw_plan':V1/'data/range_plan.json','cluster_psd':V1/'data/processed/all_psd.npz',
        'external_download':V2/'data/external/acquisition_plan.json','external_qc':V2/'data/external/quality_audit.json',
        'external_publication':V2/'data/external/processed/manifest.json','external_handoff':V2/'data/external/handoff.json',
        'cluster_cnn_template':V2/'experiments/cnn/cluster_config.json','imagenet_weights':WEIGHT}
    state={k:{'path':str(p),'present':(root/p).is_file()} for k,p in required.items()}
    if (root/WEIGHT).is_file():state['imagenet_weights']['hash_matches']=sha(root/WEIGHT)==WEIGHT_HASH
    result=dict(timestamp_utc=now(),status='source_setup_passed',source_files_verified=len(m['source_files_sha256']),
        scripts=scripts,packages_observed=packages,inputs=state,all_stage_commands_exist=all(v is None or (root/v[0]).is_file() for v in STEPS.values()),
        network_used=False,producer_executed=False,training_executed=False,target_inference_executed=False,
        caveat='This validates setup/source syntax and reports missing staged inputs; it does not claim extraction or training completed.')
    path=root/'reproduction'/('preflight_'+str(time.time_ns())+'.json');write(path,result)
    print(json.dumps(dict(status=result['status'],receipt=str(path),missing_inputs=[k for k,v in state.items() if not v['present']]),indent=2))

def handoff(root):
    """Validate this run's publication, then mint new handoff and source folds."""
    import numpy as np
    import csv
    data=root/V2/'data/external';cnn=root/V2/'experiments/cnn';research=root/V2/'research'
    for p in [data/'handoff.json',data/'publication_checks.json',cnn/'external_split_preflight.json']:
        require(not p.exists(),'Refusing to overwrite new-run binding: '+str(p))
    for p in [cnn/'external/selected.json',cnn/'external/selection.json',root/V2/'experiments/calibration/external/selected.json']:
        require(not p.exists(),'Handoff must precede this run\'s model selection')
    decision=read(research/'external_validity_decision.json');freeze=read(research/'external_validity_decision.freeze.json')
    require(sha(research/'external_validity_decision.json')==freeze['sha256'],'New decision freeze mismatch')
    for rel,digest in decision['input_sha256'].items():require(sha(root/V2/relative(rel))==digest,'New QC input changed: '+rel)
    publication=read(data/'processed/manifest.json')
    require(publication['status']=='complete' and publication['rows']==3550 and
            publication['all_eligible_cnn_tensors_bitwise_equal_to_blocked_run'] and
            not publication['classification_outcomes_computed'],'Publication is not eligible for source configuration')
    require(publication['decision_sha256']==sha(research/'external_validity_decision.json'),'Publication bound to another decision')
    require(publication['final_eligibility_audit_sha256']==sha(data/'final_eligibility_audit.json'),'Publication audit changed')
    for name,record in publication['artifacts'].items():
        require(sha(data/'processed'/relative(name))==record['sha256'],'Published artifact changed: '+name)
    labels=np.load(data/'processed/labels.npy',allow_pickle=False)
    mask=np.load(data/'processed/final_valid_mask.npy',allow_pickle=False)
    psd=np.load(data/'processed/psd.npz',allow_pickle=False)
    images=np.load(data/'processed/input.npy',allow_pickle=False,mmap_mode='r')
    require(labels.shape==(3550,3) and images.shape==(3550,3,128,64) and images.dtype==np.float16,'Publication shape mismatch')
    require(np.array_equal(psd['labels'],labels) and np.isfinite(psd['psd']).all() and (psd['psd']>=0).all(),'PSD label/value mismatch')
    require(np.array_equal(np.flatnonzero(mask),decision['eligible_original_rows']),'Decision mask mismatch')
    with (data/'processed/metadata.csv').open() as stream:rows=list(csv.DictReader(stream))
    require([r['id'] for r in rows]==decision['eligible_ids'],'Metadata decision IDs mismatch')
    require(np.array_equal(labels,np.array([[int(r[k]) for k in ['class_id','speed','trial']] for r in rows])),'Metadata label mismatch')
    for offset in range(0,len(images),128):
        chunk=images[offset:offset+128]
        require(np.isfinite(chunk).all() and chunk.min()>=-1 and chunk.max()<=1,'CNN input range/finite check failed')
    source=[r for r in rows if int(r['speed']) in [30,35]]
    require(len(source)==2354 and sum(int(r['speed'])==40 for r in rows)==1196,'Source/target counts differ')
    require(all(int(r['source'])==int(int(r['speed']) in [30,35]) and r['cv_fold']==str(int(r['trial'])%3) for r in rows),'Frozen fold/source assignment changed')
    folds={}
    for held in range(3):
        tr=[r for r in source if int(r['trial'])%3!=held];va=[r for r in source if int(r['trial'])%3==held]
        require({int(r['class_id']) for r in tr}=={int(r['class_id']) for r in va}==set(range(12)),'Source fold lacks a class')
        folds[str(held)]={'training_n':len(tr),'validation_n':len(va),
            'training_ids_sha256':hashlib.sha256('\n'.join(r['id'] for r in tr).encode()).hexdigest(),
            'validation_ids_sha256':hashlib.sha256('\n'.join(r['id'] for r in va).encode()).hexdigest()}
    checks=dict(timestamp_utc=now(),all_passed=True,manifest_sha256=sha(data/'processed/manifest.json'),
        checks={'artifact_sha256':True,'shapes_and_dtype':True,'cnn_values_finite_and_bounded':True,
            'psd_finite_nonnegative':True,'metadata_label_alignment':True,'decision_row_alignment':True,
            'common_mask_alignment':True,'cnn_folds_unchanged':True,'new_qc_input_hashes':True},
        classification_outcomes_computed=False,validator_sha256=sha(__file__))
    write(data/'publication_checks.json',checks)
    write(cnn/'external_split_preflight.json',dict(timestamp_utc=now(),stage='new-run before external classifier fitting',
        eligibility_decision_sha256=sha(research/'external_validity_decision.json'),source_n=2354,target_n=1196,
        fold_rule='actual original trial integer modulo 3',folds=folds,all_classes_in_each_training_and_validation_fold=True))
    files={name:data/'processed'/name for name in ['input.npy','labels.npy','metadata.csv','psd.npz','final_valid_mask.npy']}
    files.update(manifest=data/'processed/manifest.json',validity_decision=research/'external_validity_decision.json',
        decision_freeze=research/'external_validity_decision.freeze.json',protocol=research/'external_protocol.md',
        protocol_freeze=research/'external_protocol.freeze.json',final_eligibility_audit=data/'final_eligibility_audit.json',
        original_quality_audit=data/'quality_audit.json',checks=data/'publication_checks.json',
        producer=root/V2/'experiments/extract_external.py',publication_helper=data/'publish_under_decision.py',
        download_helper=root/V2/'experiments/download_external.py')
    write(data/'handoff.json',dict(timestamp_utc=now(),status='sealed',rows=3550,source=2354,target=1196,checks_passed=True,
        files={k:{'path':str(p),'sha256':sha(p)} for k,p in files.items()},classification_outcomes_computed=False,
        origin='new run recomputed publication and QC bindings; no historical release reused',preparer_sha256=sha(__file__)))

def ensure_budget(root):
    """Bind the copied 180-minute addendum to this run's own ledger at a boundary."""
    cnn=root/V2/'experiments/cnn';ledger=cnn/'compute_budget.json'
    data=read(ledger) if ledger.exists() else {'limit_seconds':9000,'consumed_seconds':0.,'history':[],'active_session':None}
    require(not data.get('active_session'),'Cannot change an active compute ledger')
    auth=cnn/'compute_budget_amendment_02.json';applied=cnn/'compute_budget_amendment_02.applied.json'
    if auth.exists() or applied.exists():
        require(auth.exists() and applied.exists() and data['limit_seconds']==10800,'Incomplete budget binding; review preserved files')
        return
    require(data['limit_seconds']==9000,'Unexpected initial fresh-run compute cap')
    old_hash=sha(ledger) if ledger.exists() else None
    if ledger.exists():shutil.copyfile(ledger,cnn/'compute_budget_before_fresh_amendment.json')
    record=dict(timestamp_utc=now(),status='new_run_same_frozen_resource_bound',previous_limit_seconds=9000,new_limit_seconds=10800,
        source_addendum_sha256=sha(cnn/'external_baseline_protocol.md'),old_ledger_sha256=old_hash,
        basis='Initialize this reproduction run under copied external addendum; no prior consumed time reset, grid or seeds changed',paid_compute=False)
    write(auth,record)
    data.update(limit_seconds=10800,last_update_utc=now(),scope='this new run only; preserved consumed time, no historical sessions imported')
    temporary=ledger.with_suffix('.json.new')
    write(temporary,data);temporary.replace(ledger)
    write(applied,dict(timestamp_utc=now(),status='applied_to_new_run',previous_limit_seconds=9000,new_limit_seconds=10800,
        consumed_seconds_preserved=data['consumed_seconds'],authorization_sha256=sha(auth),amended_ledger_sha256=sha(ledger),preparer_sha256=sha(__file__)))

def qc_preflight(root):
    data=root/V2/'data/external';rows=read(data/'trial_integrity.json');quality=read(data/'quality_audit.json')
    expected={(c,s,t) for c in range(12) for s in [30,35,40] for t in range(1,101)}
    require(len(rows)==3600 and {tuple(r['labels']) for r in rows}==expected,'QC episode inventory differs from fixed public dataset')
    require(sum(not r['valid'] for r in rows)==quality['invalid']==49,'QC exclusions differ; no automatic policy relaxation')
    groups=collections.defaultdict(list)
    for r in rows:
        if r['valid']:groups[r['normalized_content_sha256']].append(r)
    excluded={r['id'] for values in groups.values() for r in sorted(values,key=lambda x:x['id'])[1:]}
    require(excluded=={'03_35_083'},'Duplicate disposition differs from fixed same-data policy')
    eligible=[r for r in rows if r['valid'] and r['id'] not in excluded]
    require(sum(r['labels'][1] in [30,35] for r in eligible)==2354 and
            sum(r['labels'][1]==40 for r in eligible)==1196,'Eligible source/target counts changed')

def execute(root,step,timeout):
    manifest(root)
    stamp=str(time.time_ns());audit=root/'reproduction'/f'{step}_{stamp}.json';log=audit.with_suffix('.log')
    record=dict(step=step,started_utc=now(),cwd=str(root),training_requested=step.startswith('select-'),
                target_prediction_requested=False,timeout_seconds=timeout)
    started=time.monotonic()
    try:
        if step=='external-handoff':
            handoff(root);record['exit_code']=0
        else:
            if step=='external-qc-decision':qc_preflight(root)
            if step=='configure-external-cnn':ensure_budget(root)
            args=[sys.executable,'-B']+STEPS[step];record['argv']=args
            with log.open('x') as handle:
                process=subprocess.run(args,cwd=root,stdout=handle,stderr=subprocess.STDOUT,timeout=timeout,
                    env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'})
            record['exit_code']=process.returncode;record['log_sha256']=sha(log)
            if process.returncode:
                q=root/V2/'data/external/quality_audit.json'
                if step=='extract-external' and q.exists() and read(q).get('status')=='needs_root_decision':
                    record['status']='needs_qc_disposition';record['next_step']='external-qc-decision'
                else:raise RuntimeError(f'{step} exited {process.returncode}; inspect {log}')
        record.setdefault('status','complete')
    except Exception:
        record.update(status='failed',exception=traceback.format_exc())
        record.setdefault('exit_code',124 if 'TimeoutExpired' in record['exception'] else 1)
    record.update(ended_utc=now(),elapsed_seconds=time.monotonic()-started)
    write(audit,record)
    print(json.dumps(dict(status=record['status'],receipt=str(audit),log=str(log) if log.exists() else None),indent=2))
    return 1 if record['status']=='failed' else (2 if record['status']=='needs_qc_disposition' else 0)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['prepare','check','run'])
    ap.add_argument('--source-root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[3])
    ap.add_argument('--project-root',type=pathlib.Path,required=True)
    ap.add_argument('--step',choices=list(STEPS))
    ap.add_argument('--timeout-seconds',type=int,default=10800)
    args=ap.parse_args();root=args.project_root.resolve()
    if args.timeout_seconds<=0:ap.error('timeout must be positive')
    if args.mode=='prepare':prepare(args.source_root.resolve(),root)
    elif args.mode=='check':check(root)
    else:
        if not args.step:ap.error('run requires --step')
        sys.exit(execute(root,args.step,args.timeout_seconds))
