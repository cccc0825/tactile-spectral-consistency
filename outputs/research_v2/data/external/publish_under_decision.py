#!/usr/bin/env python3
"""Publish common inputs under a separately frozen, outcome-blind root decision.

The original producer, per-episode eligibility criteria, protocol, and blocked QC
are read-only. Reconstructs features with the frozen producer and verifies every
published CNN tensor against the corresponding original blocked tensor.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'

DATA=Path(__file__).resolve().parent
ROOT=DATA.parents[1]
BASE=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import numpy as np
import scipy
import torch
import extract_external as producer
from download_external import sha_file,write_json


def utc():return datetime.now(timezone.utc).isoformat()


def main():
    torch.set_num_threads(1)
    decision_path=ROOT/'research/external_validity_decision.json'
    freeze_path=ROOT/'research/external_validity_decision.freeze.json'
    decision=json.loads(decision_path.read_text())
    freeze=json.loads(freeze_path.read_text())
    decision_hash=sha_file(decision_path)
    if freeze['sha256']!=decision_hash:
        raise ValueError('Root validity decision does not match its freeze')
    expected_hashes=decision['input_sha256']
    for rel,digest in expected_hashes.items():
        if sha_file(ROOT/rel)!=digest:
            raise ValueError(f'Root decision input changed: {rel}')
    audit=json.loads((DATA/'quality_audit.json').read_text())
    records=json.loads((DATA/'trial_integrity.json').read_text())
    if audit['status']!='needs_root_decision' or audit['invalid']!=49 or len(records)!=3600:
        raise ValueError('Original blocked QC differs from the documented decision')
    if audit['settings']!=producer.SETTINGS:
        raise ValueError('Producer settings changed after blocked QC')
    blocked=[p for p in (DATA/'processed').glob('staging_*')
             if (p/'stopped.json').exists() and sha_file(p/'stopped.json')==sha_file(DATA/'quality_audit.json')]
    if len(blocked)!=1:raise ValueError('Expected exactly one matching original blocked run')
    blocked=blocked[0]
    producer_hash=sha_file(producer.__file__)
    running=json.loads((blocked/'running.json').read_text())
    if running['extractor_sha256']!=producer_hash:
        raise ValueError('Original producer source changed')
    selected=np.array(decision['eligible_original_rows'],dtype=np.int64)
    if len(selected)!=3550 or len(set(selected))!=3550 or np.any(np.diff(selected)<=0):
        raise ValueError('Invalid explicit row list in root decision')
    ids=[records[i]['id'] for i in selected]
    if ids!=decision['eligible_ids'] or any(not records[i]['valid'] for i in selected):
        raise ValueError('Decision rows do not match common original QC')
    if '03_35_083' in ids or '03_35_001' not in ids:
        raise ValueError('Exact-duplicate decision was not applied')
    for field in ['source_pickle_sha256','normalized_content_sha256']:
        hashes=[records[i][field] for i in selected]
        if len(set(hashes))!=len(hashes):
            raise ValueError('Duplicate source content remains among published rows')
    labels=np.array([records[i]['labels'] for i in selected],dtype=np.int64)
    source=int(np.isin(labels[:,1],[30,35]).sum())
    target=int((labels[:,1]==40).sum())
    if (source,target)!=(2354,1196):raise ValueError('Decision source/target counts differ')
    cell_counts={f'{c:02d}_{s}':int(np.sum((labels[:,0]==c)&(labels[:,1]==s)))
                 for c in range(12) for s in [30,35,40]}
    if cell_counts!=decision['eligible_by_class_speed']:
        raise ValueError('Decision class/speed cell counts differ')
    plan=json.loads((DATA/'acquisition_plan.json').read_text())
    if plan['status']!='complete' or len(plan['blocks'])!=444 or any(b.get('status')!='complete' for b in plan['blocks']):
        raise ValueError('Full raw acquisition is not sealed')
    entries=producer.available_members()  # verifies all acquisition chunk hashes
    if len(entries)!=3600 or any(m['name']!=r['member'] for (m,b),r in zip(entries,records)):
        raise ValueError('Raw acquisition order differs from original QC')
    out=DATA/'processed'
    if (out/'manifest.json').exists():raise ValueError('Refusing to overwrite a sealed publication')
    attempt=out/('publication_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    attempt.mkdir()
    bindings={'decision_file':str(decision_path.relative_to(ROOT)), 'decision_sha256':decision_hash,
              'decision_freeze_sha256':sha_file(freeze_path),
              'producer_file':str(Path(producer.__file__).relative_to(ROOT)), 'producer_sha256':producer_hash,
              'publication_helper_file':str(Path(__file__).relative_to(ROOT)), 'publication_helper_sha256':sha_file(__file__),
              'download_helper_file':'experiments/download_external.py',
              'download_helper_sha256':sha_file(ROOT/'experiments/download_external.py'),
              'original_quality_audit_sha256':sha_file(DATA/'quality_audit.json'),
              'trial_integrity_sha256':sha_file(DATA/'trial_integrity.json'),
              'protocol_file':'research/external_protocol.md','protocol_sha256':sha_file(ROOT/'research/external_protocol.md'),
              'protocol_freeze_sha256':sha_file(ROOT/'research/external_protocol.freeze.json'),
              'acquisition_plan_sha256':sha_file(DATA/'acquisition_plan.json'),
              'blocked_tensor_file':str((blocked/'input_all.npy').relative_to(ROOT)),
              'blocked_tensor_sha256':sha_file(blocked/'input_all.npy')}
    write_json(attempt/'running.json',{'utc':utc(),'status':'running',**bindings})
    old_images=np.load(blocked/'input_all.npy',mmap_mode='r')
    if old_images.shape!=(3600,3,128,64) or old_images.dtype!=np.float16:
        raise ValueError('Blocked CNN tensor shape changed')
    new_images=np.lib.format.open_memmap(attempt/'input.npy',mode='w+',dtype=np.float16,shape=(3550,3,128,64))
    psd=np.empty((3550,129),dtype=np.float64)
    start=time.monotonic()
    for row,original in enumerate(selected):
        member,block=entries[original]
        raw=producer.member_payload(member,block)
        if hashlib.sha256(raw).hexdigest()!=records[original]['source_pickle_sha256']:
            raise ValueError('Source pickle changed since blocked QC')
        frame,_=producer.load_passive_dataframe(raw)
        q,windows,content_sha=producer.quality_and_windows(frame)
        if not q['valid'] or content_sha!=records[original]['normalized_content_sha256']:
            raise ValueError('Frozen producer eligibility or normalized content changed')
        frequency,psd[row],cnn=producer.spectral_features(windows)
        if not np.array_equal(cnn,old_images[original]):
            raise ValueError('Replayed CNN input differs from original blocked tensor')
        new_images[row]=cnn
        if (row+1)%500==0:
            new_images.flush()
            print(json.dumps({'replayed_eligible':row+1,'total':3550,'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
    new_images.flush()
    if not np.isfinite(psd).all() or np.any(psd<0) or not np.array_equal(frequency,np.arange(129)*500/256):
        raise ValueError('PSD output invariant failed')
    np.savez_compressed(attempt/'psd.npz',psd=psd,frequency=frequency,labels=labels)
    np.save(attempt/'labels.npy',labels)
    final_mask=np.zeros(3600,dtype=bool);final_mask[selected]=True
    np.save(attempt/'final_valid_mask.npy',final_mask)
    with (attempt/'metadata.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['row','row_original','id','class_id','speed','trial','source','cv_fold','domain'])
        writer.writeheader()
        for row,(original,lab) in enumerate(zip(selected,labels)):
            c,s,t=map(int,lab)
            writer.writerow({'row':row,'row_original':int(original),'id':records[original]['id'],
                             'class_id':c,'speed':s,'trial':t,'source':int(s in [30,35]),
                             'cv_fold':str(t%3),'domain':'speed' if s==40 else 'source'})
    for rel,digest in expected_hashes.items():
        if sha_file(ROOT/rel)!=digest:raise ValueError('Root decision input changed during publication')
    if sha_file(decision_path)!=decision_hash or sha_file(producer.__file__)!=producer_hash:
        raise ValueError('Decision or producer changed during publication')
    del old_images,new_images
    artifacts={}
    for name in ['input.npy','psd.npz','metadata.csv','labels.npy','final_valid_mask.npy']:
        artifacts[name]={'sha256':sha_file(attempt/name),'bytes':(attempt/name).stat().st_size}
    final_audit={'status':'eligible_under_frozen_root_decision','utc':utc(),
                 'original_quality_gate_status':'needs_root_decision','original_quality_excluded':49,
                 'additional_exact_duplicate_excluded':1,'additional_excluded_ids':['03_35_083'],
                 'eligible':3550,'source':2354,'target':1196,'class_speed_counts':cell_counts,
                 'eligibility_criteria_changed':False,'exact_duplicates_among_eligible':0,
                 'canonical_duplicates_among_eligible':0,'source_target_duplicates':0,
                 'all_eligible_cnn_tensors_bitwise_equal_to_blocked_run':True,
                 'all_eligible_original_source_sha_and_canonical_sha_reverified':True,
                 'final_mask_sha256':artifacts['final_valid_mask.npy']['sha256'],
                 'classification_outcomes_computed':False,**bindings}
    write_json(attempt/'final_eligibility_audit.json',final_audit)
    for name in artifacts:os.replace(attempt/name,out/name)
    os.replace(attempt/'final_eligibility_audit.json',DATA/'final_eligibility_audit.json')
    manifest={'status':'complete','utc':utc(),'rows':3550,'source_rows':2354,'target_rows':1196,
              'cnn_shape':[3550,3,128,64],'cnn_dtype':'float16','psd_shape':[3550,129],
              'labels_columns':['class_id0to11','nominal_speed_mm_s','episode_id1to100'],
              'artifacts':artifacts,'final_eligibility_audit_file':'data/external/final_eligibility_audit.json',
              'final_eligibility_audit_sha256':sha_file(DATA/'final_eligibility_audit.json'),
              'settings':producer.SETTINGS,'all_eligible_cnn_tensors_bitwise_equal_to_blocked_run':True,
              'classification_outcomes_computed':False,'full_archive_sha256_verified':False,
              'elapsed_seconds':time.monotonic()-start,
              'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
                             'scipy':scipy.__version__,'torch':torch.__version__},**bindings}
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'status':'complete','rows':3550,'source':2354,'target':1196,
                      'manifest_sha256':sha_file(out/'manifest.json'),'elapsed_seconds':manifest['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        with (DATA/'publication_failures.jsonl').open('a') as stream:
            stream.write(json.dumps({'utc':utc(),'exception':repr(exc),'traceback':traceback.format_exc()})+'\n')
        raise
