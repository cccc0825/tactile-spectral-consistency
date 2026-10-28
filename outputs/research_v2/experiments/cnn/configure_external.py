"""Verify the independently sealed common data and freeze external CNN settings."""
import sys,pathlib,importlib.util,json,csv,hashlib
sys.dont_write_bytecode=True
path=pathlib.Path(__file__).resolve().parents[1]/'cnn_baseline.py'
spec=importlib.util.spec_from_file_location('frozen_cnn',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
device=m.setup();configpath=m.OUT/'external_config.json';smokepath=m.OUT/'external_smoke_checks.json'
if configpath.exists() or smokepath.exists():raise RuntimeError('Refusing to overwrite an external prefit configuration/audit')
data=m.BASE/'outputs/research_v2/data/external';handoff_path=data/'handoff.json';handoff=json.loads(handoff_path.read_text())
assert handoff['status']=='sealed' and handoff['checks_passed'] and not handoff['classification_outcomes_computed']
for name,rec in handoff['files'].items():
    if m.sha(rec['path'])!=rec['sha256']:raise ValueError('Sealed producer artifact changed: '+name)
files=handoff['files'];manifest_path=pathlib.Path(files['manifest']['path']);manifest=json.loads(manifest_path.read_text())
decision=json.loads(pathlib.Path(files['validity_decision']['path']).read_text())
assert manifest['status']=='complete' and manifest['all_eligible_cnn_tensors_bitwise_equal_to_blocked_run'] and not manifest['classification_outcomes_computed']
with pathlib.Path(files['metadata.csv']['path']).open() as f:rows=list(csv.DictReader(f))
labels=m.np.load(files['labels.npy']['path']);inputs=m.np.load(files['input.npy']['path'],mmap_mode='r')
assert [r['id'] for r in rows]==decision['eligible_ids']
assert len(rows)==3550 and inputs.shape==(3550,3,128,64) and inputs.dtype==m.np.float16
assert len({r['id'] for r in rows})==3550 and '03_35_083' not in {r['id'] for r in rows}
csvlabels=m.np.array([[int(r['class_id']),int(r['speed']),int(r['trial'])] for r in rows])
assert m.np.array_equal(labels,csvlabels)
for i,r in enumerate(rows):
    assert int(r['row'])==i
    assert int(r['source'])==int(int(r['speed']) in [30,35])
    assert r['id']==f"{int(r['class_id']):02d}_{int(r['speed'])}_{int(r['trial']):03d}"
    if int(r['source']):assert r['cv_fold']==str(int(r['trial'])%3)
for start in range(0,len(inputs),128):
    chunk=inputs[start:start+128]
    assert m.np.isfinite(chunk).all() and chunk.min()>=-1 and chunk.max()<=1
source=m.np.array([int(r['source'])==1 for r in rows]);assert source.sum()==2354 and (~source).sum()==1196
foldpre=json.loads((m.OUT/'external_split_preflight.json').read_text())
for held in ['0','1','2']:
    val=[r['id'] for r in rows if int(r['source']) and r['cv_fold']==held]
    train=[r['id'] for r in rows if int(r['source']) and r['cv_fold']!=held]
    assert hashlib.sha256('\n'.join(val).encode()).hexdigest()==foldpre['folds'][held]['validation_ids_sha256']
    assert hashlib.sha256('\n'.join(train).encode()).hexdigest()==foldpre['folds'][held]['training_ids_sha256']
config=json.loads((m.OUT/'cluster_config.json').read_text())
extra=[pathlib.Path(v['path']) for k,v in files.items() if k not in ['input.npy','metadata.csv','manifest']]
extra += [handoff_path,data/'trial_integrity.json',data/'extraction_smoke_checks.json',m.OUT/'external_baseline_protocol.md',m.OUT/'external_split_preflight.json',m.OUT/'source_eval_audit.py',m.OUT/'compute_budget_amendment_02.json',m.OUT/'compute_budget_amendment_02.applied.json',pathlib.Path(__file__),smokepath]
config.update(dataset='external',created_utc=m.now(),research_status='External same-panel speed validation: 30/35 source to 40 mm/s target. Local protocol and final QC disposition fixed before external fitting; twelve physical texture specimens, not new-device or new-class generalization.',input_file=m.rel(files['input.npy']['path']),metadata_file=m.rel(files['metadata.csv']['path']),input_sha256=files['input.npy']['sha256'],metadata_sha256=files['metadata.csv']['sha256'],data_manifest_file=m.rel(manifest_path),data_manifest_sha256=files['manifest']['sha256'],num_classes=12,expected_rows=3550,expected_source_rows=2354,folds=['0','1','2'],additional_context_files=sorted({m.rel(p) for p in extra}),source_selection_metric='unweighted mean of ordinary validation accuracy over the three original-episode-modulo folds; unchanged frozen choice rule',target_reporting_metric='macro_accuracy across twelve classes; ordinary accuracy also retained; QC exclusions cause mild class imbalance',external_protocol_file=m.rel(files['protocol']['path']),external_protocol_sha256=files['protocol']['sha256'],external_baseline_addendum_file=m.rel(m.OUT/'external_baseline_protocol.md'),external_baseline_addendum_sha256=m.sha(m.OUT/'external_baseline_protocol.md'),validity_decision_sha256=files['validity_decision']['sha256'],external_global_gate='Wait for parent global seal of all nine SVM selections and all three CNN models before either external target phase',active_compute_limit_seconds=10800,active_compute_ledger_file=m.rel(m.OUT/'compute_budget.json'),formal_external_fits_before_configuration=0,external_classification_outcomes_seen=False)
temp=configpath.with_suffix('.json.prefit');m.savej(temp,config)
cfg,_,_,src,_=m.read_config(temp)
model=m.model_init(cfg,cfg['cv_seed'],device);state=m.torch.load(m.WEIGHTS,map_location='cpu',weights_only=True)
assert m.torch.equal(model.conv1.weight.detach().cpu(),state['conv1.weight']);del state
model.eval()
with m.torch.inference_mode():output=model(m.torch.zeros(2,3,128,64,device=device))
assert tuple(output.shape)==(2,12) and m.torch.isfinite(output).all()
temp.replace(configpath)
audit={'timestamp_utc':m.now(),'status':'pass','checks':{'all_handoff_file_hashes_match':True,'exact_root_eligible_ID_order':True,'CSV_labels_equal_sealed_labels':True,'input_finite_and_in_unit_interval':True,'source_assignment_exact_30_35':True,'cv_fold_actual_original_trial_modulo3':True,'exact_duplicate_removed':True,'each_fold_has_all_classes':True,'pretrained_conv1_retained':True,'dummy_12_class_output_shape':[2,12]},'source_n':int(source.sum()),'target_n':int((~source).sum()),'data_shape':list(inputs.shape),'formal_external_fits':0,'real_external_data_forward_passes':0,'configuration_sha256':m.sha(configpath),'data_manifest_sha256':files['manifest']['sha256'],'environment':m.environment(),'config_generator_sha256':m.sha(__file__)}
m.savej(smokepath,audit)
print(json.dumps({'status':'external_configuration_frozen','config':m.rel(configpath),'sha256':m.sha(configpath),'source_n':2354,'target_n':1196,'smoke_checks':audit['status']}),flush=True)
