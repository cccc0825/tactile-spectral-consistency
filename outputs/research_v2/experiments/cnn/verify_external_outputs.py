"""Read-only numeric/lineage checks from saved logits; no fitting or inference."""
import sys,pathlib,importlib.util,argparse,json,csv,traceback
sys.dont_write_bytecode=True
path=pathlib.Path(__file__).resolve().parents[1]/'cnn_baseline.py'
spec=importlib.util.spec_from_file_location('frozen_cnn',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['source','complete']);args=ap.parse_args()
m.setup();cfgpath=m.OUT/'external_config.json';cfg,metadata,y,source,fold=m.read_config(cfgpath)
dest=m.OUT/cfg['dataset'];output=dest/(args.phase+'_integrity_check.json')
if output.exists():raise RuntimeError('Refusing to overwrite a completed verification')
budget=m.Budget('verify_external_'+args.phase);status='failed'
try:
    freeze=json.loads((dest/'freeze.json').read_text())
    for p,h in freeze['artifacts_sha256'].items():assert m.sha(m.BASE/p)==h
    for p,h in freeze['context']['files_sha256'].items():assert m.sha(m.BASE/p)==h
    runs=json.loads((dest/'runs.json').read_text());planned=json.loads((dest/'planned_runs.json').read_text())['runs']
    assert len(runs)==12 and len({r['run_id'] for r in runs})==12 and all(r['status']=='complete' for r in runs)
    assert {r['run_id'] for r in runs}==set(planned)
    fold_scores={};validated_predictions=0
    for run in [r for r in runs if r['phase']=='source_cv']:
        data=m.np.load(dest/run['validation_file']);idx=data['row_indices'];truth=data['truth'];logits=data['logits']
        expected=m.np.flatnonzero(source&(fold==run['held_fold']))
        assert m.np.array_equal(idx,expected) and m.np.array_equal(truth,y[idx])
        assert logits.shape==(6,len(idx),12) and m.np.isfinite(logits).all()
        for epoch in range(6):
            pred=logits[epoch].argmax(1);acc=float((pred==truth).mean())
            mac=float(m.np.mean([(pred[truth==c]==c).mean() for c in range(12)]))
            rec=run['epochs'][epoch]['validation']
            assert abs(acc-rec['accuracy'])<1e-12 and abs(mac-rec['macro_accuracy'])<1e-12
            fold_scores.setdefault((run['learning_rate'],epoch+1),[]).append(acc)
            validated_predictions+=len(idx)
        budget.check()
    candidates=[{'learning_rate':lr,'epoch':epoch,'mean':float(m.np.mean(scores))} for (lr,epoch),scores in fold_scores.items()]
    chosen=sorted(candidates,key=lambda d:(-d['mean'],d['epoch'],d['learning_rate']))[0]
    assert chosen['learning_rate']==freeze['selected']['learning_rate'] and chosen['epoch']==freeze['selected']['epoch']
    assert abs(chosen['mean']-freeze['selected']['mean_cv_accuracy'])<1e-12
    final=[r for r in runs if r['phase']=='final_refit'];assert sorted(r['seed'] for r in final)==sorted(cfg['final_seeds'])
    record={'timestamp_utc':m.now(),'phase':args.phase,'status':'pass','checks':{'all_frozen_code_data_models_hashes_match':True,'all_12_planned_runs_complete':True,'source_CV_indices_and_labels_match_frozen_metadata':True,'all_54_epoch_metrics_recomputed_from_saved_logits':True,'LR_epoch_selection_recomputed_from_saved_logits':True,'all_three_predeclared_final_seeds_present':True},'source_n':int(source.sum()),'target_n':int((~source).sum()),'validated_source_prediction_rows_across_epochs':validated_predictions,'selection_freeze_sha256':m.sha(dest/'freeze.json'),'verification_script_sha256':m.sha(__file__)}
    if args.phase=='source':
        assert not (dest/'predictions').exists() or not list((dest/'predictions').iterdir())
        record['target_predictions_started']=False
    else:
        records=json.loads((dest/'predictions/manifest.json').read_text());target=m.np.flatnonzero(~source);per_seed=[]
        assert sorted(r['seed'] for r in records)==sorted(cfg['final_seeds'])
        for rec in records:
            assert m.sha(m.BASE/rec['prediction_file'])==rec['prediction_sha256'] and m.sha(m.BASE/rec['logits_file'])==rec['logits_sha256']
            data=m.np.load(m.BASE/rec['logits_file']);assert m.np.array_equal(data['row_indices'],target) and m.np.array_equal(data['truth'],y[target])
            with (m.BASE/rec['prediction_file']).open() as f:rows=list(csv.DictReader(f))
            assert [r['id'] for r in rows]==[metadata[i]['id'] for i in target]
            pred=m.np.array([int(r['prediction']) for r in rows]);truth=y[target]
            assert m.np.array_equal(pred,data['logits'].argmax(1))
            assert m.np.array_equal(m.np.array([int(r['correct']) for r in rows]),(pred==truth).astype(int))
            ordinary=float((pred==truth).mean());macro=float(m.np.mean([(pred[truth==c]==c).mean() for c in range(12)]))
            assert abs(ordinary-rec['metrics']['accuracy'])<1e-12 and abs(macro-rec['metrics']['macro_accuracy'])<1e-12
            per_seed.append({'seed':rec['seed'],'ordinary_accuracy':ordinary,'macro_accuracy':macro,'n':len(truth)})
            budget.check()
        record['checks'].update(all_target_CSV_predictions_match_saved_logits=True,all_target_IDs_labels_and_counts_match_frozen_outer_split=True,ordinary_and_macro_metrics_recomputed=True)
        record['per_seed']=per_seed;record['macro_accuracy_seed_mean']=float(m.np.mean([r['macro_accuracy'] for r in per_seed]));record['ordinary_accuracy_seed_mean']=float(m.np.mean([r['ordinary_accuracy'] for r in per_seed]));record['interpretation']='Mean across all three fixed optimization seeds; these are not independent data specimens.'
        record['external_global_release_sha256']=m.sha(dest/'external_global_release.json')
    m.savej(output,record);print(json.dumps(record,indent=2),flush=True);status='complete'
except BaseException as e:
    m.appendj(m.OUT/'verification_failures.jsonl',{'timestamp_utc':m.now(),'phase':args.phase,'error':repr(e),'traceback':traceback.format_exc()});raise
finally:budget.finish(status)
