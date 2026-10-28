"""Post-selection, pre-target correctness audit; never changes a saved model."""
import sys,pathlib,importlib.util,argparse,json
sys.dont_write_bytecode=True
base=pathlib.Path(__file__).resolve().parents[4]
path=base/'outputs/research_v2/experiments/cnn_baseline.py'
spec=importlib.util.spec_from_file_location('frozen_cnn',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ap=argparse.ArgumentParser();ap.add_argument('--config',type=pathlib.Path,default=m.OUT/'cluster_config.json');args=ap.parse_args()
device=m.setup();cfg,rows,y,source,fold=m.read_config(args.config)
dest=m.OUT/cfg['dataset'];freeze=json.loads((dest/'freeze.json').read_text())
if (dest/'predictions').exists() and list((dest/'predictions').iterdir()):raise RuntimeError('This audit must precede target predictions')
context=m.context(args.config,cfg)
if cfg.get('additional_context_files'):
    extra=[m.OUT/'run_external_baseline.py']+[m.BASE/p for p in cfg['additional_context_files']]
    context['files_sha256'].update({m.rel(p):m.sha(p) for p in extra})
    context['external_global_gate']='all nine SVM selections and all three CNN final models must be sealed by the parent before either target phase'
assert context==freeze['context']
for p,h in freeze['artifacts_sha256'].items():assert m.sha(m.BASE/p)==h
resultpath=dest/'source_eval_audit.json'
if resultpath.exists():raise RuntimeError('Refusing to overwrite a source diagnostic')
ledger=json.loads((dest/'runs.json').read_text());final=[r for r in ledger if r['phase']=='final_refit']
records=[];budget=m.Budget('source_eval_audit_'+cfg['dataset']);status='failed'
try:
    loader=m.make_loader(cfg,y,m.np.flatnonzero(source),False,cfg['cv_seed'])
    for rec in final:
        model=m.model_init(cfg,rec['seed'],device)
        state=m.torch.load(dest/rec['checkpoint'],map_location='cpu',weights_only=True)
        model.load_state_dict(state['model_state']);del state
        logits,truth,idx,metrics=m.predict(model,loader,device,budget)
        raw=dest/f'source_eval_seed{rec["seed"]}.npz'
        if raw.exists():raise RuntimeError('Refusing to overwrite a source diagnostic prediction')
        m.np.savez_compressed(raw,logits=logits,truth=truth,row_indices=idx)
        row={'seed':rec['seed'],'source_eval_metrics':metrics,'raw_prediction_file':m.rel(raw),'raw_prediction_sha256':m.sha(raw),'last_online_training_metrics':rec['epochs'][-1]['training'],'checkpoint_sha256':rec['checkpoint_sha256'],'saved_model_unchanged':m.sha(dest/rec['checkpoint'])==rec['checkpoint_sha256']}
        records.append(row);del model
    result={'timestamp_utc':m.now(),'status':'complete','role':'pre-target correctness diagnostic, not a model-selection metric','target_inputs_or_metrics_used':False,'models_or_selection_changed':False,'note':'online training accuracy uses train() and evolving parameters; frozen source accuracy uses eval() and final parameters, so their difference is diagnostic rather than a paired performance estimate','records':records,'selection_freeze_sha256':m.sha(dest/'freeze.json'),'audit_script_sha256':m.sha(__file__)}
    m.savej(resultpath,result);print(json.dumps(result,indent=2),flush=True);status='complete'
finally:budget.finish(status)
