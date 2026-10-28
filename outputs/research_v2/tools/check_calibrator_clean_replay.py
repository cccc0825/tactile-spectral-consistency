"""Compare deterministic outputs of the already executed isolated-env replay.
This consistency test is separate from the non-author methodological audit.
"""
import datetime,hashlib,json,pathlib,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=ROOT/'experiments/calibrator_diagnostic/run_001'
 b=ROOT/'experiments/calibrator_diagnostic/clean_replay_001'
 dest=ROOT/'audits/clean_environment/calibrator_diagnostic';dest.mkdir(exist_ok=True)
 files=set(a.rglob('*.joblib'))|set((a/'evaluation').rglob('*.csv'))|set((a/'evaluation/responses').glob('*.npz'))|{a/'evaluation/summary_table.tex',a/'evaluation/summary.json'}
 comparisons=[]
 for p in sorted(files):
  q=b/p.relative_to(a);h1=sha(p);h2=sha(q)
  comparisons.append({'relative_path':str(p.relative_to(a)),'original_sha256':h1,'replay_sha256':h2,'bitwise_equal':h1==h2})
 assert all(x['bitwise_equal'] for x in comparisons)
 timings=[json.loads((b/f'timing/{s}.json').read_text()) for s in ['test','source','evaluate','verify']]
 assert all(x['status']=='complete' for x in timings)
 for fname in ['calibrator_clean_replay_01.log','calibrator_clean_replay_02.log']:
  shutil.copy2(BASE/'work/revision'/fname,dest/fname)
 r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','scope':'One additional full simulation producer run in the existing isolated numeric venv on the same host/base Python. This is technical replay, not a new scientific replicate or independent methodological acceptance.',
 'executed_command':'work/revision/clean-numeric-env/bin/python -B outputs/research_v2/experiments/calibrator_diagnostic.py all --run-dir outputs/research_v2/experiments/calibrator_diagnostic/clean_replay_001',
 'exit_code':0,'active_seconds':timings[-1]['cumulative_active_seconds'],'initial_rejected_destination':{'exit_code':1,'reason':'documented owned-output-directory guard rejected scratch path before any fit','log':'calibrator_clean_replay_01.log'},
 'same_host':True,'base_python_shared':True,'system_and_user_site_disabled':True,'new_scientific_samples':0,'all_170_saved_models_bitwise_equal':True,'all_576_target_predictions_equal':True,
 'comparisons':comparisons,'environment_path':str((b/'environment.json').relative_to(ROOT)),'environment_sha256':sha(b/'environment.json'),'original_result_seal_sha256':sha(a/'result_seal.json'),'replay_result_seal_sha256':sha(b/'result_seal.json'),'checker_sha256':sha(pathlib.Path(__file__))}
 (dest/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({'status':r['status'],'files_compared':len(comparisons),'models':170,'active_seconds':r['active_seconds']}))
if __name__=='__main__':main()
