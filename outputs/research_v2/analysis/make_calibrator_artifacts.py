"""Generate post-review simulation table/macros only after independent acceptance."""
import csv,datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/calibrator_diagnostic/run_001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 gate=ROOT/'audits/calibrator_diagnostic/acceptance_receipt.json'
 assert json.loads(gate.read_text())['status']=='passed', 'Independent diagnostic gate is not passed'
 source=RUN/'evaluation/summary.csv'
 with source.open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==8
 def one(part,design,obs,method):
  selected=[x for x in rows if (x['part'],x['design'],x['observation'],x['method'])==(part,design,obs,method)]
  assert len(selected)==1
  return selected[0]
 mapping={
 'CalibratorIdealAccuracy':('B','Cluster','ideal','corrected','target_macro_accuracy',100,'.2f'),
 'CalibratorFilteredAccuracy':('B','Cluster','lowpass_200Hz','corrected','target_macro_accuracy',100,'.2f'),
 'CalibratorAdditiveClusterError':('A','Cluster','exact_additive','corrected','target_response_rms_gauge_adjusted',1,'.5f'),
 'CalibratorAdditiveLimaError':('A','Lima','exact_additive','corrected','target_response_rms_gauge_adjusted',1,'.5f'),
 'CalibratorFiniteIdealError':('B','Cluster','ideal','corrected','target_response_rms_gauge_adjusted',1,'.5f'),
 'CalibratorFiniteFilteredError':('B','Cluster','lowpass_200Hz','corrected','target_response_rms_gauge_adjusted',1,'.5f')}
 macros=[];lineage={}
 for key, (part,design,obs,method,col,mult,fmt) in mapping.items():
  row=one(part,design,obs,method);value=float(row[col])*mult;text=format(value,fmt)
  macros.append('\\newcommand{\\'+key+'}{'+text+'}')
  lineage[key]={'part':part,'design':design,'observation':obs,'method':method,'column':col,'multiplier':mult,'raw_value':row[col],'rendered':text}
 out=ROOT/'paper/calibrator_numbers.tex';out.write_text('% Generated from the accepted post-review simulation CSV.\n'+'\n'.join(macros)+'\n')
 table=ROOT/'paper/calibrator_table.tex'
 table.write_text((RUN/'evaluation/summary_table.tex').read_text().replace(r'exact\_additive','additive').replace(r'lowpass\_200Hz','low-pass, 200 Hz'))
 rec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Explicit post-review simulation; no physical-data values changed','generator_sha256':sha(pathlib.Path(__file__)),'source_path':str(source.relative_to(ROOT)),'source_sha256':sha(source),'independent_acceptance_sha256':sha(gate),'original_result_seal_sha256':sha(RUN/'result_seal.json'),'macros':lineage,'outputs_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [out,table]},'table_policy':'Producer table with only human-readable observation labels substituted; numeric tokens and all eight rows retained'}
 (ROOT/'analysis/calibrator_artifact_lineage.json').write_text(json.dumps(rec,indent=2)+'\n')
 print(json.dumps({'macros':len(macros),'table_rows':len(rows),'status':'generated_after_independent_acceptance'}))
if __name__=='__main__':main()
