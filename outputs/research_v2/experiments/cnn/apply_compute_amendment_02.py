"""Apply the parent-authorized compute-only amendment at a closed phase boundary."""
import datetime,hashlib,json,pathlib
out=pathlib.Path(__file__).resolve().parent
clock=out/'compute_budget.json';authorization=out/'compute_budget_amendment_02.json'
applied=out/'compute_budget_amendment_02.applied.json'
if applied.exists():raise RuntimeError('Compute amendment already applied; refusing duplicate mutation')
raw=clock.read_bytes();ledger=json.loads(raw);amendment=json.loads(authorization.read_text())
if ledger.get('active_session'):raise RuntimeError('Cannot mutate an active in-memory budget writer')
if ledger['limit_seconds']!=amendment['previous_limit_seconds']:raise RuntimeError('Unexpected prior compute limit')
snapshot=out/'compute_budget_before_amendment_02_closed.json'
if snapshot.exists():raise RuntimeError('Closed prior-budget snapshot already exists')
snapshot.write_bytes(raw)
utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
record={'timestamp_utc':utc,'status':'applied','previous_limit_seconds':ledger['limit_seconds'],'new_limit_seconds':amendment['new_limit_seconds'],'consumed_seconds_preserved':ledger['consumed_seconds'],'authorization_sha256':hashlib.sha256(authorization.read_bytes()).hexdigest(),'closed_previous_ledger_sha256':hashlib.sha256(raw).hexdigest(),'fit_configuration_changed':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
ledger['limit_seconds']=amendment['new_limit_seconds'];ledger['last_update_utc']=utc;ledger.setdefault('amendments',[]).append(record)
tmp=clock.with_suffix('.json.partial');tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(clock)
record['amended_ledger_sha256']=hashlib.sha256(clock.read_bytes()).hexdigest()
applied.write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
