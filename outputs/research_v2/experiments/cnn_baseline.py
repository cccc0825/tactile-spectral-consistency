"""Source-selected ImageNet ResNet34 baseline with separate target evaluation.

Adapted protocol, not a reproduction of the published random-split accuracy.
See baseline_protocol.md. Supports any frozen 3-channel 128x64 .npy dataset
with a CSV metadata table (class_id, source, cv_fold, domain, id).
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='2'
import argparse,contextlib,csv,datetime,hashlib,itertools,json,math,pathlib,platform,random,resource,sys,time,traceback
import numpy as np
import scipy
import torch
import torchvision
from torch.utils.data import DataLoader,Dataset

BASE=pathlib.Path(__file__).resolve().parents[3]
OUT=BASE/'outputs/research_v2/experiments/cnn'
PROTOCOL=pathlib.Path(__file__).with_name('baseline_protocol.md')
EXTRACTOR=pathlib.Path(__file__).with_name('extract_cluster_spectrograms.py')
WEIGHTS=OUT/'resnet34-b627a593.pth'
OFFICIAL=BASE/'work/revision/vendor/cluster-official'
OFFICIAL_COMMIT='e05d6b022d127e24f73583146f0aa229c6934449'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def savej(path,obj):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def appendj(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(obj)+'\n');f.flush()
def rel(path):return str(pathlib.Path(path).resolve().relative_to(BASE))
def resource_record():
    r=resource.getrusage(resource.RUSAGE_SELF)
    d={'peak_rss_bytes':r.ru_maxrss*(1 if sys.platform=='darwin' else 1024),'cpu_user_seconds':r.ru_utime,'cpu_system_seconds':r.ru_stime}
    if torch.backends.mps.is_available():d.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory())
    return d

class Budget:
    """Persistent active execution time shared across all dataset training/evaluation."""
    def __init__(self,label):
        self.path=OUT/'compute_budget.json';self.start=time.monotonic();self.last=0
        self.data=json.loads(self.path.read_text()) if self.path.exists() else {'limit_seconds':9000,'consumed_seconds':0.,'history':[],'scope':'Cluster and external source CV, refit, validation, target evaluation, checkpoint I/O; excludes extraction and synthetic resource benchmark'}
        if self.data.get('active_session'):
            raise RuntimeError('Previous execution did not close its budget ledger; review the preserved failure before recovery')
        self.base=self.data['consumed_seconds'];self.label=label;self.data['active_session']={'label':label,'start_utc':now()};self.check(force=True)
    def check(self,force=False):
        elapsed=time.monotonic()-self.start;used=self.base+elapsed
        r=resource_record()
        if force or elapsed-self.last>=5:
            self.data['consumed_seconds']=used;self.data['last_update_utc']=now();self.data['last_resource']=r;savej(self.path,self.data);self.last=elapsed
        if r['peak_rss_bytes']>3_000_000_000:raise MemoryError('3 GB process peak RSS budget exceeded')
        if used>self.data['limit_seconds']:raise TimeoutError('150 minute cumulative active compute budget exceeded')
    def finish(self,status):
        elapsed=time.monotonic()-self.start
        self.data['consumed_seconds']=self.base+elapsed
        self.data['history'].append({'label':self.label,'status':status,'end_utc':now(),'elapsed_seconds':elapsed,**resource_record()})
        self.data['active_session']=None;self.data['last_update_utc']=now();savej(self.path,self.data)

def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.backends.mps.is_available():torch.mps.manual_seed(seed)
def environment():
    return {'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'torchvision':torchvision.__version__,'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'mps_available':torch.backends.mps.is_available(),'mps_built':torch.backends.mps.is_built(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}

def create_cluster_config():
    configpath=OUT/'cluster_config.json'
    if configpath.exists():raise RuntimeError('Refusing to overwrite the frozen dataset configuration')
    dp=BASE/'outputs/research_v2/data/cluster_spectrograms.npy'
    lp=dp.with_suffix('.labels.npy');mp=dp.with_suffix('.manifest.json')
    manifest=json.loads(mp.read_text());labels=np.load(lp)
    if labels.shape!=(18880,5) or sha(dp)!=manifest['file_sha256'] or sha(lp)!=manifest['labels_sha256']:raise ValueError('Data hash or label schema mismatch')
    t,d,s,f,r=labels.T
    a=~np.isin(s,[30,40,50]);b=~np.isin(d,[0,90,180,270]);c=f!=500
    source=(~a)&(~b)&(~c)&(r==0)
    if source.sum()!=1416:raise ValueError('Source sample count changed')
    rows=[]
    for i,label in enumerate(labels):
        domain='_'.join(k for k,flag in [('speed',a[i]),('direction',b[i]),('load',c[i])] if flag) or 'id'
        rows.append({'row':i,'id':'_'.join(map(str,label)),'class_id':int(t[i]),'texture':int(t[i]),'direction':int(d[i]),'speed':int(s[i]),'force':int(f[i]),'repeat':int(r[i]),'source':int(source[i]),'cv_fold':str(s[i]) if source[i] else '', 'domain':domain})
    table=OUT/'cluster_metadata.csv'
    with table.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    config={'dataset':'cluster','created_utc':now(),'research_status':'exploratory revision; v1 targets have previously been inspected','input_file':rel(dp),'metadata_file':rel(table),'input_sha256':sha(dp),'metadata_sha256':sha(table),'data_manifest_file':rel(mp),'data_manifest_sha256':sha(mp),'num_classes':118,'expected_rows':18880,'expected_source_rows':1416,'folds':['30','40','50'],'learning_rates':[0.0001,0.0003,0.001],'max_epochs':6,'batch_size':16,'weight_decay':0.0001,'cv_seed':20270913,'final_seeds':[20270913,20270914,20270915],'optimizer':'Adam','scheduler':None,'augmentation':None,'selection':'largest mean source-CV accuracy; ties earlier epoch, then smaller learning rate','weights_file':rel(WEIGHTS),'weights_sha256':sha(WEIGHTS),'protocol_sha256':sha(PROTOCOL),'checkpoint_policy':'exact fp32 terminal model for each CV run and final refit; every epoch metrics and every validation logit; failure state checkpoint; no optimizer-state archival for successful runs'}
    if not config['weights_sha256'].startswith('b627a593'):raise ValueError('Wrong torchvision ImageNet ResNet34 weight archive')
    savej(configpath,config)
    print(json.dumps({'status':'configuration_frozen','path':rel(configpath),'source_rows':1416,'target_rows':17464}),flush=True)

class IndexedInputs(Dataset):
    def __init__(self,path,labels,indices):self.path=path;self.labels=labels;self.indices=np.asarray(indices);self.inputs=None
    def __len__(self):return len(self.indices)
    def __getitem__(self,j):
        if self.inputs is None:self.inputs=np.load(self.path,mmap_mode='r')
        i=int(self.indices[j]);x=np.array(self.inputs[i],dtype=np.float32,copy=True)
        return torch.from_numpy(x),int(self.labels[i]),i

def read_config(path):
    cfg=json.loads(path.read_text())
    for field,key in [('input_file','input_sha256'),('metadata_file','metadata_sha256'),('data_manifest_file','data_manifest_sha256'),('weights_file','weights_sha256')]:
        if sha(BASE/cfg[field])!=cfg[key]:raise ValueError('Frozen artifact changed: '+cfg[field])
    if sha(PROTOCOL)!=cfg['protocol_sha256']:raise ValueError('Baseline protocol changed since configuration freeze')
    with (BASE/cfg['metadata_file']).open() as f:rows=list(csv.DictReader(f))
    x=np.load(BASE/cfg['input_file'],mmap_mode='r')
    if x.shape!=(cfg['expected_rows'],3,128,64) or x.dtype not in [np.float16,np.float32]:raise ValueError('Wrong input tensor shape/dtype')
    if len(rows)!=len(x) or len({r['id'] for r in rows})!=len(rows):raise ValueError('Metadata count or duplicate acquisition ID')
    y=np.array([int(r['class_id']) for r in rows],dtype=np.int64);source=np.array([int(r['source'])==1 for r in rows]);fold=np.array([r['cv_fold'] for r in rows])
    if not np.array_equal(np.unique(y),np.arange(cfg['num_classes'])) or source.sum()!=cfg['expected_source_rows']:raise ValueError('Class coverage or source count mismatch')
    if set(fold[source])!=set(cfg['folds']):raise ValueError('Unexpected CV folds')
    for held in cfg['folds']:
        for mask in [source&(fold==held),source&(fold!=held)]:
            if len(np.unique(y[mask]))!=cfg['num_classes']:raise ValueError('Missing classes in a source training/validation fold')
    return cfg,rows,y,source,fold

def make_loader(cfg,y,indices,shuffle,seed):
    gen=torch.Generator();gen.manual_seed(seed)
    ds=IndexedInputs(BASE/cfg['input_file'],y,indices)
    return DataLoader(ds,batch_size=cfg['batch_size'],shuffle=shuffle,num_workers=0,pin_memory=False,drop_last=False,generator=gen)
def model_init(cfg,seed,device):
    seed_everything(seed)
    model=torchvision.models.resnet34(weights=None)
    state=torch.load(BASE/cfg['weights_file'],map_location='cpu',weights_only=True)
    model.load_state_dict(state);del state
    # The original three pretrained input channels are deliberately retained.
    model.fc=torch.nn.Linear(model.fc.in_features,cfg['num_classes'])
    return model.to(device)
def checkpoint(path,model,rec,optimizer=None):
    if path.exists():raise RuntimeError('Refusing to overwrite a checkpoint: '+str(path))
    payload={'format':'source_selected_resnet34_v1','model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'record':rec}
    if optimizer is not None:payload['optimizer_state']=optimizer.state_dict()
    partial=path.with_suffix('.partial');torch.save(payload,partial);partial.replace(path)
    return sha(path)
def accuracy(y,yp):return float(np.mean(np.asarray(y)==np.asarray(yp)))
def macro(y,yp,k):return float(np.mean([np.mean(yp[y==i]==i) for i in range(k)]))

def train_epoch(model,optimizer,loader,device,budget):
    model.train();loss_sum=0.;correct=0;n=0;start=time.monotonic()
    for inputs,labels,_ in loader:
        budget.check();inputs=inputs.to(device);labels=labels.to(device)
        optimizer.zero_grad(set_to_none=True);logits=model(inputs)
        loss=torch.nn.functional.cross_entropy(logits,labels);loss.backward();optimizer.step()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        batch=len(labels);loss_sum+=float(loss.detach().cpu())*batch
        correct+=int((logits.detach().argmax(1)==labels).sum().cpu());n+=batch
    budget.check()
    return {'loss':loss_sum/n,'accuracy':correct/n,'n':n,'seconds':time.monotonic()-start}
def predict(model,loader,device,budget):
    model.eval();all_logits=[];all_y=[];indices=[];start=time.monotonic()
    with torch.inference_mode():
        for inputs,labels,idx in loader:
            budget.check();logits=model(inputs.to(device)).cpu().numpy()
            if not np.isfinite(logits).all():raise FloatingPointError('Nonfinite inference output')
            all_logits.append(logits);all_y.append(labels.numpy());indices.append(idx.numpy())
    logits=np.concatenate(all_logits);y=np.concatenate(all_y);idx=np.concatenate(indices);yp=logits.argmax(1)
    return logits,y,idx,{'accuracy':accuracy(y,yp),'macro_accuracy':macro(y,yp,logits.shape[1]),'n':len(y),'seconds':time.monotonic()-start}

def context(cfgpath,cfg):
    files=[pathlib.Path(__file__),EXTRACTOR,PROTOCOL,cfgpath,BASE/cfg['input_file'],BASE/cfg['metadata_file'],BASE/cfg['data_manifest_file'],BASE/cfg['weights_file']]
    files += [OFFICIAL/p for p in ['LICENSE','train/classification/ResnetClassification.py','train/datasets/accel_dataset_mel.py','train/haptics_texture_classification.py']]
    return {'files_sha256':{rel(p):sha(p) for p in files},'official_code_commit':OFFICIAL_COMMIT,'environment':environment()}

def select(cfgpath,device):
    cfg,rows,y,source,fold=read_config(cfgpath);dest=OUT/cfg['dataset'];dest.mkdir(parents=True,exist_ok=True)
    cp=dest/'selection_context.json';ctx=context(cfgpath,cfg)
    if cp.exists():
        if json.loads(cp.read_text())!=ctx:raise RuntimeError('Selection context drift; preserve previous run and create a separate version')
    else:savej(cp,ctx)
    if (dest/'freeze.json').exists():raise RuntimeError('Selection already frozen; use evaluate')
    ledgerpath=dest/'runs.json';ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else []
    planned=[f'cv_lr{lr:g}_fold{held}_seed{cfg["cv_seed"]}' for lr in cfg['learning_rates'] for held in cfg['folds']]
    planned += [f'final_seed{seed}' for seed in cfg['final_seeds']]
    savej(dest/'planned_runs.json',{'timestamp_utc':now(),'runs':planned,'max_epochs':cfg['max_epochs'],'all_required_before_target_inference':True})
    budget=Budget('select_'+cfg['dataset']);status='failed'
    try:
        for lr in cfg['learning_rates']:
            for held in cfg['folds']:
                runid=f'cv_lr{lr:g}_fold{held}_seed{cfg["cv_seed"]}'
                prior=[r for r in ledger if r['run_id']==runid]
                if prior:
                    if len(prior)==1 and prior[0]['status']=='complete' and sha(dest/prior[0]['checkpoint'])==prior[0]['checkpoint_sha256']:continue
                    raise RuntimeError('Preserved incomplete/failed run requires explicit recovery: '+runid)
                rec={'run_id':runid,'phase':'source_cv','status':'running','start_utc':now(),'learning_rate':lr,'held_fold':held,'seed':cfg['cv_seed'],'epochs':[]}
                ledger.append(rec);savej(ledgerpath,ledger);model=None;optimizer=None
                try:
                    budget.check();tridx=np.flatnonzero(source&(fold!=held));vaidx=np.flatnonzero(source&(fold==held))
                    rec.update(train_n=len(tridx),validation_n=len(vaidx),train_ids_sha256=hashlib.sha256(tridx.tobytes()).hexdigest(),validation_ids_sha256=hashlib.sha256(vaidx.tobytes()).hexdigest())
                    model=model_init(cfg,cfg['cv_seed'],device);optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=cfg['weight_decay'])
                    tr=make_loader(cfg,y,tridx,True,cfg['cv_seed']);va=make_loader(cfg,y,vaidx,False,cfg['cv_seed']);all_logits=[]
                    for epoch in range(1,cfg['max_epochs']+1):
                        training=train_epoch(model,optimizer,tr,device,budget);logits,truth,idx,validation=predict(model,va,device,budget);all_logits.append(logits)
                        e={'run_id':runid,'epoch':epoch,'timestamp_utc':now(),'training':training,'validation':validation,**resource_record()};rec['epochs'].append(e)
                        appendj(dest/'epochs.jsonl',e);savej(ledgerpath,ledger)
                        print(json.dumps({'phase':'source_cv','dataset':cfg['dataset'],'run':runid,'epoch':epoch,'training_accuracy':training['accuracy'],'validation_accuracy':validation['accuracy'],'epoch_seconds':training['seconds']+validation['seconds']}),flush=True)
                    logits_path=dest/(runid+'_validation.npz');np.savez_compressed(logits_path,logits=np.stack(all_logits),truth=truth,row_indices=idx,epochs=np.arange(1,cfg['max_epochs']+1))
                    modelpath=dest/(runid+'.pt');rec['checkpoint_sha256']=checkpoint(modelpath,model,{'run_id':runid,'config_sha256':sha(cfgpath),'epoch':cfg['max_epochs']})
                    rec.update(status='complete',end_utc=now(),checkpoint=modelpath.name,validation_file=logits_path.name,validation_sha256=sha(logits_path))
                    savej(ledgerpath,ledger);del optimizer,model,tr,va,all_logits;optimizer=None;model=None
                except BaseException as e:
                    rec.update(status='failed',end_utc=now(),error=repr(e));savej(ledgerpath,ledger)
                    if model is not None:
                        failure=dest/(runid+'_failure.pt')
                        if not failure.exists():rec['failure_checkpoint_sha256']=checkpoint(failure,model,{'run_id':runid,'completed_epochs':len(rec['epochs']),'error':repr(e)},optimizer)
                    savej(ledgerpath,ledger);raise
        cv=[r for r in ledger if r['phase']=='source_cv' and r['status']=='complete'];candidates=[]
        if len(cv)!=len(cfg['learning_rates'])*len(cfg['folds']):raise RuntimeError('Incomplete source CV matrix')
        for lr in cfg['learning_rates']:
            candidates_for_lr=[r for r in cv if r['learning_rate']==lr]
            for epoch in range(1,cfg['max_epochs']+1):
                scores=[r['epochs'][epoch-1]['validation']['accuracy'] for r in candidates_for_lr]
                candidates.append({'learning_rate':lr,'epoch':epoch,'mean_cv_accuracy':float(np.mean(scores)),'fold_accuracy':scores,'fold_order':[r['held_fold'] for r in candidates_for_lr]})
        selected=sorted(candidates,key=lambda r:(-r['mean_cv_accuracy'],r['epoch'],r['learning_rate']))[0]
        savej(dest/'selection.json',{'timestamp_utc':now(),'candidates':candidates,'selected':selected,'rule':cfg['selection'],'source_only':True})
        for seed in cfg['final_seeds']:
            runid=f'final_seed{seed}';prior=[r for r in ledger if r['run_id']==runid]
            if prior:
                if len(prior)==1 and prior[0]['status']=='complete' and sha(dest/prior[0]['checkpoint'])==prior[0]['checkpoint_sha256']:continue
                raise RuntimeError('Preserved incomplete final refit requires explicit recovery: '+runid)
            rec={'run_id':runid,'phase':'final_refit','status':'running','start_utc':now(),'learning_rate':selected['learning_rate'],'seed':seed,'selected_epoch':selected['epoch'],'train_n':int(source.sum()),'epochs':[]}
            ledger.append(rec);savej(ledgerpath,ledger);model=None;optimizer=None
            try:
                budget.check();model=model_init(cfg,seed,device);optimizer=torch.optim.Adam(model.parameters(),lr=selected['learning_rate'],weight_decay=cfg['weight_decay'])
                tr=make_loader(cfg,y,np.flatnonzero(source),True,seed)
                for epoch in range(1,selected['epoch']+1):
                    training=train_epoch(model,optimizer,tr,device,budget)
                    e={'run_id':runid,'epoch':epoch,'timestamp_utc':now(),'training':training,**resource_record()};rec['epochs'].append(e);appendj(dest/'epochs.jsonl',e);savej(ledgerpath,ledger)
                    print(json.dumps({'phase':'final_refit','dataset':cfg['dataset'],'run':runid,'epoch':epoch,'epoch_seconds':training['seconds']}),flush=True)
                modelpath=dest/(runid+'.pt');rec['checkpoint_sha256']=checkpoint(modelpath,model,{'run_id':runid,'config_sha256':sha(cfgpath),'epoch':selected['epoch']})
                rec.update(status='complete',end_utc=now(),checkpoint=modelpath.name);savej(ledgerpath,ledger);del optimizer,model,tr;optimizer=None;model=None
            except BaseException as e:
                rec.update(status='failed',end_utc=now(),error=repr(e));savej(ledgerpath,ledger)
                if model is not None:
                    failure=dest/(runid+'_failure.pt')
                    if not failure.exists():rec['failure_checkpoint_sha256']=checkpoint(failure,model,{'run_id':runid,'completed_epochs':len(rec['epochs']),'error':repr(e)},optimizer)
                savej(ledgerpath,ledger);raise
        if {r['run_id'] for r in ledger if r['status']=='complete'}!=set(planned):raise RuntimeError('Incomplete planned model matrix')
        budget.check();files=[cp,ledgerpath,dest/'selection.json',dest/'planned_runs.json',dest/'epochs.jsonl']
        for r in ledger:
            files.append(dest/r['checkpoint'])
            if r.get('validation_file'):files.append(dest/r['validation_file'])
        savej(dest/'freeze.json',{'timestamp_utc':now(),'status':'all_source_selection_and_final_models_sealed_before_target_inference','target_prediction_started':False,'context':ctx,'selected':selected,'model_seeds':cfg['final_seeds'],'artifacts_sha256':{rel(p):sha(p) for p in files}})
        status='complete';print('SOURCE_SELECTION_FROZEN '+cfg['dataset'],flush=True)
    finally:
        budget.finish(status)
        if status!='complete':savej(dest/'missing_runs.json',{'timestamp_utc':now(),'planned':planned,'complete':[r['run_id'] for r in ledger if r['status']=='complete'],'incomplete':[p for p in planned if p not in {r['run_id'] for r in ledger if r['status']=='complete'}],'target_evaluation_permitted':False})

def evaluate(cfgpath,device):
    cfg,rows,y,source,fold=read_config(cfgpath);dest=OUT/cfg['dataset'];freeze_path=dest/'freeze.json';freeze=json.loads(freeze_path.read_text())
    if context(cfgpath,cfg)!=freeze['context']:raise ValueError('Frozen code/data/config/environment context changed before target evaluation')
    for p,h in freeze['artifacts_sha256'].items():
        if sha(BASE/p)!=h:raise ValueError('Frozen model/selection artifact changed: '+p)
    ledger=json.loads((dest/'runs.json').read_text());final=[r for r in ledger if r['phase']=='final_refit'];pred_dir=dest/'predictions';pred_dir.mkdir(exist_ok=True)
    if list(pred_dir.iterdir()):raise RuntimeError('Refusing to overwrite or mix target predictions')
    budget=Budget('evaluate_'+cfg['dataset']);status='failed';records=[]
    try:
        target=np.flatnonzero(~source);loader=make_loader(cfg,y,target,False,cfg['cv_seed'])
        for rec in final:
            budget.check();model=model_init(cfg,rec['seed'],device);state=torch.load(dest/rec['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(state['model_state']);del state
            logits,truth,idx,metrics=predict(model,loader,device,budget);yp=logits.argmax(1)
            csvpath=pred_dir/f'seed{rec["seed"]}.csv';logitpath=pred_dir/f'seed{rec["seed"]}_logits.npz'
            with csvpath.open('w',newline='') as f:
                fields=list(rows[0])+['prediction','correct','seed'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                for i,pred in zip(idx,yp):writer.writerow({**rows[int(i)],'prediction':int(pred),'correct':int(pred==y[i]),'seed':rec['seed']})
            np.savez_compressed(logitpath,logits=logits,truth=truth,row_indices=idx)
            domain_scores={}
            for domain in sorted({rows[int(i)]['domain'] for i in idx}):
                mask=np.array([rows[int(i)]['domain']==domain for i in idx]);domain_scores[domain]={'n':int(mask.sum()),'accuracy':accuracy(truth[mask],yp[mask]),'macro_accuracy':macro(truth[mask],yp[mask],cfg['num_classes'])}
            record={'seed':rec['seed'],'timestamp_utc':now(),'status':'complete','metrics':metrics,'domain_metrics':domain_scores,'checkpoint_sha256':rec['checkpoint_sha256'],'prediction_file':rel(csvpath),'prediction_sha256':sha(csvpath),'logits_file':rel(logitpath),'logits_sha256':sha(logitpath),'selection_freeze_sha256':sha(freeze_path)}
            records.append(record);savej(pred_dir/'manifest.json',records);del model
            print(json.dumps({'phase':'target_prediction_complete','dataset':cfg['dataset'],'seed':rec['seed'],'n':len(target),'seconds':metrics['seconds']}),flush=True)
        if len(records)!=len(cfg['final_seeds']):raise RuntimeError('Missing final seed evaluation')
        summary={'timestamp_utc':now(),'status':'complete','dataset':cfg['dataset'],'interpretation':'seed averages describe model stochasticity; seeds are not independent data samples','research_status':cfg['research_status'],'seeds':cfg['final_seeds'],'selected':freeze['selected'],'overall_accuracy_seed_mean':float(np.mean([r['metrics']['accuracy'] for r in records])),'domain_accuracy_seed_mean':{d:float(np.mean([r['domain_metrics'][d]['accuracy'] for r in records])) for d in records[0]['domain_metrics']},'per_seed':records,'selection_freeze_sha256':sha(freeze_path),'resource':resource_record()}
        savej(dest/'summary.json',summary);budget.check();savej(pred_dir/'completion.json',{'timestamp_utc':now(),'status':'complete','summary_sha256':sha(dest/'summary.json'),'manifest_sha256':sha(pred_dir/'manifest.json'),'target_rows_per_seed':len(target),'all_planned_seeds_present':True});status='complete'
    finally:budget.finish(status)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['configure-cluster','select','evaluate']);ap.add_argument('--config',type=pathlib.Path,default=OUT/'cluster_config.json');args=ap.parse_args();started=time.monotonic()
    try:
        device=setup()
        if args.phase=='configure-cluster':create_cluster_config()
        elif args.phase=='select':select(args.config.resolve(),device)
        else:evaluate(args.config.resolve(),device)
    except BaseException as e:
        appendj(OUT/'failures.jsonl',{'timestamp_utc':now(),'phase':args.phase,'config':str(args.config),'error':repr(e),'traceback':traceback.format_exc()});raise
    finally:appendj(OUT/'process_resources.jsonl',{'timestamp_utc':now(),'phase':args.phase,'process_elapsed_seconds':time.monotonic()-started,**resource_record()})
