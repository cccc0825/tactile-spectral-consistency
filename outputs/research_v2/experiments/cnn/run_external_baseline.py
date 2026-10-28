"""Reuse the frozen ResNet trainer and additionally seal external-specific code/QC.

No optimizer, model, preprocessing, selection, or metric function is altered.
Target invocation is reserved for the parent's external global-freeze release.
"""
import sys,pathlib,importlib.util,argparse,json,time,traceback
sys.dont_write_bytecode=True
path=pathlib.Path(__file__).resolve().parents[1]/'cnn_baseline.py'
spec=importlib.util.spec_from_file_location('frozen_cnn',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_context=m.context

def external_context(cfgpath,cfg):
    result=original_context(cfgpath,cfg)
    extra=[pathlib.Path(__file__)]+[m.BASE/p for p in cfg['additional_context_files']]
    result['files_sha256'].update({m.rel(p):m.sha(p) for p in extra})
    result['external_global_gate']='all nine SVM selections and all three CNN final models must be sealed by the parent before either target phase'
    return result

m.context=external_context
ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['select','evaluate']);ap.add_argument('--config',type=pathlib.Path,default=m.OUT/'external_config.json');ap.add_argument('--global-release',type=pathlib.Path)
args=ap.parse_args();started=time.monotonic()
try:
    device=m.setup()
    if args.phase=='select':m.select(args.config.resolve(),device)
    else:
        if args.global_release is None or not args.global_release.exists():
            raise RuntimeError('External target phase requires the parent-provided global freeze/release artifact')
        # Its schema belongs to the parent. Preserve exact bytes/hash as a
        # prerequisite record; the parent independently validates global hashes.
        cfg=json.loads(args.config.read_text());dest=m.OUT/cfg['dataset']
        release_record=dest/'external_global_release.json'
        if release_record.exists():raise RuntimeError('A target release record already exists; review preserved state before retry')
        m.savej(release_record,{'timestamp_utc':m.now(),'release_file':m.rel(args.global_release),'release_sha256':m.sha(args.global_release),'release_document':json.loads(args.global_release.read_text()),'authority':'Parent explicitly released both fixed external evaluations after all SVM and CNN model hashes were sealed'})
        m.evaluate(args.config.resolve(),device)
except BaseException as e:
    m.appendj(m.OUT/'failures.jsonl',{'timestamp_utc':m.now(),'phase':'external_'+args.phase,'config':str(args.config),'error':repr(e),'traceback':traceback.format_exc()});raise
finally:m.appendj(m.OUT/'process_resources.jsonl',{'timestamp_utc':m.now(),'phase':'external_'+args.phase,'process_elapsed_seconds':time.monotonic()-started,**m.resource_record()})
