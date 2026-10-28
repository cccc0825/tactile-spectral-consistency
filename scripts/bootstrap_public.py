#!/usr/bin/env python3
"""Create local compatibility metadata for the unchanged research downloaders.

Uses only bundled public citation fields. Does not download data or train models.
The generated HTML is explicitly a metadata extract, not an archived webpage.
"""
from pathlib import Path
import argparse,datetime,gzip,hashlib,html,importlib.metadata,json,platform

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.project_root.resolve()
    index=ROOT/'outputs/research/corpus/archive_index.json'
    expected='7a3bfb2ee8c96f99c48a96812c33c36a289ae90100306b558c66773025821550'
    if not index.exists():
        raw=gzip.decompress(index.with_suffix('.json.gz').read_bytes())
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Archive index hash mismatch')
        with index.open('xb') as stream:stream.write(raw)
    elif hashlib.sha256(index.read_bytes()).hexdigest()!=expected:
        raise ValueError('Existing archive index differs; refusing to overwrite')
    meta=json.loads((ROOT/'release_metadata/external_dataset.json').read_text())
    snapshot=meta['snapshot']
    versions={}
    for name in ['numpy','scipy','pandas','scikit-learn','matplotlib','torch','torchvision']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    runtime={'kind':'new local environment observation, not historical experiment evidence',
             'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'python':platform.python_version(),'system':platform.system(),
             'machine':platform.machine(),'packages':versions}
    payloads={
       'work/revision/mendeley_state.js':'window.__PUBLIC_METADATA__ = '+json.dumps({'dataset':{'snapshot':snapshot}},indent=2)+';\n',
       'work/revision/mendeley_files.json':json.dumps(meta['files'],indent=2)+'\n',
       'work/revision/mendeley_landing.html':'<!doctype html><meta charset="utf-8"><title>Public dataset metadata extract</title><p>Generated compatibility document; not an archived webpage or a new source retrieval.</p><a href="'+html.escape(meta['source_url'],quote=True)+'">Dataset source</a><pre>'+html.escape(json.dumps(snapshot,indent=2))+'</pre>\n',
       'outputs/research_v2/environment/runtime.json':json.dumps(runtime,indent=2)+'\n',
       'outputs/research/experiments/environment.json':json.dumps(runtime,indent=2)+'\n'}
    created=[];existing=[]
    for relative,content in payloads.items():
        p=root/relative
        if p.exists():existing.append(relative);continue
        p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x') as stream:stream.write(content)
        created.append(relative)
    print(json.dumps({'created':created,'preserved_existing':existing,'network_used':False},indent=2))
if __name__=='__main__':main()
