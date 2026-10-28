#!/usr/bin/env python3
"""Verify tracked source release files without scientific dependencies."""
from pathlib import Path
import ast,gzip,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'release_metadata/release_manifest.json').read_text())
errors=[];scripts=0
for name,record in manifest['files'].items():
    p=ROOT/name
    if not p.is_file():errors.append(name+': missing');continue
    b=p.read_bytes()
    if hashlib.sha256(b).hexdigest()!=record['sha256']:errors.append(name+': hash mismatch')
    if p.suffix=='.py':
        try:ast.parse(b.decode('utf-8'));scripts+=1
        except SyntaxError:errors.append(name+': syntax error')
raw=gzip.decompress((ROOT/'outputs/research/corpus/archive_index.json.gz').read_bytes())
if hashlib.sha256(raw).hexdigest()!='7a3bfb2ee8c96f99c48a96812c33c36a289ae90100306b558c66773025821550':errors.append('uncompressed index mismatch')
print(json.dumps({'status':'pass' if not errors else 'fail','verified_files':len(manifest['files']),'parsed_python_files':scripts,'errors':errors},indent=2))
sys.exit(bool(errors))
