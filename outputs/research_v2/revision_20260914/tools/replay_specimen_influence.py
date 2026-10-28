#!/usr/bin/env python3
"""Replay frozen descriptive analysis in a new directory, never in the archive."""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    revision = Path(__file__).resolve().parents[1]
    root = revision.parents[2]
    dest = args.destination.resolve()
    if dest.exists():
        parser.error('Destination must not exist; archived outputs are never overwritten.')
    original = revision / 'analysis'
    plan = json.loads((original / 'analysis_plan.json').read_text())
    stage = dest / original.relative_to(root)
    stage.mkdir(parents=True)
    copied = {}
    for src in [original/'run_specimen_influence.py', original/'analysis_plan.json'] + [root/p for p in plan['inputs']]:
        target = dest / src.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied[str(src.relative_to(root))] = sha(src)
    before = {str(p.relative_to(original)): sha(p) for p in original.rglob('*') if p.is_file()}
    env = dict(os.environ, MPLCONFIGDIR=str(dest/'matplotlib_cache'))
    result = subprocess.run([sys.executable, str(stage/'run_specimen_influence.py')], cwd=dest, env=env, capture_output=True, text=True)
    (dest/'stdout.log').write_text(result.stdout)
    (dest/'stderr.log').write_text(result.stderr)
    result.check_returncode()
    checks = {}
    for name in ['per_specimen_effects.csv', 'specimen_influence_table.tex', 'specimen_influence_numbers.tex']:
        checks[name] = sha(stage/name) == sha(original/name)
    old, new = [json.loads((p/'summary.json').read_text()) for p in [original, stage]]
    old.pop('created_utc'); new.pop('created_utc')
    checks['all_summary_fields_except_timestamp'] = old == new
    checks['archive_unchanged'] = before == {str(p.relative_to(original)): sha(p) for p in original.rglob('*') if p.is_file()}
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), status='pass' if all(checks.values()) else 'fail', checks=checks, input_sha256=copied, python=sys.executable, destination=str(dest), scope='Deterministic replay of post-hoc analysis, not a new independent experiment.')
    (dest/'replay_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
