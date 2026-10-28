#!/usr/bin/env python3
"""Outcome-blind range acquisition for Mendeley n666tk4mw9 version 1.

No pickle deserialization, notebook execution, training, or outcome analysis.
Retains official metadata, central directory, HTTP request records and chunk hashes.
The full archive SHA256 is a publisher assertion, NOT verified by partial acquisition.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data' / 'external'
REPO = ROOT.parents[1]
SCRATCH = REPO / 'work' / 'revision' / 'external'
SOURCE_URL = 'https://data.mendeley.com/datasets/n666tk4mw9/1'
ZIP_SIZE = 3831798836
ZIP_SHA256 = '56468a6bc7191b46e12f09fece605d117bf29e8c678ae458b885e01a41917572'
ZIP_URL = ('https://data.mendeley.com/public-files/datasets/n666tk4mw9/files/'
           '70c5500e-2ef3-40af-a8f4-d4eac7bb3119/file_downloaded')
MAX_EXTERNAL_RAW = 4_000_000_000
ZIP_DIRECT_URL = ('https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/'
                  '2518144f-4552-4ecd-83ea-8b880546f385')
LOG_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def log_event(value):
    DEST.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        with (DEST / 'request_log.jsonl').open('a') as stream:
            stream.write(json.dumps({'utc': now(), **value}, sort_keys=True) + '\n')


def fetch(url, output, start=None, end=None, expected_sha=None, max_time=240):
    """curl argv, no shell evaluation; TLS verification remains enabled."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        digest = sha_file(output)
        if expected_sha and digest != expected_sha:
            raise ValueError(f'Existing file hash mismatch: {output}')
        if start is not None and output.stat().st_size != end - start + 1:
            raise ValueError(f'Existing range length mismatch: {output}')
        return output
    request_id = f'{time.time_ns()}_{output.stem}'
    headers_path = DEST / 'requests' / f'{request_id}.headers'
    stderr_path = DEST / 'requests' / f'{request_id}.stderr'
    headers_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + f'.{request_id}.partial')
    maximum = end - start + 1 if start is not None else 2_000_000
    args = ['curl', '--fail', '--location', '--silent', '--show-error',
            '--connect-timeout', '30', '--max-time', str(max_time),
            '--max-filesize', str(maximum), '--dump-header', str(headers_path),
            '--output', str(partial), '--write-out', '%{json}']
    if start is not None:
        args += ['--range', f'{start}-{end}']
    args.append(url)
    started = time.monotonic()
    result = subprocess.run(args, capture_output=True, text=True)
    stderr_path.write_text(result.stderr)
    try:
        curl_meta = json.loads(result.stdout)
    except json.JSONDecodeError:
        curl_meta = {'raw_stdout': result.stdout}
    record = {'request_id': request_id, 'url': url, 'range_start': start,
              'range_end': end, 'output': str(output.relative_to(DEST)),
              'returncode': result.returncode, 'elapsed_seconds': time.monotonic() - started,
              'headers_path': str(headers_path.relative_to(DEST)),
              'stderr_path': str(stderr_path.relative_to(DEST)), 'curl': curl_meta,
              'status': 'failed'}
    if partial.exists():
        record['received_bytes'] = partial.stat().st_size
        record['received_sha256'] = sha_file(partial)
    error = None
    if result.returncode != 0:
        error = f'curl exit {result.returncode}: {result.stderr.strip()}'
    elif start is not None:
        header_text = headers_path.read_text(errors='replace')
        ranges = re.findall(r'(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)', header_text)
        if not ranges or tuple(map(int, ranges[-1])) != (start, end, ZIP_SIZE):
            error = f'Content-Range mismatch: {ranges!r}'
        elif partial.stat().st_size != maximum:
            error = 'Downloaded range length mismatch'
    if error is None and expected_sha and record.get('received_sha256') != expected_sha:
        error = 'Official SHA256 mismatch'
    if error is not None:
        record['error'] = error
        log_event(record)
        raise RuntimeError(error)
    os.replace(partial, output)
    record['status'] = 'complete'
    log_event(record)
    return output


def preserve_sources():
    source = DEST / 'source'
    source.mkdir(parents=True, exist_ok=True)
    input_state = REPO / 'work' / 'revision' / 'mendeley_state.js'
    text = input_state.read_text()
    # Parse the first JSON value; never execute the surrounding page JavaScript.
    state, _ = json.JSONDecoder().raw_decode(text.split('=', 1)[1].lstrip())
    snapshot = state['dataset']['snapshot']
    if snapshot['doi'] != '10.17632/n666tk4mw9.1' or snapshot['licence']['short_name'] != 'CC BY 4.0':
        raise ValueError('Unexpected dataset identity or license')
    for name in ['mendeley_files.json', 'mendeley_landing.html']:
        target = source / name
        if not target.exists():
            shutil.copyfile(REPO / 'work' / 'revision' / name, target)
    write_json(source / 'dataset_metadata.json', snapshot)
    write_json(source / 'source_manifest.json', {
        'checked_utc': now(), 'dataset_landing_url': SOURCE_URL,
        'article_doi': '10.1016/j.dib.2023.109590',
        'dataset_doi': snapshot['doi'], 'license': snapshot['licence'],
        'attribution': snapshot['contributors'],
        'archive_url': ZIP_URL, 'archive_bytes': ZIP_SIZE,
        'publisher_archive_sha256': ZIP_SHA256,
        'whole_archive_sha256_verified': False,
        'acquisition': 'HTTP byte ranges of official public archive; IMU members only',
        'known_ambiguity': ('Repository landing-page prose says 30/40/45 mm/s; '
                            'actual ZIP folder labels are 30/35/40, consistent with the '
                            'article specifications table. Preserve actual folder labels as '
                            'nominal speeds, while retaining the unresolved landing-page discrepancy.'),
        'source_hashes': {p.name: sha_file(p) for p in source.iterdir() if p.is_file()
                          and p.name != 'source_manifest.json'},
    })
    metadata = json.loads((source / 'mendeley_files.json').read_text())
    notebook = next(x for x in metadata if x['filename'].endswith('.ipynb'))
    content = notebook['content_details']
    fetch(content['download_url'], source / notebook['filename'],
          expected_sha=content['sha256_hash'])


def index_archive():
    raw = DEST / 'raw'
    tail_len = 65557
    tail = fetch(ZIP_URL, raw / 'zip_tail.bin', ZIP_SIZE-tail_len, ZIP_SIZE-1).read_bytes()
    eocd_offset = tail.rfind(b'PK\x05\x06')
    if eocd_offset < 0:
        raise ValueError('No ZIP end-of-central-directory record')
    sig, disk, cd_disk, disk_entries, entries, cd_size, cd_offset, comment_len = struct.unpack_from(
        '<4s4H2IH', tail, eocd_offset)
    if disk or cd_disk or disk_entries != entries or eocd_offset + 22 + comment_len != len(tail):
        raise ValueError('Unsupported multi-disk ZIP or invalid EOCD')
    if 0xffff in (disk_entries, entries) or 0xffffffff in (cd_size, cd_offset):
        raise ValueError('ZIP64 central directory unsupported')
    central_file = fetch(ZIP_URL, raw / 'central_directory.bin', cd_offset, cd_offset+cd_size-1)
    central = central_file.read_bytes()
    members = []
    offset = 0
    while offset < len(central):
        values = struct.unpack_from('<4s6H3I5H2I', central, offset)
        (sig, version, need_version, flags, method, mt, md, crc, csize, usize,
         nlen, xlen, clen, start_disk, internal, external, local) = values
        if sig != b'PK\x01\x02' or start_disk != 0:
            raise ValueError('Invalid central-directory member')
        name = central[offset+46:offset+46+nlen].decode('utf-8' if flags & 0x800 else 'cp437')
        extra = central[offset+46+nlen:offset+46+nlen+xlen]
        cursor, zip64_values = 0, None
        while cursor < len(extra):
            tag, field_size = struct.unpack_from('<HH', extra, cursor)
            cursor += 4
            field = extra[cursor:cursor+field_size]
            if len(field) != field_size:
                raise ValueError('Truncated ZIP extra field')
            if tag == 0x0001:
                zip64_values = field
            cursor += field_size
        # ZIP64 can occur for individual members even when the entire archive is <4 GiB.
        if 0xffffffff in (usize, csize, local):
            if zip64_values is None:
                raise ValueError('Missing ZIP64 member sizes')
            cursor = 0
            if usize == 0xffffffff:
                usize = struct.unpack_from('<Q', zip64_values, cursor)[0]
                cursor += 8
            if csize == 0xffffffff:
                csize = struct.unpack_from('<Q', zip64_values, cursor)[0]
                cursor += 8
            if local == 0xffffffff:
                local = struct.unpack_from('<Q', zip64_values, cursor)[0]
                cursor += 8
        if name.startswith('/') or '..' in Path(name).parts or flags & 1:
            raise ValueError('Unsafe member path or encrypted ZIP')
        if method not in (0, 8):
            raise ValueError(f'Unsupported ZIP compression: {method}')
        members.append({'name': name, 'flags': flags, 'compression': method,
                        'crc32': f'{crc:08x}', 'compressed_bytes': csize,
                        'uncompressed_bytes': usize, 'local_header_offset': local})
        offset += 46+nlen+xlen+clen
    if len(members) != entries or offset != len(central):
        raise ValueError('Central-directory count or size mismatch')
    sorted_members = sorted(members, key=lambda x: x['local_header_offset'])
    for i, m in enumerate(sorted_members):
        m['next_header_offset'] = (sorted_members[i+1]['local_header_offset']
                                   if i+1 < len(sorted_members) else cd_offset)
    imu = [m for m in members if '/full_imu/' in m['name'].lower() and not m['name'].endswith('/')]
    counts = Counter('/'.join(m['name'].split('/')[:-1]) for m in imu)
    write_json(DEST / 'zip_index.json', {'parser_schema_version': 2, 'archive_bytes': ZIP_SIZE,
               'central_offset': cd_offset, 'central_bytes': cd_size, 'members': members})
    summary = {'utc': now(), 'all_member_count': len(members), 'imu_member_count': len(imu),
               'imu_compressed_bytes': sum(x['compressed_bytes'] for x in imu),
               'imu_uncompressed_bytes': sum(x['uncompressed_bytes'] for x in imu),
               'imu_folder_counts': dict(sorted(counts.items())),
               'free_disk_bytes': shutil.disk_usage(DEST).free,
               'max_external_raw_bytes': MAX_EXTERNAL_RAW,
               'archive_sha256_not_verified': True,
               'central_sha256': sha_file(central_file)}
    write_json(DEST / 'resource_audit.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return members


def selected_members(members):
    return [m for m in members if '/full_imu/' in m['name'].lower() and not m['name'].endswith('/')]


def acquire(members, limit=None, workers=1, max_block_bytes=2*(1<<20), direct_s3=False):
    selected = sorted(selected_members(members), key=lambda m:m['local_header_offset'])
    if limit:
        selected = selected[:limit]
    # Adjacent IMU entries share a range; never silently add barometer files.
    # Cap each request at 2 MiB to retain bounded retry and resource costs.
    blocks = []
    for member in selected:
        start, end = member['local_header_offset'], member['next_header_offset']-1
        if (blocks and blocks[-1]['end']+1 == start
                and end-blocks[-1]['start']+1 <= max_block_bytes):
            blocks[-1]['end'] = end
            blocks[-1]['members'].append(member['name'])
        else:
            blocks.append({'start': start, 'end': end, 'members': [member['name']]})
    total = sum(b['end']-b['start']+1 for b in blocks)
    # Three attempts retain at most three range copies including failed partials.
    if total*3 > MAX_EXTERNAL_RAW:
        raise ValueError(f'Compressed acquisition exceeds raw budget: {total}')
    current = sum(p.stat().st_size for p in (DEST / 'raw').glob('*') if p.is_file())
    if shutil.disk_usage(DEST).free < total-current+(1<<30):
        raise ValueError('Insufficient disk reserve')
    plan = {'utc': now(), 'status': 'planned', 'total_bytes': total, 'member_count': len(selected),
            'block_count': len(blocks), 'blocks': blocks, 'workers': workers,
            'max_block_bytes': max_block_bytes, 'request_attempts_max': 3,
            'url': ZIP_DIRECT_URL if direct_s3 else ZIP_URL}
    plan_path = DEST / ('acquisition_sample.json' if limit else 'acquisition_plan.json')
    if plan_path.exists():
        prior = json.loads(plan_path.read_text())
        if prior.get('status') != 'complete':
            archive = DEST / 'prior_acquisition_plans'
            archive.mkdir(exist_ok=True)
            shutil.copyfile(plan_path, archive / f'{time.time_ns()}_{plan_path.name}')
    write_json(plan_path, plan)

    def get_block(block):
        filename = f"imu_{block['start']:010d}_{block['end']:010d}.range"
        errors = []
        for attempt in range(3):
            try:
                path = fetch(plan['url'], DEST / 'raw' / filename, block['start'], block['end'])
                return {'path': str(path.relative_to(DEST)), 'sha256': sha_file(path),
                        'bytes': path.stat().st_size, 'status': 'complete',
                        'failed_attempts': errors}
            except (RuntimeError, ValueError) as exc:
                errors.append({'attempt': attempt+1, 'error': repr(exc), 'utc': now()})
        return {'status': 'failed', 'failed_attempts': errors}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(get_block, block): block for block in blocks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            block = futures[future]
            block.update(future.result())
            write_json(plan_path, plan)
            print(f"block {i+1}/{len(blocks)} {block['start']}-{block['end']} {block['status']}", flush=True)
    plan['status'] = 'complete' if all(b['status']=='complete' for b in blocks) else 'failed'
    plan['completed_utc'] = now()
    write_json(plan_path, plan)
    if plan['status'] != 'complete':
        raise RuntimeError('Some planned ranges failed; preserve ledger and rerun exact plan to resume')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['index', 'sample', 'all'], required=True)
    parser.add_argument('--sample-count', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-block-mib', type=int, choices=[1,2], default=2)
    parser.add_argument('--direct-s3', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Workers must be between1and8')
    DEST.mkdir(parents=True, exist_ok=True)
    lock = (DEST / '.downloader.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another downloader process owns this dataset')
    try:
        preserve_sources()
        index_path = DEST / 'zip_index.json'
        old_index = json.loads(index_path.read_text()) if index_path.exists() else None
        if old_index and old_index.get('parser_schema_version', 1) < 2:
            revision = DEST / 'failed_metadata_parses'
            revision.mkdir(exist_ok=True)
            for filename in ['zip_index.json', 'resource_audit.json']:
                shutil.copyfile(DEST / filename, revision / filename)
            write_json(revision / 'failure.json', {
                'utc': now(), 'reason': 'ZIP64 sentinel values were not yet expanded from member extra fields',
                'reported_byte_totals_valid': False, 'member_names_and_offsets_valid': True,
                'disposition': 'Correct parser and regenerate from retained official central bytes',
            })
        members = old_index['members'] if old_index and old_index.get('parser_schema_version', 1) >= 2 else index_archive()
        if args.mode != 'index':
            acquire(members, args.sample_count if args.mode == 'sample' else None,
                    workers=args.workers, max_block_bytes=args.max_block_mib*(1<<20),
                    direct_s3=args.direct_s3)
    except BaseException as exc:
        log_event({'stage': args.mode, 'status': 'failed', 'exception': repr(exc)})
        raise


if __name__ == '__main__':
    main()
