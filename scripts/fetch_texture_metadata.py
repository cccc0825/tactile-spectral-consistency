#!/usr/bin/env python3
"""Retrieve the original public Cluster metadata workbook using one ZIP range."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import struct
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://ndownloader.figshare.com/files/61118716'
MEMBER = 'cluster_haptic_texture_dataset/texture_list.xlsx'
START, END, ARCHIVE_BYTES = 8534231, 8550848, 15220748922
EXPECTED_SHA256 = '6cd7dfc06e61852d49a2256d7fe8d44d222f3619f48fe9a4031f677d1b0eb4ff'


def verify(data):
    if len(data) != 19016 or zlib.crc32(data) != 4242404218:
        raise ValueError('Workbook size or archive-member CRC mismatch')
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Workbook SHA-256 mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.project_root / 'outputs/research/corpus/texture_list.xlsx'
    if output.exists():
        verify(output.read_bytes())
        print(json.dumps({'status': 'verified_existing', 'sha256': EXPECTED_SHA256}))
        return
    request = urllib.request.Request(URL, headers={'Range': f'bytes={START}-{END}'})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {START}-{END}/{ARCHIVE_BYTES}':
            raise ValueError('Server did not honor the exact range; full-archive download refused')
        raw = response.read(END - START + 2)
    if len(raw) != END - START + 1:
        raise ValueError('Truncated or oversized ZIP member range')
    header = struct.unpack('<4s5H3I2H', raw[:30])
    _, _, flags, compression, _, _, crc, compressed, size, name_len, extra_len = header
    if header[0] != b'PK\x03\x04' or flags & 1 or compression != 8:
        raise ValueError('Unexpected ZIP member header')
    if raw[30:30 + name_len].decode('utf-8') != MEMBER:
        raise ValueError('Wrong ZIP member')
    data_start = 30 + name_len + extra_len
    # Bit 3 denotes a trailing data descriptor; local CRC/size fields may be
    # zero. The fixed compressed length comes from the public central index.
    payload_bytes = 16492
    if compressed not in (0, payload_bytes):
        raise ValueError('Unexpected compressed member length')
    data = zlib.decompress(raw[data_start:data_start + payload_bytes], -15)
    verify(data)
    if size not in (0, len(data)) or crc not in (0, zlib.crc32(data)):
        raise ValueError('Local header and decoded workbook disagree')
    if flags & 8:
        descriptor = struct.unpack('<4s3I', raw[data_start + payload_bytes:])
        if descriptor != (b'PK\x07\x08', 4242404218, payload_bytes, 19016):
            raise ValueError('ZIP data descriptor mismatch')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(data)
    receipt = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'url': URL, 'member': MEMBER, 'http_range': [START, END],
               'downloaded_bytes': len(raw), 'workbook_bytes': len(data),
               'sha256': EXPECTED_SHA256, 'member_crc32': zlib.crc32(data),
               'license': 'CC BY 4.0; attribution in THIRD_PARTY.md',
               'scope': 'Original upstream file, including its original Office metadata; full archive hash not verified.'}
    output.with_suffix('.download.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'downloaded_and_verified', **receipt}, indent=2))


if __name__ == '__main__':
    main()
