#!/usr/bin/env python3
"""Strict, outcome-blind IMU acquisition audit and frozen spectral extraction.

Author notebook is read as JSON only. External pickle bytecode is inspected with
pickletools; only explicitly mapped passive pandas/numpy reconstruction globals
are allowed. No generic pickle.load or pandas.read_pickle is used.
"""
from __future__ import annotations

import argparse
import csv
import collections
import hashlib
import io
import json
import os
import pickle
import pickletools
import platform
import re
import resource
import struct
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

for _thread_key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_thread_key] = '1'

import numpy as np
import pandas as pd
from pandas._libs.internals import _unpickle_block
from pandas.core.indexes.base import _new_Index
from pandas.core.indexes.datetimes import _new_DatetimeIndex
from pandas.core.internals.managers import BlockManager

from download_external import DEST, write_json, sha_file

SETTINGS = {
    'source_speeds': [30,35], 'target_speed': 40, 'classes': 12, 'episodes_per_cell': 100,
    'window_seconds': 1, 'centers_fraction_of_duration': [1/8,3/8,5/8,7/8],
    'interpolation_hz': 1500, 'lowpass_order': 6, 'lowpass_hz': 100,
    'resample_poly_up': 1, 'resample_poly_down': 3, 'resample_poly_window': ['kaiser',5.0],
    'sampling_hz': 500, 'samples_per_window': 500,
    'welch_nperseg': 256, 'welch_noverlap': 128, 'welch_detrend': 'constant',
    'welch_window': 'hann', 'welch_scaling': 'density',
    'psd_aggregation': 'sum three axes, then mean four windows',
    'stft_nfft': 256, 'stft_hop': 64, 'stft_window': 'hann',
    'stft_detrend': 'constant', 'stft_boundary': None, 'stft_padded': False,
    'stft_scaling': 'spectrum', 'mel_scale': 'HTK', 'mel_count': 32,
    'mel_fmin_hz': 5, 'mel_fmax_hz': 100, 'mel_normalization': 'unnormalized triangles',
    'cnn_log': 'per-axis per-leg maximum reference;10log10 power;clip[-80,0];map db/40+1',
    'cnn_aggregation': 'concatenate four legs along time, bilinear resize to3x128x64',
    'cnn_resize_align_corners': False, 'cnn_resize_antialias': False, 'cnn_dtype': 'float16',
    'min_duration_seconds': 4.4, 'max_window_timestamp_gap_seconds': .05,
    'max_invalid_fraction': .01, 'duplicate_timestamp_policy': 'average acceleration axes',
    'physical_unit_conversion': 'none; retain supplied numeric units',
    'grouping': 'one full original square episode; never split legs across folds',
}


# Classes and constructors map directly to imported objects. No dynamic import,
# attribute traversal, filesystem/network function, or arbitrary callable allowed.
ALLOWED_GLOBALS = {
    ('pandas.core.frame', 'DataFrame'): pd.DataFrame,
    ('pandas.core.internals.managers', 'BlockManager'): BlockManager,
    ('pandas.core.internals', 'BlockManager'): BlockManager,
    ('pandas._libs.internals', '_unpickle_block'): _unpickle_block,
    ('pandas.core.indexes.base', '_new_Index'): _new_Index,
    ('pandas.core.indexes.base', 'Index'): pd.Index,
    ('pandas.core.index', '_new_Index'): _new_Index,
    ('pandas.core.index', 'Index'): pd.Index,
    ('pandas.tseries.index', 'DatetimeIndex'): pd.DatetimeIndex,
    ('pandas.tseries.index', '_new_DatetimeIndex'): _new_DatetimeIndex,
    ('pandas.core.indexes.range', 'RangeIndex'): pd.RangeIndex,
    ('numpy', 'ndarray'): np.ndarray,
    ('numpy', 'dtype'): np.dtype,
    ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
    ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
    ('numpy.core.multiarray', 'scalar'): np._core.multiarray.scalar,
    ('numpy._core.multiarray', 'scalar'): np._core.multiarray.scalar,
    ('numpy.core.numeric', '_frombuffer'): np._core.numeric._frombuffer,
    ('numpy._core.numeric', '_frombuffer'): np._core.numeric._frombuffer,
    ('builtins', 'slice'): slice,
    ('__builtin__', 'slice'): slice,
}


def inspect_pickle(blob):
    """Symbolically inspect opcodes; never call an opcode's constructor."""
    if len(blob) > 32 * (1 << 20):
        raise ValueError('Pickle exceeds per-episode byte limit')
    marker, opaque = object(), object()
    stack, memo, globals_seen = [], {}, set()
    counts = collections.Counter()

    def popmark():
        while stack:
            if stack.pop() is marker:
                return
        raise ValueError('Missing pickle MARK')

    for op, arg, pos in pickletools.genops(blob):
        name = op.name
        counts[name] += 1
        if name in ('PROTO', 'FRAME'):
            continue
        if name == 'STOP':
            if pos != len(blob) - 1:
                raise ValueError('Trailing content after pickle STOP')
            break
        if name in ('EXT1', 'EXT2', 'EXT4', 'PERSID', 'BINPERSID', 'NEXT_BUFFER', 'READONLY_BUFFER', 'INST', 'OBJ'):
            raise ValueError(f'Forbidden pickle opcode: {name}')
        if name == 'GLOBAL':
            module, class_name = arg.split(' ', 1)
            globals_seen.add((module, class_name))
            stack.append(('global', module, class_name))
        elif name == 'STACK_GLOBAL':
            class_name, module = stack.pop(), stack.pop()
            if not isinstance(class_name, str) or not isinstance(module, str):
                raise ValueError('Non-literal STACK_GLOBAL')
            globals_seen.add((module, class_name))
            stack.append(('global', module, class_name))
        elif name in ('BINUNICODE', 'SHORT_BINUNICODE', 'BINUNICODE8', 'UNICODE', 'STRING', 'BINSTRING', 'SHORT_BINSTRING'):
            stack.append(arg)
        elif name == 'MARK':
            stack.append(marker)
        elif name in ('NONE', 'NEWTRUE', 'NEWFALSE', 'INT', 'BININT', 'BININT1', 'BININT2',
                      'LONG', 'LONG1', 'LONG4', 'FLOAT', 'BINFLOAT', 'BINBYTES', 'SHORT_BINBYTES',
                      'BINBYTES8', 'BYTEARRAY8', 'EMPTY_TUPLE', 'EMPTY_LIST', 'EMPTY_DICT', 'EMPTY_SET'):
            stack.append(opaque)
        elif name in ('BINPUT', 'LONG_BINPUT', 'PUT'):
            memo[int(arg)] = stack[-1]
        elif name == 'MEMOIZE':
            memo[len(memo)] = stack[-1]
        elif name in ('BINGET', 'LONG_BINGET', 'GET'):
            stack.append(memo[int(arg)])
        elif name in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
            del stack[-int(name[-1]):]
            stack.append(opaque)
        elif name in ('TUPLE', 'LIST', 'DICT', 'FROZENSET'):
            popmark()
            stack.append(opaque)
        elif name in ('APPENDS', 'SETITEMS', 'ADDITEMS'):
            popmark()
        elif name in ('APPEND', 'BUILD', 'POP'):
            stack.pop()
        elif name == 'SETITEM':
            del stack[-2:]
        elif name in ('REDUCE', 'NEWOBJ'):
            del stack[-2:]
            stack.append(opaque)
        elif name == 'NEWOBJ_EX':
            del stack[-3:]
            stack.append(opaque)
        elif name == 'DUP':
            stack.append(stack[-1])
        elif name == 'POP_MARK':
            popmark()
        else:
            raise ValueError(f'Unrecognized pickle opcode: {name}')
        if len(stack) > 100000 or len(memo) > 100000:
            raise ValueError('Pickle symbolic resource limit exceeded')
    forbidden = globals_seen - ALLOWED_GLOBALS.keys()
    report = {'globals': [list(x) for x in sorted(globals_seen)],
              'opcode_counts': dict(counts), 'bytes': len(blob),
              'forbidden_globals': [list(x) for x in sorted(forbidden)]}
    if forbidden:
        raise ValueError(f'Forbidden pickle globals: {sorted(forbidden)}')
    return report


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        key = module, name
        if key not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f'Forbidden global: {module}.{name}')
        return ALLOWED_GLOBALS[key]

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('Persistent references are forbidden')


def load_passive_dataframe(blob):
    inspection = inspect_pickle(blob)
    # The source was serialized under Python 2; latin1 preserves byte values in
    # NumPy's legacy array state, as distinct from executing any source code.
    value = RestrictedUnpickler(io.BytesIO(blob), encoding='latin1').load()
    if type(value) is not pd.DataFrame:
        raise ValueError(f'Expected exact pandas DataFrame, got {type(value)}')
    if value.shape[0] > 2_000_000 or value.shape[1] > 64:
        raise ValueError('DataFrame exceeds schema bounds')
    if any(not isinstance(c, str) for c in value.columns):
        raise ValueError('Unexpected non-string column')
    return value, inspection


def member_payload(member, block):
    """Validate local header, decompress in memory, check size and official CRC."""
    path = DEST / block['path']
    with path.open('rb') as stream:
        stream.seek(member['local_header_offset'] - block['start'])
        header = stream.read(30)
        sig, version, flags, method, mt, md, crc, csize, usize, nlen, xlen = struct.unpack('<4s5H3I2H', header)
        if sig != b'PK\x03\x04' or flags != member['flags'] or method != member['compression']:
            raise ValueError('Local/central ZIP header mismatch')
        name = stream.read(nlen).decode('utf-8' if flags & 0x800 else 'cp437')
        if name != member['name']:
            raise ValueError('ZIP member name mismatch')
        stream.read(xlen)
        compressed = stream.read(member['compressed_bytes'])
    if len(compressed) != member['compressed_bytes'] or member['uncompressed_bytes'] > 32*(1<<20):
        raise ValueError('Member length violates bounds')
    if method == 0:
        blob = compressed
    else:
        obj = zlib.decompressobj(-15)
        blob = obj.decompress(compressed, member['uncompressed_bytes']+1)
        if not obj.eof or obj.unused_data or obj.unconsumed_tail:
            raise ValueError('Malformed ZIP deflate stream')
    actual_crc = f'{zlib.crc32(blob) & 0xffffffff:08x}'
    if len(blob) != member['uncompressed_bytes'] or actual_crc != member['crc32']:
        raise ValueError('ZIP member CRC or uncompressed size mismatch')
    return blob


def available_members():
    index = json.loads((DEST / 'zip_index.json').read_text())['members']
    member_by_name = {m['name']: m for m in index}
    plans = []
    # Full acquisition supersedes the single-episode probe when a member appears in both.
    for filename in ['acquisition_sample.json', 'acquisition_plan.json']:
        path = DEST / filename
        if path.exists():
            plans.append(json.loads(path.read_text()))
    located = {}
    for plan in plans:
        for block in plan['blocks']:
            if block.get('status') != 'complete':
                continue
            path = DEST / block['path']
            if sha_file(path) != block['sha256']:
                raise ValueError(f'Acquisition chunk SHA256 mismatch: {path}')
            for name in block['members']:
                located[name] = (member_by_name[name], block)
    return [located[name] for name in sorted(located)]


def dataframe_schema(frame):
    result = {'shape': list(frame.shape), 'columns': list(frame.columns),
              'dtypes': {str(c): str(t) for c,t in frame.dtypes.items()},
              'index_type': type(frame.index).__name__, 'index_name': frame.index.name,
              'index_head': [str(x) for x in frame.index[:5]],
              'index_tail': [str(x) for x in frame.index[-5:]],
              'head': frame.head(3).astype(str).to_dict(orient='records'),
              'tail': frame.tail(3).astype(str).to_dict(orient='records'),
              'missing_per_column': {str(c): int(x) for c,x in frame.isna().sum().items()},
              'time_candidates': {}}
    candidates = [(str(c), frame[c].to_numpy()) for c in frame.columns
                  if any(s in str(c).lower() for s in ('time', 'stamp'))]
    if isinstance(frame.index, pd.DatetimeIndex):
        native = frame.index.as_unit('ns').asi8
        candidates.append(('__datetime_index_elapsed_seconds__', (native-native[0])*1e-9))
        result['timestamp_encoding'] = {'source_dtype': str(frame.index.dtype),
                                        'conversion': 'DatetimeIndex.as_unit(ns).asi8; subtract first; multiply1e-9',
                                        'reported_units_after_conversion': 'seconds'}
    if pd.api.types.is_numeric_dtype(frame.index.dtype):
        candidates.append(('__index__', frame.index.to_numpy()))
    for name, value in candidates:
        try:
            values = np.asarray(value, dtype=np.float64)
            delta = np.diff(values)
            positive = delta[delta>0]
            result['time_candidates'][name] = {
                'first': float(values[0]), 'last': float(values[-1]),
                'span_native_units': float(values[-1]-values[0]),
                'delta_min': float(np.min(delta)), 'delta_max': float(np.max(delta)),
                'delta_median': float(np.median(delta)),
                'delta_quantiles_01_05_95_99': np.quantile(delta, [.01,.05,.95,.99]).tolist(),
                'positive_median': float(np.median(positive)) if len(positive) else None,
                'negative_delta_count': int(np.sum(delta<0)), 'zero_delta_count': int(np.sum(delta==0)),
            }
        except (TypeError, ValueError):
            result['time_candidates'][name] = {'numeric': False}
    return result


def audit_notebook():
    path = DEST / 'source' / 'reading-pickle-file.ipynb'
    notebook = json.loads(path.read_text())
    # Deliberately no nbconvert, exec, eval, import-from-notebook or notebook kernel.
    cells = [{'cell_type': c.get('cell_type'), 'source': ''.join(c.get('source', []))}
             for c in notebook.get('cells', [])]
    text = '\n\n'.join(f"CELL {i} ({c['cell_type']})\n{c['source']}" for i,c in enumerate(cells))
    output = DEST / 'source' / 'notebook_source_readonly.txt'
    output.write_text(text)
    return {'path': str(path.relative_to(DEST)), 'sha256': sha_file(path),
            'source_text_path': str(output.relative_to(DEST)), 'executed': False,
            'cell_count': len(cells)}


def audit(sample_count):
    notebook = audit_notebook()
    records = []
    for member, block in available_members()[:sample_count]:
        blob = member_payload(member, block)
        frame, inspection = load_passive_dataframe(blob)
        records.append({'member': member, 'pickle_sha256': hashlib.sha256(blob).hexdigest(),
                        'crc_verified': True, 'pickle_inspection': inspection,
                        'schema': dataframe_schema(frame)})
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
              'phase': 'outcome_blind_schema_audit', 'notebook': notebook, 'records': records,
              'environment': {'python': sys.version, 'numpy': np.__version__,
                              'pandas': pd.__version__, 'platform': platform.platform()},
              'preprocessing_specification_pending': True}
    write_json(DEST / 'schema_audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


def episode_labels(name):
    match = re.fullmatch(r'pickles_(30|35|40)/texture_(\d{2})/full_imu/imu_?(\d{2})(\d+)\.pkl', name)
    if not match or match[2] != match[3]:
        raise ValueError(f'Unexpected episode identifier: {name}')
    speed, texture, trial = int(match[1]), int(match[2])-1, int(match[4])
    if not (0 <= texture < 12 and 1 <= trial <= 100):
        raise ValueError('Episode label outside frozen factorial design')
    return [texture,speed,trial]


def quality_and_windows(frame):
    from scipy.signal import butter, sosfiltfilt, resample_poly
    expected = ['imu_ax','imu_ay','imu_az','imu_gx','imu_gy','imu_gz','imu_mx','imu_my','imu_mz']
    if list(frame.columns) != expected or not isinstance(frame.index,pd.DatetimeIndex):
        return {'valid': False, 'reasons': ['schema_mismatch']}, None, None
    if frame.index.hasnans:
        return {'valid': False, 'reasons': ['missing_timestamp']}, None, None
    ns = frame.index.as_unit('ns').asi8
    ns = ns-ns[0]
    t = ns.astype(np.float64)*1e-9
    values = frame.to_numpy(dtype=np.float64)
    accel = values[:,:3]
    finite = bool(np.isfinite(accel).all())
    delta = np.diff(t)
    duration = float(t[-1])
    reasons = []
    if not finite: reasons.append('nonfinite_acceleration')
    if np.any(delta<0): reasons.append('decreasing_timestamp')
    if duration < SETTINGS['min_duration_seconds']: reasons.append('duration_below4.4s')
    centers = duration*np.asarray(SETTINGS['centers_fraction_of_duration'])
    maxgaps = []
    for center in centers:
        intersects = (t[:-1]<center+.5)&(t[1:]>center-.5)
        maxgaps.append(float(np.max(delta[intersects])) if np.any(intersects) else None)
    if any(g is None or g > .05 for g in maxgaps): reasons.append('window_timestamp_gap_above50ms')
    q = {'valid': not reasons, 'reasons': reasons, 'n_rows': len(frame),
         'duration_seconds': duration, 'finite_acceleration': finite,
         'finite_all_nine_channels': bool(np.isfinite(values).all()),
         'negative_timestamp_steps': int(np.sum(delta<0)),
         'duplicate_timestamp_count': int(np.sum(delta==0)),
         'timestamp_delta_median_seconds': float(np.median(delta)),
         'timestamp_delta_max_seconds': float(np.max(delta)),
         'window_centers_seconds': centers.tolist(), 'window_max_timestamp_gap_seconds': maxgaps,
         'consecutive_acceleration_hold_fraction': float(np.mean(np.all(np.diff(accel,axis=0)==0,axis=1))),
         'clipping_audit': {'hardware_limits_known': False, 'amplitude_exclusion_applied': False,
                           'interpretation': 'Observed extrema and recurrence only; not certified clipping'}}
    if finite:
        lo,hi = accel.min(axis=0),accel.max(axis=0)
        q['clipping_audit'].update({'axis_min': lo.tolist(), 'axis_max': hi.tolist(),
                                   'observed_extrema_fraction': np.mean((accel==lo)|(accel==hi),axis=0).tolist()})
    # Canonical content hash changes neither units nor amplitudes. It ignores
    # serialization details and absolute epoch offsets, but retains every sample.
    h = hashlib.sha256()
    h.update(json.dumps(expected,separators=(',',':')).encode())
    h.update(np.asarray(ns,dtype='<i8').tobytes())
    canonical = np.array(values,dtype='<f8',order='C',copy=True)
    canonical[canonical==0] = 0.0
    h.update(canonical.tobytes())
    content_sha = h.hexdigest()
    if reasons:
        return q,None,content_sha
    if q['duplicate_timestamp_count']:
        unique, inverse, counts = np.unique(ns,return_inverse=True,return_counts=True)
        merged = np.zeros((len(unique),3),dtype=np.float64)
        np.add.at(merged,inverse,accel)
        accel = merged/counts[:,None]
        t = unique*1e-9
    grid = np.arange(int(np.floor(duration*1500))+1,dtype=np.float64)/1500
    x1500 = np.stack([np.interp(grid,t,accel[:,i]) for i in range(3)])
    sos = butter(6,100,btype='lowpass',fs=1500,output='sos')
    filtered = sosfiltfilt(sos,x1500,axis=-1)
    x500 = resample_poly(filtered,1,3,axis=-1,window=('kaiser',5.0),padtype='constant')
    starts = np.rint((centers-.5)*500).astype(int)
    if any(start < 0 or start+500>x500.shape[-1] for start in starts):
        raise ValueError('Frozen leg window unexpectedly outside full recording')
    windows = np.stack([x500[:,start:start+500] for start in starts])
    q['actual_window_starts_seconds'] = (starts/500).tolist()
    q['full_interpolated_samples'] = len(grid)
    q['full_resampled_samples'] = x500.shape[-1]
    if windows.shape != (4,3,500) or not np.isfinite(windows).all():
        raise ValueError('Window extraction shape/finiteness failure')
    return q,windows,content_sha


def spectral_features(windows):
    from scipy.signal import welch,stft
    import torch
    import torch.nn.functional as functional
    frequency,axes_psd = welch(windows,fs=500,window='hann',nperseg=256,
                               noverlap=128,nfft=256,detrend='constant',
                               return_onesided=True,scaling='density',axis=-1)
    psd = axes_psd.sum(axis=1).mean(axis=0)
    f,_,z = stft(windows,fs=500,window='hann',nperseg=256,noverlap=192,nfft=256,
                 detrend='constant',return_onesided=True,boundary=None,padded=False,
                 axis=-1,scaling='spectrum')
    mel = lambda value:2595*np.log10(1+np.asarray(value)/700)
    hz = lambda value:700*(10**(np.asarray(value)/2595)-1)
    edges = hz(np.linspace(mel(5),mel(100),34))
    bank = np.maximum(0,np.minimum((f[None,:]-edges[:-2,None])/(edges[1:-1,None]-edges[:-2,None]),
                                  (edges[2:,None]-f[None,:])/(edges[2:,None]-edges[1:-1,None])))
    if not np.all(bank.sum(axis=1)>0): raise ValueError('Empty frozen mel filter')
    power = (np.abs(z)**2).astype(np.float32)
    mp = np.einsum('mf,lcft->lcmt',bank.astype(np.float32),power,optimize=True)
    ref = np.maximum(mp.max(axis=(2,3),keepdims=True),np.float32(1e-20))
    db = np.clip(10*np.log10(np.maximum(mp/ref,np.float32(1e-8))),-80,0)
    joined = np.concatenate([db[leg] for leg in range(4)],axis=-1)
    tensor = torch.from_numpy((joined/40+1).astype(np.float32))[None]
    image = functional.interpolate(tensor,size=(128,64),mode='bilinear',
                                   align_corners=False,antialias=False)[0].numpy().astype(np.float16)
    if psd.shape!=(129,) or image.shape!=(3,128,64) or not np.isfinite(psd).all() or not np.isfinite(image).all():
        raise ValueError('Frozen spectrum shape/finiteness failure')
    return frequency,psd,image


def extraction_smoke():
    import torch
    torch.set_num_threads(1)
    t = np.arange(12*1500)/1500
    values = np.zeros((len(t),9),dtype=np.float64)
    values[:,0] = np.sin(2*np.pi*31.25*t)
    frame = pd.DataFrame(values,columns=['imu_ax','imu_ay','imu_az','imu_gx','imu_gy','imu_gz','imu_mx','imu_my','imu_mz'],
                         index=pd.to_datetime(np.rint(t*1e9).astype(np.int64)))
    q,w,h = quality_and_windows(frame)
    freq,p,img = spectral_features(w)
    checks = {'source_independent_sinusoid_peak31.25Hz': bool(freq[np.argmax(p)]==31.25),
              'all_four_legs_one_tensor': w.shape==(4,3,500),
              'cnn_shape_and_range': bool(img.shape==(3,128,64) and img.min()>=-1 and img.max()<=1),
              'quality_accepts_regular_time': bool(q['valid'])}
    # A gap crossing the first leg must be a common invalid episode.
    removed = frame[(t<1.45)|(t>1.55)]
    qbad,_,_ = quality_and_windows(removed)
    checks['window_gap_rejected'] = not qbad['valid'] and 'window_timestamp_gap_above50ms' in qbad['reasons']
    report = {'checks': checks, 'all_passed': all(checks.values()),
              'settings': SETTINGS, 'extractor_sha256': sha_file(__file__),
              'classification_outcomes_computed': False}
    write_json(DEST/'extraction_smoke_checks.json',report)
    if not report['all_passed']: raise ValueError('Extraction smoke failed')
    print(json.dumps(report,indent=2))


def extract(protocol_file,freeze_file,follow_download=False):
    import torch
    torch.set_num_threads(1)
    protocol,freeze = Path(protocol_file),Path(freeze_file)
    if not protocol.is_file() or not freeze.is_file():
        raise ValueError('Root protocol and freeze must exist before extraction')
    protocol_hash,freeze_hash = sha_file(protocol),sha_file(freeze)
    frozen = json.loads(freeze.read_text())
    def literals(value):
        if isinstance(value,dict):
            for item in value.values(): yield from literals(item)
        elif isinstance(value,list):
            for item in value: yield from literals(item)
        elif isinstance(value,str): yield value
    if protocol_hash not in set(literals(frozen)):
        raise ValueError('Protocol current SHA256 is not represented in root freeze')
    smoke = json.loads((DEST/'extraction_smoke_checks.json').read_text())
    if not smoke['all_passed'] or smoke['extractor_sha256']!=sha_file(__file__):
        raise ValueError('Passing smoke must refer to the current extractor')
    plan = json.loads((DEST/'acquisition_plan.json').read_text())
    if plan['status']!='complete' and not follow_download:
        raise ValueError('Full acquisition not complete; use explicit follow-download to pipeline verified blocks')
    index = json.loads((DEST/'zip_index.json').read_text())['members']
    by_name = {m['name']:m for m in index}
    member_locations = {}
    for block in plan['blocks']:
        for name in block['members']:
            member_locations[name] = (by_name[name],block)
    entries = [member_locations[name] for name in sorted(member_locations)]
    expected_block_members = {b['start']:tuple(b['members']) for b in plan['blocks']}
    verified_blocks = {}

    def ready_block(expected):
        if expected['start'] in verified_blocks:
            return verified_blocks[expected['start']]
        last_notice = 0
        while True:
            current_plan = json.loads((DEST/'acquisition_plan.json').read_text())
            current_blocks = {b['start']:b for b in current_plan['blocks']}
            block = current_blocks.get(expected['start'])
            if block is None or tuple(block['members'])!=expected_block_members[expected['start']]:
                raise ValueError('Acquisition member-to-block mapping changed during extraction')
            if block.get('status')=='complete':
                if sha_file(DEST/block['path'])!=block['sha256']:
                    raise ValueError('Completed acquisition block SHA256 mismatch')
                verified_blocks[block['start']] = block
                return block
            if block.get('status')=='failed' or current_plan['status']=='failed':
                raise ValueError('Acquisition failed; partial extraction retained for explicit restart')
            if not follow_download: raise ValueError('Required acquisition block not complete')
            if time.monotonic()-last_notice>=60:
                print(json.dumps({'status':'waiting_for_verified_block','start':expected['start'],
                                  'utc':datetime.now(timezone.utc).isoformat()}),flush=True)
                last_notice = time.monotonic()
            time.sleep(5)
    labels = np.array([episode_labels(m['name']) for m,b in entries],dtype=np.int64)
    expected = {(c,s,t) for c in range(12) for s in [30,35,40] for t in range(1,101)}
    if len(entries)!=3600 or set(map(tuple,labels))!=expected:
        raise ValueError('Full3600-episode factorial inventory mismatch')
    out = DEST/'processed'
    out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists(): raise ValueError('Refusing to overwrite published extraction')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    staging = out/f'staging_{run_id}'
    staging.mkdir()
    started = time.monotonic()
    write_json(staging/'running.json',{'status':'running','settings':SETTINGS,'protocol_sha256':protocol_hash,
                                     'freeze_sha256':freeze_hash,'extractor_sha256':sha_file(__file__)})
    images = np.lib.format.open_memmap(staging/'input_all.npy',mode='w+',dtype=np.float16,shape=(3600,3,128,64))
    spectra = np.full((3600,129),np.nan,dtype=np.float64)
    valid = np.zeros(3600,dtype=bool)
    records = []
    for row,((member,block),label) in enumerate(zip(entries,labels)):
        block = ready_block(block)
        record = {'row_original':row,'id':f'{label[0]:02d}_{label[1]}_{label[2]:03d}',
                  'labels':label.tolist(),'member':member['name'],'source_crc32':member['crc32']}
        try:
            blob = member_payload(member,block)
            record['source_pickle_sha256'] = hashlib.sha256(blob).hexdigest()
            frame,inspection = load_passive_dataframe(blob)
            q,windows,content_hash = quality_and_windows(frame)
            record.update(q)
            record['normalized_content_sha256'] = content_hash
            record['pickle_globals'] = inspection['globals']
            if q['valid']:
                frequency,spectra[row],images[row] = spectral_features(windows)
                valid[row] = True
        except (ValueError,pickle.UnpicklingError,TypeError,IndexError,KeyError) as exc:
            record.update({'valid':False,'reasons':['ingestion_or_extraction_failure'],'error':repr(exc)})
        records.append(record)
        if (row+1)%100==0:
            images.flush()
            write_json(staging/'records_in_progress.json',records)
            print(json.dumps({'processed':row+1,'valid':int(valid.sum()),'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    images.flush()
    # Last member can finish just before the downloader seals its manifest.
    for attempt in range(12):
        final_plan = json.loads((DEST/'acquisition_plan.json').read_text())
        if final_plan['status']=='complete': break
        if final_plan['status']=='failed': raise ValueError('Acquisition seal failed')
        time.sleep(5)
    if final_plan['status']!='complete' or len(verified_blocks)!=len(final_plan['blocks']):
        raise ValueError('Cannot seal extraction before every acquisition block is verified')
    duplicate_groups = {}
    for field in ['source_pickle_sha256','normalized_content_sha256']:
        groups = collections.defaultdict(list)
        for record in records:
            if record.get(field): groups[record[field]].append(record)
        duplicate_groups[field] = [
            {'sha256':h,'ids':[r['id'] for r in group],'speeds':sorted(set(r['labels'][1] for r in group))}
            for h,group in groups.items() if len(group)>1]
    cross_speed = [g for groups in duplicate_groups.values() for g in groups if len(g['speeds'])>1]
    cross_source_target = [g for g in cross_speed if 40 in g['speeds'] and any(s in g['speeds'] for s in [30,35])]
    cell_counts = {f'{c:02d}_{s}':int(np.sum(valid&(labels[:,0]==c)&(labels[:,1]==s))) for c in range(12) for s in [30,35,40]}
    reason_counts = collections.Counter(reason for record in records if not record['valid'] for reason in record['reasons'])
    gate_pass = valid.mean()>=.99 and min(cell_counts.values())>0 and not cross_source_target
    audit_report = {'status':'pass' if gate_pass else 'needs_root_decision','total':3600,'valid':int(valid.sum()),
                    'invalid':int((~valid).sum()),'invalid_fraction':float((~valid).mean()),
                    'invalid_reasons':dict(reason_counts),'valid_class_speed_counts':cell_counts,
                    'duplicates':duplicate_groups,'cross_speed_duplicate_group_count':len(cross_speed),
                    'source_target_duplicate_group_count':len(cross_source_target),
                    'duplicate_hash_definition':'Exact source pickle bytes and canonical elapsedns+all9float64columns; no amplitude normalization',
                    'target_classification_outcomes_computed':False,'settings':SETTINGS,
                    'protocol_sha256':protocol_hash,'freeze_sha256':freeze_hash}
    write_json(DEST/'quality_audit.json',audit_report)
    write_json(DEST/'trial_integrity.json',records)
    if not gate_pass:
        write_json(staging/'stopped.json',audit_report)
        raise ValueError('Dataset quality gate needs root decision; no common feature files published')
    selected = np.flatnonzero(valid)
    final_images = np.lib.format.open_memmap(staging/'input.npy',mode='w+',dtype=np.float16,shape=(len(selected),3,128,64))
    for output_row,original_row in enumerate(selected): final_images[output_row]=images[original_row]
    final_images.flush()
    np.savez_compressed(staging/'psd.npz',psd=spectra[valid],frequency=frequency,labels=labels[valid])
    with (staging/'metadata.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['row','id','class_id','speed','trial','source','cv_fold','domain'])
        writer.writeheader()
        for output_row,original_row in enumerate(selected):
            c,s,t = map(int,labels[original_row])
            writer.writerow({'row':output_row,'id':records[original_row]['id'],'class_id':c,'speed':s,
                             'trial':t,'source':int(s in [30,35]),'cv_fold':str(t%3),
                             'domain':'speed' if s==40 else 'source'})
    if sha_file(protocol)!=protocol_hash or sha_file(freeze)!=freeze_hash:
        raise ValueError('Protocol or freeze changed during extraction')
    del final_images,images
    artifacts = {}
    for name in ['input.npy','psd.npz','metadata.csv']:
        path = staging/name
        artifacts[name] = {'sha256':sha_file(path),'bytes':path.stat().st_size}
        os.replace(path,out/name)
    manifest = {'status':'complete','run_id':run_id,'rows':len(selected),'source_rows':int(np.sum(labels[valid,1]!=40)),
                'target_rows':int(np.sum(labels[valid,1]==40)),'cnn_shape':[len(selected),3,128,64],
                'psd_shape':[len(selected),129],'labels_columns':['class_id0to11','nominal_speed_mm_s','episode_id1to100'],
                'artifacts':artifacts,'quality_audit_sha256':sha_file(DEST/'quality_audit.json'),
                'trial_integrity_sha256':sha_file(DEST/'trial_integrity.json'),
                'acquisition_plan_sha256':sha_file(DEST/'acquisition_plan.json'),
                'extractor_sha256':sha_file(__file__),'protocol_sha256':protocol_hash,'freeze_sha256':freeze_hash,
                'settings':SETTINGS,'elapsed_seconds':time.monotonic()-started,
                'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'target_classification_outcomes_computed':False,'full_archive_sha256_verified':False}
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'status':'complete','rows':len(selected),'elapsed_seconds':manifest['elapsed_seconds']}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['audit','smoke','extract'], required=True)
    parser.add_argument('--sample-count', type=int, default=1)
    parser.add_argument('--protocol-file')
    parser.add_argument('--freeze-file')
    parser.add_argument('--follow-download',action='store_true')
    args = parser.parse_args()
    try:
        if args.mode=='audit': audit(args.sample_count)
        elif args.mode=='smoke': extraction_smoke()
        else:
            if not args.protocol_file or not args.freeze_file:
                parser.error('Extraction requires protocol-file and freeze-file')
            extract(args.protocol_file,args.freeze_file,args.follow_download)
    except BaseException as exc:
        with (DEST / 'extraction_failures.jsonl').open('a') as stream:
            stream.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                    'exception': repr(exc), 'mode': args.mode})+'\n')
        raise


if __name__ == '__main__':
    main()
