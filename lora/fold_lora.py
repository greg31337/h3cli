#!/usr/bin/env python3
"""Strict, bounded-memory offline folding of ordinary H3 LoRAs.

Only NumPy and safetensors are required. No inference framework is imported.
The normal checkpoint bytes and inference engine remain unchanged in format.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile

try:
    import numpy as np
    from safetensors import safe_open
except ImportError as exc:
    raise SystemExit("Install dependencies in lora/.venv: python -m pip install -r lora/requirements.txt") from exc

VERSION = '1.0.0'
MANIFEST = 'h3_lora_manifest.json'
TURBO_SHA = '5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3'
PROFILES = {
    'larryvrh-turbo-v4-step600-ema': {
        'sha256': TURBO_SHA, 'type': 'turbo', 'recommended_steps': [6, 8],
        'supported_step_range': [4, 8], 'recommended_scheduler': 'simple (native H3 dual AV schedule)',
        'reuse_compatible': False, 'core_reuse_compatible': False,
        'source': 'https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora',
        'notes': 'Use strength 1.0 and 8 steps (6 faster). Four-step fast motion can smear. '
                 'Reuse/core-reuse combinations are unvalidated; leave both disabled. '
                 'Profile advice does not change h3cli sampling or prove Ref2VA quality.'},
}
SIZES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2, 'BF16': 2, 'F16': 2,
         'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}
FLOATS = {'BF16', 'F16', 'F32'}


class FoldError(ValueError):
    pass


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FoldError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(raw):
    try:
        return json.loads(raw, object_pairs_hook=unique_object,
                          parse_constant=lambda x: (_ for _ in ()).throw(FoldError(f'invalid JSON number {x}')))
    except (ValueError, UnicodeError) as exc:
        raise FoldError(f'invalid JSON: {exc}') from exc


def sha256(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    s = Path(path).stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


@dataclass(frozen=True)
class Tensor:
    path: Path
    name: str
    dtype: str
    shape: tuple
    offset: int
    size: int

    def scalar(self):
        if math.prod(self.shape) != 1 or self.dtype == 'BOOL':
            raise FoldError(f'{self.name}: alpha must be a numeric scalar')
        if self.dtype in FLOATS:
            return self.read().item()
        formats = {'F64': '<d', 'I8': '<b', 'U8': '<B', 'I16': '<h', 'U16': '<H',
                   'I32': '<i', 'U32': '<I', 'I64': '<q', 'U64': '<Q'}
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            raw = stream.read(self.size)
        if self.dtype not in formats or len(raw) != self.size:
            raise FoldError(f'{self.name}: invalid scalar dtype/payload')
        return struct.unpack(formats[self.dtype], raw)[0]

    def read(self, start=0, stop=None):
        """Return owned FP32 storage; slicing happens before conversion."""
        dtype = {'BF16': '<u2', 'F16': '<f2', 'F32': '<f4'}.get(self.dtype)
        if dtype is None:
            raise FoldError(f'{self.name}: unsupported arithmetic dtype {self.dtype}')
        raw = np.memmap(self.path, dtype=dtype, mode='r', offset=self.offset, shape=self.shape or ())
        view = raw[start:stop] if self.shape else raw
        if self.dtype == 'BF16':
            result = (np.asarray(view, dtype=np.uint32) << 16).view(np.float32)
        else:
            result = np.array(view, dtype=np.float32, copy=True)
        if not np.isfinite(result).all():
            raise FoldError(f'{self.name}: non-finite tensor values')
        return result


class Shard:
    def __init__(self, path):
        self.path = Path(path)
        self.stamp = stamp(self.path)
        self.size = self.stamp[2]
        with self.path.open('rb') as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise FoldError(f'{path}: truncated safetensors header')
            length, = struct.unpack('<Q', prefix)
            if length < 2 or length > 100_000_000 or length > self.size - 8:
                raise FoldError(f'{path}: invalid safetensors header length')
            self.header = prefix + f.read(length)
        data = read_json(self.header[8:])
        if not isinstance(data, dict):
            raise FoldError(f'{path}: header must be an object')
        self.metadata = data.get('__metadata__', {})
        if not isinstance(self.metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                    for k, v in self.metadata.items()):
            raise FoldError(f'{path}: metadata must contain strings')
        self.tensors = {}
        ranges = []
        for name, desc in data.items():
            if name == '__metadata__':
                continue
            if not name or not isinstance(desc, dict) or set(desc) != {'dtype', 'shape', 'data_offsets'}:
                raise FoldError(f'{path}: invalid tensor descriptor {name}')
            dtype, shape, offsets = desc['dtype'], desc['shape'], desc['data_offsets']
            if not isinstance(dtype, str) or dtype not in SIZES:
                raise FoldError(f'{name}: unsupported dtype {dtype}')
            if not isinstance(shape, list) or len(shape) > 16 or any(type(x) is not int or x < 0 or x > 2**63-1 for x in shape):
                raise FoldError(f'{name}: invalid shape')
            size = math.prod(shape) * SIZES[dtype]
            if size > 2**63-1 or not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
                raise FoldError(f'{name}: overflow/invalid offsets')
            lo, hi = offsets
            if lo < 0 or hi < lo or hi - lo != size or hi > self.size - len(self.header):
                raise FoldError(f'{name}: payload bounds/shape size mismatch')
            if lo % SIZES[dtype]:
                raise FoldError(f'{name}: misaligned tensor-relative payload offset')
            self.tensors[name] = Tensor(self.path, name, dtype, tuple(shape), len(self.header)+lo, size)
            ranges.append((lo, hi))
        cursor = 0
        for lo, hi in sorted(ranges):
            if lo != cursor:
                raise FoldError(f'{path}: overlapping tensors or holes in payload')
            cursor = hi
        if cursor != self.size - len(self.header):
            raise FoldError(f'{path}: trailing/truncated payload')
        # Independent Rust parser validates the file, without loading tensor data.
        try:
            with safe_open(str(self.path), framework='numpy') as sf:
                if set(sf.keys()) != set(self.tensors):
                    raise FoldError(f'{path}: safetensors tensor enumeration mismatch')
        except Exception as exc:
            raise FoldError(f'{path}: safetensors validation failed: {exc}') from exc
        # Released H3 files can have non-16-byte-aligned absolute payloads.
        # Preserve them exactly; h3cli has a safe owned-buffer fallback.
        self.unaligned = [t.name for t in self.tensors.values() if t.offset % 16]


class Checkpoint:
    def __init__(self, directory):
        self.path = Path(directory).resolve(strict=True)
        if not self.path.is_dir():
            raise FoldError('--checkpoint must be a transformer directory')
        self.files = sorted(self.path.iterdir())
        if any(not p.is_file() for p in self.files):
            raise FoldError('transformer directory must contain only regular files (file symlinks are supported)')
        self.stamps = {p.name: stamp(p) for p in self.files}
        self.shards = {p.name: Shard(p) for p in self.files if p.suffix == '.safetensors'}
        if not self.shards:
            raise FoldError('no safetensors shards in transformer directory')
        self.tensors = {}
        for shard in self.shards.values():
            for name, tensor in shard.tensors.items():
                if name in self.tensors:
                    raise FoldError(f'duplicate checkpoint tensor: {name}')
                self.tensors[name] = tensor
        index = self.path / 'model.safetensors.index.json'
        if index.exists():
            data = read_json(index.read_bytes())
            expected = {n: t.path.name for n, t in self.tensors.items()}
            if not isinstance(data, dict) or data.get('weight_map') != expected:
                raise FoldError('shard index weight_map does not exactly match discovered tensors')
        config = self.path / 'config.json'
        self.config = read_json(config.read_bytes()) if config.exists() else {}
        if not isinstance(self.config, dict):
            raise FoldError('transformer config must be a JSON object')

    def unchanged(self):
        if set(p.name for p in self.path.iterdir()) != set(self.stamps):
            raise FoldError('source checkpoint files changed during folding')
        for name, value in self.stamps.items():
            if stamp(self.path / name) != value:
                raise FoldError(f'source checkpoint changed during folding: {name}')


def parse_lora(value):
    # An existing literal filename wins, including colons. Otherwise the last
    # colon is the scale delimiter; punctuation elsewhere is never split.
    if Path(value).is_file():
        return Path(value).resolve(), 1.0
    path, sep, suffix = value.rpartition(':')
    if not sep:
        path, suffix = value, '1.0'
    try:
        scale = float(suffix)
    except ValueError as exc:
        raise FoldError(f'invalid LoRA scale {suffix!r} in {value!r}') from exc
    if not path or not math.isfinite(scale) or abs(scale) > np.finfo(np.float32).max:
        raise FoldError(f'invalid/non-finite LoRA scale in {value!r}')
    p = Path(path).resolve(strict=True)
    if not p.is_file():
        raise FoldError(f'LoRA is not a file: {p}')
    return p, scale


PAIR = re.compile(r'^(.*)\.(lora_A|lora_B|lora_down|lora_up)(?:\.default)?\.weight$')
PREFIXES = ('base_model.model.', 'model.diffusion_model.', 'diffusion_model.', 'transformer.')


def normalize_target(module, checkpoint):
    """Naming only. No adapter math or shape-based guesses belong here."""
    original = module
    convention = 'native'
    for prefix in PREFIXES:
        if module.startswith(prefix):
            module = module[len(prefix):]
            convention = 'peft/native'
    if module.startswith('lora_unet_'):
        flat = module[len('lora_unet_'):]
        matches = [n[:-7] for n in checkpoint.tensors if n.endswith('.weight') and n[:-7].replace('.', '_') == flat]
        if len(matches) != 1:
            raise FoldError(f'unknown/ambiguous ComfyUI target: {original}')
        module, convention = matches[0], 'comfyui'
    module = re.sub(r'^transformer_blocks\.', 'blocks.', module)
    split = re.fullmatch(r'((?:token_refiner\.)?blocks\.\d+\.attn)\.(to_q|to_k|to_v)', module)
    if split:
        name = split[1] + '.qkv_proj.weight'
        tensor = checkpoint.tensors.get(name)
        if tensor is None or len(tensor.shape) != 2 or not all(tensor.shape) or tensor.shape[0] % 3:
            raise FoldError(f'cannot unambiguously split fused H3 QKV: {original}')
        # H3 has equal Q/K/V widths, despite the Qwen text encoder using GQA.
        width = tensor.shape[0] // 3
        heads = checkpoint.config.get('num_attention_heads')
        dim = checkpoint.config.get('attention_head_dim')
        if heads is not None and dim is not None:
            if type(heads) is not int or type(dim) is not int or heads <= 0 or dim <= 0 or heads * dim != width:
                raise FoldError(f'fused QKV dimensions disagree with H3 config: {name}')
        lo = {'to_q': 0, 'to_k': width, 'to_v': 2*width}[split[2]]
        return name, lo, lo+width, 'diffusers-qkv'
    module = re.sub(r'\.attn\.to_out\.0$', '.attn.out_proj', module)
    name = module + '.weight'
    if name not in checkpoint.tensors:
        raise FoldError(f'unmatched adapter target: {original} -> {name}')
    tensor = checkpoint.tensors[name]
    if len(tensor.shape) != 2 or not all(tensor.shape):
        raise FoldError(f'{name}: LoRA target must be a nonempty matrix')
    return name, 0, tensor.shape[0], convention


@dataclass
class Target:
    tensor: str
    start: int
    stop: int
    a: Tensor
    b: Tensor
    rank: int
    alpha: float
    alpha_source: str
    user_scale: float
    adapter: int
    convention: str

    @property
    def scale(self):
        return self.user_scale * self.alpha / self.rank

    def record(self):
        return dict(target=self.tensor, rows=[self.start, self.stop], a=self.a.name, b=self.b.name,
                    rank=self.rank, alpha=self.alpha, alpha_source=self.alpha_source,
                    native_scale=self.alpha/self.rank, user_scale=self.user_scale,
                    effective_scale=self.scale, format=self.convention)


def finite_number(value, label):
    if isinstance(value, bool):
        raise FoldError(f'invalid {label}: boolean')
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise FoldError(f'invalid {label}: {value!r}') from exc
    if not math.isfinite(result):
        raise FoldError(f'non-finite {label}')
    return result


def adapter_targets(path, scale, checkpoint, adapter=0, explicit_profile=None):
    shard = Shard(path)
    meta = dict(shard.metadata)
    architectural = r'(?i)(?:^|[._ /-])(?:pdd|vdn|controlnet|dora|lora_magnitude_vector|output_heads|diffusion_head)(?:$|[._ /-])'
    architecture_fields = ('format', 'architecture', 'adapter_type', 'peft_type', 'ss_network_module', 'method', 'algorithm')
    descriptions = [meta[k] for k in architecture_fields if k in meta]
    if any(re.search(architectural, value) for value in [*shard.tensors, *meta, *descriptions]):
        raise FoldError('Unsupported architectural adapter (PDD/VDN/DoRA/ControlNet). '
                        'PDD requires special output heads and inference behavior; backbone LoRA folding alone is incomplete.')
    for field in ('adapter_config', 'ss_network_args'):
        if field in meta:
            nested = read_json(meta[field])
            if not isinstance(nested, dict):
                raise FoldError(f'invalid {field}')
            for key, value in nested.items():
                if key in meta and str(meta[key]) != str(value):
                    raise FoldError(f'inconsistent adapter metadata: {key}')
                meta[key] = value
    if any(meta.get(k) not in (None, False, 'false', 'False', '0') for k in ('use_rslora', 'use_dora', 'fan_in_fan_out')):
        raise FoldError('RS-LoRA/DoRA/transposed base layouts require dedicated scaling/architecture handlers')
    if 'peft_type' in meta and meta['peft_type'] != 'LORA':
        raise FoldError(f"unsupported PEFT architecture: {meta['peft_type']}")
    if any(k in meta for k in ('rank_pattern', 'alpha_pattern')):
        raise FoldError('per-module PEFT metadata patterns are unsupported; export explicit per-target alpha tensors')
    pairs = defaultdict(dict)
    alphas = {}
    extras = []
    for name, tensor in shard.tensors.items():
        match = PAIR.fullmatch(name)
        if match:
            module, part = match.groups()
            part = 'A' if part in ('lora_A', 'lora_down') else 'B'
            if part in pairs[module]:
                raise FoldError(f'duplicate {part} pair: {module}')
            pairs[module][part] = tensor
        elif name.endswith('.alpha'):
            alphas[name[:-6]] = tensor
        else:
            extras.append(name)
    if extras:
        raise FoldError('unsupported adapter tensors (nothing was folded): ' + ', '.join(extras))
    if not pairs:
        raise FoldError('adapter contains no LoRA A/B pairs')
    digest = sha256(path)
    profile = explicit_profile or next((k for k, v in PROFILES.items() if v['sha256'] == digest), None)
    targets = []
    occupied = defaultdict(list)
    for module, pair in sorted(pairs.items()):
        if set(pair) != {'A', 'B'}:
            raise FoldError(f'missing A/B pair: {module}')
        a, b = pair['A'], pair['B']
        if a.dtype not in FLOATS or b.dtype not in FLOATS or len(a.shape) != 2 or len(b.shape) != 2:
            raise FoldError(f'{module}: A/B must be BF16/F16/F32 matrices')
        rank = a.shape[0]
        name, lo, hi, convention = normalize_target(module, checkpoint)
        base = checkpoint.tensors[name]
        if base.dtype != 'BF16':
            raise FoldError(f'{name}: base target must be BF16')
        if rank < 1 or b.shape[1] != rank or a.shape[1] != base.shape[1] or b.shape[0] != hi-lo:
            raise FoldError(f'{module}: rank/orientation/shape mismatch A{a.shape}, B{b.shape}, target{base.shape}, rows {lo}:{hi}')
        if any(lo < end and hi > start for start, end in occupied[name]):
            raise FoldError(f'duplicate/overlapping adapter target: {name}')
        occupied[name].append((lo, hi))
        for field in ('rank', 'r', 'lora_rank', 'ss_network_dim', 'network_dim'):
            if field in meta and finite_number(meta[field], field) != rank:
                raise FoldError(f'{module}: rank metadata {field} disagrees with tensors')
        values = [(field, finite_number(meta[field], field)) for field in ('alpha', 'lora_alpha', 'ss_network_alpha', 'network_alpha') if field in meta]
        if module in alphas:
            tensor = alphas.pop(module)
            values.append(('tensor', finite_number(tensor.scalar(), 'alpha')))
        if values and any(value != values[0][1] for _, value in values):
            raise FoldError(f'{module}: inconsistent alpha metadata')
        alpha = values[0][1] if values else float(rank)
        if alpha < 0:
            raise FoldError(f'{module}: alpha must be nonnegative')
        target = Target(name, lo, hi, a, b, rank, alpha,
                        '+'.join(k for k, _ in values) if values else 'absent: explicit fallback alpha=rank (plain BA)',
                        scale, adapter, convention)
        if not math.isfinite(target.scale) or abs(target.scale) > np.finfo(np.float32).max:
            raise FoldError(f'{module}: effective scale overflows FP32')
        targets.append(target)
    if alphas:
        raise FoldError('orphan alpha tensors: ' + ', '.join(alphas))
    record = dict(path=str(path), sha256=digest, user_scale=scale, metadata=shard.metadata,
                  profile=profile, profile_source='explicit' if explicit_profile else ('sha256' if profile else None),
                  profile_recommendations=PROFILES.get(profile),
                  target_families=dict(Counter(re.sub(r'\.\d+\.', '.N.', t.tensor) for t in targets)),
                  tensors=[dict(name=t.name, shape=t.shape, dtype=t.dtype) for t in shard.tensors.values()],
                  targets=[t.record() for t in targets])
    return targets, record, shard.stamp


def round_bf16(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise FoldError('non-finite FP32 fold result')
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype('<u2')
    if np.any((rounded & 0x7fff) == 0x7f80):
        raise FoldError('BF16 overflow in folded weights')
    return rounded


class ProbeMetrics:
    """Stream statistics so even extremely tall matrices cannot retain probes."""
    def __init__(self):
        self.count = 0
        self.maximum = self.absolute = self.squared = 0.0
        self.actual_squared = self.expected_squared = self.dot = 0.0

    def add(self, actual, expected):
        a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
        diff = a-b
        self.count += a.size
        self.maximum = max(self.maximum, float(np.max(np.abs(diff), initial=0)))
        self.absolute += float(np.sum(np.abs(diff)))
        self.squared += float(np.dot(diff, diff))
        self.actual_squared += float(np.dot(a, a))
        self.expected_squared += float(np.dot(b, b))
        self.dot += float(np.dot(a, b))

    def finish(self):
        norm = math.sqrt(self.actual_squared * self.expected_squared)
        return dict(max_absolute_error=self.maximum,
                    mean_absolute_error=self.absolute/self.count,
                    relative_l2=math.sqrt(self.squared)/max(math.sqrt(self.expected_squared), 1e-30),
                    cosine_similarity=self.dot/norm if norm else (1.0 if self.squared == 0 else 0.0))


def metrics(actual, expected):
    result = ProbeMetrics()
    result.add(actual, expected)
    return result.finish()


def validate_probe(actual, expected, bound, label):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FoldError(f'{label}: non-finite numerical probe')
    # Componentwise error bound handles near-zero/cancelling outputs where
    # relative error and cosine alone become meaningless. BF16 adds at most
    # half an ulp per stored element; reduction roundoff gets a small allowance.
    if np.any(np.abs(actual.astype(np.float64)-expected) > bound):
        raise FoldError(f'{label}: numerical parity failed: {metrics(actual, expected)}')


def fold_tensor(base, targets, destination, memory_mib):
    columns = base.shape[1]
    budget = int(memory_mib * (1 << 20))
    a_bytes = sum(math.prod(t.a.shape)*4 for t in targets)
    per_row = 32*columns + 8*sum(t.rank for t in targets) + 512
    rows = min(base.shape[0], 2048, (budget - 3*a_bytes - columns*64) // per_row)
    if rows < 1:
        raise FoldError(f'{base.name}: --memory-mib too small for adapter A matrices and one row')
    rng = np.random.default_rng(0)
    x = rng.standard_normal((columns, 3), dtype=np.float32) / np.sqrt(np.float32(columns))
    ax = [(t, t.a.read()) for t in targets]
    updates = [(t, a, a @ x, np.abs(a) @ np.abs(x)) for t, a in ax]
    pre_metrics, post_metrics = ProbeMetrics(), ProbeMetrics()
    output_hash = hashlib.sha256()
    with destination.open('r+b', buffering=0) as out:
        for start in range(0, base.shape[0], rows):
            stop = min(start + rows, base.shape[0])
            w = base.read(start, stop)
            expected = w @ x
            # Conservative FP32 BLAS reassociation allowance, distinct from BF16.
            magnitude = np.abs(w) @ np.abs(x)
            for t, a, projected, absolute_projected in updates:
                lo, hi = max(start, t.start), min(stop, t.stop)
                if lo >= hi:
                    continue
                b = t.b.read(lo-t.start, hi-t.start)
                if t.scale == 0:
                    continue
                section = slice(lo-start, hi-start)
                expected[section] += np.float32(t.scale) * (b @ projected)
                magnitude[section] += abs(t.scale) * (np.abs(b) @ absolute_projected)
                delta = b @ a
                delta *= np.float32(t.scale)
                w[section] += delta
            pre = w @ x
            pre_bound = 2e-5 * magnitude + 2e-7
            validate_probe(pre, expected, pre_bound, base.name + ' FP32')
            rounded = round_bf16(w)
            # BF16 round-to-nearest bound: 2^-8 relative per element (normal
            # values), plus subnormal floor; deliberately independent of delta.
            bound = pre_bound + (np.abs(w) @ np.abs(x)) * (1/256) + 1e-7
            out.seek(base.offset + start*columns*2)
            raw = rounded.tobytes()
            if out.write(raw) != len(raw):
                raise FoldError('short checkpoint write')
            output_hash.update(raw)
            # Validate the actual on-disk rounded matrix, not an FP32 surrogate.
            written = Tensor(destination, base.name, 'BF16', (stop-start, columns),
                             base.offset+start*columns*2, len(raw)).read()
            post = written @ x
            validate_probe(post, expected, bound, base.name + ' BF16')
            pre_metrics.add(pre, expected)
            post_metrics.add(post, expected)
        os.fsync(out.fileno())
    return dict(fp32=pre_metrics.finish(), bf16=post_metrics.finish(),
                sha256=output_hash.hexdigest(), block_rows=rows)


def clone_file(source, destination, allow_full_copy):
    if sys.platform == 'darwin':
        libc = ctypes.CDLL(None, use_errno=True)
        clone = libc.clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source.resolve()), os.fsencode(destination), 0) == 0:
            return 'clonefile'
        reason = os.strerror(ctypes.get_errno())
    elif sys.platform == 'linux':
        import fcntl
        # Linux FICLONE creates independent inodes backed by shared extents.
        # Open exclusively so an existing destination is never truncated.
        with source.open('rb') as src, destination.open('xb') as dst:
            try:
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())
            except OSError as exc:
                reason = str(exc)
            else:
                return 'reflink'
        # A rejected clone leaves our destination behind. Remove it before
        # reporting failure or taking the explicitly permitted copy fallback.
        destination.unlink()
    else:
        reason = 'no copy-on-write implementation for this platform'
    if not allow_full_copy:
        raise FoldError(f'copy-on-write unavailable: {reason}; use --allow-full-copy for a deliberate full copy')
    print(f'WARNING: FULL COPY {source.name}: {source.stat().st_size / (1<<30):.2f} GiB logical disk space', flush=True)
    if shutil.disk_usage(destination.parent).free < source.stat().st_size + (16 << 20):
        raise FoldError('insufficient disk space for full copy')
    shutil.copyfile(source, destination)
    return 'full-copy'


def verify_files(checkpoint, stage, modified):
    """Hash all source/output bytes; compare every byte outside changed tensors."""
    identities = []
    for source in checkpoint.files:
        if source.name == MANIFEST:
            continue  # Parent manifest is retained inside the new manifest.
        destination = stage / source.name
        if destination.stat().st_size != source.stat().st_size:
            raise FoldError(f'{source.name}: file size changed')
        if source.name in checkpoint.shards:
            original = checkpoint.shards[source.name]
            reopened = Shard(destination)
            if reopened.header != original.header or reopened.size != original.size:
                raise FoldError(f'{source.name}: header/shape/dtype/offset/tensor count changed')
        ranges = sorted((t.offset, t.offset+t.size, t.name) for t in checkpoint.tensors.values()
                        if t.path == source and t.name in modified)
        source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
        with source.open('rb') as a, destination.open('rb') as b:
            cursor = 0
            for lo, hi, name in ranges + [(source.stat().st_size, source.stat().st_size, None)]:
                for end, compare in ((lo, True), (hi, False)):
                    target_hash = hashlib.sha256()
                    while cursor < end:
                        n = min(8 << 20, end-cursor)
                        before, after = a.read(n), b.read(n)
                        if len(before) != n or len(after) != n or (compare and before != after):
                            raise FoldError(f'{source.name}: unmodified bytes changed or short read at {cursor}')
                        source_hash.update(before)
                        output_hash.update(after)
                        target_hash.update(after)
                        cursor += n
                    if not compare and name and target_hash.hexdigest() != modified[name]['sha256']:
                        raise FoldError(f'{name}: written tensor digest mismatch')
        identities.append(dict(file=source.name, bytes=source.stat().st_size,
                               sha256=source_hash.hexdigest(), folded_sha256=output_hash.hexdigest()))
    checkpoint.unchanged()
    return identities


def output_path(checkpoint, out, adapters, overwrite):
    raw = Path(out).absolute()
    if raw.is_symlink():
        raise FoldError('output must not be a symlink')
    path = raw.resolve()
    if path == checkpoint.path or path in checkpoint.path.parents or checkpoint.path in path.parents:
        raise FoldError('output aliases, contains, or is inside source checkpoint')
    if any(path == p.resolve() or path in p.resolve().parents for p in checkpoint.files):
        raise FoldError('output contains a resolved source checkpoint file')
    if any(path == p or path in p.parents for p, _ in adapters):
        raise FoldError('output contains an input adapter')
    if path.exists() and (not path.is_dir() or not overwrite):
        raise FoldError('output already exists; use --overwrite for an existing directory')
    return path


def fold(checkpoint_path, adapters, out=None, *, dry_run=False, overwrite=False,
         allow_full_copy=False, memory_mib=512, profiles=None, verbose=False):
    if not math.isfinite(memory_mib) or memory_mib <= 0:
        raise FoldError('--memory-mib must be positive and finite')
    checkpoint = Checkpoint(checkpoint_path)
    if not adapters:
        raise FoldError('at least one --lora is required')
    if profiles and len(profiles) != len(adapters):
        raise FoldError('when --profile is used, supply one profile per --lora (auto for unknown)')
    destination = output_path(checkpoint, out, adapters, overwrite) if out else None
    if destination is None and not dry_run:
        raise FoldError('--out is required unless --dry-run/--inspect is used')
    records, all_targets, adapter_stamps = [], [], []
    for i, (path, scale) in enumerate(adapters):
        profile = profiles[i] if profiles and profiles[i] != 'auto' else None
        if profile and profile not in PROFILES:
            raise FoldError(f'unknown profile: {profile}')
        targets, record, identity = adapter_targets(path, scale, checkpoint, i, profile)
        records.append(record)
        all_targets.extend(targets)
        adapter_stamps.append(identity)
    grouped = defaultdict(list)
    for target in all_targets:
        grouped[target.tensor].append(target)
    shard_names = sorted({checkpoint.tensors[n].path.name for n in grouped})
    print(f'Adapters: {len(records)}\nLoRA pairs: {len(all_targets)}\nTargets matched: {len(grouped)}\n'
          f'Targets unsupported: 0\nShards modified: {len(shard_names)}', flush=True)
    for record in records:
        scales = sorted({t['effective_scale'] for t in record['targets']})
        print(f"{Path(record['path']).name}: formats={sorted({t['format'] for t in record['targets']})} "
              f"ranks={dict(Counter(t['rank'] for t in record['targets']))} effective_scales={scales} "
              f"alpha={sorted({t['alpha'] for t in record['targets']})} profile={record['profile'] or 'unknown'}", flush=True)
        if any(t['alpha_source'].startswith('absent:') for t in record['targets']):
            print('  No alpha metadata for some/all targets: using alpha=rank, native scale=1 (plain BA).', flush=True)
        if record['profile']:
            print('  ' + PROFILES[record['profile']]['notes'], flush=True)
        if verbose:
            print(json.dumps(record, indent=2))
    logical = sum(p.stat().st_size for p in checkpoint.files)
    unaligned = sum(len(s.unaligned) for s in checkpoint.shards.values())
    print(f'Logical checkpoint: {logical/(1<<30):.2f} GiB; prefer CoW (APFS clones/Linux reflinks); full copy opt-in={allow_full_copy}. '
          f'Preserving {unaligned} payloads not aligned to 16 bytes.', flush=True)
    plan = dict(schema_version=1, tool_version=VERSION, created_at=datetime.now(timezone.utc).isoformat(),
                base_path=str(checkpoint.path), base_mode=checkpoint.path.parent.name,
                adapters=records, modified_tensors=sorted(grouped), modified_shards=shard_names,
                logical_bytes=logical, memory_limit_mib=memory_mib)
    if dry_run:
        print('Dry run: validation and mapping PASS; no files written. Numerical probes run during folding.', flush=True)
        return plan
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.tmp-', dir=destination.parent))
    backup = None
    try:
        copies = {}
        for source in checkpoint.files:
            if source.name != MANIFEST:
                copies[source.name] = clone_file(source, stage/source.name, allow_full_copy)
        results = {}
        # CLI adapter order, then lexical module order: deterministic addition.
        for n, name in enumerate(sorted(grouped), 1):
            print(f'Folding {n}/{len(grouped)} {name}', flush=True)
            results[name] = fold_tensor(checkpoint.tensors[name], grouped[name],
                                        stage/checkpoint.tensors[name].path.name, memory_mib)
            print('  BF16 ' + json.dumps(results[name]['bf16']), flush=True)
        identities = verify_files(checkpoint, stage, results)
        for (path, _), identity, record in zip(adapters, adapter_stamps, records):
            if stamp(path) != identity or sha256(path) != record['sha256']:
                raise FoldError(f'adapter changed during folding: {path}')
        plan.update(base_files=identities, base_sha256=hashlib.sha256(json.dumps(
            [{k:v for k,v in x.items() if k != 'folded_sha256'} for x in identities], sort_keys=True).encode()).hexdigest(),
                    copy_methods=copies, validation=results)
        parent_manifest = checkpoint.path/MANIFEST
        if parent_manifest.exists():
            plan['parent_manifest'] = read_json(parent_manifest.read_bytes())
        with (stage/MANIFEST).open('w') as f:
            json.dump(plan, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        checkpoint.unchanged()
        if destination.exists():
            if not overwrite:
                raise FoldError('output appeared during folding; refusing to overwrite')
            backup = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.backup-', dir=destination.parent))
            backup.rmdir()
            os.rename(destination, backup)
        try:
            os.rename(stage, destination)
        except BaseException:
            if backup is not None:
                os.rename(backup, destination)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
        print(f'BF16 validation: PASS\nManifest written: {destination/MANIFEST}', flush=True)
        return plan
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='H3 transformer directory, not the complete model tree')
    parser.add_argument('--lora', action='append', required=True, metavar='FILE[:SCALE]')
    parser.add_argument('--out')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--inspect', action='store_true', help='verbose dry-run including every adapter tensor')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--allow-full-copy', action='store_true')
    parser.add_argument('--memory-mib', type=float, default=512, help='bounded matrix working-set budget (default 512 MiB)')
    parser.add_argument('--profile', action='append', choices=['auto', *PROFILES])
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args(argv)
    try:
        fold(args.checkpoint, [parse_lora(v) for v in args.lora], args.out,
             dry_run=args.dry_run or args.inspect, overwrite=args.overwrite,
             allow_full_copy=args.allow_full_copy, memory_mib=args.memory_mib,
             profiles=args.profile, verbose=args.verbose or args.inspect)
    except (FoldError, OSError, MemoryError) as exc:
        print(f'fold_lora: error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
