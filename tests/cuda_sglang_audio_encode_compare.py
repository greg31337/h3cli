#!/usr/bin/env python3
"""Compare bounded soundtrack-encoder captures; no weights or GPU are accessed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cuda_sglang_compare import metric, oracle


BOUNDARIES = {
    'enc-initial': 'encoder.block.0.output',
    **{f'enc-stage{i}': f'encoder.block.{i}.output' for i in range(1, 6)},
    'enc-final-conv': 'encoder.block.7.output',
    'enc-proj-norm': 'pre_block.norm3.output',
    'enc-proj': 'pre_block.proj.output',
    'enc-attn-norm': 'pre_block.norm1.output',
    'enc-query': 'pre_block.attn.attn.q',
    'enc-key': 'pre_block.attn.attn.k',
    'enc-value': 'pre_block.attn.attn.v',
    'enc-attended': 'pre_block.attn.attn.output',
    'enc-pooled': 'pre_block.attn.proj.input',
    'enc-attn-proj': 'pre_block.attn.proj.output',
    'enc-base-attn': 'pre_block.norm2.input',
    'enc-mlp-norm2': 'pre_block.norm2.output',
    'enc-mlp': 'pre_block.mlp.output',
    'enc-base-mlp': 'pre_block.output',
    'latent': 'rows',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('oracle', 'native', 'output'):
        p.add_argument(name, type=Path)
    p.add_argument('--previous-oracle', type=Path,
                   help='Also require the extended hooks to preserve old final rows')
    a = p.parse_args()
    contract_path = Path(__file__).with_name('cuda_sglang_contract.json')
    gate = json.loads(contract_path.read_text())['gates']['operation']
    records = []
    for name, reference in BOUNDARIES.items():
        row = dict(name=name, oracle_boundary=reference, passed=False)
        try:
            path = a.native / (name + '.f32')
            meta = json.loads((a.oracle / (reference + '.json')).read_text())
            if (meta['dtype'] != 'torch.float32' or meta['bytes'] > 64 * 1024**2
                    or path.stat().st_size != meta['bytes'] or meta['bytes'] <= 0):
                raise ValueError('invalid capture dtype, size or bounded geometry')
            raw = path.read_bytes()
            x = np.frombuffer(raw, dtype='<f4')
            y = oracle(a.oracle, reference)  # checks oracle size and checksum
            if reference.startswith('encoder.'):
                y = y.transpose(0, 2, 1)
            if reference == 'rows':
                if y.ndim != 2 or y.shape[0] % 2 or y.shape[1] != 32:
                    raise ValueError('invalid stereo audio rows')
                x = x.reshape(32, y.shape[0]).T
            else:
                x = x.reshape(y.shape)
            row.update(metric(x, y), native_sha256=hashlib.sha256(raw).hexdigest(),
                       oracle_sha256=meta['sha256'])
            row['absolute_limit'] = gate['absolute_floor'] + gate['absolute_max_per_reference_rms'] * row.get('reference_rms', 0)
            row['passed'] = bool(row.get('finite') and row.get('compatible')
                and row['relative_l2'] <= gate['relative_l2_max']
                and row['cosine'] >= gate['cosine_min']
                and row['max_abs'] <= row['absolute_limit'])
        except (OSError, ValueError, KeyError) as e:
            row['error'] = str(e)
        records.append(row)
    if a.previous_oracle:
        row = dict(name='oracle_hooks_unchanged', passed=False)
        try:
            row.update(metric(oracle(a.oracle, 'rows'), oracle(a.previous_oracle, 'rows')))
            row['passed'] = bool(row.get('exact') and row.get('finite'))
        except (OSError, ValueError, KeyError) as e:
            row['error'] = str(e)
        records.append(row)
    result = dict(comparison='reference soundtrack encoder operations',
        oracle=str(a.oracle), native=str(a.native), results=records,
        passed=all(r['passed'] for r in records),
        bitwise_equal=all(r.get('exact', False) for r in records),
        first_failing_boundary=next((r['name'] for r in records if not r['passed']), None),
        contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'bitwise_equal', 'first_failing_boundary')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
