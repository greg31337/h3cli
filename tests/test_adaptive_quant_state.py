#!/usr/bin/env python3
"""Validate the coupled recipes in a real mixed-precision cache checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from test_sampler_file import entries, build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('state', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    original = args.state.read_bytes()
    parts = entries(original)
    assert {34, 40, 41, 42, 44} <= {p[0] for p in parts}
    assert struct.unpack_from('<2I', next(p[-1] for p in parts if p[0] == 34))[1] == 3
    assert struct.unpack_from('<2I', next(p[-1] for p in parts if p[0] == 40)) == (1, 2)
    records = []
    with tempfile.TemporaryDirectory(prefix='h3cli-adaptive-quant-state-') as temp:
        path = Path(temp) / 'state.h3sample'

        def check(name, data, accepted=False):
            path.write_bytes(data)
            result = subprocess.run([str(binary), '--load', str(path)], capture_output=True, text=True)
            records.append(dict(name=name, accepted=result.returncode == 0,
                                expected=accepted, error=result.stderr.strip()))
            assert (result.returncode == 0) == accepted, (name, result.stderr)

        check('original', original, True)
        for kind in [34, 40, 41, 42, 44]:
            check(f'missing-{kind}', build([p for p in parts if p[0] != kind]))
            for field, value in [(1, 99), (2, 0)]:
                changed = copy.deepcopy(parts)
                next(p for p in changed if p[0] == kind)[field] = value
                check(f'section-{kind}-field-{field}', build(changed))
        for kind, offset, fmt, value in [
            (34, 0, 'I', 0), (34, 0, 'I', 99), (34, 4, 'I', 2), (34, 4, 'I', 99),
            (40, 0, 'I', 0), (40, 0, 'I', 2), (40, 4, 'I', 1), (40, 4, 'I', 99),
            (40, 12, 'I', 2), (40, 16, 'I', 1), (40, 20, 'i', -1),
            (40, 28, 'Q', 2**63), (41, 0, 'H', 0x7fc0), (42, 0, 'H', 0x7f80),
        ]:
            changed = copy.deepcopy(parts)
            struct.pack_into('<' + fmt, next(p[-1] for p in changed if p[0] == kind), offset, value)
            check(f'payload-{kind}-{offset}-{value}', build(changed))
        changed = bytearray(original)
        changed[-1] ^= 1
        check('checksum', changed)
    args.out.write_text(json.dumps(dict(passed=True, cases=records), indent=2) + '\n')
    print(f'PASS {len(records)} mixed-precision checkpoint integrity cases')


if __name__ == '__main__':
    main()
