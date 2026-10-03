#!/usr/bin/env python3
"""Verify offline, pinned native kernel dependencies without downloading files."""
import hashlib
import json
from pathlib import Path
import sys

base=Path(__file__).resolve().parent.parent/'third_party/sageattention'
cutlass=Path(sys.argv[1])
for manifest,root,key in [('upstream.json',base,'native_sha256'),('cutlass-headers.json',cutlass,None)]:
    data=json.loads((base/manifest).read_text())
    for name,item in data['files'].items():
        expected=item[key] if key else item
        p=root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise SystemExit(f'Sage dependency differs from pinned source: {p}')
print('Verified pinned native SageAttention and CUTLASS sources')
