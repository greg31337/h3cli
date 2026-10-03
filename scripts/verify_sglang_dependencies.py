#!/usr/bin/env python3
"""Verify the small source-only CUTLASS dependency; never inspect model weights."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cutlass',type=Path);a=p.parse_args()
    manifest=Path(__file__).resolve().parents[1]/'third_party/flash-attention/cutlass-manifest.json'
    record=json.loads(manifest.read_text());root=a.cutlass.resolve(strict=True)
    for name,digest in record['files'].items():
        path=(root/name).resolve(strict=True)
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise SystemExit('Changed reference CUTLASS dependency: '+name)
    print('Verified reference CUTLASS '+record['revision']+' ('+str(len(record['files']))+' source files)')


if __name__=='__main__':main()
