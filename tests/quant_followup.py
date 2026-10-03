"""Read-only historical metrics; rendering uses cuda_single_run.py."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import struct
import sys
from quant_qualify import inspect, sha

MODES=('off','fp8','nvfp4')

def decisions(directory):
    """Never inherit base-gallery approval or approval of replaced media."""
    manifest_path=directory/'review-manifest.json'
    manifest=json.loads(manifest_path.read_text())
    acceptance_path=directory/'acceptance.json'
    acceptance=json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
    matching=acceptance.get('gallery_manifest_sha256')==sha(manifest_path)
    result={}
    for scope,cases in [('continuation',('continuation-hard','continuation-bridge')),('turbo',('turbo-strength-1',))]:
        rows=[row for row in manifest['rows'] if row['case'] in cases]
        expected={row['case']:{mode:entry['sha256'] for mode,entry in row['outputs'].items()} for row in rows}
        decision=acceptance.get('decisions',{}).get(scope,{})
        valid=(matching and len(rows)==len(cases) and all(set(row['outputs'])==set(MODES) for row in rows)
               and decision.get('status')=='accepted' and decision.get('media')==expected
               and all((directory/entry['path']).is_file() and sha(directory/entry['path'])==entry['sha256']
                       for row in rows for entry in row['outputs'].values()))
        result[scope]=dict(status='accepted' if valid else 'pending-or-stale',media=expected)
    return result
