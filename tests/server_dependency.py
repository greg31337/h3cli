#!/usr/bin/env python3
"""Offline verification of the pinned HTTP dependency, including local changes."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'third_party/civetweb'
manifest=json.loads((root/'manifest.json').read_text())
assert manifest['version']=='v1.16'
for name,digest in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert 'MIT' in (root/'LICENSE.md').read_text()
print(f"CivetWeb {manifest['version']}: {len(manifest['files'])} pinned files verified")
