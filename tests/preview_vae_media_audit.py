#!/usr/bin/env python3
"""Check generated smoke deliveries against their persisted presentation contract."""
import hashlib
import json
import os
from pathlib import Path
import struct
from preview_vae_gallery import inspect

root=Path(os.environ.get('H3_PREVIEW_TEST_ROOT','outputs/preview-vae/metal'));records=[]
for sidecar in sorted(root.rglob('*.h3av.presentation')):
    state=Path(str(sidecar).removesuffix('.presentation'));video=state.with_suffix('.mp4')
    if not video.exists():continue  # Gallery states have separately validated named reconstructions.
    blob=state.read_bytes();fields={line.split()[0]:line.split()[1:] for line in sidecar.read_text().splitlines()}
    assert fields['state'][0]==hashlib.sha256(blob).hexdigest()
    width,height=map(int,fields['output']);trim=int(fields['trim'][0]);frames=struct.unpack_from('<I',blob,32)[0]-trim
    metadata=inspect(video,frames,width,height)
    records.append(dict(path=str(video),sha256=hashlib.sha256(video.read_bytes()).hexdigest(),frames=frames,width=width,height=height,media=metadata))
assert len(records)>=int(os.environ.get('H3_PREVIEW_TEST_MIN_MEDIA','25')),len(records)
(root/'media-audit.json').write_text(json.dumps(records,indent=2)+'\n')
print(f'PASS {len(records)} complete generated MP4s: dimensions, counts, timing, stereo audio, presentation association')
