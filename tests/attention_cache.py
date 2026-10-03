#!/usr/bin/env python3
"""Bound validation disk usage without deleting pre-existing packed artifacts."""
import json
from pathlib import Path

def prepare(cache,source):
    cache=Path(cache).resolve();source=Path(source).resolve()
    if cache==source:raise ValueError('bounded test cache must differ from source cache')
    marker=cache/'attention-test-cache.json'
    expected={'schema':1,'purpose':'h3-attention-validation','source':str(source)}
    cache.mkdir(parents=True,exist_ok=True)
    if marker.exists():
        if json.loads(marker.read_text())!=expected:raise RuntimeError('different test-cache owner')
    elif any(cache.iterdir()):raise RuntimeError('refusing to own a nonempty cache directory')
    else:marker.write_text(json.dumps(expected)+'\n')
    for path in source.glob('*.h3q'):
        target=cache/path.name
        if not target.exists():target.symlink_to(path)

def clear_generated(cache):
    cache=Path(cache);marker=cache/'attention-test-cache.json'
    record=json.loads(marker.read_text())
    if record.get('purpose')!='h3-attention-validation':raise RuntimeError('unowned test cache')
    removed=0
    for path in cache.glob('*.h3q'):
        if path.is_symlink():continue
        removed+=path.stat().st_size;path.unlink()
    return removed
