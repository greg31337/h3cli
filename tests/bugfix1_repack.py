#!/usr/bin/env python3
"""Create an explicitly requested development copy with odd tensor offsets.

Only destination files are written. Non-weight assets are symlinked to source.
Adds one JSON padding space; tensor descriptors and payload bytes are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if destination == source or source in destination.parents:
        parser.error('destination must be outside source')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'repack-manifest.json'
    records = json.loads(manifest.read_text()) if manifest.exists() else {}
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
            continue
        if path.suffix != '.safetensors':
            if not target.exists():
                target.symlink_to(path)
            continue
        if str(relative) in records and target.exists() and target.stat().st_size == records[str(relative)]['output_bytes']:
            continue
        start = time.monotonic()
        tmp = target.with_suffix('.partial')
        digest = hashlib.sha256()
        with path.open('rb') as src, tmp.open('wb') as out:
            size = struct.unpack('<Q', src.read(8))[0]
            header = src.read(size)
            assert len(header) == size
            json.loads(header)
            # Make data_start odd even if input is already odd.
            padding = 1 if (8 + size) % 2 == 0 else 2
            out.write(struct.pack('<Q', size + padding))
            out.write(header + b' ' * padding)
            for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
                digest.update(block)
                out.write(block)
        tmp.replace(target)
        records[str(relative)] = dict(source=str(path), input_bytes=path.stat().st_size,
                                     output_bytes=target.stat().st_size, payload_sha256=digest.hexdigest(),
                                     header_padding_added=padding, seconds=time.monotonic()-start)
        manifest.write_text(json.dumps(records, indent=2)+'\n')
        print('repacked', relative, flush=True)


if __name__ == '__main__':
    main()
