#!/usr/bin/env python3
"""Build pinned FFmpeg/FFprobe locally, without replacing system executables."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONFIGURE = ['--disable-debug', '--disable-doc', '--disable-autodetect', '--enable-gpl',
             '--enable-libx264', '--enable-zlib', '--enable-static', '--disable-shared']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build(prefix, version='9.0.2', jobs=8):
    prefix = Path(prefix).resolve()
    lock = json.loads((ROOT / 'scripts/linux/lock.json').read_text())
    name = f'ffmpeg-{version}.tar.xz'
    pin = lock['sources'][name]
    expected = dict(version=version, source_sha256=pin['sha256'], configure=CONFIGURE)
    stamp = prefix / 'build.json'
    if stamp.is_file():
        record = json.loads(stamp.read_text())
        if all(record.get(k) == v for k, v in expected.items()) and all(
                (prefix / 'bin' / tool).is_file() and sha(prefix / 'bin' / tool) == record.get(tool)
                for tool in ('ffmpeg', 'ffprobe')):
            print(f'Keeping verified FFmpeg {version}: {prefix}', flush=True)
            return prefix
    if prefix.exists():
        raise ValueError(f'Incomplete or changed FFmpeg installation; move it aside before retrying: {prefix}')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cache = prefix.parent / 'downloads'
    cache.mkdir(exist_ok=True)
    archive = cache / name
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        try:
            with urllib.request.urlopen(pin['url'], timeout=120) as source, temporary.open('wb') as dest:
                shutil.copyfileobj(source, dest)
            if sha(temporary) != pin['sha256']:
                raise ValueError('FFmpeg source checksum mismatch')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if sha(archive) != pin['sha256']:
        raise ValueError('FFmpeg source checksum mismatch')
    with tempfile.TemporaryDirectory(prefix='ffmpeg-build-', dir=prefix.parent) as tmp:
        with tarfile.open(archive) as tar:
            # Only the hash-verified upstream archive reaches extraction.
            tar.extractall(tmp)
        source = Path(tmp) / f'ffmpeg-{version}'
        command = ['./configure', '--prefix=' + str(prefix), *CONFIGURE]
        subprocess.run(command, cwd=source, check=True)
        subprocess.run(['make', '-j' + str(jobs)], cwd=source, check=True)
        subprocess.run(['make', 'install'], cwd=source, check=True)
        for tool in ('ffmpeg', 'ffprobe'):
            first = subprocess.check_output([str(prefix / 'bin' / tool), '-version'], text=True).splitlines()[0]
            if first.split()[2] != version:
                raise ValueError('Incorrect installed version: ' + first)
            expected[tool] = sha(prefix / 'bin' / tool)
        stamp.write_text(json.dumps(expected, sort_keys=True, indent=2) + '\n')
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=ROOT / 'outputs/setup/ffmpeg')
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('H3_BUILD_JOBS', '8')))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    build(args.prefix, jobs=args.jobs)


if __name__ == '__main__':
    main()
