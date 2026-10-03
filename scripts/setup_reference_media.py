#!/usr/bin/env python3
"""Install historical codecs for SGLang regression tests; never changes defaults."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import reference_media
import setup_ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, default=ROOT / 'outputs/setup/reference-media')
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('H3_BUILD_JOBS', '8')))
    args = parser.parse_args()
    if sys.platform != 'linux' or os.uname().machine != 'x86_64':
        parser.error('The recorded CUDA codecs require Linux x86-64')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    prefix = args.prefix.resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    decoder = setup_ffmpeg.build(prefix / 'decoder-build', '6.1.1', args.jobs)
    venv = prefix / 'venv'
    if not venv.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    python = venv / 'bin/python'
    subprocess.run([str(python), '-m', 'pip', 'install', '--only-binary=:all:', '--no-deps',
                    '--require-hashes', '-r', str(ROOT / 'scripts/requirements-reference-media.txt')], check=True)
    site = Path(subprocess.check_output([str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    for source, relative in ((site / 'imageio_ffmpeg/binaries/ffmpeg-linux64-v4.2.2', 'output/ffmpeg'),
                             (decoder / 'bin/ffmpeg', 'input/ffmpeg'), (decoder / 'bin/ffprobe', 'input/ffprobe')):
        dest = prefix / relative
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, dest)
        dest.chmod(0o755)
    print('Regression media profile: ' + str(reference_media.write_profile(prefix)))


if __name__ == '__main__':
    main()
