"""Produce a pinned OCI builder containing the compiled Linux dependencies."""
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_process
import dependency_bundle as bundles


def produce(source, cache, work, payload, jobs, offline=False):
    import build
    import environment
    inputs = build.fetch(cache, offline)
    work.mkdir(parents=True);(work/'build').mkdir();(work/'evidence').mkdir()
    payload.mkdir(parents=True)
    # Shared build volumes may inherit setgid from their parent. The portable
    # archive deliberately rejects special permission bits, including on dirs.
    payload.chmod(0o755)
    image = payload/'images/builder'
    shutil.copytree(cache/'images/builder', image)
    image.joinpath('origin.json').unlink(missing_ok=True)
    build_process.run(['umoci', 'unpack', '--rootless', '--image', str(image)+':base', work/'builder'], check=True)
    root = work/'builder/rootfs'
    build.run_builder(cache, root, source, inputs, work, jobs, ['bash', '-c',
        'export DEBIAN_FRONTEND=noninteractive; cp /inputs/debs/*.deb /var/cache/apt/archives/ && apt-get --no-download --no-install-recommends -y install /inputs/debs/*.deb'])
    build.run_builder(cache, root, source, inputs, work, jobs, ['bash', 'scripts/linux/build_dependencies.sh'])
    # All required headers, notices and generated configurations live under the
    # image prefix. Application builds need no producer's /build bind mount.
    build.run_builder(cache, root, source, inputs, work, jobs, ['bash', '-c',
        'rm -f /var/cache/apt/archives/*.deb; source /opt/h3deps/environment.sh; '
        'python3 scripts/verify_sglang_dependencies.py "$SGLANG_CUTLASS_PATH"; '
        'python3 scripts/verify_sage_dependencies.py "$SAGE_CUTLASS_PATH"; '
        'ffmpeg -version; /opt/h3deps/reference-media/bin/ffmpeg -version'])
    build_process.run(['umoci', 'repack', '--no-history', '--image', str(image)+':base', work/'builder'], check=True)
    build_process.run(['umoci', 'gc', '--layout', image], check=True)
    target = payload/'inputs';target.mkdir()
    # Source materials accompany binary reuse. Wheels/debs are already installed
    # in the image; retaining their original downloads here would duplicate GBs.
    lock = environment.LOCK
    for name in lock['sources'] | lock.get('materials', {}):
        shutil.copyfile(inputs/name, target/name)
    shutil.copytree(inputs/'source-packages', target/'source-packages')
    (payload/'tools').mkdir();shutil.copyfile(cache/'tools/proot', payload/'tools/proot')
    (payload/'tools/proot').chmod(0o755)
    for blob in (image/'blobs/sha256').iterdir():
        if bundles.sha(blob) != blob.name:raise ValueError('Changed dependency OCI blob')
    return {'platform': 'linux-x86_64', 'builder': lock['images']['builder'],
            'oci_index_sha256': bundles.sha(image/'index.json'),
            'glibc_max': lock['glibc_max'], 'driver_min': lock['driver_min']}
