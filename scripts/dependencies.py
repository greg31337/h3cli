#!/usr/bin/env python3
"""Build, qualify and inspect reusable dependencies. Never publish a release."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import build_process
import dependency_bundle as bundles

ROOT = Path(__file__).resolve().parents[1]


def host(target):
    expected = ('Linux', 'x86_64') if target == 'linux-x86_64' else ('Darwin', 'arm64')
    if (platform.system(), platform.machine()) != expected:
        raise ValueError('This operation requires '+' '.join(expected))


def command(argv, log, **kwargs):
    print('+ '+' '.join(map(str, argv)), flush=True)
    with Path(log).open('a') as file:
        file.write('\n'+repr(list(map(str, argv)))+'\n');file.flush()
        build_process.run(argv, stdout=file, stderr=subprocess.STDOUT, check=True, **kwargs)


def build(args):
    host(args.platform)
    work, output = args.work_dir.resolve(), args.output.resolve()
    if work.exists() or output.exists():
        raise ValueError('Use fresh dependency work and output directories; failed attempts retain evidence')
    if work == output or work in output.parents or output in work.parents:
        raise ValueError('Dependency work/output directories must not overlap')
    recipe = bundles.recipe(ROOT, args.platform)
    output.mkdir(parents=True)
    start = time.monotonic()
    producer = bundles.adapter(ROOT, args.platform)
    payload = output/'payload'
    audit = producer.produce(ROOT, args.cache.expanduser().resolve(), work, payload, args.jobs, args.offline)
    recipes = payload/'recipes'
    shutil.copytree(ROOT/'scripts', recipes/'scripts', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT/'third_party', recipes/'third_party', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(ROOT/'LICENSE', recipes/'LICENSE')
    bundles.write(payload/'source-materials.json', dict(schema=1, recipe=recipe,
        purpose='Dependency source archives, notices and exact recipes; vendor binaries retain their original terms.',
        rebuild='Run recipes/scripts/dependencies.py build on the matching host with a provisioned source/toolchain download cache.'))
    if bundles.recipe(ROOT, args.platform) != recipe:
        raise ValueError('Dependency recipe changed during production')
    archive = output/('dependencies-'+args.platform+'-'+bundles.recipe_id(recipe)+'.tar')
    manifest = bundles.pack(payload, archive, recipe, audit)
    pin = dict(schema=1, platform=args.platform, recipe_id=manifest['recipe_id'], archive=bundles.artifact(archive))
    bundles.write(output/'candidate.json', pin)
    bundles.write(output/'build.json', dict(schema=1, recipe_id=pin['recipe_id'],
        archive_sha256=pin['archive']['sha256'], seconds=time.monotonic()-start, audit=audit))
    print('Built unqualified dependency candidate: '+str(output/'candidate.json'), flush=True)
    print('Run the qualify command before selecting this bundle for releases.', flush=True)


def source_identity():
    sys.path.insert(0, str(ROOT/'tests'))
    import cuda_reference_regression as gate
    digest = gate.fingerprint(gate.source_files(ROOT))
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT))
    except subprocess.CalledProcessError:
        commit = 'source-archive:'+digest;dirty = None
    return dict(source_sha256=digest, reference_commit=commit, dirty=dirty)


def qualify(args):
    # Child commands run in the reference checkout, which may differ from the
    # caller's directory. Resolve model locations before changing directories.
    args.model = args.model.expanduser().resolve()
    args.models_path = args.models_path.expanduser().resolve()
    pin = bundles.reference(args.bundle)
    target = pin['platform'];host(target)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    identity = source_identity();start = time.monotonic()
    recipe = bundles.recipe(ROOT, target)
    candidate = 'qualification' not in pin
    payload, dependency = bundles.load(pin, args.dependency_cache.resolve(), recipe, args.offline, candidate=candidate)
    platform_name = 'linux' if target == 'linux-x86_64' else 'macos'
    release = out/'release'
    command(['bash', ROOT/('scripts/build_'+platform_name+'.sh'),
        '--dependency-candidate' if candidate else '--dependencies', args.bundle.resolve(),
        '--dependency-cache', args.dependency_cache.resolve(), '--offline', '--cache', args.cache.resolve(),
        '--jobs', str(args.jobs), '--work-dir', out/'build', '--output', release], out/'build.log', cwd=ROOT)
    record = bundles.read(release/'build.json')
    if record['dependencies']['archive_sha256'] != pin['archive']['sha256']:
        raise ValueError('Application was built with a different dependency bundle')
    runtime = release/(platform_name+'-runtime')
    binary = release/('h3cli-'+target)
    tests = out/'tests';tests.mkdir()
    env = dict(os.environ, H3CLI_RUNTIME_CACHE=str(out/'runtime-cache'))
    for name in ('LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH','DYLD_LIBRARY_PATH','DYLD_INSERT_LIBRARIES',
                 'H3CLI_RUNTIME_ROOT','H3CLI_BUNDLE_INFO'):
        env.pop(name, None)
    checks = {}
    # Exercise the media bytes actually packaged in this candidate.
    media = tests/'media.mp4'
    command([runtime/'tools/ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=32x32:r=1',
        '-frames:v', '1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', media], tests/'media.log', env=env)
    result = build_process.run([runtime/'tools/ffprobe', '-v', 'error', '-show_streams', '-of', 'json', media],
        capture_output=True, text=True, check=True, env=env)
    stream = json.loads(result.stdout)['streams'][0]
    if (stream['width'], stream['height'], stream['codec_name']) != (32, 32, 'h264'):
        raise ValueError('Dependency media qualification failed')
    checks['media'] = dict(passed=True, ffmpeg_sha256=bundles.sha(runtime/'tools/ffmpeg'),
                          ffprobe_sha256=bundles.sha(runtime/'tools/ffprobe'))
    relocated = out/'relocated directory';relocated.mkdir()
    moved = relocated/'h3cli';shutil.copyfile(binary, moved);moved.chmod(0o755)
    command([moved, '--help'], tests/'relocation.log', cwd=relocated,
        env=env | {'PATH':'/usr/bin:/bin:/usr/sbin:/sbin', 'H3_OFFLINE':'1'})
    checks['relocation'] = dict(passed=True, binary_sha256=bundles.sha(moved))
    checks['artifact'] = dict(passed=True, binary_sha256=bundles.sha(binary),
        build_sha256=bundles.sha(release/'build.json'), dependencies=record['dependencies'])
    if platform_name == 'linux':
        command(['python3', ROOT/'tests/cuda_reference_regression.py', '--source', ROOT, '--runtime', runtime,
            '--validation', release/'linux-validation', '--artifact', binary, '--model', args.model,
            '--out', tests/'parity'], tests/'parity.log', env=env, cwd=ROOT)
        parity = bundles.read(tests/'parity/result.json')
        if parity.get('passed') is not True or len(parity.get('files', {})) != 204:
            raise ValueError('Full dependency reference qualification did not pass')
        checks['parity'] = dict(passed=True, outputs=204, report_sha256=bundles.sha(tests/'parity/result.json'))
        command(['python3', ROOT/'tests/linux_distribution.py', '--source', ROOT, '--release', release,
            '--cache', args.cache.resolve(), '--model', args.model, '--out', tests/'distros'], tests/'distros.log', env=env, cwd=ROOT)
        distros = bundles.read(tests/'distros/result.json')
        rows = distros.get('checks', [])
        if distros.get('passed') is not True or len(rows) != 4 or any(not r['passed'] or r['output_count'] != 204 for r in rows):
            raise ValueError('Incomplete dependency distro qualification')
        checks['distros'] = dict(passed=True, outputs=[204]*4, report_sha256=bundles.sha(tests/'distros/result.json'))
        command(['python3', ROOT/'tests/linux_runtime_smoke.py', '--release', release, '--cache', args.cache.resolve(),
            '--model', args.model, '--out', tests/'runtime'], tests/'runtime.log', env=env, cwd=ROOT)
        if bundles.read(tests/'runtime/results.json').get('passed') is not True:
            raise ValueError('Dependency runtime qualification failed')
        checks['runtime'] = dict(passed=True, report_sha256=bundles.sha(tests/'runtime/results.json'))
    else:
        command(['python3', ROOT/'tests/test_macos_bundle.py'], tests/'host.log', cwd=ROOT, env=env)
    command(['python3', ROOT/'tests/model_downloads_render.py', '--binary', binary,
        '--backend', 'cuda' if platform_name == 'linux' else 'metal', '--models-path', args.models_path,
        '--model', args.model, '--ffmpeg', runtime/'tools/ffmpeg', '--ffprobe', runtime/'tools/ffprobe',
        '--server', '--out', tests/'features'], tests/'features.log', env=env, cwd=ROOT)
    features = bundles.read(tests/'features/results.json')
    if features.get('passed') is not True:raise ValueError('Dependency feature qualification failed')
    checks['features'] = dict(passed=True, report_sha256=bundles.sha(tests/'features/results.json'))
    if source_identity() != identity or bundles.recipe(ROOT, target) != recipe:
        raise ValueError('Reference application or dependency recipe changed during qualification')
    # Recheck shared bytes after compilers, tests and macOS signing have run.
    bundles.load(pin, args.dependency_cache.resolve(), recipe, offline=True, candidate=candidate)
    report = dict(schema=1, passed=True, platform=target, recipe_id=pin['recipe_id'],
        archive_sha256=pin['archive']['sha256'], checks=checks, **identity,
        seconds=time.monotonic()-start)
    bundles.write(out/'qualification.json', report)
    pin['qualification'] = bundles.artifact(out/'qualification.json')
    bundles.write(out/'bundle.json', pin)
    print('Qualified dependency bundle: '+str(out/'bundle.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    identity = commands.add_parser('identity', help='Print the dependency recipe identity without building')
    identity.add_argument('--platform', choices=bundles.PLATFORMS, required=True)
    b = commands.add_parser('build', help='Explicitly build dependencies once; does not qualify or publish them')
    b.add_argument('--platform', choices=bundles.PLATFORMS, required=True)
    b.add_argument('--work-dir', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    q = commands.add_parser('qualify', help='Build/test a reference application and emit a qualified bundle pin')
    q.add_argument('--bundle', type=Path, required=True)
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--model', type=Path, required=True)
    q.add_argument('--models-path', type=Path, required=True)
    q.add_argument('--dependency-cache', type=Path, default=ROOT/'outputs/dependencies/cache')
    q.set_defaults(offline=False)
    b.add_argument('--offline', action='store_true', help='Require locally provisioned dependency inputs')
    for p in (b, q):
        p.add_argument('--cache', type=Path, required=True, help='Pinned source/toolchain download cache')
        p.add_argument('--jobs', type=int, default=8)
    a = parser.parse_args()
    if hasattr(a, 'jobs') and not 1 <= a.jobs <= 32:parser.error('Jobs must be in 1..32')
    if a.command == 'identity':print(bundles.recipe_id(bundles.recipe(ROOT, a.platform)))
    elif a.command == 'build':build(a)
    else:qualify(a)


if __name__ == '__main__':
    try:main()
    except KeyboardInterrupt:sys.exit(130)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit('Dependencies: '+str(exc))
