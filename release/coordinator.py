#!/usr/bin/env python3
"""Build both platforms from one commit and finish with a verified GitHub draft."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys

import driver


# These programs run through SSH with separately shell-quoted arguments. No
# credentials or arbitrary user commands are sent as code.
REMOTE_PREPARE = '''
import json, pathlib, platform, subprocess, sys
base, identity = pathlib.Path(sys.argv[1]), json.loads(sys.argv[2])
if sys.version_info < (3, 10) or (platform.system(), platform.machine()) != ('Linux', 'x86_64'):
    sys.exit('Linux x86-64 and Python 3.10+ are required')
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
if 'RTX PRO 5000' not in gpu:
    sys.exit('Release qualification requires the RTX PRO 5000')
marker = base / 'candidate.json'
if base.exists():
    if not marker.is_file() or json.loads(marker.read_text()) != identity:
        sys.exit('Remote candidate already exists with different settings; choose a new --run')
else:
    base.mkdir(parents=True)
    marker.write_text(json.dumps(identity, sort_keys=True) + '\\n')
'''

REMOTE_CHECKOUT = '''
import pathlib, subprocess, sys
base, commit, repo = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
source = base / 'source'
def git(*args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
if not source.exists():
    subprocess.run(['git', 'clone', '--no-checkout', str(base / 'source.bundle'), str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '--detach', commit], check=True)
    subprocess.run(['git', '-C', str(source), 'remote', 'set-url', 'origin', 'https://github.com/' + repo + '.git'], check=True)
if git('rev-parse', 'HEAD') != commit or git('status', '--porcelain'):
    sys.exit('Remote release checkout changed; preserve it and choose a new --run')
if git('remote', 'get-url', 'origin') != 'https://github.com/' + repo + '.git':
    sys.exit('Remote release repository changed')
'''


def local_path(value):
    path = Path(value).expanduser()
    driver.require(path.is_absolute(), 'Local machine paths must be absolute or start with ~/')
    return str(path.resolve())


def remote_path(value):
    driver.require(isinstance(value, str) and re.fullmatch(r'/[A-Za-z0-9_./-]+', value)
                   and '..' not in PurePosixPath(value).parts,
                   'Linux paths must be absolute, without spaces or shell characters')
    return value.rstrip('/') or '/'


def settings(path):
    config = driver.read(path)
    driver.require(config.get('schema') == 1, 'Unsupported machine settings')
    mac, linux = config['macos'], config['linux']
    driver.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*', linux['host']),
                   'Use an SSH host or user@host, not SSH options')
    for key in ('root', 'models'):
        linux[key] = remote_path(linux[key])
    if linux.get('model'):
        linux['model'] = remote_path(linux['model'])
    linux['identity_file'] = local_path(linux['identity_file']) if linux.get('identity_file') else ''
    mac['models'] = local_path(mac['models'])
    mac['model'] = local_path(mac['model']) if mac.get('model') else str(
        Path(mac['models']) / ('MiniMax-H3' if (Path(mac['models'])/'MiniMax-H3').exists() else 'MiniMaxH3'))
    mac['secret_key'] = local_path(mac.get('secret_key', '~/.config/h3cli-release/h3cli-release.key'))
    mac.setdefault('notary_profile', 'h3cli')
    mac.setdefault('apple_identity', '')
    for machine in (mac, linux):
        machine.setdefault('jobs', 8)
        driver.require(type(machine['jobs']) is int and 1 <= machine['jobs'] <= 32, 'Jobs must be in 1..32')
    return config


class Coordinator:
    def __init__(self, args, root=driver.ROOT):
        self.root = Path(root).resolve()
        self.args = args
        self.settings = settings(args.settings)
        self.mac, self.linux = self.settings['macos'], self.settings['linux']
        self.dry = args.dry_run
        for value in (args.version, args.run or args.version + '-01'):
            driver.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value), 'Use a simple version/run name')
        driver.require(not args.commit.startswith('-'), 'Commit must be a revision, not a Git option')
        # Dry runs only execute local read-only Git queries.
        self.commit = self.git('rev-parse', '--verify', args.commit + '^{commit}')
        self.repo = driver.repo_name(self.git('remote', 'get-url', 'origin'))
        self.run = args.run or args.version + '-01'
        self.base = self.root/'outputs/release/candidates'/self.run
        self.source = self.base/'source'
        self.remote_base = self.linux['root'] + '/' + self.run
        self.remote_source = self.remote_base + '/source'
        self.identity = dict(repo=self.repo, tag=args.version, commit=self.commit, run=self.run)
        self.manifest = dict(schema=1, identity=self.identity, settings=self.settings)
        self.changes = args.notes.read_text().strip() if args.notes else ''
        driver.require('REPLACE_' not in self.changes, 'Finish the supplied release notes')
        config = dict(schema=1, root=str(self.source), **self.identity,
                      jobs=self.mac['jobs'], models=self.mac['models'], model=self.mac['model'],
                      apple_identity=self.mac['apple_identity'], notary_profile=self.mac['notary_profile'],
                      secret_key=self.mac['secret_key'], linux_host=self.linux['host'],
                      linux_root=self.remote_source, linux_identity_file=self.linux['identity_file'], linux_handoff='')
        # Runner normally reads an on-disk config. Dry runs must not create one.
        self.config = config
        self.runner = None

    def git(self, *args, cwd=None):
        return subprocess.check_output(['git', *args], cwd=cwd or self.root, text=True).strip()

    def command(self, argv, **kwargs):
        print('+ ' + shlex.join(list(map(str, argv))), flush=True)
        if not self.dry:
            subprocess.run(list(map(str, argv)), check=True, **kwargs)

    def remote(self, argv):
        self.command(['ssh', *driver.ssh_options(self.linux['identity_file']), self.linux['host'],
                      shlex.join(list(map(str, argv)))])

    def preflight(self):
        if self.dry:
            print('Would require a clean checkout and check Mac tools, signing keys, notarization, GitHub access and the remote GPU.')
            return
        driver.require(not self.git('status', '--porcelain'), 'Commit/review local changes before releasing')
        driver.require((driver.platform.system(), driver.platform.machine()) == ('Darwin', 'arm64'),
                       'Start this command on an Apple Silicon Mac')
        missing = [name for name in ('git', 'ssh', 'scp', 'gh', 'minisign', 'xcrun') if not shutil.which(name)]
        for label, path in (('Minisign secret key', self.mac['secret_key']),
                            ('committed release public key', self.root/'docs/release/h3cli-release.pub'),
                            ('Mac model directory', self.mac['model'])):
            if not Path(path).exists():
                missing.append(label)
        if self.linux['identity_file'] and not Path(self.linux['identity_file']).is_file():
            missing.append('SSH identity file')
        identities = subprocess.check_output(['security', 'find-identity', '-v', '-p', 'codesigning'], text=True)
        available = re.findall(r'"(Developer ID Application: [^"]+)"', identities)
        if not self.mac['apple_identity'] and len(available) == 1:
            self.mac['apple_identity'] = available[0]
            self.config['apple_identity'] = available[0]
        if self.mac['apple_identity'] not in available:
            missing.append('Developer ID Application identity (set macos.apple_identity if several exist)')
        driver.require(not missing, 'Release setup missing: ' + ', '.join(missing))
        driver.require(not Path(self.mac['secret_key']).is_relative_to(self.root), 'Keep the signing secret outside the checkout')
        self.command(['gh', 'auth', 'status', '--hostname', 'github.com'])
        self.command(['xcrun', 'notarytool', 'history', '--keychain-profile', self.mac['notary_profile'],
                      '--output-format', 'json'], stdout=subprocess.DEVNULL)

    def checkout(self):
        marker = self.base/'candidate.json'
        if not self.dry:
            if self.base.exists():
                driver.require(marker.is_file() and driver.read(marker) == self.manifest,
                               'Candidate already exists with different settings; choose a new --run')
            else:
                self.base.mkdir(parents=True)
                driver.write(marker, self.manifest)
        if not self.source.exists():
            self.command(['git', 'clone', '--no-hardlinks', '--no-checkout', self.root, self.source])
            self.command(['git', '-C', self.source, 'checkout', '--detach', self.commit])
            self.command(['git', '-C', self.source, 'remote', 'set-url', 'origin',
                          'https://github.com/' + self.repo + '.git'])
        if self.dry:
            print('Would save immutable per-machine configuration and verify the selected checkout.')
            return
        driver.require(self.git('rev-parse', 'HEAD', cwd=self.source) == self.commit and
                       not self.git('status', '--porcelain', cwd=self.source),
                       'Local release checkout changed; choose a new --run')
        driver.require(driver.repo_name(self.git('remote', 'get-url', 'origin', cwd=self.source)) == self.repo,
                       'Local release repository changed')
        driver.require((self.source/'release/coordinator.py').is_file(),
                       'Selected commit predates the coordinator; commit this implementation before releasing')
        driver.require((self.source/'docs/release/h3cli-release.pub').is_file(),
                       'Selected commit must contain the public release key')
        path = self.source/'outputs/release/config.json'
        driver.require(not path.exists() or driver.read(path) == self.config, 'Candidate configuration changed')
        driver.write(path, self.config)
        self.runner = driver.Runner(path, root=self.source)
        existing = self.runner.remote_release()
        driver.require(existing is None or existing.get('draft') is True,
                       'Release is already public; choose a new version')

    def linux_run(self):
        self.remote(['python3', '-c', REMOTE_PREPARE, self.remote_base,
                     json.dumps(self.manifest, sort_keys=True)])
        bundle = self.base/'source.bundle'
        self.command(['git', '-C', self.source, 'bundle', 'create', bundle, 'HEAD'])
        self.command(['scp', *driver.ssh_options(self.linux['identity_file']), bundle,
                      self.linux['host'] + ':' + self.remote_base + '/source.bundle'])
        self.remote(['python3', '-c', REMOTE_CHECKOUT, self.remote_base, self.commit, self.repo])
        self.remote(['bash', self.remote_source + '/release/linux/step1-prerequisites.sh'])
        configure = ['python3', self.remote_source + '/release/driver.py', 'configure',
                     '--repo', self.repo, '--tag', self.args.version, '--commit', self.commit,
                     '--run', self.run, '--jobs', self.linux['jobs'], '--models-path', self.linux['models']]
        if self.linux.get('model'):
            configure.extend(['--model', self.linux['model']])
        self.remote(configure)
        self.remote(['bash', self.remote_source + '/release/linux/run.sh'])

    def notes(self):
        r = self.runner
        linux = driver.read(r.incoming/'qualification.json')
        mac = r.completed('macos-test', required=True)
        # Verify reports before describing any test as passed. Machine paths and
        # raw logs stay private; notes contain only public release facts.
        r.check_handoff()
        driver.require(mac['reports']['features']['passed'] is True and
                       linux['reports']['features']['passed'] is True, 'Missing passing feature reports')
        pending = [name for name in driver.MANUAL if not r.completed('manual-' + name)]
        changes = self.changes or 'Build of the source commit listed below.'
        body = f'''# h3cli {self.args.version}

{changes}

## Downloads

- Linux/NVIDIA: `h3cli-linux-x86_64`.
- Apple Silicon: `h3cli-macos-arm64.dmg` (signed, notarized and stapled); a signed standalone executable is also attached.
- Both platforms include matching application/dependency source archives and license notices.
- Model weights are separate; use `--models-path` to select their location.

## Verification

Source commit: `{self.commit}`.
Apple signing identity: {self.mac['apple_identity']}.
Verify `SHA256SUMS.minisig` and `SHA256SUMS` using the independently trusted project public key.
The committed key is available in [the release source](https://github.com/{self.repo}/blob/{self.commit}/docs/release/h3cli-release.pub).

## Automated qualification

- Linux RTX PRO 5000: all {len(linux['reports']['parity']['files'])} recorded parity outputs passed, along with all four distro gates, runtime checks, and feature/server tests.
- Mac: signed-package feature/server tests and Apple signature/notarization checks passed.
- Uploaded assets are downloaded and checked against the tested artifacts before this command reports success.

## Additional qualification

'''
        body += ('Pending checks: ' + ', '.join('`' + name + '`' for name in pending) + '.\n'
                 if pending else 'All additional qualification reports are recorded for these artifacts.\n')
        r.notes.write_text(body)
        r.notes_check()

    def mac_run(self):
        r = self.runner
        r.build('macos')
        r.sign()
        r.test('macos')
        r.collect()
        r.stage_files()
        self.notes()
        r.draft()
        r.verify()
        remote = r.remote_release()
        driver.require(remote and remote.get('draft') is True, 'Release is no longer a draft')
        print('\nVerified GitHub draft: ' + remote['html_url'])

    def run_all(self):
        self.preflight()
        self.checkout()
        if self.dry:
            self.linux_run()
            print('Would build/sign/test Mac, collect Linux, sign checksums, generate notes, upload a draft, and verify downloads.')
            print('Publication is a separate command.')
            return
        with self.runner.lock():
            # Check the selected commit's locked toolchain before the remote build.
            self.runner.prerequisites('macos')
            self.linux_run()
            self.mac_run()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', '--tag', required=True, help='New GitHub release version, e.g. v0.2.0')
    p.add_argument('--settings', type=Path, default=driver.ROOT/'outputs/release/machines.json',
                   help='Private machine settings; see release/machines.example.json')
    p.add_argument('--commit', default='HEAD', help='Exact source revision; defaults to HEAD')
    p.add_argument('--run', help='Candidate name, defaults to VERSION-01; reuse to retry or choose a fresh name after a partial build')
    p.add_argument('--notes', type=Path, help='Optional change description; verified test results are appended automatically')
    p.add_argument('--dry-run', action='store_true', help='Read local settings/Git and print actions without writes or network access')
    return p


def main(argv=None):
    driver.require(sys.version_info >= (3, 10), 'Python 3.10+ is required')
    Coordinator(parser().parse_args(argv)).run_all()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        sys.exit('Release stopped: ' + str(exc))
