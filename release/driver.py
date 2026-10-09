#!/usr/bin/env python3
"""Release orchestration. Build recipes remain in scripts/{linux,macos}."""
import argparse
from contextlib import contextmanager
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
LINUX = ('h3cli-linux-x86_64', 'h3cli-linux-x86_64.sha256',
         'h3cli-linux-x86_64-sources.tar.gz', 'h3cli-linux-x86_64-sources.tar.gz.sha256')
MACOS = ('h3cli-macos-arm64', 'h3cli-macos-arm64.dmg', 'h3cli-macos-arm64-sources.tar.gz')
PUBLIC = (*LINUX, *MACOS, 'release-info.txt', 'h3cli-release.pub')
ASSETS = (*PUBLIC, 'SHA256SUMS', 'SHA256SUMS.minisig')
STEPS = {'linux': ('prerequisites', 'build', 'models', 'test', 'export'),
         'macos': ('prerequisites', 'build', 'sign', 'test', 'collect', 'stage',
                   'draft', 'verify', 'publish')}
MANUAL = ('source-tests', 'macos-clean-runtime',
          'macos-fresh-online', 'macos-fresh-offline', 'linux-downloaded')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A killed writer must not leave a completed-looking receipt/config.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(f.name, path)


def hashes(directory, names):
    result = {}
    for name in names:
        require(Path(name).name == name, 'Unsafe artifact name: ' + name)
        path = Path(directory) / name
        require(path.is_file() and not path.is_symlink(), 'Missing/linked artifact: ' + str(path))
        result[name] = sha(path)
    return result


def checksum_list(path, expected):
    rows = {}
    for line in Path(path).read_text().splitlines():
        m = re.fullmatch(r'([0-9a-f]{64})  ([A-Za-z0-9_.-]+)', line)
        require(m is not None, 'Malformed checksum line')
        digest, name = m.groups()
        require(name not in rows, 'Duplicate checksum: ' + name)
        rows[name] = digest
    require(set(rows) == set(expected), 'Unexpected or missing checksum entries')
    return rows


def verify_checksums(directory, manifest, names):
    require(checksum_list(Path(directory)/manifest, names) == hashes(directory, names),
            'Checksum mismatch in ' + str(directory))


def repo_name(value):
    m = re.fullmatch(r'(?:https://github\.com/|git@github\.com:)?([\w.-]+/[\w.-]+?)(?:\.git)?/?', value)
    require(m is not None and not m[1].startswith('-'), 'Repository must be OWNER/REPO or a GitHub URL')
    return m[1]


def ssh_options(identity_file=''):
    return ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            *(['-i', str(identity_file), '-o', 'IdentitiesOnly=yes'] if identity_file else [])]


class Runner:
    def __init__(self, config, dry=False, root=ROOT):
        self.root = Path(root).resolve()
        self.config = Path(config).resolve()
        self.c = read(self.config)
        require(self.c.get('schema') == 1, 'Unsupported release config')
        require(self.c['root'] == str(self.root), 'Config belongs to another checkout; configure locally')
        require(all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', self.c[k]) for k in ('run', 'tag')),
                'Invalid run/tag in config')
        require(re.fullmatch(r'[0-9a-f]{40}', self.c['commit']), 'Invalid commit in config')
        require(repo_name(self.c['repo']) == self.c['repo'], 'Invalid repository in config')
        self.dry = dry
        self.identity = {k: self.c[k] for k in ('repo', 'tag', 'commit', 'run')}
        self.base = self.root/'outputs/release'/self.c['run']
        self.linux = self.root/'bin'/('linux-' + self.c['run'])
        self.dev = self.root/'bin'/('macos-dev-' + self.c['run'])
        self.mac = self.root/'bin'/('macos-signed-' + self.c['run'])
        self.handoff = self.root/'bin'/('linux-handoff-' + self.c['run'])
        self.incoming = self.root/'bin'/('incoming-linux-' + self.c['run'])
        self.stage = self.root/'bin'/('publish-' + self.c['run'])
        self.notes = self.base/'release-notes.md'
        self.key = self.root/'docs/release/h3cli-release.pub'

    def command(self, argv, *, capture=False, cwd=None, env=None, interactive=False):
        argv = list(map(str, argv))
        print('+ ' + shlex.join(argv), flush=True)
        if self.dry:
            return ''
        cwd = cwd or self.root
        if capture:
            return subprocess.check_output(argv, cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT).strip()
        if interactive:
            subprocess.run(argv, cwd=cwd, env=env, check=True)
            return ''
        logs = self.base/'logs'
        logs.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', prefix=time.strftime('%Y%m%d-%H%M%S-'),
                                         suffix='.log', dir=logs, delete=False) as log:
            self.last_log = Path(log.name)
            print('  Log: ' + log.name, flush=True)
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        require(result.returncode == 0, f'Command failed ({result.returncode}); see {log.name}')
        return ''

    @contextmanager
    def lock(self):
        if self.dry:
            yield
            return
        self.base.mkdir(parents=True, exist_ok=True)
        with (self.base/'automation.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another release step is running for this candidate') from None
            yield

    def clean(self):
        if self.dry:
            print('Would require the configured commit and a clean checkout.')
            return
        actual = self.command(['git', 'rev-parse', 'HEAD'], capture=True)
        require(actual == self.c['commit'], 'Checkout moved away from the configured release commit')
        require(not self.command(['git', 'status', '--porcelain'], capture=True),
                'Release needs a clean checkout. Commit/review changes first; do not discard them.')

    def host(self, target):
        if not self.dry:
            expected = ('Linux', 'x86_64') if target == 'linux' else ('Darwin', 'arm64')
            require((platform.system(), platform.machine()) == expected,
                    f'This step requires {expected[0]} {expected[1]}')

    def receipt(self, name, files=(), **extra):
        if not self.dry:
            write(self.base/(name + '.json'), dict(schema=1, identity=self.identity,
                  config_sha256=sha(self.config), files={str(p): sha(p) for p in files},
                  completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra))

    def completed(self, name, required=False):
        if self.dry:
            return False
        path = self.base/(name + '.json')
        if not path.exists():
            require(not required, f'Run the {name} step first')
            return False
        row = read(path)
        require(row.get('identity') == self.identity and row.get('config_sha256') == sha(self.config),
                'Stale completion record: ' + name + '; use a new run/config')
        require(all(Path(p).is_file() and sha(p) == h for p, h in row['files'].items()),
                'Files changed after ' + name + '; retain evidence and use a new run')
        return row

    def fresh(self, path):
        if self.dry:
            return
        require(not path.exists(), 'Existing incomplete output: ' + str(path) + '; use a new run name')
        path.parent.mkdir(parents=True, exist_ok=True)

    def linux_build_check(self, directory):
        if self.dry:
            return
        row = read(directory/'build.json')
        require(row.get('commit') == self.c['commit'] and row.get('dirty') is False,
                'Linux build is dirty or from another commit')
        verify_checksums(directory, LINUX[1], [LINUX[0]])
        verify_checksums(directory, LINUX[3], [LINUX[2]])

    def mac_build_check(self, signed=False, directory=None):
        if self.dry:
            return
        row = read(self.dev/'build.json')
        source = row['source']['files']
        # Compare the actual archived source inventory, not a platform-specific
        # fingerprint or the unrelated current HEAD label alone.
        for name, digest in source.items():
            path = self.root/name
            require(path.resolve().is_relative_to(self.root), 'Unsafe source inventory')
            require(path.is_file() and sha(path) == digest, 'Mac build source differs: ' + name)
        record = row
        directory = directory or (self.mac if signed else self.dev)
        if signed:
            record = read(directory/'build.json')
            require(record['input_build_sha256'] == sha(self.dev/'build.json'), 'Wrong Mac signing input')
            require(all(record.get(phase, {}).get('status') == 'Accepted' for phase in ('inner', 'outer')),
                    'Mac notarization was not accepted')
        for name, item in record['artifacts'].items():
            require(Path(name).name == name and sha(directory/name) == item['sha256'],
                    'Mac artifact changed: ' + name)

    def prerequisites(self, target, install=False):
        self.command(['bash', self.root/'release'/target/'step1-prerequisites.sh',
                      *(['--install'] if install else [])], interactive=install)

    def build(self, target):
        self.host(target)
        self.clean()
        if self.completed(target + '-build'):
            print('Verified completed build; reusing it.')
            return
        output = self.linux if target == 'linux' else self.dev
        cache = self.root/'outputs'/(target + '-build')/'cache'
        work = cache.parent/self.c['run']
        self.fresh(output)
        self.fresh(work)
        script = 'scripts/build_' + target + '.sh'
        self.command(['bash', script, '--fetch-only', '--cache', cache])
        self.command(['bash', script, '--offline', '--jobs', self.c['jobs'],
                      '--cache', cache, '--work-dir', work, '--output', output])
        self.clean()
        if target == 'linux':
            self.linux_build_check(output)
            self.command([output/LINUX[0]], env=dict(os.environ, H3CLI_BUNDLE_INFO='1'))
            names = (*LINUX, 'build.json')
        else:
            self.mac_build_check()
            names = (*MACOS, 'build.json', 'SHA256SUMS', 'libh3.a')
            self.command(['python3', 'tests/test_macos_bundle.py'])
        self.receipt(target + '-build', [output/n for n in names])

    def models(self, target):
        self.host(target)
        self.completed('linux-build' if target == 'linux' else 'macos-sign', required=True)
        binary = self.linux/LINUX[0] if target == 'linux' else self.mac/MACOS[0]
        self.command([binary, '--models-path', self.c['models'], '-d', self.c['model'],
                      '--download-models', 'all'])

    def sign(self):
        self.host('macos')
        self.clean()
        self.completed('macos-build', required=True)
        if self.completed('macos-sign'):
            print('Verified completed Apple signing; reusing it.')
            return
        identity = self.c['apple_identity']
        require(identity.startswith('Developer ID Application: '), 'Configure --apple-identity first')
        self.mac_build_check()
        self.fresh(self.mac)
        candidate = self.mac.parent/('macos-sign-attempt-' + uuid.uuid4().hex)
        self.command(['python3', 'scripts/macos/release.py', '--input', self.dev,
                      '--output', candidate, '--identity', identity,
                      '--keychain-profile', self.c['notary_profile']], interactive=True)
        self.mac_build_check(signed=True, directory=candidate)
        if not self.dry:
            candidate.rename(self.mac)
        self.receipt('macos-sign', [self.mac/n for n in (*MACOS, 'build.json', 'SHA256SUMS', 'libh3.a')])

    def apple_verify(self, directory):
        self.command(['xcrun', 'stapler', 'validate', directory/MACOS[1]])
        self.command(['spctl', '--assess', '--type', 'open', '--context',
                      'context:primary-signature', '--verbose=2', directory/MACOS[1]])
        self.command(['codesign', '--verify', '--strict', '--verbose=4',
                      '--test-requirement', '=notarized', '--check-notarization', directory/MACOS[0]])
        details = self.command(['codesign', '--display', '--verbose=4', directory/MACOS[0]], capture=True)
        if not self.dry:
            team = re.search(r'\(([^()]+)\)$', self.c['apple_identity'])
            require(team is not None and 'TeamIdentifier=' + team[1] in details.splitlines(),
                    'Unexpected Apple signing team')

    def test(self, target):
        self.host(target)
        self.clean()
        self.completed('linux-build' if target == 'linux' else 'macos-sign', required=True)
        if self.completed(target + '-test'):
            print('Verified completed qualification; reusing it.')
            return
        output = self.base/target/'tests'
        # Failed test attempts remain intact. A retry gets a fresh directory.
        if not self.dry:
            output.parent.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix='tests-', dir=output.parent))
        release = self.linux if target == 'linux' else self.mac
        runtime = release/(target + '-runtime')
        binary = release/(LINUX[0] if target == 'linux' else MACOS[0])
        before = {} if self.dry else hashes(release, LINUX if target == 'linux' else MACOS)
        reports = []
        if target == 'linux':
            self.linux_build_check(release)
            self.command(['nvidia-smi'])
            self.command(['python3', 'scripts/linux/package.py', 'seal', runtime])
            self.command(['python3', 'tests/cuda_reference_regression.py', '--source', self.root,
                          '--runtime', runtime, '--validation', release/'linux-validation',
                          '--artifact', binary, '--model', self.c['model'], '--out', output/'parity'])
            cache = self.root/'outputs/linux-build/cache'
            self.command(['python3', 'tests/linux_distribution.py', '--source', self.root,
                          '--release', release, '--cache', cache, '--model', self.c['model'],
                          '--out', output/'distros'])
            self.command(['python3', 'tests/linux_runtime_smoke.py', '--release', release,
                          '--cache', cache, '--model', self.c['model'], '--out', output/'runtime'])
            reports = [output/'parity/result.json', output/'distros/result.json', output/'runtime/results.json']
        else:
            self.mac_build_check(signed=True)
            self.apple_verify(release)
            self.models('macos')
        self.command(['python3', 'tests/model_downloads_render.py', '--binary', binary,
                      '--backend', 'cuda' if target == 'linux' else 'metal',
                      '--models-path', self.c['models'], '--model', self.c['model'],
                      '--ffmpeg', runtime/'tools/ffmpeg', '--ffprobe', runtime/'tools/ffprobe',
                      '--server', '--out', output/'features'])
        reports.append(output/'features/results.json')
        if not self.dry:
            require(all(read(p).get('passed') is True for p in reports), 'A qualification report did not pass')
            if target == 'linux':
                require(len(read(reports[0]).get('files', {})) == 204, 'Expected all 204 parity outputs')
                rows = read(reports[1]).get('checks', [])
                require(len(rows) == 4 and all(r.get('passed') and r.get('output_count') == 204 for r in rows),
                        'Incomplete distro parity qualification')
            require(read(reports[-1])['binary_sha256'] == sha(binary), 'Feature report tested another executable')
            require(before == hashes(release, before), 'Artifacts changed during qualification')
        self.clean()
        self.receipt(target + '-test', [*reports, *(release/n for n in before)],
                     artifacts=before, reports={p.parent.name: read(p) for p in reports} if not self.dry else {})

    def export(self):
        self.clean()
        record = self.completed('linux-test', required=True)
        self.linux_build_check(self.linux)
        if self.completed('linux-export'):
            print('Verified Linux handoff already exists.')
            return
        self.fresh(self.handoff)
        if self.dry:
            print('Would export the four public Linux files, build.json and qualification.json to ' + str(self.handoff))
            return
        self.handoff.mkdir()
        for name in (*LINUX, 'build.json'):
            shutil.copy2(self.linux/name, self.handoff/name)
        write(self.handoff/'qualification.json', record)
        self.receipt('linux-export', [self.handoff/n for n in (*LINUX, 'build.json', 'qualification.json')])

    def check_handoff(self, directory=None):
        if self.dry:
            return
        directory = directory or self.incoming
        self.linux_build_check(directory)
        q = read(directory/'qualification.json')
        require(q.get('identity') == self.identity, 'Linux handoff belongs to another release')
        require(q.get('artifacts') == hashes(directory, LINUX), 'Linux qualification has different artifacts')
        reports = q.get('reports', {})
        require(set(reports) == {'parity', 'distros', 'runtime', 'features'} and
                all(r.get('passed') is True for r in reports.values()), 'Incomplete Linux qualification')
        require(len(reports['parity'].get('files', {})) == 204, 'Incomplete 204-output parity report')
        rows = reports['distros'].get('checks', [])
        require(len(rows) == 4 and all(r.get('passed') and r.get('output_count') == 204 for r in rows),
                'Incomplete distro parity report')
        require(reports['features'].get('binary_sha256') == sha(directory/LINUX[0]) and
                reports['runtime'].get('release_sha256') == sha(directory/LINUX[0]),
                'Linux test report identity mismatch')

    def collect(self):
        self.clean()
        if self.completed('macos-collect'):
            self.check_handoff()
            print('Verified Linux handoff already collected.')
            return
        self.fresh(self.incoming)
        local = self.c['linux_handoff']
        names = (*LINUX, 'build.json', 'qualification.json')
        incoming = self.incoming
        if not self.dry:
            incoming = Path(tempfile.mkdtemp(prefix='incoming-attempt-', dir=self.incoming.parent))
        if local:
            if not self.dry:
                for name in names:
                    shutil.copy2(Path(local)/name, incoming/name)
        else:
            host, remote = self.c['linux_host'], self.c['linux_root']
            require(host and remote, 'Configure --linux-host/--linux-root or --linux-handoff')
            require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*', host) and
                    re.fullmatch(r'/[A-Za-z0-9_./-]+', remote), 'Use a simple SSH alias and remote path without spaces')
            remote = remote.rstrip('/') + '/bin/linux-handoff-' + self.c['run']
            self.command(['scp', *ssh_options(self.c.get('linux_identity_file', '')),
                          *[host + ':' + remote + '/' + n for n in names], str(incoming) + '/'])
        self.check_handoff(incoming)
        if not self.dry:
            incoming.rename(self.incoming)
        self.receipt('macos-collect', [self.incoming/n for n in names])

    def stage_files(self):
        self.clean()
        self.completed('macos-test', required=True)
        self.completed('macos-collect', required=True)
        self.check_handoff()
        self.mac_build_check(signed=True)
        if self.completed('macos-stage'):
            self.verify_directory(self.stage)
            print('Verified signed staging files already exist.')
            return
        self.fresh(self.stage)
        candidate = self.stage.parent/('publish-attempt-' + uuid.uuid4().hex)
        if not self.dry:
            require(self.key.is_file(), 'Review and commit the public release key before building')
            candidate.mkdir()
            for directory, names in ((self.incoming, LINUX), (self.mac, MACOS)):
                for name in names:
                    shutil.copy2(directory/name, candidate/name)
            shutil.copy2(self.key, candidate/self.key.name)
            (candidate/'release-info.txt').write_text(self.info())
            (candidate/'SHA256SUMS').write_text(''.join(h + '  ' + n + '\n' for n, h in hashes(candidate, PUBLIC).items()))
            require(all((candidate/n).stat().st_size < 2**31 for n in PUBLIC), 'GitHub assets must be under 2 GiB')
        self.command(['minisign', '-Sm', candidate/'SHA256SUMS', '-s', self.c['secret_key'],
                      '-t', f"h3cli {self.c['tag']} commit {self.c['commit']}"], interactive=True)
        self.verify_directory(candidate)
        if not self.dry:
            candidate.rename(self.stage)
        if not self.dry and not self.notes.exists():
            template = (self.root/'docs/release/release-notes-template.md').read_text()
            template = template.replace('REPLACE_VERSION', self.c['tag']).replace('REPLACE_FULL_COMMIT_ID', self.c['commit'])
            self.notes.write_text(template)
        self.receipt('macos-stage', [self.stage/n for n in ASSETS])
        print('Edit release notes before the draft step: ' + str(self.notes))

    def info(self):
        return ''.join(f'{key}={self.c[value]}\n' for key, value in
                       (('H3_RELEASE_REPO', 'repo'), ('H3_RELEASE_TAG', 'tag'), ('H3_RELEASE_COMMIT', 'commit')))

    def verify_directory(self, directory):
        # Never trust a public key supplied only by the download itself.
        self.command(['minisign', '-Vm', directory/'SHA256SUMS', '-p', self.key])
        if self.dry:
            print('Would verify the exact public asset list, all SHA-256 hashes and release identity.')
            return
        require(set(p.name for p in directory.iterdir()) == set(ASSETS), 'Unexpected/missing public release files')
        verify_checksums(directory, 'SHA256SUMS', PUBLIC)
        require((directory/'release-info.txt').read_text() == self.info(), 'Wrong release version/commit')
        require(sha(directory/self.key.name) == sha(self.key), 'Wrong public signing key')

    def gh(self, *args, capture=False):
        return self.command(['gh', *args, '--repo', self.c['repo']], capture=capture)

    def remote_release(self):
        if self.dry:
            return None
        # Distinguish 404 from authentication/network errors; never create on an unknown error.
        argv = ['gh', 'api', 'repos/' + self.c['repo'] + '/releases/tags/' + self.c['tag']]
        proc = subprocess.run(argv, cwd=self.root, capture_output=True, text=True)
        if proc.returncode:
            require('HTTP 404' in proc.stderr, 'Cannot inspect GitHub release: ' + proc.stderr.strip())
            return None
        return json.loads(proc.stdout)

    def tag(self):
        repo = self.c['repo']
        tag = self.c['tag']
        remote = self.command(['git', 'remote', 'get-url', 'origin'], capture=True)
        if not self.dry:
            require(repo_name(remote) == repo, 'origin differs from the configured GitHub repository')
        self.command(['git', 'fetch', 'origin', '--tags'])
        if self.dry:
            print('Would create the tag only if absent, reject a different commit, and push only that tag.')
            return
        local = subprocess.run(['git', 'rev-parse', '--verify', 'refs/tags/' + tag + '^{commit}'],
                               cwd=self.root, capture_output=True, text=True)
        if local.returncode == 0:
            require(local.stdout.strip() == self.c['commit'], 'Existing tag points at another commit')
        else:
            self.command(['git', 'tag', '-a', tag, self.c['commit'], '-m', 'h3cli ' + tag])
        self.command(['git', 'push', 'origin', 'refs/tags/' + tag])

    def remote_tag_check(self):
        ref = 'refs/tags/' + self.c['tag']
        output = self.command(['git', 'ls-remote', 'https://github.com/' + self.c['repo'] + '.git',
                               ref, ref + '^{}'], capture=True)
        if not self.dry:
            refs = dict((name, commit) for commit, name in (line.split() for line in output.splitlines()))
            require(refs.get(ref + '^{}', refs.get(ref)) == self.c['commit'],
                    'Remote release tag moved or is missing')

    def notes_check(self):
        if not self.dry:
            require(self.notes.is_file() and self.notes.read_text().strip() and
                    'REPLACE_' not in self.notes.read_text(), 'Finish release notes: ' + str(self.notes))

    def draft(self):
        self.clean()
        self.completed('macos-stage', required=True)
        self.verify_directory(self.stage)
        self.notes_check()
        existing = self.remote_release()
        require(existing is None or existing.get('draft') is True, 'Release is already public; never overwrite it')
        self.tag()
        if existing is None:
            self.gh('release', 'create', self.c['tag'], '--verify-tag', '--draft',
                    '--title', 'h3cli ' + self.c['tag'], '--notes-file', self.notes)
        else:
            self.gh('release', 'edit', self.c['tag'], '--notes-file', self.notes)
        remote_names = set()
        if existing:
            # Inspect existing bytes before resuming an interrupted upload.
            require(len(existing['assets']) == len({a['name'] for a in existing['assets']}), 'Duplicate remote assets')
            remote_names = {a['name'] for a in existing['assets']}
            require(remote_names <= set(ASSETS), 'Unexpected asset in draft; inspect it manually')
            if remote_names:
                check = Path(tempfile.mkdtemp(prefix='existing-draft-', dir=self.base))
                self.gh('release', 'download', self.c['tag'], '--dir', check)
                require(hashes(check, remote_names) == hashes(self.stage, remote_names),
                        'Existing draft bytes differ; refusing to replace them')
        missing = [self.stage/n for n in ASSETS if n not in remote_names]
        if missing:
            self.gh('release', 'upload', self.c['tag'], *missing)
        self.receipt('macos-draft', [self.stage/n for n in ASSETS])
        self.gh('release', 'view', self.c['tag'])

    def verify(self):
        self.completed('macos-stage', required=True)
        self.completed('macos-draft', required=True)
        self.remote_tag_check()
        directory = self.base/'downloaded'
        if not self.dry:
            directory = Path(tempfile.mkdtemp(prefix='downloaded-', dir=self.base))
            remote = self.remote_release()
            require(remote is not None and {a['name'] for a in remote['assets']} == set(ASSETS)
                    and len(remote['assets']) == len(ASSETS), 'Wrong GitHub asset list')
        self.gh('release', 'download', self.c['tag'], '--dir', directory)
        self.verify_directory(directory)
        if not self.dry:
            require(hashes(directory, ASSETS) == hashes(self.stage, ASSETS), 'Downloaded bytes differ from the tested stage')
        self.apple_verify(directory)
        self.receipt('macos-verify', [directory/n for n in ASSETS], download=str(directory))

    def record_check(self, check, report):
        self.completed('macos-stage', required=True)
        if self.dry:
            print(f'Would record maintainer-reported {check} evidence from {report}, bound to these artifacts.')
            return
        report = Path(report).resolve()
        require(report.is_file() and report.stat().st_size > 0, 'Supply a nonempty actual test report')
        folder = self.base/'manual'
        folder.mkdir(exist_ok=True)
        dest = folder/(check + '.txt')
        require(not dest.exists(), 'Evidence already recorded; use a new candidate to replace it')
        shutil.copyfile(report, dest)
        self.receipt('manual-' + check, [dest, *(self.stage/n for n in ASSETS)],
                     reported_by=getpass.getuser(), note='Maintainer-reported check; not an automated test result')

    def source_tests(self, target, environment=None):
        self.host(target)
        self.clean()
        if environment:
            environment = Path(environment).expanduser().resolve()
            require(environment.is_file(), 'Native setup environment file does not exist')
        targets = ['all', 'test', 'test-server-http']
        if target == 'macos':
            targets.append('test-macos-package')
        # Only a user-selected, trusted setup file is sourced. Values are passed
        # as positional arguments, never interpolated into shell program text.
        program = ('set -euo pipefail; if [ -n "$1" ]; then source "$1"; fi; '
                   'export H3_MODEL_DIR="$2" H3_OFFLINE=1; shift 2; exec make "$@"')
        self.command(['bash', '-c', program, 'h3cli-source-tests', str(environment or ''),
                      self.c['model'], '-j' + str(self.c['jobs']), *targets])
        self.clean()
        self.receipt(target + '-source-tests', [] if self.dry else [self.last_log])

    def publish(self, channel):
        require(channel in ('stable', 'prerelease'), 'Choose --channel stable or --channel prerelease explicitly')
        self.clean()
        self.completed('macos-stage', required=True)
        self.completed('macos-verify', required=True)
        self.notes_check()
        checks = MANUAL if channel == 'stable' else ('source-tests',)
        for check in checks:
            self.completed('manual-' + check, required=True)
        # Re-download immediately before publishing; a previous verification
        # receipt alone cannot detect later server-side asset replacement.
        self.verify()
        remote = self.remote_release()
        if not self.dry:
            require(remote and remote['draft'], 'Only an existing draft can be published')
            if channel == 'prerelease':
                missing = [check for check in MANUAL if not self.completed('manual-' + check)]
                notes = self.notes.read_text()
                require(all(check in notes for check in missing),
                        'Prerelease notes must name pending check IDs: ' + ', '.join(missing))
        self.gh('release', 'edit', self.c['tag'], '--notes-file', self.notes)
        self.gh('release', 'edit', self.c['tag'], '--draft=false',
                '--prerelease=' + ('false' if channel == 'stable' else 'true'),
                '--latest=' + ('true' if channel == 'stable' else 'false'))
        self.receipt('macos-publish', [self.stage/n for n in ASSETS], channel=channel)


def configure(args):
    root = ROOT
    commit = subprocess.check_output(['git', 'rev-parse', args.commit + '^{commit}'], cwd=root, text=True).strip()
    repository = args.repo or subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=root, text=True).strip()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.tag), 'Use a simple version tag, e.g. v0.1.0')
    run = args.run or args.tag + '-01'
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run), 'Run name must be a simple directory name')
    require(1 <= args.jobs <= 32, 'Jobs must be in 1..32')
    if args.linux_host:
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*', args.linux_host), 'Use an SSH host/alias, not SSH options')
    if args.linux_root:
        require(re.fullmatch(r'/[A-Za-z0-9_./-]+', args.linux_root), 'Use an absolute remote checkout path without spaces')
    models = Path(args.models_path).expanduser().resolve()
    default_model = models/'MiniMaxH3'
    if not default_model.exists() and (models/'MiniMax-H3').exists():
        default_model = models/'MiniMax-H3'
    model = Path(args.model).expanduser().resolve() if args.model else default_model
    require(not Path(args.secret_key).expanduser().resolve().is_relative_to(root), 'Keep the secret key outside the checkout')
    config = dict(schema=1, root=str(root), repo=repo_name(repository), tag=args.tag, commit=commit,
                  run=run, jobs=args.jobs, models=str(models), model=str(model),
                  apple_identity=args.apple_identity, notary_profile=args.notary_profile,
                  secret_key=str(Path(args.secret_key).expanduser().resolve()),
                  linux_host=args.linux_host, linux_root=args.linux_root,
                  linux_handoff=str(Path(args.linux_handoff).expanduser().resolve()) if args.linux_handoff else '')
    if args.linux_identity_file:
        config['linux_identity_file'] = str(Path(args.linux_identity_file).expanduser().resolve())
    if args.dry_run:
        print(json.dumps(config, indent=2))
    else:
        path = Path(args.config)
        require(not path.exists() or read(path) == config, 'Config already exists with different values; use a new --config and --run')
        write(path, config)
        print('Saved configuration: ' + str(path.resolve()))
        print('Release steps can now run independently from any working directory.')


def clone(args):
    repo = repo_name(args.repo)
    require(re.fullmatch(r'[0-9a-f]{40}', args.commit), 'Use the full reviewed commit ID')
    dest = Path(args.destination).expanduser().resolve()
    require(not dest.exists(), 'Clone destination already exists')
    commands = [['git', 'clone', 'https://github.com/' + repo + '.git', str(dest)],
                ['git', '-C', str(dest), 'checkout', '--detach', args.commit]]
    for cmd in commands:
        print('+ ' + shlex.join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


def setup_signing(runner, args):
    runner.host('macos')
    c = runner.c
    if args.github_login:
        runner.command(['gh', 'auth', 'login', '--hostname', 'github.com', '--git-protocol', 'https', '--web'], interactive=True)
        runner.command(['gh', 'auth', 'setup-git', '--hostname', 'github.com'])
    if args.store_credentials:
        require(args.apple_id and args.team_id, 'Supply --apple-id and --team-id; the password is prompted locally')
        runner.command(['xcrun', 'notarytool', 'store-credentials', c['notary_profile'],
                        '--apple-id', args.apple_id, '--team-id', args.team_id], interactive=True)
    if args.generate_key:
        secret = Path(c['secret_key'])
        public = secret.with_suffix('.pub')
        require(not secret.exists() and not public.exists() and not runner.key.exists(), 'A release key already exists; refusing to replace it')
        require(not secret.is_relative_to(runner.root), 'Secret key must be outside the checkout')
        if not runner.dry:
            secret.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        runner.command(['minisign', '-G', '-s', secret, '-p', public], interactive=True)
        if not runner.dry:
            shutil.copy2(public, runner.key)
        print('Review and commit docs/release/h3cli-release.pub, then configure the final release commit in a fresh clone.')
    runner.command(['security', 'find-identity', '-v', '-p', 'codesigning'])
    runner.command(['xcrun', 'notarytool', 'history', '--keychain-profile', c['notary_profile']])
    runner.command(['gh', 'auth', 'status', '--hostname', 'github.com'])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    def common(name, **kwargs):
        child = commands.add_parser(name, **kwargs)
        child.add_argument('--config', type=Path, default=ROOT/'outputs/release/config.json')
        child.add_argument('--dry-run', action='store_true', help='Preview actions without changing files or calling external services')
        return child
    c = common('configure', help='Save release settings once on each machine')
    c.add_argument('--repo', default='')
    c.add_argument('--tag', required=True)
    c.add_argument('--commit', default='HEAD')
    c.add_argument('--run', default='')
    c.add_argument('--models-path', required=True)
    c.add_argument('--model')
    c.add_argument('--jobs', type=int, default=8)
    c.add_argument('--apple-identity', default='')
    c.add_argument('--notary-profile', default='h3cli')
    c.add_argument('--secret-key', default='~/.config/h3cli-release/h3cli-release.key')
    c.add_argument('--linux-host', default='')
    c.add_argument('--linux-root', default='')
    c.add_argument('--linux-handoff', default='')
    c.add_argument('--linux-identity-file', default='', help='SSH private-key path on the Mac (never copied)')
    c = commands.add_parser('clone', help='Clone a reviewed commit into a new directory')
    c.add_argument('--repo', required=True)
    c.add_argument('--commit', required=True)
    c.add_argument('--destination', required=True)
    c.add_argument('--dry-run', action='store_true')
    for name in ('step', 'run'):
        c = common(name)
        c.add_argument('--platform', choices=STEPS, required=True)
        c.add_argument('--install', action='store_true', help='Install missing host helpers in prerequisites step')
        if name == 'step':
            c.add_argument('--name', required=True)
            c.add_argument('--channel', choices=('stable', 'prerelease'))
        else:
            c.add_argument('--through', type=int, help='Last numbered step; defaults to Linux 5 or Mac 4 (no upload)')
    c = common('record-check', help='Attach an actual human-reviewed test report to the staged artifacts')
    c.add_argument('--check', choices=MANUAL, required=True)
    c.add_argument('--report', type=Path, required=True)
    c = common('setup-signing', help='One-time interactive credentials/key setup; never publishes')
    c.add_argument('--github-login', action='store_true')
    c.add_argument('--store-credentials', action='store_true')
    c.add_argument('--apple-id')
    c.add_argument('--team-id')
    c.add_argument('--generate-key', action='store_true')
    common('status', help='Show recorded steps without changing anything')
    c = common('source-tests', help='Run the native source test suite using an already configured development toolchain')
    c.add_argument('--platform', choices=STEPS, required=True)
    c.add_argument('--environment', type=Path, help='Trusted environment file generated by the normal source setup script')
    c = commands.add_parser('verify-downloads', help='Verify release downloads using an independently trusted public key')
    c.add_argument('--directory', type=Path, required=True)
    c.add_argument('--public-key', type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(sys.version_info >= (3, 10), 'Python 3.10 or newer is required')
    if args.command == 'configure':
        configure(args)
        return
    if args.command == 'clone':
        clone(args)
        return
    if args.command == 'verify-downloads':
        subprocess.run(['minisign', '-Vm', str(args.directory/'SHA256SUMS'), '-p', str(args.public_key)], check=True)
        require(set(p.name for p in args.directory.iterdir()) == set(ASSETS), 'Unexpected/missing public release files')
        verify_checksums(args.directory, 'SHA256SUMS', PUBLIC)
        require(sha(args.directory/'h3cli-release.pub') == sha(args.public_key), 'Wrong downloaded public key')
        print('Signature and all nine file checksums verified.')
        return
    r = Runner(args.config, args.dry_run)
    if args.command == 'status':
        for path in sorted(r.base.glob('*.json')):
            print(path.name)
        print('Notes: ' + str(r.notes))
        return
    with r.lock():
        if args.command == 'setup-signing':
            setup_signing(r, args)
        elif args.command == 'record-check':
            r.record_check(args.check, args.report)
        elif args.command == 'source-tests':
            r.source_tests(args.platform, args.environment)
        else:
            steps = STEPS[args.platform]
            if args.command == 'run':
                through = args.through if args.through is not None else (5 if args.platform == 'linux' else 4)
                require(1 <= through <= (5 if args.platform == 'linux' else 8),
                        'run stops before publication; invoke step9-publish.sh explicitly')
                selected = steps[:through]
            else:
                require(args.name in steps, 'Unknown platform step')
                selected = [args.name]
            for step in selected:
                print(f'\n{args.platform}: {step}', flush=True)
                if step == 'prerequisites':
                    r.prerequisites(args.platform, args.install)
                elif step in ('build', 'models', 'test'):
                    getattr(r, step)(args.platform)
                elif step == 'stage':
                    r.stage_files()
                elif step == 'publish':
                    r.publish(args.channel)
                else:
                    getattr(r, step)()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        sys.exit('Release stopped: ' + str(exc))
