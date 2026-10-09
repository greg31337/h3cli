"""Coordinator control tests using real temporary Git repositories and mocked services."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coordinator
import driver


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h3-release-coordinator-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base/'checkout with spaces'
        self.root.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Release Test')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'commit.gpgsign', 'false')
        self.git('remote', 'add', 'origin', 'https://github.com/example/h3cli.git')
        (self.root/'.gitignore').write_text('outputs/\n')
        (self.root/'release').mkdir()
        (self.root/'release/coordinator.py').write_text('# fixture\n')
        (self.root/'docs/release').mkdir(parents=True)
        (self.root/'docs/release/h3cli-release.pub').write_text('test public key')
        self.git('add', '.')
        self.git('commit', '-qm', 'source fixture')
        self.commit = self.git('rev-parse', 'HEAD')
        self.profile = self.base/'machines.json'
        self.config = dict(schema=1, macos=dict(models=str(self.base/'models'),
                           apple_identity='Developer ID Application: Example (TEAM123)',
                           secret_key=str(self.base/'secret.key')),
                           linux=dict(host='gpu-test', identity_file=str(self.base/'key with spaces'),
                                      root='/work/releases', models='/models'))
        driver.write(self.profile, self.config)

    def git(self, *args, cwd=None):
        return subprocess.check_output(['git', *args], cwd=cwd or self.root, text=True).strip()

    def make(self, *extra):
        args = coordinator.parser().parse_args(['--version', 'v0.2.0', '--settings', str(self.profile), *extra])
        return coordinator.Coordinator(args, root=self.root)

    def checkout(self, c):
        with patch.object(driver.Runner, 'remote_release', return_value=None):
            c.checkout()

    def test_dry_run_is_read_only_and_contains_explicit_ssh_identity(self):
        c = self.make('--dry-run')
        before = sorted(self.base.rglob('*'))
        with patch.object(subprocess, 'run', side_effect=AssertionError('external command')):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                c.run_all()
        self.assertEqual(before, sorted(self.base.rglob('*')))
        self.assertIn('IdentitiesOnly=yes', output.getvalue())
        self.assertIn('key with spaces', output.getvalue())
        self.assertIn('verify downloads', output.getvalue())
        self.assertNotIn('--draft=false', output.getvalue())

    def test_isolated_checkout_bundle_and_remote_checkout_pin_the_commit(self):
        c = self.make()
        self.checkout(c)
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=c.source), self.commit)
        self.assertEqual(self.git('status', '--porcelain'), '')
        self.assertEqual(c.runner.c['linux_identity_file'], self.config['linux']['identity_file'])
        remote = self.base/'remote'
        remote.mkdir()
        self.git('bundle', 'create', str(remote/'source.bundle'), 'HEAD', cwd=c.source)
        argv = ['python3', '-c', coordinator.REMOTE_CHECKOUT, str(remote), c.commit, c.repo]
        subprocess.run(argv, check=True, capture_output=True, text=True)
        subprocess.run(argv, check=True, capture_output=True, text=True)
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=remote/'source'), self.commit)
        (remote/'source/local-edit.txt').write_text('preserve this')
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((remote/'source/local-edit.txt').read_text(), 'preserve this')

    def test_retry_reuses_candidate_but_rejects_changed_source_and_settings(self):
        c = self.make()
        self.checkout(c)
        self.checkout(c)
        changed = self.make()
        changed.manifest['settings']['linux']['models'] = '/different/models'
        with self.assertRaisesRegex(ValueError, 'different settings'):
            self.checkout(changed)
        (c.source/'local-edit').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'checkout changed'):
            self.checkout(c)
        self.assertEqual((c.source/'local-edit').read_text(), 'keep')

    def test_existing_public_release_rejected_before_remote_work(self):
        c = self.make()
        with patch.object(c, 'preflight'), patch.object(c, 'linux_run') as linux, \
             patch.object(driver.Runner, 'remote_release', return_value={'draft': False}):
            with self.assertRaisesRegex(ValueError, 'already public'):
                c.run_all()
        linux.assert_not_called()

    def test_linux_failure_prevents_mac_build_and_github_upload(self):
        c = self.make()
        self.checkout(c)
        with patch.object(c, 'preflight'), patch.object(c, 'checkout'), \
             patch.object(c.runner, 'prerequisites'), \
             patch.object(c, 'linux_run', side_effect=subprocess.CalledProcessError(1, 'ssh')), \
             patch.object(c, 'mac_run') as mac:
            with self.assertRaises(subprocess.CalledProcessError):
                c.run_all()
        mac.assert_not_called()

    def test_only_successful_verification_reports_success_and_never_publishes(self):
        c = self.make()
        c.runner = MagicMock()
        c.runner.remote_release.return_value = {'draft': True, 'html_url': 'https://github.com/example/h3cli/releases/tag/untagged-test'}
        c.runner.verify.side_effect = ValueError('bad uploaded bytes')
        with patch.object(c, 'notes'), contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(ValueError, 'bad uploaded bytes'):
                c.mac_run()
        self.assertNotIn('Verified GitHub draft:', output.getvalue())
        c.runner.publish.assert_not_called()
        c.runner.verify.side_effect = None
        with patch.object(c, 'notes'), contextlib.redirect_stdout(io.StringIO()) as output:
            c.mac_run()
        self.assertIn('Verified GitHub draft:', output.getvalue())
        self.assertIn('untagged-test', output.getvalue())
        c.runner.publish.assert_not_called()

    def test_notes_use_verified_results_and_disclose_pending_checks(self):
        c = self.make()
        self.checkout(c)
        r = c.runner
        r.base.mkdir(parents=True, exist_ok=True)
        driver.write(r.incoming/'qualification.json', dict(reports=dict(
            parity=dict(files={str(i): 'digest' for i in range(204)}), features=dict(passed=True))))
        r.receipt('macos-test', reports=dict(features=dict(passed=True)))
        with patch.object(r, 'check_handoff'):
            c.notes()
        body = r.notes.read_text()
        self.assertIn('204 recorded parity outputs passed', body)
        self.assertIn('Pending checks: `source-tests`', body)
        self.assertIn('`macos-fresh-offline`', body)
        self.assertNotIn('visual-review', body)
        self.assertNotIn('REPLACE_', body)
        self.assertNotIn(str(self.base), body)
        r.receipt('macos-test', reports=dict(features=dict(passed=False)))
        with patch.object(r, 'check_handoff'), self.assertRaisesRegex(ValueError, 'passing feature'):
            c.notes()

    def test_bad_settings_and_revision_cannot_be_interpreted_as_commands(self):
        for host in ('-oProxyCommand=evil', 'host;touch bad', 'host\ncommand'):
            self.config['linux']['host'] = host
            driver.write(self.profile, self.config)
            with self.assertRaisesRegex(ValueError, 'SSH host'):
                self.make()
        self.config['linux']['host'] = 'gpu-test'
        for path in ('relative/path', '/work/../elsewhere', '/work/$(touch bad)'):
            self.config['linux']['root'] = path
            driver.write(self.profile, self.config)
            with self.assertRaisesRegex(ValueError, 'Linux paths'):
                self.make()
        self.config['linux']['root'] = '/work/releases'
        driver.write(self.profile, self.config)
        with self.assertRaisesRegex(ValueError, 'Git option'):
            self.make('--commit=--help')

    def test_remote_prepare_rejects_wrong_gpu_and_existing_unowned_directory(self):
        c = self.make()
        remote = self.base/'remote'
        argv = ['remote', str(remote), json.dumps(c.manifest)]
        with patch.object(sys, 'argv', argv), patch.object(coordinator.driver.platform, 'system', return_value='Linux'), \
             patch.object(coordinator.driver.platform, 'machine', return_value='x86_64'), \
             patch.object(subprocess, 'check_output', return_value='NVIDIA other GPU'):
            with self.assertRaisesRegex(SystemExit, 'RTX PRO 5000'):
                exec(coordinator.REMOTE_PREPARE, {})
        self.assertFalse(remote.exists())
        remote.mkdir()
        with patch.object(sys, 'argv', argv), patch.object(coordinator.driver.platform, 'system', return_value='Linux'), \
             patch.object(coordinator.driver.platform, 'machine', return_value='x86_64'), \
             patch.object(subprocess, 'check_output', return_value='NVIDIA RTX PRO 5000 72GB Blackwell'):
            with self.assertRaisesRegex(SystemExit, 'different settings'):
                exec(coordinator.REMOTE_PREPARE, {})
        self.assertFalse((remote/'candidate.json').exists())


if __name__ == '__main__':
    unittest.main()
