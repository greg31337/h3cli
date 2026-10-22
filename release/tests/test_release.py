#!/usr/bin/env python3
"""Release control tests: no GPU, credentials, network, signing or publication."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1]/'driver.py'
spec = importlib.util.spec_from_file_location('h3_release', MODULE)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h3cli-release-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = (Path(self.tmp.name)/'checkout with spaces').resolve()
        self.root.mkdir()
        self.config = self.root/'outputs/release/config.json'
        self.c = dict(schema=1, root=str(self.root), repo='example/h3cli', tag='v0.1.0',
                      commit='a'*40, run='v0.1.0-01', jobs=8, models='/models', model='/models/MiniMaxH3',
                      apple_identity='Developer ID Application: Example (TEAM123)', notary_profile='h3cli',
                      secret_key=str(Path(self.tmp.name)/'private.key'), linux_host='gpu-alias',
                      linux_root='/work/h3cli', linux_handoff='')
        release.write(self.config, self.c)
        self.r = release.Runner(self.config, root=self.root)
        self.r.base.mkdir(parents=True)

    def public_files(self, folder=None):
        folder = folder or self.r.stage
        folder.mkdir(parents=True)
        self.r.key.parent.mkdir(parents=True, exist_ok=True)
        self.r.key.write_text('public key fixture\n')
        for name in release.PUBLIC:
            (folder/name).write_text('fixture ' + name + '\n')
        (folder/'release-info.txt').write_text(self.r.info())
        (folder/'h3cli-release.pub').write_text(self.r.key.read_text())
        (folder/'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n, h in release.hashes(folder, release.PUBLIC).items()))
        (folder/'SHA256SUMS.minisig').write_text('signature fixture\n')
        return folder

    def handoff(self):
        directory = self.r.incoming
        directory.mkdir(parents=True)
        for name in (release.LINUX[0], release.LINUX[2]):
            (directory/name).write_text(name)
            (directory/(name+'.sha256')).write_text(release.sha(directory/name)+'  '+name+'\n')
        release.write(directory/'build.json', dict(commit=self.c['commit'], dirty=False))
        digest = release.sha(directory/release.LINUX[0])
        q = dict(identity=self.r.identity, artifacts=release.hashes(directory, release.LINUX), reports={
            'parity': dict(passed=True, files={str(i): 'hash' for i in range(204)}),
            'distros': dict(passed=True, checks=[dict(passed=True, output_count=204) for _ in range(4)]),
            'runtime': dict(passed=True, release_sha256=digest),
            'features': dict(passed=True, binary_sha256=digest)})
        release.write(directory/'qualification.json', q)
        return directory, q

    def test_checksum_manifest_roundtrip_and_tampering(self):
        directory = self.public_files()
        release.verify_checksums(directory, 'SHA256SUMS', release.PUBLIC)
        (directory/release.LINUX[0]).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
            release.verify_checksums(directory, 'SHA256SUMS', release.PUBLIC)

    def test_checksum_manifest_rejects_traversal_duplicates_omissions(self):
        directory = self.public_files()
        p = directory/'SHA256SUMS'
        original = p.read_text()
        for bad in (original + original.splitlines()[0]+'\n',
                    original.replace('h3cli-linux-x86_64\n', '../secret\n'),
                    '\n'.join(original.splitlines()[1:])+'\n'):
            p.write_text(bad)
            with self.assertRaises(ValueError):
                release.checksum_list(p, release.PUBLIC)

    def test_public_verification_rejects_extra_file(self):
        directory = self.public_files()
        with patch.object(self.r, 'command'):
            self.r.verify_directory(directory)
            (directory/'private.key').write_text('must not upload')
            with self.assertRaisesRegex(ValueError, 'Unexpected/missing'):
                self.r.verify_directory(directory)

    def test_public_verification_rejects_other_release(self):
        directory = self.public_files()
        (directory/'release-info.txt').write_text('other version')
        (directory/'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n, h in release.hashes(directory, release.PUBLIC).items()))
        with patch.object(self.r, 'command'), self.assertRaisesRegex(ValueError, 'Wrong release'):
            self.r.verify_directory(directory)

    def test_public_verification_stops_on_bad_signature_before_checksums(self):
        self.public_files()
        with patch.object(self.r, 'command', side_effect=ValueError('Bad signature')), \
             patch.object(release, 'verify_checksums') as checksums:
            with self.assertRaisesRegex(ValueError, 'Bad signature'):
                self.r.verify_directory(self.r.stage)
            checksums.assert_not_called()

    def test_artifact_symlink_rejected(self):
        self.public_files()
        p = self.r.stage/release.LINUX[0]
        p.unlink()
        p.symlink_to(self.r.key)
        with self.assertRaisesRegex(ValueError, 'linked'):
            release.hashes(self.r.stage, release.PUBLIC)

    def test_linux_handoff_requires_all_results_and_same_identity(self):
        directory, q = self.handoff()
        self.r.check_handoff()
        for change in ('identity', 'count', 'failed', 'binary', 'distro'):
            bad = json.loads(json.dumps(q))
            if change == 'identity':
                bad['identity']['commit'] = 'b'*40
            if change == 'count':
                bad['reports']['parity']['files'].pop('0')
            if change == 'failed':
                bad['reports']['features']['passed'] = False
            if change == 'binary':
                bad['reports']['runtime']['release_sha256'] = '0'*64
            if change == 'distro':
                bad['reports']['distros']['checks'].pop()
            release.write(directory/'qualification.json', bad)
            with self.assertRaises(ValueError, msg=change):
                self.r.check_handoff()

    def test_receipt_detects_stale_config_and_modified_artifact(self):
        p = self.root/'file'
        p.write_text('original')
        self.r.receipt('build', [p])
        self.assertTrue(self.r.completed('build'))
        p.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Files changed'):
            self.r.completed('build')
        p.write_text('original')
        self.config.write_text(self.config.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'Stale completion'):
            self.r.completed('build')

    def test_config_rejects_path_traversal(self):
        self.c['run'] = '../../escape'
        release.write(self.config, self.c)
        with self.assertRaisesRegex(ValueError, 'Invalid run'):
            release.Runner(self.config, root=self.root)

    def test_dirty_or_wrong_commit_fails(self):
        with patch.object(self.r, 'command', side_effect=[self.c['commit'], ' M src/h3cli.c']):
            with self.assertRaisesRegex(ValueError, 'clean checkout'):
                self.r.clean()
        with patch.object(self.r, 'command', return_value='b'*40):
            with self.assertRaisesRegex(ValueError, 'moved away'):
                self.r.clean()

    def test_existing_partial_build_is_preserved(self):
        self.r.linux.mkdir(parents=True)
        (self.r.linux/'partial').write_text('keep evidence')
        with patch.object(self.r, 'host'), patch.object(self.r, 'clean'), patch.object(self.r, 'command') as command:
            with self.assertRaisesRegex(ValueError, 'Existing incomplete'):
                self.r.build('linux')
            command.assert_not_called()
        self.assertEqual((self.r.linux/'partial').read_text(), 'keep evidence')

    def test_completed_build_is_verified_without_rebuild(self):
        self.r.linux.mkdir(parents=True)
        p = self.r.linux/release.LINUX[0]
        p.write_text('built')
        self.r.receipt('linux-build', [p])
        with patch.object(self.r, 'host'), patch.object(self.r, 'clean'), patch.object(self.r, 'command') as command:
            self.r.build('linux')
            command.assert_not_called()

    def test_dry_run_does_not_spawn_or_write(self):
        r = release.Runner(self.config, dry=True, root=self.root)
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        with patch.object(subprocess, 'run', side_effect=AssertionError('spawn')), \
             patch.object(subprocess, 'check_output', side_effect=AssertionError('spawn')), \
             contextlib.redirect_stdout(io.StringIO()):
            for target in release.STEPS:
                r.prerequisites(target)
                r.build(target)
                r.models(target)
                r.test(target)
            r.sign()
            r.export()
            r.collect()
            r.stage_files()
            r.draft()
            r.verify()
            r.publish('stable')
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))

    def test_publish_needs_explicit_channel(self):
        with patch.object(self.r, 'command') as command:
            with self.assertRaisesRegex(ValueError, 'Choose --channel'):
                self.r.publish(None)
            command.assert_not_called()

    def test_stable_requires_manual_fresh_trust_evidence(self):
        self.r.receipt('macos-stage')
        self.r.receipt('macos-verify')
        self.notes()
        with patch.object(self.r, 'clean'), patch.object(self.r, 'gh') as gh:
            with self.assertRaisesRegex(ValueError, 'manual-source-tests'):
                self.r.publish('stable')
            gh.assert_not_called()

    def notes(self):
        self.r.notes.write_text('Release notes without placeholders')

    def test_publish_rechecks_downloads_and_refuses_public_release(self):
        self.notes()
        for name in ('macos-stage', 'macos-verify', *('manual-'+c for c in release.MANUAL)):
            self.r.receipt(name)
        with patch.object(self.r, 'clean'), patch.object(self.r, 'verify') as verify, \
             patch.object(self.r, 'remote_release', return_value={'draft': False}), \
             patch.object(self.r, 'gh') as gh:
            with self.assertRaisesRegex(ValueError, 'existing draft'):
                self.r.publish('stable')
            verify.assert_called_once()
            gh.assert_not_called()

    def test_prerelease_must_disclose_deferred_check_ids(self):
        self.notes()
        for name in ('macos-stage', 'macos-verify', 'manual-source-tests'):
            self.r.receipt(name)
        with patch.object(self.r, 'clean'), patch.object(self.r, 'verify'), \
             patch.object(self.r, 'remote_release', return_value={'draft': True}), patch.object(self.r, 'gh') as gh:
            with self.assertRaisesRegex(ValueError, 'pending check IDs'):
                self.r.publish('prerelease')
            gh.assert_not_called()

    def test_publication_no_longer_requires_video_review(self):
        self.public_files()
        self.notes()
        for name in ('macos-stage', 'macos-verify', *('manual-'+c for c in release.MANUAL)):
            self.r.receipt(name)
        self.assertNotIn('visual-review', release.MANUAL)
        self.assertFalse((self.r.base/'manual-visual-review.json').exists())
        with patch.object(self.r, 'clean'), patch.object(self.r, 'verify'), \
             patch.object(self.r, 'remote_release', return_value={'draft': True}), patch.object(self.r, 'gh') as gh:
            for channel in ('stable', 'prerelease'):
                self.r.publish(channel)
                self.assertTrue(any('--draft=false' in call.args for call in gh.call_args_list))

    def test_collection_uses_configured_ssh_key(self):
        self.r.c['linux_identity_file'] = '/path with spaces/release.key'
        with patch.object(self.r, 'clean'), patch.object(self.r, 'check_handoff'), \
             patch.object(self.r, 'receipt'), patch.object(self.r, 'command') as command:
            self.r.collect()
        argv = command.call_args.args[0]
        self.assertEqual(argv[argv.index('-i')+1], self.r.c['linux_identity_file'])
        self.assertIn('BatchMode=yes', argv)

    def test_remote_tag_mismatch_rejected(self):
        for output in ('', 'b'*40+'\trefs/tags/v0.1.0\n'):
            with patch.object(self.r, 'command', return_value=output):
                with self.assertRaisesRegex(ValueError, 'tag moved'):
                    self.r.remote_tag_check()
        with patch.object(self.r, 'command', return_value='b'*40+'\trefs/tags/v0.1.0\n'+'a'*40+'\trefs/tags/v0.1.0^{}\n'):
            self.r.remote_tag_check()

    def test_notary_rejection_cannot_get_completion_receipt(self):
        self.r.receipt('macos-build')
        with patch.object(self.r, 'host'), patch.object(self.r, 'clean'), patch.object(self.r, 'mac_build_check'), \
             patch.object(self.r, 'command', side_effect=ValueError('Rejected by Apple')):
            with self.assertRaisesRegex(ValueError, 'Rejected by Apple'):
                self.r.sign()
        self.assertFalse((self.r.base/'macos-sign.json').exists())

    def test_draft_retry_uploads_only_missing_matching_assets(self):
        self.public_files()
        self.notes()
        self.r.receipt('macos-stage', [self.r.stage/n for n in release.ASSETS])
        existing_names = [release.LINUX[0], 'SHA256SUMS']
        existing = {'draft': True, 'assets': [{'name': n} for n in existing_names]}
        def github(*args, **kwargs):
            if args[:2] == ('release', 'download'):
                directory = Path(args[args.index('--dir')+1])
                for name in existing_names:
                    (directory/name).write_bytes((self.r.stage/name).read_bytes())
        with patch.object(self.r, 'clean'), patch.object(self.r, 'tag'), \
             patch.object(self.r, 'verify_directory'), patch.object(self.r, 'remote_release', return_value=existing), \
             patch.object(self.r, 'gh', side_effect=github) as gh:
            self.r.draft()
        uploads = [c.args for c in gh.call_args_list if c.args[:2] == ('release', 'upload')]
        self.assertEqual(len(uploads), 1)
        self.assertEqual({Path(p).name for p in uploads[0][3:]}, set(release.ASSETS)-set(existing_names))
        self.assertTrue(self.r.completed('macos-draft'))

    def test_draft_retry_refuses_different_remote_bytes(self):
        self.public_files()
        self.notes()
        self.r.receipt('macos-stage')
        existing = {'draft': True, 'assets': [{'name': release.LINUX[0]}]}
        def github(*args, **kwargs):
            if args[:2] == ('release', 'download'):
                (Path(args[args.index('--dir')+1])/release.LINUX[0]).write_text('different')
        with patch.object(self.r, 'clean'), patch.object(self.r, 'tag'), \
             patch.object(self.r, 'verify_directory'), patch.object(self.r, 'remote_release', return_value=existing), \
             patch.object(self.r, 'gh', side_effect=github) as gh:
            with self.assertRaisesRegex(ValueError, 'draft bytes differ'):
                self.r.draft()
        self.assertFalse(any(c.args[:2] == ('release', 'upload') for c in gh.call_args_list))
        self.assertFalse((self.r.base/'macos-draft.json').exists())

    def test_draft_refuses_existing_public_release_before_tag_push(self):
        self.public_files()
        self.notes()
        self.r.receipt('macos-stage')
        with patch.object(self.r, 'clean'), patch.object(self.r, 'verify_directory'), \
             patch.object(self.r, 'remote_release', return_value={'draft': False}), \
             patch.object(self.r, 'tag') as tag:
            with self.assertRaisesRegex(ValueError, 'already public'):
                self.r.draft()
            tag.assert_not_called()

    def test_apple_team_must_match_config(self):
        with patch.object(self.r, 'command', return_value='TeamIdentifier=OTHERTEAM'):
            with self.assertRaisesRegex(ValueError, 'signing team'):
                self.r.apple_verify(self.r.mac)
        with patch.object(self.r, 'command', return_value='TeamIdentifier=TEAM123'):
            self.r.apple_verify(self.r.mac)

    def test_failed_transfer_is_not_published_and_can_retry(self):
        self.r.incoming.parent.mkdir(parents=True)
        with patch.object(self.r, 'clean'), patch.object(self.r, 'command', side_effect=ValueError('transfer failed')):
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, 'transfer failed'):
                    self.r.collect()
        self.assertFalse(self.r.incoming.exists())
        self.assertEqual(len(list(self.r.incoming.parent.glob('incoming-attempt-*'))), 2)

    def test_configuration_is_idempotent_and_detects_old_model_name(self):
        model_root = self.root/'models with spaces'
        (model_root/'MiniMax-H3').mkdir(parents=True)
        destination = self.root/'outputs/new-config.json'
        args = release.parser().parse_args(['configure', '--repo', 'example/h3cli', '--tag', 'v0.2.0',
                                           '--models-path', str(model_root), '--config', str(destination)])
        with patch.object(release, 'ROOT', self.root), patch.object(subprocess, 'check_output', return_value='c'*40+'\n'):
            release.configure(args)
            release.configure(args)
            self.assertEqual(release.read(destination)['model'], str(model_root/'MiniMax-H3'))
            args.jobs = 16
            with self.assertRaisesRegex(ValueError, 'already exists'):
                release.configure(args)

    def test_stage_sign_and_download_flow_uses_exact_artifact_bytes(self):
        directory, _ = self.handoff()
        self.r.receipt('macos-collect', [directory/n for n in (*release.LINUX, 'build.json', 'qualification.json')])
        self.r.key.parent.mkdir(parents=True)
        self.r.key.write_text('trusted public key fixture')
        (self.r.key.parent/'release-notes-template.md').write_text('Release REPLACE_VERSION commit REPLACE_FULL_COMMIT_ID')
        self.r.dev.mkdir(parents=True)
        self.r.mac.mkdir()
        (self.root/'Makefile').write_text('fixture source')
        for name in (*release.MACOS, 'libh3.a'):
            (self.r.dev/name).write_text('development '+name)
            (self.r.mac/name).write_text('signed '+name)
        def artifacts(folder):
            return {n: {'sha256': release.sha(folder/n)} for n in (*release.MACOS, 'libh3.a')}
        release.write(self.r.dev/'build.json', dict(source={'files': {'Makefile': release.sha(self.root/'Makefile')}},
                                                  artifacts=artifacts(self.r.dev)))
        release.write(self.r.mac/'build.json', dict(input_build_sha256=release.sha(self.r.dev/'build.json'),
                     inner={'status': 'Accepted'}, outer={'status': 'Accepted'}, artifacts=artifacts(self.r.mac)))
        self.r.receipt('macos-test', [self.r.mac/n for n in release.MACOS])
        def command(argv, **kwargs):
            if argv[:2] == ['minisign', '-Sm']:
                Path(str(argv[2])+'.minisig').write_text('test signature')
            if argv[:2] == ['codesign', '--display']:
                return 'TeamIdentifier=TEAM123'
            return ''
        with patch.object(self.r, 'clean'), patch.object(self.r, 'command', side_effect=command):
            self.r.stage_files()
            self.assertEqual(set(p.name for p in self.r.stage.iterdir()), set(release.ASSETS))
            self.assertEqual((self.r.stage/release.MACOS[0]).read_bytes(), (self.r.mac/release.MACOS[0]).read_bytes())
            self.assertTrue(self.r.completed('macos-stage'))
            # A completed stage is verified and reused without signing it again.
            self.r.stage_files()
            self.r.receipt('macos-draft')
            remote = {'draft': True, 'assets': [{'name': n} for n in release.ASSETS]}
            def download(*args, **kwargs):
                dest = Path(args[args.index('--dir')+1])
                for name in release.ASSETS:
                    (dest/name).write_bytes((self.r.stage/name).read_bytes())
            with patch.object(self.r, 'remote_release', return_value=remote), \
                 patch.object(self.r, 'remote_tag_check'), patch.object(self.r, 'gh', side_effect=download):
                self.r.verify()
            self.assertTrue(self.r.completed('macos-verify'))
            verified = Path(release.read(self.r.base/'macos-verify.json')['download'])
            (verified/release.MACOS[0]).write_text('changed downloaded bytes')
            with self.assertRaisesRegex(ValueError, 'Files changed'):
                self.r.completed('macos-verify')

    def test_shell_metacharacters_remain_single_argv_values(self):
        value = '/path with spaces/$(touch NEVER); `echo x`'
        with patch.object(release.build_process, 'run', return_value=subprocess.CompletedProcess(['tool'],0,'ok','')) as run:
            self.r.command(['tool', value], capture=True)
        self.assertEqual(run.call_args.args[0], ['tool', value])
        self.assertNotIn('shell', run.call_args.kwargs)

    def test_build_requires_explicit_qualified_dependencies(self):
        with patch.object(self.r,'host'),patch.object(self.r,'clean'),patch.object(self.r,'command') as run:
            with self.assertRaisesRegex(ValueError,'qualified --dependencies'):self.r.build('linux')
            run.assert_not_called()

    def test_build_dependency_identity_is_checked(self):
        pin=dict(platform='linux-x86_64',recipe_id='a'*64,archive=dict(sha256='b'*64),qualification=dict(sha256='c'*64))
        self.r.c['dependency_bundle']=pin
        row=dict(dependencies=dict(mode='qualified',recipe_id='a'*64,archive_sha256='b'*64,qualification_sha256='c'*64))
        self.r.dependency_check(row,'linux')
        for key in ('mode','recipe_id','archive_sha256','qualification_sha256'):
            changed=dict(dependencies=row['dependencies']|{key:'different'})
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'unqualified dependencies'):
                self.r.dependency_check(changed,'linux')

    def test_handoff_checks_pinned_dependency_reference(self):
        directory,q=self.handoff()
        self.r.c['linux_dependency_reference_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'dependency reference'):self.r.check_handoff(directory)
        q['dependency_reference_sha256']='d'*64;release.write(directory/'qualification.json',q)
        self.r.check_handoff(directory)

    def test_repo_normalization(self):
        for value in ('example/h3cli', 'https://github.com/example/h3cli.git', 'git@github.com:example/h3cli.git'):
            self.assertEqual(release.repo_name(value), 'example/h3cli')
        for value in ('--upload-pack=evil', 'https://other.invalid/example/h3cli', 'example/h3cli;cmd'):
            with self.assertRaises(ValueError):
                release.repo_name(value)

    def test_all_shell_entrypoints_are_executable_and_syntax_valid(self):
        for script in MODULE.parent.rglob('*.sh'):
            self.assertTrue(os.access(script, os.X_OK), script)
            subprocess.run(['bash', '-n', str(script)], check=True)

    def test_prerequisite_bootstrap_dry_run_needs_no_configuration(self):
        for target in ('linux', 'macos'):
            script = MODULE.parent/target/'step1-prerequisites.sh'
            result = subprocess.run([str(script), '--install', '--dry-run', '--config', str(self.root/'absent.json')],
                                    cwd=self.tmp.name, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((self.root/'absent.json').exists())
            self.assertIn('apt-get' if target=='linux' else 'brew install', result.stdout)


class SigningSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h3-signing-setup-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base/'checkout'
        self.root.mkdir()
        self.config = self.root/'outputs/release/config.json'
        self.settings = self.root/'outputs/release/machines.json'
        self.secret = self.base/'private keys/release.key'

    def runner(self, *argv):
        args = release.parser().parse_args(['setup-signing', *argv])
        return release.SigningRunner(args, root=self.root), args

    def test_missing_config_dry_run_previews_login_and_key_without_side_effects(self):
        before = list(self.root.rglob('*'))
        with patch.object(release, 'ROOT', self.root), \
             patch.object(subprocess, 'run', side_effect=AssertionError('external command')), \
             patch.object(subprocess, 'check_output', side_effect=AssertionError('external command')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            release.main(['setup-signing', '--github-login', '--generate-key', '--dry-run',
                          '--secret-key', str(self.secret)])
        self.assertIn('gh auth login', output.getvalue())
        self.assertIn('minisign -G', output.getvalue())
        self.assertIn('--keychain-profile h3cli', output.getvalue())
        self.assertEqual(before, list(self.root.rglob('*')))
        self.assertFalse(self.secret.parent.exists())
        self.assertFalse(self.config.exists())

    def test_bootstrap_uses_machine_settings_and_copies_only_public_key(self):
        release.write(self.settings, dict(schema=1, macos=dict(
            notary_profile='test-profile', secret_key=str(self.secret))))
        calls = []
        def command(runner, argv, **kwargs):
            calls.append(list(map(str, argv)))
            if argv[:2] == ['minisign', '-G']:
                Path(argv[argv.index('-s')+1]).write_text('secret fixture')
                Path(argv[argv.index('-p')+1]).write_text('public fixture')
            return ''
        with patch.object(release, 'ROOT', self.root), patch.object(release.Runner, 'host'), \
             patch.object(release.SigningRunner, 'command', command):
            release.main(['setup-signing', '--github-login', '--generate-key'])
        self.assertEqual(self.secret.read_text(), 'secret fixture')
        self.assertEqual((self.root/'docs/release/h3cli-release.pub').read_text(), 'public fixture')
        self.assertFalse(self.config.exists())
        self.assertIn(['xcrun', 'notarytool', 'history', '--keychain-profile', 'test-profile'], calls)
        self.assertIn(['gh', 'auth', 'login', '--hostname', 'github.com', '--git-protocol', 'https', '--web'], calls)
        self.assertFalse(any(p.read_text() == 'secret fixture' for p in self.root.rglob('*') if p.is_file()))

    def test_defaults_need_neither_release_nor_machine_config(self):
        runner, _ = self.runner()
        self.assertEqual(runner.c['notary_profile'], 'h3cli')
        self.assertEqual(runner.c['secret_key'], str(Path('~/.config/h3cli-release/h3cli-release.key').expanduser().resolve()))

    def test_existing_release_config_takes_precedence_over_machine_settings(self):
        release.write(self.config, dict(schema=1, root=str(self.root),
                      notary_profile='release-profile', secret_key=str(self.secret)))
        release.write(self.settings, dict(schema=1, macos=dict(notary_profile='machine-profile')))
        runner, _ = self.runner()
        self.assertEqual(runner.c['notary_profile'], 'release-profile')
        self.assertEqual(runner.c['secret_key'], str(self.secret))

    def test_explicit_settings_and_overrides_take_precedence(self):
        release.write(self.config, dict(schema=1, root=str(self.root), notary_profile='release-profile'))
        custom = self.base/'custom-machines.json'
        release.write(custom, dict(schema=1, macos=dict(notary_profile='machine-profile')))
        runner, _ = self.runner('--settings', str(custom))
        self.assertEqual(runner.c['notary_profile'], 'machine-profile')
        runner, _ = self.runner('--settings', str(custom), '--notary-profile', 'override',
                                '--secret-key', str(self.secret))
        self.assertEqual(runner.c, dict(notary_profile='override', secret_key=str(self.secret)))

    def test_explicit_missing_files_do_not_silently_use_defaults(self):
        release.write(self.settings, dict(schema=1, macos={}))
        for option in ('--config', '--settings'):
            with self.subTest(option=option), self.assertRaises(FileNotFoundError):
                self.runner(option, str(self.base/'missing.json'))

    def test_invalid_configuration_is_rejected(self):
        release.write(self.config, dict(schema=1, root='/another/checkout'))
        with self.assertRaisesRegex(ValueError, 'another checkout'):
            self.runner()
        release.write(self.settings, dict(schema=2, macos={}))
        with self.assertRaisesRegex(ValueError, 'Unsupported Mac'):
            self.runner('--settings', str(self.settings))
        with self.assertRaisesRegex(ValueError, 'Choose --config or --settings'):
            self.runner('--config', str(self.config), '--settings', str(self.settings))

    def test_existing_keys_are_rejected_before_interactive_login(self):
        runner, args = self.runner('--github-login', '--generate-key', '--secret-key', str(self.secret))
        for path in (self.secret, self.secret.with_suffix('.pub'), runner.key):
            with self.subTest(path=path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('existing key fixture')
                with patch.object(runner, 'host'), patch.object(runner, 'command') as command:
                    with self.assertRaisesRegex(ValueError, 'refusing to replace'):
                        release.setup_signing(runner, args)
                    command.assert_not_called()
                self.assertEqual(path.read_text(), 'existing key fixture')
                path.unlink()
        runner.key.symlink_to(self.base/'absent-public-key')
        with patch.object(runner, 'host'), patch.object(runner, 'command') as command:
            with self.assertRaisesRegex(ValueError, 'refusing to replace'):
                release.setup_signing(runner, args)
            command.assert_not_called()

    def test_secret_in_checkout_is_rejected_before_interactive_login(self):
        runner, args = self.runner('--github-login', '--generate-key', '--secret-key', str(self.root/'private.key'))
        with patch.object(runner, 'host'), patch.object(runner, 'command') as command:
            with self.assertRaisesRegex(ValueError, 'outside the checkout'):
                release.setup_signing(runner, args)
            command.assert_not_called()

    def test_missing_apple_account_arguments_fail_before_interactive_login(self):
        runner, args = self.runner('--github-login', '--store-credentials')
        with patch.object(runner, 'host'), patch.object(runner, 'command') as command:
            with self.assertRaisesRegex(ValueError, '--apple-id and --team-id'):
                release.setup_signing(runner, args)
            command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
