"""Failure/retry contracts of the seven-action release interface; no services."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

RELEASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RELEASE))
import common
import flow
import worker
import noexec


class FlowFixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h3-flow-')
        self.root=Path(self.tmp.name)
        self.patch=patch.object(flow,'ROOT',self.root);self.patch.start()

    def tearDown(self):
        self.patch.stop();self.tmp.cleanup()

    def profile(self,targets=('linux',)):
        return dict(schema=2,version='v0.0.1',repo='example/test',commit='a'*40,targets=list(targets),
                    machines={t:dict(root=str(self.root/t),models='/models',model='/models/main',jobs=2) for t in targets},
                    signing={},offline=False,channel='prerelease',notes='',public_key_sha256='x')


class FlowTests(FlowFixture):
    def test_exact_seven_public_actions(self):
        parser=flow.parser()
        self.assertEqual(set(parser._subparsers._group_actions[0].choices),
                         {'setup-signing','configure',*flow.STAGES})

    def test_dry_run_never_claims_pass_or_writes(self):
        profile=self.root/'profile.json';common.write(profile,self.profile())
        before=set(self.root.rglob('*'));out=io.StringIO()
        with contextlib.redirect_stdout(out),patch.object(flow,'run',side_effect=AssertionError('no actions')):
            self.assertEqual(flow.main(['publish','--profile',str(profile),'--dry-run']),0)
        self.assertIn('PLAN publish:',out.getvalue());self.assertNotIn('PASS',out.getvalue())
        self.assertEqual(set(self.root.rglob('*')),before)

    def test_missing_profile_failure_is_nonzero_and_retryable(self):
        output=io.StringIO()
        with contextlib.redirect_stderr(output):
            self.assertEqual(flow.main(['test-local','--profile',str(self.root/'missing')]),1)
        self.assertIn('FAIL test-local:',output.getvalue());self.assertIn('retry:',output.getvalue())
        self.assertIn('--profile '+str(self.root/'missing'),output.getvalue())
        logs=list((self.root/'outputs/release/logs').glob('*/error.log'))
        self.assertEqual(len(logs),1);self.assertIn('FAIL test-local:',logs[0].read_text())

    def test_failed_dry_run_is_also_read_only(self):
        before=set(self.root.rglob('*'))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(flow.main(['test-local','--dry-run','--profile',str(self.root/'missing')]),1)
        self.assertEqual(set(self.root.rglob('*')),before)

    def test_receipts_reject_changed_bytes_even_if_size_unchanged(self):
        file=self.root/'artifact';file.write_bytes(b'abc')
        marker=self.root/'receipt.json'
        common.receipt(marker,{'commit':'a'},{},[file])
        self.assertIsNotNone(common.completed(marker,{'commit':'a'}))
        file.write_bytes(b'xyz')
        self.assertIsNone(common.completed(marker,{'commit':'a'}))

    def test_receipts_reject_configuration_changes_and_failed_reports(self):
        marker=self.root/'receipt.json';common.receipt(marker,{'jobs':1},{},[])
        self.assertIsNone(common.completed(marker,{'jobs':2}))
        row=common.read(marker);row['passed']=False;common.write(marker,row)
        self.assertIsNone(common.completed(marker,{'jobs':1}))

    def test_later_stage_rejects_tampered_native_binary(self):
        f=flow.Flow(self.profile(),self.root/'profile.json');f.base.mkdir(parents=True)
        binary=f.base/'native';binary.write_text('binary')
        common.receipt(f.base/'build-local.json',f.inputs('build-local'),{},[binary])
        self.assertIn('build-local',f.inputs('test-local'))
        binary.write_text('different')
        with self.assertRaisesRegex(ValueError,'changed recorded'):f.inputs('test-local')

    def test_downstream_receipt_rejects_replaced_test_receipt(self):
        f=flow.Flow(self.profile(),self.root/'profile.json');f.base.mkdir(parents=True)
        common.receipt(f.base/'build-local.json',f.inputs('build-local'),{},[])
        common.receipt(f.base/'test-local.json',f.inputs('test-local'),{},[])
        common.receipt(f.base/'build-local.json',f.inputs('build-local'),{'new':'attempt'},[])
        with self.assertRaisesRegex(ValueError,'Stale test-local'):f.inputs('build-packages')

    def test_attempts_are_fresh_and_preserve_failure_logs(self):
        a=common.attempt(self.root,'test');(a/'failure.log').write_text('evidence')
        b=common.attempt(self.root,'test')
        self.assertNotEqual(a,b);self.assertEqual((a/'failure.log').read_text(),'evidence')

    def test_import_existing_machine_and_signing_settings(self):
        common.write(self.root/'outputs/release/machines.json',dict(schema=1,macos={'secret_key':'/private/key','models':'/models'},linux={'host':'cuda-test'}))
        row=flow.profile_defaults(self.root/'missing')
        self.assertEqual(row['signing']['secret_key'],'/private/key')
        self.assertEqual(set(row['machines']),{'macos','linux'})

    def test_single_platform_and_two_platform_jobs(self):
        for targets in [('macos',),('linux',),('macos','linux')]:
            f=flow.Flow(self.profile(targets),self.root/'profile.json')
            self.assertEqual(set(f.p['targets']),set(targets))
            self.assertNotIn('signing',f.job(targets[0]))

    def test_stage_summary_is_not_pass_when_selected_host_fails(self):
        profile=self.root/'profile.json';common.write(profile,self.profile(('macos','linux')))
        output=io.StringIO()
        with patch.object(flow.Flow,'prepare'),patch.object(flow.Flow,'worker',side_effect=ValueError('linux unavailable')),contextlib.redirect_stderr(output):
            self.assertEqual(flow.main(['build-local','--profile',str(profile)]),1)
        self.assertIn('FAIL build-local',output.getvalue());self.assertNotIn('PASS',output.getvalue())

    def test_cancel_returns_130(self):
        profile=self.root/'profile.json';common.write(profile,self.profile())
        with patch.object(flow.Flow,'execute',side_effect=KeyboardInterrupt),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(flow.main(['build-local','--profile',str(profile)]),130)

    def test_verified_retry_keeps_receipt_identity(self):
        f=flow.Flow(self.profile(),self.root/'profile.json');f.base.mkdir(parents=True)
        binary=self.root/'binary';binary.write_text('binary')
        result=dict(result={'binary':str(binary)},files=common.identities([binary]))
        with patch.object(f,'prepare'),patch.object(f,'worker',return_value=result),contextlib.redirect_stdout(io.StringIO()):
            f.execute('build-local');before=common.sha(f.base/'build-local.json')
            f.execute('build-local')
        self.assertEqual(before,common.sha(f.base/'build-local.json'))

    def test_remote_command_quotes_paths_and_uses_lifeline(self):
        with patch.object(flow,'run') as run:
            flow.remote({'host':'cuda-test'},['python3','/path with spaces/worker.py','--value','$(bad)'],supervised='/path with spaces')
        argv=run.call_args.args[0]
        import shlex
        decoded=shlex.split(argv[-1])
        self.assertEqual(decoded[:3],['python3','/path with spaces/scripts/build_process.py','--'])
        self.assertEqual(decoded[-1],'$(bad)')
        self.assertEqual(run.call_args.kwargs['stdin'],subprocess.PIPE)

    def test_draft_lookup_falls_back_to_all_release_pages(self):
        f=flow.Flow(self.profile(),self.root/'profile.json')
        draft={'tag_name':f.p['version'],'draft':True,'assets':[]}
        responses=[subprocess.CompletedProcess([],1,'','HTTP 404'),
                   subprocess.CompletedProcess([],0,json.dumps([[{'tag_name':'older'}],[draft]]),'')]
        with patch.object(flow.subprocess,'run',side_effect=responses) as run:
            self.assertEqual(f.remote_release(),draft)
        self.assertIn('--paginate',run.call_args.args[0])
        self.assertIn('--slurp',run.call_args.args[0])

    def test_failed_draft_listing_is_not_treated_as_a_missing_release(self):
        f=flow.Flow(self.profile(),self.root/'profile.json')
        responses=[subprocess.CompletedProcess([],1,'','HTTP 404'),
                   subprocess.CompletedProcess([],1,'','HTTP 403')]
        with patch.object(flow.subprocess,'run',side_effect=responses):
            with self.assertRaisesRegex(ValueError,'Cannot inspect GitHub drafts'):f.remote_release()

    def test_cleanup_retries_a_zombie_group_probe_until_child_is_reaped(self):
        process=Mock(pid=12345)
        probes=[]
        def killpg(pid,sig):
            if sig==0:
                probes.append(sig)
                if len(probes)==1:raise PermissionError(1,'unreaped zombie group')
                raise ProcessLookupError(3,'group reaped')
        with patch.object(common.build_process.os,'killpg',side_effect=killpg),patch.object(common.build_process.time,'sleep'):
            common.build_process._stop(process)
        self.assertEqual(process.poll.call_count,2)
        process.wait.assert_called_once_with()

    def test_cleanup_does_not_hide_permission_failure_sending_term(self):
        with patch.object(common.build_process.os,'killpg',side_effect=PermissionError(1,'denied')):
            with self.assertRaises(PermissionError):common.build_process._stop(Mock(pid=12345))

    def test_dependency_warm_hit_avoids_build_and_qualification(self):
        pin=self.root/'bundle.json';common.write(pin,{})
        job={'target':'macos','dependency_cache':str(self.root/'cache'),'dependencies':str(pin)}
        with patch.object(worker.bundles,'recipe',return_value={}),patch.object(worker.bundles,'recipe_id',return_value='r'),\
             patch.object(worker.bundles,'reference',return_value={'recipe_id':'r','qualification':{}}),\
             patch.object(worker.bundles,'load',return_value=(self.root,{})),patch.object(worker,'run',side_effect=AssertionError('must reuse')):
            self.assertEqual(worker.dependencies(job,self.root)[0],pin)

    def test_offline_dependency_miss_fails_without_fetch(self):
        job={'target':'macos','dependency_cache':str(self.root/'cache'),'offline':True}
        with patch.object(worker.bundles,'recipe',return_value={}),patch.object(worker,'run',side_effect=AssertionError('no network')):
            with self.assertRaisesRegex(ValueError,'offline cache'):worker.dependencies(job,self.root)

    def test_snapshot_excludes_dirty_and_untracked_changes(self):
        def git(*argv):return subprocess.check_output(['git','-C',str(self.root),*argv],text=True,stderr=subprocess.DEVNULL).strip()
        git('init');git('config','user.name','Test');git('config','user.email','test@example.invalid')
        (self.root/'release').mkdir();(self.root/'release/flow.py').write_text('# committed\n')
        git('add','.');git('commit','-qm','fixture');commit=git('rev-parse','HEAD')
        (self.root/'release/flow.py').write_text('# dirty\n');(self.root/'private.txt').write_text('untracked')
        base=self.root/'outputs';base.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):source=flow.snapshot(dict(commit=commit),base)
        self.assertEqual((source/'release/flow.py').read_text(),'# committed\n')
        self.assertFalse((source/'private.txt').exists())

    def test_empty_legacy_identity_uses_discovered_default(self):
        import argparse
        self.assertEqual(flow.value(argparse.Namespace(apple_identity=None),{'apple_identity':''},
                                    'apple_identity','discovered'),'discovered')

    def test_apple_verifier_reads_identity_from_codesign_stderr(self):
        profile=self.profile(('macos',));profile['signing']['apple_identity']='Developer ID Application: Test (TEAM)'
        f=flow.Flow(profile,self.root/'profile.json')
        def command(argv,**kwargs):
            if kwargs.get('capture'):
                return common.run([sys.executable,'-c','import sys;sys.stderr.write("TeamIdentifier=TEAM\\n")'],
                                  capture=True,stderr=kwargs['stderr'])
        with patch.object(flow,'run',side_effect=command),contextlib.redirect_stdout(io.StringIO()):
            f.apple_verify(self.root)

    def test_scratch_falls_back_from_work_volume_to_cache(self):
        job=dict(target='linux',root=str(self.root/'work'),dependency_cache=str(self.root/'cache/bundles'),models=str(self.root/'models'))
        count=0
        def clone(destination,operation,source):
            nonlocal count
            count+=1
            if count==1:raise OSError('No reflinks on work volume')
            os.write(destination,os.pread(source,100,0))
        with patch.object(worker.fcntl,'ioctl',side_effect=clone):
            chosen=worker.scratch_root(job)
        self.assertEqual(chosen,self.root/'cache/native-scratch')
        self.assertFalse(list(chosen.glob('clone-check-*')))

    def test_no_cloning_volume_fails_configuration(self):
        job=dict(target='linux',root=str(self.root/'work'),dependency_cache=str(self.root/'cache/bundles'),models=str(self.root/'models'))
        with patch.object(worker.fcntl,'ioctl',side_effect=OSError('unsupported')):
            with self.assertRaisesRegex(ValueError,'cloning-capable'):worker.scratch_root(job)

    def test_native_libraries_include_locked_media_but_keep_host_glibc(self):
        from types import SimpleNamespace
        import re
        source=self.root/'source';(source/'bin').mkdir(parents=True);(source/'bin/h3cli').write_bytes(b'ELF')
        work=self.root/'work';lib=work/'builder/rootfs/usr/lib/x86_64-linux-gnu';lib.mkdir(parents=True)
        (lib/'libx264.so.163').write_bytes(b'locked x264 bytes')
        openssl=work/'builder/rootfs/usr/bin/openssl';openssl.parent.mkdir();openssl.write_bytes(b'locked OpenSSL tool')
        package=SimpleNamespace(HOST=re.compile(r'libc\.so\.6'),elf=lambda p:True,nvidia_library=lambda n:False,
            dynamic=lambda p:'(NEEDED) [libx264.so.163]\n(NEEDED) [libc.so.6]\n(NEEDED) [libstdc++.so.6]' if p.name=='h3cli' else '')
        with patch.object(worker,'module',return_value=package):out=worker.host_libraries(source,work)
        self.assertEqual((out/'libx264.so.163').read_bytes(),b'locked x264 bytes')
        self.assertFalse((out/'libc.so.6').exists());self.assertFalse((out/'libstdc++.so.6').exists())

    def test_noexec_scratch_falls_back_to_an_existing_mount(self):
        from types import SimpleNamespace
        first=self.root/'exec';second=self.root/'noexec';first.mkdir();second.mkdir()
        with patch.object(noexec.os,'statvfs',side_effect=lambda p:SimpleNamespace(f_flag=0 if p==first else getattr(os,'ST_NOEXEC',8))):
            self.assertEqual(noexec.scratch([first,second]),second)

    def test_noexec_rejects_changed_artifact_before_running_it(self):
        binary=self.root/'binary';binary.write_bytes(b'changed')
        with patch.object(noexec,'scratch',side_effect=AssertionError('must verify first')):
            with self.assertRaisesRegex(ValueError,'executable changed'):
                noexec.main(['--binary',str(binary),'--sha256','a'*64,'--package-module',str(self.root/'package.py'),
                             '--report',str(self.root/'result.json')])

    def test_added_package_check_replaces_the_old_receipt(self):
        f=flow.Flow(self.profile(),self.root/'profile.json');f.base.mkdir(parents=True)
        (self.root/'release').mkdir();(self.root/'release/worker.py').write_text('verifier')
        binary=f.base/'binary';binary.write_bytes(b'final')
        proof=f.base/'additional-check.json';common.write(proof,dict(passed=True))
        common.write(f.base/'build-packages.json',dict(result={'linux':{'binary':str(binary)}}))
        row=dict(result={'binary_sha256':common.sha(binary)},files={})
        with patch.object(f,'prepare'),patch.object(f,'inputs',return_value={}),patch.object(f,'verify_packages'),\
             patch.object(f,'files',return_value=[binary]),patch.object(f,'worker',return_value=row),\
             patch.object(f,'linux_noexec',side_effect=[[],[proof]]),contextlib.redirect_stdout(io.StringIO()):
            f.execute('test-packages');before=common.sha(f.base/'test-packages.json')
            f.execute('test-packages')
        self.assertNotEqual(before,common.sha(f.base/'test-packages.json'))
        self.assertIn(str(proof.resolve()),common.read(f.base/'test-packages.json')['files'])

    def test_package_cache_paths_follow_each_launcher_protocol(self):
        info=dict(cache='/cache',runtime_id='runtime-id')
        self.assertEqual(worker.extracted_runtime(info,'macos'),Path('/cache/runtime-id'))
        self.assertEqual(worker.extracted_runtime(info,'linux'),Path('/cache'))

    def test_package_verifier_uses_selected_source_without_modifying_it(self):
        source=self.root/'selected/release';source.mkdir(parents=True)
        (source/'common.py').write_text('ROOT="selected source"\n')
        verifier=self.root/'verifier.py';verifier.write_text('import common,sys;print(common.ROOT,sys.argv[1])\n')
        p=subprocess.run([sys.executable,'-c',flow.PACKAGE_WORKER,str(source),str(verifier),'test-packages'],capture_output=True,text=True,check=True)
        self.assertEqual(p.stdout.strip(),'selected source test-packages')


class PublicationTests(FlowFixture):
    def publication(self,existing=None):
        f=flow.Flow(self.profile(),self.root/'profile.json');f.base.mkdir(parents=True);f.packages.mkdir()
        asset=f.packages/'h3cli-linux-x86_64';asset.write_bytes(b'tested bytes')
        notes=f.base/'release-notes.md';notes.write_text('complete notes')
        state={'release':existing,'uploads':0,'edits':0,'checks':0}
        def gh(*argv,**kw):
            if argv[:2]==('release','create'):
                state['release']=dict(draft=True,prerelease=True,body=notes.read_text(),assets=[],html_url='https://github.com/example/test/releases/tag/v0.0.1')
            elif argv[:2]==('release','upload'):
                state['uploads']+=1;state['release']['assets'].append({'name':Path(argv[-1]).name})
            elif argv[:2]==('release','edit'):
                state['edits']+=1;state['release']['draft']=False
            elif argv[:2]==('release','download'):
                target=Path(argv[-1])
                for item in state['release']['assets']:shutil.copy2(f.packages/item['name'],target/item['name'])
        f.gh=gh;f.remote_release=lambda:state['release'];f.notes=lambda:notes
        f.verify_packages=lambda *a:None;f.tag=lambda **kw:None
        f.worker=lambda *a,**kw:state.__setitem__('checks',state['checks']+1)
        return f,state

    def test_publish_creates_verifies_and_finishes_public(self):
        f,state=self.publication();row=f.publish({})
        self.assertTrue(row['passed']);self.assertFalse(state['release']['draft'])
        self.assertEqual((state['uploads'],state['edits'],state['checks']),(1,1,1))

    def test_partial_draft_uploads_only_missing_assets(self):
        f,state=self.publication(dict(draft=True,prerelease=True,body='complete notes',assets=[],html_url='url'))
        f.publish({});self.assertEqual(state['uploads'],1)

    def test_empty_failed_upload_is_retried_in_matching_draft(self):
        asset=dict(name='h3cli-linux-x86_64',state='starter',size=0,id=12)
        f,state=self.publication(dict(draft=True,prerelease=True,body='complete notes',assets=[asset],html_url='url'))
        def remove(argv,**kw):
            self.assertEqual(argv,['gh','api','--method','DELETE','repos/example/test/releases/assets/12'])
            state['release']['assets'].remove(asset)
        with patch.object(flow,'run',side_effect=remove):f.publish({})
        self.assertEqual(state['uploads'],1);self.assertFalse(state['release']['draft'])

    def test_incomplete_public_or_nonempty_asset_is_never_deleted(self):
        for draft,size in ((False,0),(True,5)):
            with self.subTest(draft=draft,size=size):
                asset=dict(name='h3cli-linux-x86_64',state='starter',size=size,id=12)
                f,state=self.publication(dict(draft=draft,prerelease=True,body='complete notes',assets=[asset],html_url='url'))
                with patch.object(flow,'run',side_effect=AssertionError('must not delete')):
                    with self.assertRaisesRegex(ValueError,'conflicting GitHub asset'):f.publish({})
                shutil.rmtree(f.base)

    def test_published_retry_only_verifies(self):
        f,state=self.publication(dict(draft=False,prerelease=True,body='complete notes',assets=[{'name':'h3cli-linux-x86_64'}],html_url='url'))
        f.publish({});self.assertEqual((state['uploads'],state['edits']),(0,0));self.assertEqual(state['checks'],1)

    def test_conflicting_public_notes_are_never_replaced(self):
        f,state=self.publication(dict(draft=False,prerelease=True,body='other notes',assets=[],html_url='url'))
        with self.assertRaisesRegex(ValueError,'notes/identity conflict'):f.publish({})
        self.assertEqual((state['uploads'],state['edits']),(0,0))

    def test_remote_asset_change_blocks_publication(self):
        f,state=self.publication(dict(draft=True,prerelease=True,body='complete notes',assets=[{'name':'h3cli-linux-x86_64'}],html_url='url'))
        def wrong(*args,**kw):
            if args[:2]==('release','download'):(Path(args[-1])/'h3cli-linux-x86_64').write_text('wrong bytes')
        f.gh=wrong
        with self.assertRaisesRegex(ValueError,'Remote asset bytes conflict'):f.publish({})
        self.assertEqual(state['edits'],0)

    def test_wrong_remote_tag_blocks_without_push(self):
        f,state=self.publication();del f.tag
        with patch.object(flow,'run',return_value='b'*40+'\trefs/tags/v0.0.1') as run:
            with self.assertRaisesRegex(ValueError,'another commit'):f.tag(create=True)
            self.assertEqual(run.call_count,1)


class CancellationTests(unittest.TestCase):
    def test_sigterm_stops_owned_child_group(self):
        with tempfile.TemporaryDirectory(prefix='h3-cancel-') as d:
            pidfile=Path(d)/'pid'
            code=('import sys,time,os;sys.path.insert(0,sys.argv[1]);import build_process;'
                  'build_process.run([sys.executable,"-c",\'import os,time,pathlib;pathlib.Path("'+str(pidfile)+'").write_text(str(os.getpid()));time.sleep(60)\'])')
            p=subprocess.Popen([sys.executable,'-c',code,str(RELEASE.parent/'scripts')],stderr=subprocess.DEVNULL)
            try:
                for _ in range(100):
                    if pidfile.exists():break
                    time.sleep(.02)
                self.assertTrue(pidfile.exists());child=int(pidfile.read_text())
                p.send_signal(signal.SIGTERM);p.wait(timeout=10)
                with self.assertRaises(ProcessLookupError):os.kill(child,0)
            finally:
                if p.poll() is None:p.kill();p.wait()


if __name__=='__main__':unittest.main()
