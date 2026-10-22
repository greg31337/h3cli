#!/usr/bin/env python3
"""Dependency trust, invalidation, offline reuse, extraction and cancellation."""
import copy
import io
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import dependency_bundle as bundles


class Bundles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = self.root/'payload';self.payload.mkdir()
        (self.payload/'tool').write_bytes(b'qualified tool');(self.payload/'tool').chmod(0o755)
        self.recipe = dict(schema=1, platform='macos-arm64', lock={'version':'1'}, recipes={})
        self.archive = self.root/'bundle.tar'
        bundles.pack(self.payload, self.archive, self.recipe, {'passed':True})
        self.pin = dict(schema=1, platform='macos-arm64', recipe_id=bundles.recipe_id(self.recipe),
                        archive=bundles.artifact(self.archive))
        self.report = self.root/'qualification.json'
        bundles.write(self.report, dict(schema=1, platform='macos-arm64', passed=True,
            archive_sha256=self.pin['archive']['sha256'], recipe_id=self.pin['recipe_id'],
            source_sha256='a'*64, reference_commit='b'*40,
            checks={name:dict(passed=True) for name in ('media','relocation','artifact','features')}))
        self.pin['qualification'] = bundles.artifact(self.report)
        self.ref = self.root/'bundle.json';bundles.write(self.ref, self.pin)
        self.cache = self.root/'cache'

    def load(self, **kwargs):
        return bundles.load(self.ref, self.cache, self.recipe, **kwargs)

    def test_cold_local_and_warm_offline_reuse(self):
        with patch.object(bundles.urllib.request, 'urlopen', side_effect=AssertionError('network')):
            path, identity = self.load(offline=True)
            stamps = {str(p):p.stat().st_mtime_ns for p in self.cache.rglob('*') if p.is_file()}
            self.archive.unlink();self.report.unlink()
            again, same = self.load(offline=True)
            self.assertEqual(path, again);self.assertEqual(identity, same)
            self.assertEqual(stamps, {str(p):p.stat().st_mtime_ns for p in self.cache.rglob('*') if p.is_file()})

    def test_miss_offline_never_fetches_https(self):
        pin = bundles.reference(self.ref);pin['qualification']['url']='https://example.invalid/qualification'
        with patch.object(bundles.urllib.request,'urlopen') as network:
            with self.assertRaisesRegex(ValueError,'Missing offline'):
                bundles.load(pin,self.cache,self.recipe,offline=True)
            network.assert_not_called()

    def test_unqualified_bundle_requires_explicit_candidate_mode(self):
        self.pin.pop('qualification');bundles.write(self.ref,self.pin)
        with self.assertRaisesRegex(ValueError,'unqualified'):self.load()
        path, row = self.load(candidate=True)
        self.assertEqual(row['mode'],'candidate');self.assertTrue((path/'tool').is_file())

    def test_qualification_must_bind_exact_archive_and_all_checks(self):
        original=bundles.read(self.report)
        for change in ({'passed':False},{'archive_sha256':'c'*64},{'checks':{}},{'reference_commit':''}):
            with self.subTest(change=change):
                bundles.write(self.report, original|change)
                self.pin['qualification']=bundles.artifact(self.report);bundles.write(self.ref,self.pin)
                with self.assertRaisesRegex(ValueError,'qualification'):self.load()

    def test_wrong_platform_or_recipe_rejected_before_download(self):
        for key,value in (('platform','linux-x86_64'),('recipe_id','0'*64)):
            pin=bundles.reference(self.ref);pin[key]=value
            with patch.object(bundles,'acquire') as fetch:
                with self.assertRaisesRegex(ValueError,'Incompatible'):
                    bundles.load(pin,self.cache,self.recipe)
                fetch.assert_not_called()

    def test_corruption_is_rejected_even_if_manifest_is_rewritten(self):
        path,_=self.load()
        (path/'tool').write_bytes(b'replaced bytes')
        manifest=path.parent/'manifest.json';row=bundles.read(manifest)
        row['files']=bundles.inventory(path);bundles.write(manifest,row)
        with self.assertRaisesRegex(ValueError,'manifest'):self.load()

    def test_cached_archive_corruption_is_not_silently_redownloaded(self):
        self.load();(self.cache/'downloads'/self.pin['archive']['sha256']).write_bytes(b'bad')
        with patch.object(bundles.urllib.request,'urlopen') as network:
            with self.assertRaisesRegex(ValueError,'Changed cached'):self.load()
            network.assert_not_called()

    def test_extra_file_and_symlink_are_rejected(self):
        path,_=self.load();extra=path/'extra';extra.write_text('unrecorded')
        with self.assertRaisesRegex(ValueError,'payload'):self.load()
        extra.unlink();(path/'tool').unlink();(path/'tool').symlink_to(self.root/'outside')
        with self.assertRaisesRegex(ValueError,'regular'):self.load()

    def test_archive_is_deterministic_and_never_overwritten(self):
        second=self.root/'second.tar';bundles.pack(self.payload,second,self.recipe,{'passed':True})
        self.assertEqual(bundles.sha(second),bundles.sha(self.archive))
        with self.assertRaisesRegex(ValueError,'already exists'):
            bundles.pack(self.payload,second,self.recipe,{})

    def test_extraction_rejects_traversal_links_devices_and_duplicates(self):
        for number,kind in enumerate(('parent','absolute','symlink','hardlink','device','duplicate')):
            archive=self.root/(kind+'.tar')
            with tarfile.open(archive,'w') as out:
                entry=tarfile.TarInfo('../outside' if kind=='parent' else '/outside' if kind=='absolute' else 'entry')
                if kind=='symlink':entry.type=tarfile.SYMTYPE;entry.linkname='../outside'
                elif kind=='hardlink':entry.type=tarfile.LNKTYPE;entry.linkname='../outside'
                elif kind=='device':entry.type=tarfile.CHRTYPE
                out.addfile(entry)
                if kind=='duplicate':out.addfile(entry)
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                bundles.extract(archive,self.root/('unpack'+str(number)))
        self.assertFalse((self.root.parent/'outside').exists())

    def test_failed_extract_never_creates_ready_cache(self):
        with patch.object(bundles,'extract',side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError,'disk full'):self.load()
        self.assertFalse((self.cache/'bundles'/self.pin['archive']['sha256']).exists())
        self.assertEqual(list((self.cache/'bundles').iterdir()),[])
        self.load()

    def test_interrupted_download_keeps_completed_cache_and_removes_partial(self):
        completed=self.cache/'downloads'/'completed';completed.parent.mkdir(parents=True)
        completed.write_bytes(b'previous bundle')
        pin=bundles.reference(self.ref)
        pin['qualification']['url']='https://example.invalid/qualification'
        class Interrupted(io.BytesIO):
            def read(self, *args):raise KeyboardInterrupt
        with patch.object(bundles.urllib.request,'urlopen',return_value=Interrupted(b'partial')):
            with self.assertRaises(KeyboardInterrupt):bundles.load(pin,self.cache,self.recipe)
        self.assertEqual(completed.read_bytes(),b'previous bundle')
        self.assertEqual(list((self.cache/'downloads').iterdir()),[completed])
        self.assertFalse((self.cache/'bundles').exists())

    def test_payload_root_cannot_be_replaced_with_symlink(self):
        path,_=self.load();outside=self.root/'outside';path.rename(outside);path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'Symlink'):self.load()

    def test_concurrent_consumers_publish_one_complete_cache(self):
        program='import sys,json;sys.path.insert(0,sys.argv[1]);import dependency_bundle as b;b.load(sys.argv[2],sys.argv[3],json.loads(sys.argv[4]),offline=True)'
        children=[subprocess.Popen([sys.executable,'-c',program,str(ROOT/'scripts'),str(self.ref),str(self.cache),json.dumps(self.recipe)],stderr=subprocess.PIPE) for _ in range(4)]
        for child in children:
            _,err=child.communicate(timeout=30);self.assertEqual(child.returncode,0,err)
        self.assertEqual(len(list((self.cache/'bundles').iterdir())),1)

    def test_recipe_excludes_application_and_release_identity(self):
        root=self.root/'source';shutil.copytree(ROOT/'scripts',root/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
        for platform in bundles.PLATFORMS:
            before=bundles.recipe(root,platform)
            (root/'application.c').write_text('changed application source')
            (root/'release-version').write_text('v999')
            self.assertEqual(before,bundles.recipe(root,platform))
            recipe=root/'scripts'/('linux' if platform.startswith('linux') else 'macos')/'dependencies.py'
            recipe.write_text(recipe.read_text()+'\n# changed dependency producer\n')
            self.assertNotEqual(before,bundles.recipe(root,platform))

    def test_dependency_versions_and_build_flags_invalidate_identity(self):
        value=copy.deepcopy(self.recipe)
        for key in ('version','compiler','sdk','architecture','flags'):
            changed=copy.deepcopy(value);changed['lock'][key]='new'
            self.assertNotEqual(bundles.recipe_id(value),bundles.recipe_id(changed))


class Cancellation(unittest.TestCase):
    def wait_for(self,path,child):
        deadline=time.monotonic()+10
        while not path.exists() and time.monotonic()<deadline:
            self.assertIsNone(child.poll());time.sleep(.05)
        self.assertTrue(path.exists())

    def assert_gone(self,pid):
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            try:os.kill(pid,0)
            except ProcessLookupError:return
            status=Path('/proc')/str(pid)/'stat'
            if status.exists() and status.read_text().split(') ',1)[1].startswith('Z '):return
            time.sleep(.05)
        self.fail('Child still running: '+str(pid))

    def test_sigint_sigterm_and_remote_stdin_eof_stop_grandchildren(self):
        for method in ('SIGINT','SIGTERM','EOF'):
            with self.subTest(method=method),tempfile.TemporaryDirectory() as tmp:
                marker=Path(tmp)/'children.json'
                worker=Path(tmp)/'worker.py'
                worker.write_text('import json,os,pathlib,subprocess,sys,time\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"])\npathlib.Path(sys.argv[1]).write_text(json.dumps([os.getpid(),p.pid]))\ntime.sleep(120)\n')
                argv=[sys.executable,str(ROOT/'scripts/build_process.py'),'--',sys.executable,str(worker),str(marker)]
                child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                try:
                    self.wait_for(marker,child)
                    pids=json.loads(marker.read_text())
                    if method=='EOF':child.stdin.close();child.stdin=None
                    else:child.send_signal(getattr(signal,method))
                    _,err=child.communicate(timeout=15)
                    self.assertNotEqual(child.returncode,0,err)
                    for pid in pids:self.assert_gone(pid)
                finally:
                    if child.poll() is None:child.kill();child.wait()

    def test_repeated_cancel_finishes_cleanup_of_stubborn_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'child'
            program=('import os,pathlib,signal,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                     'pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(120)')
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts/build_process.py'),'--',
                sys.executable,'-c',program,str(marker)],stdin=subprocess.PIPE)
            try:
                self.wait_for(marker,child);pid=int(marker.read_text())
                child.send_signal(signal.SIGINT);time.sleep(.2);child.send_signal(signal.SIGINT)
                child.wait(timeout=10);self.assertEqual(child.returncode,130);self.assert_gone(pid)
            finally:
                if child.poll() is None:child.kill();child.wait()
                child.stdin.close()


if __name__=='__main__':unittest.main()
