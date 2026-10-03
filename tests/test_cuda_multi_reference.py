"""Host checks for campaign safety limits and baseline attribution."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from cuda_multi_reference import aligned, canvas, make_cases, validate_case, compare, command, cg_accounting, long_cases, final_cases, run_environment, records, sample_cases
from cuda_multi_reference_audit import transfer_manifest
from cuda_multi_reference_report import black_frame_check


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.assets = {f'I{i:02d}': {'kind':'image','sha256':str(i),'path':f'i{i}.jpg'} for i in range(1,10)}
        for key,n in [('V1a',72),('V2a',72),('V2b',72),('V3a',48),('V3b',48),('V3c',48),('A1',72)]:
            self.assets[key] = {'kind':'video','sha256':key,'path':key+'.mp4',
                                'frames':n,'duration':n/24,'fps':'24/1','audio':key=='A1'}
        self.cases=make_cases()

    def test_complete_matrix_and_limits(self):
        self.assertEqual(len(self.cases),76)
        self.assertEqual(sum(c['phase']=='smoke' for c in self.cases),10)
        self.assertEqual(sum(c.get('baseline',False) for c in self.cases),22)
        for c in self.cases: validate_case(c,self.assets)
        self.assertEqual(aligned(240),243)
        self.assertEqual(aligned(144),158)

    def test_invalid_request_is_never_launched(self):
        c=copy.deepcopy(next(c for c in self.cases if c['name']=='X93-max'))
        for key,value in [('steps',7),('frames',243),('width',1920)]:
            d=dict(c,**{key:value})
            with self.assertRaises(AssertionError): validate_case(d,self.assets)
        assets=copy.deepcopy(self.assets)
        assets['V3a']['frames']=49
        assets['V3a']['duration']=49/24
        with self.assertRaises(AssertionError): validate_case(c,assets)
        assets=copy.deepcopy(self.assets)
        assets['V3a']['frames']=47
        assets['V3a']['duration']=47/24
        with self.assertRaises(AssertionError): validate_case(c,assets)
        with self.assertRaises(AssertionError): validate_case(dict(c,frames=73),self.assets)

    def test_distinct_baseline_groups_and_failures(self):
        c=next(c for c in self.cases if c['name']=='X93-max')
        def row(t,key=None,status='complete',identity='frozen'):
            return {'case':dict(c,baseline=True,baseline_key=key or c['baseline_key']),
                    'status':status,'wall_seconds':t,'max_vram_gib':40.,'manifest_identity':identity}
        reference=[row(10),row(11),row(12),row(1,'wrong-resolution'),row(2,status='failed'),row(3,identity='stale')]
        candidate=row(22);candidate['max_vram_gib']=45.
        v=compare(candidate,reference)
        self.assertEqual(v['baseline_samples'],3)
        self.assertEqual(v['wall_delta_seconds'],11.)
        self.assertEqual(v['wall_over_baseline'],2.)
        self.assertEqual(v['vram_delta_gib'],5.)
        self.assertTrue(v['baseline_unstable'])
        candidate['status']='failed'
        self.assertFalse(compare(candidate,reference)['available'])

    def test_order_and_audio_flags(self):
        m={'settings':{'model':'model'},'assets':self.assets}
        c=next(c for c in self.cases if c['name']=='O91')
        self.assertEqual(c['references'][4],'V1a')
        args=command(c,m,'result.mp4')
        self.assertEqual(args.count('--ref-image'),9)
        self.assertEqual(args.count('--ref-silent-video'),1)
        a=next(c for c in self.cases if c['name']=='A11')
        self.assertIn('--ref-video',command(a,m,'result.mp4'))

    def test_sizing_exercises_both_modes(self):
        self.assertEqual(canvas(1365,1821,640,480,'match'),[480,640])
        self.assertEqual(canvas(1365,1821,1344,768,'match'),[864,1184])
        self.assertEqual(canvas(1365,1821,1344,768,'max'),[1376,1824])

    def test_file_cache_is_reclaimable_but_dirty_memory_is_not(self):
        d=cg_accounting(1000,990,{'total_inactive_file':800,'total_dirty':20,'total_writeback':10})
        self.assertEqual(d['available_estimate'],780)
        self.assertEqual(d['working_set'],220)
        self.assertEqual(cg_accounting(1000,990,{})['available_estimate'],10)
        self.assertEqual(cg_accounting(1000,990,{'inactive_file':800,'file_dirty':800})['available_estimate'],10)

    def test_only_explicit_long_extension_can_exceed_ten_seconds(self):
        extension=long_cases()
        self.assertEqual(len(extension),10)
        self.assertEqual(sum(c.get('baseline',False) for c in extension),6)
        for c in extension:validate_case(c,self.assets)
        c=next(c for c in extension if not c.get('baseline'))
        with self.assertRaises(AssertionError):validate_case(dict(c,scope='main'),self.assets)
        with self.assertRaises(AssertionError):validate_case(dict(c,references=c['references'][:8]),self.assets)
        with self.assertRaises(AssertionError):validate_case(dict(c,frames=379),self.assets)

    def test_final_matrix_uses_ten_steps_and_separate_baselines(self):
        final=final_cases(self.cases+long_cases())
        self.assertEqual(len(final),76)
        self.assertTrue(all(c['steps']==10 and '-S10-' in c['baseline_key'] for c in final))
        self.assertEqual(sum(c.get('scope')=='long-362' for c in final),10)
        for c in final:validate_case(c,self.assets)
        args=command(final[0],{'settings':{'model':'model'},'assets':self.assets},'output.mp4')
        self.assertEqual(args[args.index('--steps')+1],'10')
        with self.assertRaises(AssertionError):validate_case(dict(final[0],steps=11),self.assets)
        with patch.dict('os.environ',{'H3_TEST_MAX_EVALUATIONS':'6','H3_INHERITED_OPTION':'1'}):
            env=run_environment(final[0])
            self.assertNotIn('H3_TEST_MAX_EVALUATIONS',env)
            self.assertNotIn('H3_INHERITED_OPTION',env)
            self.assertEqual(run_environment(self.cases[0])['H3_TEST_MAX_EVALUATIONS'],'6')

    def test_budget_sample_preserves_geometry_and_never_mixes_step_counts(self):
        cases=sample_cases()
        self.assertEqual(len(cases),4)
        for width in (640,1344):
            pair=[c for c in cases if c['width']==width]
            self.assertEqual(len(pair),2)
            self.assertEqual(pair[0]['baseline_key'],pair[1]['baseline_key'])
            self.assertEqual(pair[0]['steps'],pair[1]['steps'])
            self.assertEqual(len(pair[1]['references']),9)
            for c in pair:
                validate_case(c,self.assets)
                self.assertEqual(c['frames'],362)
                self.assertEqual(run_environment(c)['H3_TEST_MAX_EVALUATIONS'],'6')
                with self.assertRaises(AssertionError):validate_case(dict(c,steps=3),self.assets)
                with self.assertRaises(AssertionError):validate_case(dict(c,frames=243),self.assets)

    def test_explicit_retry_preserves_and_excludes_prior_attempt(self):
        with tempfile.TemporaryDirectory() as folder,patch('cuda_multi_reference.ROOT',Path(folder)):
            root=Path(folder)
            for n in (1,2):
                directory=root/'R640-test'/f'attempt-{n}'
                directory.mkdir(parents=True)
                (directory/'metrics.json').write_text(json.dumps({'artifact_directory':str(directory.relative_to(root))}))
            self.assertEqual(len(records()),2)
            (root/'retries.json').write_text(json.dumps([{'prior_directory':'R640-test/attempt-1'}]))
            self.assertEqual([r['artifact_directory'] for r in records()],['R640-test/attempt-2'])
            self.assertEqual(len(records(include_superseded=True)),2)

    def test_transfer_audit_detects_changed_download(self):
        with tempfile.TemporaryDirectory() as folder,patch('cuda_multi_reference_audit.ROOT',Path(folder)):
            root=Path(folder)
            identities={'manifest_identity':'base','final_identity':'final'}
            (root/'results.json').write_text(json.dumps(dict(identities,coverage={'pending':[]})))
            artifact=root/'output.mp4'
            artifact.write_bytes(b'original')
            transfer_manifest(write=True)
            with patch('cuda_multi_reference_audit.load_manifest',return_value={'identity':'base','final_identity':'final'}):
                transfer_manifest()
                artifact.write_bytes(b'modified')  # Equal length; a size-only audit misses this.
                with self.assertRaisesRegex(AssertionError,'checksum mismatch'):transfer_manifest()

    def test_transfer_snapshot_rejects_running_or_unfinished_campaign(self):
        import fcntl
        with tempfile.TemporaryDirectory() as folder,patch('cuda_multi_reference_audit.ROOT',Path(folder)):
            root=Path(folder)
            (root/'results.json').write_text(json.dumps({'coverage':{'pending':['remaining']}}))
            with self.assertRaisesRegex(AssertionError,'Finish the campaign'):transfer_manifest(write=True)
            with (root/'runner.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):transfer_manifest(write=True)

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg is required for media screening fixtures')
    def test_black_screen_flags_dark_output_and_invalidates_changed_media(self):
        with tempfile.TemporaryDirectory() as folder,patch('cuda_multi_reference_report.ROOT',Path(folder)):
            movie=Path(folder)/'output.mp4'
            identities=[]
            for color in ('black','white'):
                subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
                    f'color=c={color}:s=128x96:r=24:d=1','-c:v','libx264',
                    '-pix_fmt','yuv420p',str(movie)],check=True)
                result=black_frame_check(movie)
                self.assertEqual(result['warning'],color=='black')
                identities.append(result['input_sha256'])
            self.assertNotEqual(*identities)


if __name__=='__main__': unittest.main()
