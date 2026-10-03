#!/usr/bin/env python3
"""CPU checks for portable configuration and strict recorded-golden comparisons."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cuda_reference_regression as gate


class ReferenceGate(unittest.TestCase):
    def test_fixtures_reject_changed_or_missing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture=root/'fixture';fixture.write_bytes(b'accepted')
            recorded={'fixture':gate.sha(fixture)}
            gate.checked_files(root,recorded)
            fixture.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'changed fixture'):gate.checked_files(root,recorded)
            fixture.unlink()
            with self.assertRaisesRegex(ValueError,'missing'):gate.checked_files(root,recorded)

    def test_outputs_require_exact_content_and_complete_case_set(self):
        expected={'x':'accepted'}
        gate.verify_outputs(dict(expected),expected)
        for actual in ({},{'x':'bad'},{'x':'accepted','extra':'bad'}):
            with self.subTest(actual=actual),self.assertRaises(ValueError):gate.verify_outputs(actual,expected)

    def test_paths_default_to_checkout_and_allow_relocation(self):
        source=Path('/tmp/checkout with spaces').resolve()
        with patch.dict(os.environ,{},clear=True):
            config=gate.regression_config(source)
            self.assertEqual(config['model'],str(source/'models/MiniMax-H3'))
            self.assertEqual(config['fixtures'],str(source/'tests/fixtures/cuda-reference'))
        with patch.dict(os.environ,{'H3_REFERENCE_MODEL':'/tmp/model','H3_REFERENCE_FIXTURES':'/tmp/fixtures'},clear=True):
            self.assertEqual(gate.regression_config(source),dict(model=str(Path('/tmp/model').resolve()),fixtures=str(Path('/tmp/fixtures').resolve())))
            self.assertEqual(gate.regression_config(source,Path('/tmp/explicit model'),Path('/tmp/explicit fixtures')),
                             dict(model=str(Path('/tmp/explicit model').resolve()),fixtures=str(Path('/tmp/explicit fixtures').resolve())))

    def test_runtime_inherits_dependencies_and_sets_budget(self):
        inherited={'PATH':'/opt/cuda/bin:/usr/bin','LD_LIBRARY_PATH':'/opt/cuda/lib64',
                   'H3_SGLANG_CUBLAS_LIBRARY':'/opt/reference/libcublas.so.13',
                   'H3_TEST_MAX_EVALUATIONS':'99'}
        with patch.dict(os.environ,inherited,clear=True):env=gate.runtime_env()
        for name in ('PATH','LD_LIBRARY_PATH','H3_SGLANG_CUBLAS_LIBRARY'):
            self.assertEqual(env[name],inherited[name])
        self.assertEqual(env['H3_TEST_MAX_EVALUATIONS'],'6')
        self.assertEqual(env['H3_TEST_REFERENCE_ENCODING'],'1')

    def test_dependency_pins_participate_in_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir()
            pins=root/'scripts/requirements-sglang-runtime.txt'
            pins.write_text('nvidia-cublas==13.1.1.3\n')
            before=gate.source_files(root)
            self.assertIn('scripts/requirements-sglang-runtime.txt',before)
            pins.write_text('nvidia-cublas==different\n')
            self.assertNotEqual(gate.fingerprint(before),gate.fingerprint(gate.source_files(root)))

    def run_recorded_case(self,actual):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            (source/'models/MiniMax-H3').mkdir(parents=True)
            fixtures=source/'tests/fixtures/cuda-reference';fixtures.mkdir(parents=True)
            fixture=fixtures/'input';fixture.write_bytes(b'recorded input')
            manifest=source/'tests/cuda_reference/manifest.json';manifest.parent.mkdir()
            gate.write(manifest,dict(schema=2,fixtures={'input':gate.sha(fixture)},goldens={'pixel':'recorded'}))
            # Candidate source may evolve; acceptance depends on recorded output.
            (source/'src').mkdir()
            (source/'src/probe.c').write_text('/* changed test source */\n')
            out=root/'result'
            argv=['gate','--source',str(source),'--out',str(out)]
            with patch.dict(os.environ,{},clear=True),patch('sys.argv',argv),\
                 patch.object(gate,'execute'),patch.object(gate,'collect',return_value={'files':actual}),\
                 patch.object(gate,'source_media_config',return_value={'_identities':{}}),\
                 contextlib.redirect_stdout(io.StringIO()):
                status=gate.main()
            return status,json.loads((out/'result.json').read_text())

    def test_runner_accepts_recorded_outputs_with_only_local_assets(self):
        status,result=self.run_recorded_case({'pixel':'recorded'})
        self.assertEqual(status,0);self.assertTrue(result['passed'])
        self.assertIn('golden_manifest_sha256',result)
        self.assertIn('src/probe.c',result['source_files'])

    def test_runner_never_reports_pass_after_output_drift(self):
        status,result=self.run_recorded_case({'pixel':'changed'})
        self.assertEqual(status,1);self.assertFalse(result['passed'])
        self.assertIn('reference drift',result['error'])

    def artifact_fixture(self,root):
        root=root.resolve()
        source=root/'source';source.mkdir();(source/'src').mkdir()
        (source/'src/a.c').write_text('source')
        runtime=root/'runtime';validation=root/'validation';runtime.mkdir();validation.mkdir()
        resources={name:'lib/'+name for name in ('H3_FFMPEG','H3_FFPROBE','H3_SGLANG_INPUT_FFMPEG','H3_SGLANG_CUBLAS_LIBRARY','H3_SGLANG_CUDNN_LIBRARY','H3_SGLANG_JPEG_LIBRARY')}
        files={}
        for name in ['bin/h3cli',*resources.values()]:
            path=runtime/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(name.encode())
            files[name]={'sha256':gate.sha(path),'size':path.stat().st_size}
        manifest=runtime/'share/h3cli/runtime.json';manifest.parent.mkdir(parents=True)
        gate.write(manifest,dict(schema=1,files=files,environment=resources))
        probes={}
        for name in gate.BUILD[1:]:
            path=validation/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(name.encode());probes[name]=gate.sha(path)
        artifact=root/'h3cli';artifact.write_bytes(b'outer bundle')
        media=validation/'media';media_files={}
        for name,relative in gate.reference_media.PATHS.items():
            path=media/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(relative.encode())
            media_files[name]=dict(path=relative,sha256=gate.sha(path),size=path.stat().st_size,
                                  version=gate.reference_media.VERSIONS[name])
        gate.write(media/'profile.json',dict(schema=1,purpose='sglang-regression',files=media_files))
        record=dict(schema=1,source_sha256=gate.fingerprint(gate.source_files(source)),
                    reference_media_sha256=gate.sha(media/'profile.json'),
                    runtime_manifest_sha256=gate.sha(manifest),probes=probes,artifact_sha256=gate.sha(artifact))
        gate.write(validation/'manifest.json',record)
        return source,runtime,validation,artifact

    def test_artifact_selects_release_and_pinned_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=self.artifact_fixture(Path(directory));config=gate.artifact_config(*paths)
            self.assertEqual(config['_binaries']['bin/h3cli'],str(paths[3]))
            self.assertEqual(config['_decoder'],str(paths[2]/'media/input/ffmpeg'))
            self.assertEqual(config['_runtime_env']['H3_FFMPEG'],str(paths[2]/'media/output/ffmpeg'))
            self.assertEqual(set(config['_binaries']),set(gate.BUILD))
            gate.verify_artifact_identities(config['_identities'])
            (paths[1]/'lib/H3_FFMPEG').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed during'):
                gate.verify_artifact_identities(config['_identities'])

    def test_artifact_rejects_source_probe_runtime_and_release_changes(self):
        for target in ('source/src/a.c','validation/bin/cuda_reference_probe','runtime/lib/H3_FFMPEG','h3cli',
                       'validation/media/profile.json','validation/media/input/ffmpeg','validation/media/output/ffmpeg'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);paths=self.artifact_fixture(root);(root/target).write_bytes(b'changed')
                with self.assertRaises(ValueError):gate.artifact_config(*paths)

    def test_source_mode_requires_explicit_historical_media(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{},clear=True):
            source=Path(directory)
            with self.assertRaisesRegex(ValueError,'setup_reference_media'):
                gate.source_media_config(source)

    def test_current_feature_validation_keeps_production_media(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=self.artifact_fixture(Path(directory))
            config=gate.artifact_config(*paths,reference=False)
            self.assertEqual(config['_decoder'],str(paths[1]/'lib/H3_SGLANG_INPUT_FFMPEG'))
            self.assertNotIn('H3_TEST_REFERENCE_MEDIA',config['_runtime_env'])

    def test_profile_rejects_escape_and_detects_mid_run_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source,runtime,validation,artifact=self.artifact_fixture(Path(directory))
            profile=validation/'media/profile.json'
            with patch.dict(os.environ,{'H3_REFERENCE_MEDIA':str(profile)}):
                config=gate.source_media_config(source)
            (validation/'media/input/ffprobe').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed during'):
                gate.verify_artifact_identities(config['_identities'])
            record=json.loads(profile.read_text());record['files']['H3_FFMPEG']['path']='../outside'
            gate.write(profile,record)
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                gate.reference_media.read_profile(profile)

    def test_artifact_rejects_incomplete_probe_set(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=self.artifact_fixture(Path(directory));p=paths[2]/'manifest.json';record=json.loads(p.read_text())
            record['probes'].pop(gate.BUILD[-1]);gate.write(p,record)
            with self.assertRaisesRegex(ValueError,'probe set'):gate.artifact_config(*paths)

    def test_artifact_rejects_escaping_runtime_resource(self):
        with tempfile.TemporaryDirectory() as directory:
            source,runtime,validation,artifact=self.artifact_fixture(Path(directory))
            p=runtime/'share/h3cli/runtime.json';manifest=json.loads(p.read_text())
            manifest['environment']['H3_FFMPEG']='../outside';gate.write(p,manifest)
            record=validation/'manifest.json';v=json.loads(record.read_text());v['runtime_manifest_sha256']=gate.sha(p);gate.write(record,v)
            with self.assertRaisesRegex(ValueError,'unsafe artifact'):gate.artifact_config(source,runtime,validation,artifact)

    def test_execute_rejects_expired_deadline_before_starting(self):
        with patch.object(gate.time,'monotonic',return_value=100),patch.object(gate.subprocess,'Popen') as child:
            with self.assertRaisesRegex(TimeoutError,'budget exhausted'):
                gate.execute(['unrelated'],Path('/tmp'),{},Path('/tmp/unused'),99)
            child.assert_not_called()

    def test_runner_rejects_preexisting_result_directory(self):
        with tempfile.TemporaryDirectory() as directory,patch('sys.argv',['gate','--source',directory,'--out',directory]),patch.object(gate,'collect') as collect:
            with self.assertRaises(FileExistsError):gate.main()
            collect.assert_not_called()

    def test_artifact_mode_does_not_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,runtime,validation,artifact=self.artifact_fixture(root)
            model=source/'models/MiniMax-H3';model.mkdir(parents=True)
            fixtures=source/'tests/fixtures/cuda-reference';fixtures.mkdir(parents=True)
            (fixtures/'input').write_bytes(b'input')
            golden=source/'tests/cuda_reference/manifest.json';golden.parent.mkdir()
            gate.write(golden,dict(schema=2,fixtures={'input':gate.sha(fixtures/'input')},goldens={'pixel':'recorded'}))
            p=validation/'manifest.json';record=json.loads(p.read_text());record['source_sha256']=gate.fingerprint(gate.source_files(source));gate.write(p,record)
            argv=['gate','--source',str(source),'--runtime',str(runtime),'--validation',str(validation),'--artifact',str(artifact),'--out',str(root/'out')]
            with patch('sys.argv',argv),patch.object(gate,'execute') as execute,patch.object(gate,'collect',return_value={'files':{'pixel':'recorded'}}) as collect,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gate.main(),0)
                execute.assert_not_called()
                self.assertEqual(collect.call_args.args[0],source)
                self.assertEqual(collect.call_args.args[2]['_binaries']['bin/h3cli'],str(artifact))

    def test_bundled_fixtures_match_recorded_content(self):
        root=Path(__file__).resolve().parents[1]
        manifest=json.loads((root/'tests/cuda_reference/manifest.json').read_text())
        gate.checked_files(root/'tests/fixtures/cuda-reference',manifest['fixtures'])


if __name__=='__main__':unittest.main()
