from common import *
from unittest.mock import patch
import workflow


class Workflow(Fixture):
    def test_assembly_and_provenance(self):
        self.adapter_file()
        manifest=self.run_fold()
        base=self.root/'original-model'
        (base/'FL2VA').mkdir(parents=True)
        (base/'FL2VA'/'tokenizer').mkdir()
        (base/'FL2VA'/'model_index.json').write_text('{}')
        (base/'modular_model_index.json').write_text('{}')
        model=self.root/'assembled'
        (model/'FL2VA').mkdir(parents=True)
        self.out.rename(model/'FL2VA'/'transformer')
        workflow.assemble(base,model,'FL2VA')
        workflow.assemble(base,model,'FL2VA')
        self.assertTrue((model/'FL2VA'/'tokenizer').is_symlink())
        self.assertFalse((model/'Ref2VA').exists())
        (model/'FL2VA'/'model_index.json').unlink()
        (model/'FL2VA'/'model_index.json').write_text('conflict')
        with self.assertRaises(f.FoldError): workflow.assemble(base,model,'FL2VA')
        state=self.root/'state.h3av'
        state.write_bytes(b'H3AV\r\n\x1a\n'+b'fixture-state')
        before=state.read_bytes()
        sidecar=workflow.record_provenance(state,model,'FL2VA')
        info=json.loads(sidecar.read_text())
        self.assertEqual(info['state_sha256'],f.sha256(state))
        self.assertEqual(info['base_sha256'],manifest['base_sha256'])
        self.assertEqual(state.read_bytes(),before)
        state.write_bytes(b'not a state')
        with self.assertRaises(f.FoldError): workflow.record_provenance(state,model,'FL2VA')

    def test_finalize_rollback(self):
        self.adapter_file()
        self.out.mkdir()
        (self.out/'keep').write_text('previous')
        rename=f.os.rename
        def fail(source,destination):
            if '.tmp-' in str(source): raise OSError('injected rename failure')
            return rename(source,destination)
        with patch.object(f.os,'rename',fail):
            with self.assertRaises(OSError): self.run_fold(overwrite=True)
        self.assertEqual((self.out/'keep').read_text(),'previous')
        self.assertFalse(list(self.root.glob('.*.tmp-*')))
        self.unchanged()

    def test_unmodified_corruption_rejected(self):
        self.adapter_file()
        fold_tensor=f.fold_tensor
        def corrupt(base,targets,dest,budget):
            result=fold_tensor(base,targets,dest,budget)
            with dest.open('r+b') as out:
                out.seek(-1,2)
                out.write(b'\xff')
            return result
        with patch.object(f,'fold_tensor',corrupt):
            with self.assertRaisesRegex(f.FoldError,'unmodified bytes'): self.run_fold()
        self.assertFalse(self.out.exists())
        self.unchanged()

    def test_dtype_variants(self):
        for dtype in ['F16','BF16']:
            a,b = ((dtype,self.a.astype('<f2')),(dtype,self.b.astype('<f2'))) if dtype=='F16' else (bf(self.a),bf(self.b))
            save(self.adapter, {'blocks.0.attn.qkv_proj.lora_A.weight':a,'blocks.0.attn.qkv_proj.lora_B.weight':b})
            shard=f.Shard(self.adapter)
            arrays=[t.read() for t in shard.tensors.values()]
            self.run_fold(overwrite=True)
            np.testing.assert_array_equal(reference_bf16(f.Checkpoint(self.out).tensors[QKV].read()),
                                          reference_bf16(self.w+arrays[1]@arrays[0]))

    def test_cli_contract(self):
        self.adapter_file()
        for extra in (['--dry-run'],['--inspect']):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(f.main(['--checkpoint',str(self.base),'--lora',str(self.adapter),*extra]),0)
        self.assertFalse(self.out.exists())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(f.main(['--checkpoint',str(self.base),'--lora',str(self.adapter)]),1)

    def test_resolved_source_files_not_overwritten(self):
        actual=self.root/'storage'
        actual.mkdir()
        original=self.base/'model.safetensors'
        original.rename(actual/'model.safetensors')
        original.symlink_to(actual/'model.safetensors')
        cp=f.Checkpoint(self.base)
        self.adapter_file()
        with self.assertRaisesRegex(f.FoldError,'resolved source'):
            f.output_path(cp,actual,[(self.adapter,1)],True)
