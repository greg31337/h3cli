from unittest.mock import patch
import errno
from common import *


class Safety(Fixture):
    def test_aligned_unaligned_preservation(self):
        self.adapter_file()
        for align in (64, 8, 1):
            save(self.base/'model.safetensors', {QKV:bf(self.w), OUT:bf(self.w2)}, alignment=align)
            original=f.Shard(self.base/'model.safetensors')
            self.run_fold(overwrite=True)
            result=f.Shard(self.out/'model.safetensors')
            self.assertEqual(result.header,original.header)
            t=original.tensors[OUT]
            self.assertEqual((self.base/'model.safetensors').read_bytes()[t.offset:t.offset+t.size],
                             (self.out/'model.safetensors').read_bytes()[t.offset:t.offset+t.size])

    def test_header_corruption(self):
        bad=[b'', struct.pack('<Q',2**63), struct.pack('<Q',2)+b'[]',
             struct.pack('<Q',13)+b'{"a":1,"a":2}']
        for raw in bad:
            self.adapter.write_bytes(raw)
            with self.assertRaises(f.FoldError): f.Shard(self.adapter)
        for desc in [
            {'dtype':'BF16','shape':[2**62,16],'data_offsets':[0,4]},
            {'dtype':'BF16','shape':[-1],'data_offsets':[0,4]},
            {'dtype':'BF16','shape':[True],'data_offsets':[0,4]},
            {'dtype':'BF16','shape':[2],'data_offsets':[0,6]},
            {'dtype':'BAD','shape':[2],'data_offsets':[0,4]},
            {'dtype':'BF16','shape':[2],'data_offsets':[1,5]},
        ]:
            h=json.dumps({'a':desc}).encode()
            self.adapter.write_bytes(struct.pack('<Q',len(h))+h+b'123456')
            with self.assertRaises(f.FoldError): f.Shard(self.adapter)

    def test_shard_index_and_duplicates(self):
        save(self.base/'second.safetensors', {QKV:bf(self.w)})
        with self.assertRaisesRegex(f.FoldError,'duplicate'): f.Checkpoint(self.base)
        (self.base/'second.safetensors').unlink()
        (self.base/'model.safetensors.index.json').write_text('{"weight_map":{}}')
        with self.assertRaisesRegex(f.FoldError,'weight_map'): f.Checkpoint(self.base)

    def test_atomic_midway_failure(self):
        self.adapter_file(extras={'blocks.0.attn.out_proj.lora_A.weight':self.a[:,:2],
                                  'blocks.0.attn.out_proj.lora_B.weight':self.b[:4]})
        self.out.mkdir()
        (self.out/'keep').write_text('old output')
        original=f.fold_tensor
        count=0
        def fail(*args):
            nonlocal count
            count+=1
            if count==2: raise OSError('injected disk failure')
            return original(*args)
        with patch.object(f,'fold_tensor',fail):
            with self.assertRaises(OSError): self.run_fold(overwrite=True)
        self.assertEqual((self.out/'keep').read_text(),'old output')
        self.assertEqual(list(self.root.glob('.*.tmp-*')),[])
        self.unchanged()

    def test_existing_and_alias_output(self):
        self.adapter_file()
        self.out.mkdir()
        with self.assertRaises(f.FoldError): self.run_fold()
        for out in (self.base,self.base/'nested',self.root):
            with self.assertRaises(f.FoldError):
                f.output_path(self.cp,out,[(self.adapter,1)],True)
        alias=self.root/'alias'
        alias.symlink_to(self.base,target_is_directory=True)
        with self.assertRaises(f.FoldError): f.output_path(self.cp,alias,[(self.adapter,1)],True)

    def test_dry_run_no_writes(self):
        self.adapter_file()
        before=set(self.root.iterdir())
        self.run_fold(dry_run=True)
        self.assertEqual(set(self.root.iterdir()),before)
        self.unchanged()

    def test_copy_policy(self):
        source = self.base/'model.safetensors'
        destination = self.root/'copy'
        for error in (errno.EOPNOTSUPP, errno.EXDEV):
            with self.subTest(error=error), patch.object(f.sys,'platform','linux'), \
                    patch('fcntl.ioctl',side_effect=OSError(error, 'clone unavailable')):
                with self.assertRaisesRegex(f.FoldError,'allow-full-copy'):
                    f.clone_file(source,destination,False)
                self.assertFalse(destination.exists())
                with contextlib.redirect_stdout(io.StringIO()):
                    method=f.clone_file(source,destination,True)
                self.assertEqual(method,'full-copy')
                self.assertEqual(destination.read_bytes(),source.read_bytes())
                self.assertNotEqual(destination.stat().st_ino,source.stat().st_ino)
                destination.unlink()
        if sys.platform=='darwin':
            self.assertEqual(f.clone_file(self.base/'model.safetensors',self.root/'clone',False),'clonefile')
            self.assertNotEqual((self.root/'clone').stat().st_ino,(self.base/'model.safetensors').stat().st_ino)
        self.unchanged()

    def test_linux_clone_existing_destination_preserved(self):
        destination = self.root/'existing'
        destination.write_bytes(b'keep existing file')
        with patch.object(f.sys,'platform','linux'):
            for allow_full_copy in (False, True):
                with self.assertRaises(FileExistsError):
                    f.clone_file(self.base/'model.safetensors',destination,allow_full_copy)
                self.assertEqual(destination.read_bytes(),b'keep existing file')
        self.unchanged()

    def test_manifest_and_all_bytes(self):
        self.adapter_file()
        manifest=self.run_fold()
        saved=json.loads((self.out/f.MANIFEST).read_text())
        self.assertEqual(saved['adapters'][0]['sha256'],f.sha256(self.adapter))
        self.assertEqual(saved['base_files'][0]['sha256'],f.sha256(self.base/'model.safetensors'))
        self.assertEqual(saved['base_files'][0]['folded_sha256'],f.sha256(self.out/'model.safetensors'))
        self.assertEqual(saved['validation'][QKV]['bf16'],manifest['validation'][QKV]['bf16'])
        self.unchanged()
