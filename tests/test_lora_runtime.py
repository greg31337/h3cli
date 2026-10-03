"""GPU-free native folding tests; fixtures need only Python's standard library."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
import time
import signal
import random
import concurrent.futures

HOST = str(Path(os.environ.get('H3_LORA_HOST', './bin/lora_host')).resolve())


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16(x):
    u = struct.unpack('<I', struct.pack('<f', x))[0]
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16) & 0xffff


def tensor_file(path, tensors, metadata=None):
    head, payload = {}, bytearray()
    if metadata is not None:
        head['__metadata__'] = metadata
    for name, shape, values, dtype in tensors:
        start = len(payload)
        for v in values:
            payload.extend(struct.pack('<H', bf16(v)) if dtype == 'BF16' else struct.pack('<e' if dtype == 'F16' else '<f', v))
        head[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(payload)])
    encoded = json.dumps(head, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)


def values(path, name):
    data = path.read_bytes()
    n, = struct.unpack('<Q', data[:8])
    t = json.loads(data[8:8+n])[name]
    lo, hi = t['data_offsets']
    return list(struct.unpack('<'+'H'*((hi-lo)//2), data[8+n+lo:8+n+hi]))


class RuntimeLoRA(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='h3-runtime-lora-')
        self.root = Path(self.tmp.name)
        self.base = self.root/'base'
        self.base.mkdir()
        (self.base/'config.json').write_text('{"num_attention_heads":1,"attention_head_dim":2}')
        self.name = 'blocks.0.attn.qkv_proj.weight'
        tensor_file(self.base/'model.safetensors', [(self.name,[6,2],[1.0]*12,'BF16'),('untouched',[2],[2,3],'BF16')])
        self.adapter = self.root/'adapter.safetensors'
        self.make_adapter()
        self.cache = self.root/'cache'
        self.original = (self.base/'model.safetensors').read_bytes()

    def tearDown(self):
        self.assertEqual((self.base/'model.safetensors').read_bytes(), self.original)
        self.tmp.cleanup()

    def make_adapter(self, path=None, module='blocks.0.attn.qkv_proj', rows=6, dtype='F32', metadata=None, a=None, b=None, parts=('lora_A','lora_B')):
        tensor_file(path or self.adapter, [(module+'.'+parts[0]+'.weight',[1,2],a or [1,2],dtype),
            (module+'.'+parts[1]+'.weight',[rows,1],b or [0.5]*rows,dtype)], metadata)

    def run_fold(self, *adapters, good=True, memory=512, env=None, base=None, cache=None, mode=False):
        args=[HOST,str(base or self.base),str(cache or self.cache),str(memory)]
        if mode: args.append('--ref2va')
        args += list(map(str,adapters or [self.adapter]))
        result=subprocess.run(args,text=True,capture_output=True,env={**os.environ,'H3_TEST_MIN_AVAILABLE_MEMORY_BYTES':'0',**(env or {})})
        if good:
            self.assertEqual(result.returncode,0,result.stderr)
            return Path(result.stdout.strip()),result
        self.assertNotEqual(result.returncode,0,result.stderr)
        self.assertFalse(list((cache or self.cache).glob('variants/*/manifest.json')))
        return result

    def test_native_and_verified_hit(self):
        out,r=self.run_fold()
        self.assertEqual(values(out/'model.safetensors',self.name),[bf16(1.5),bf16(2)]*6)
        self.assertEqual(values(out/'model.safetensors','untouched'),[bf16(2),bf16(3)])
        again,r=self.run_fold();self.assertEqual(out,again);self.assertIn('verified cache hit',r.stderr)
        self.assertNotEqual(os.stat(out/'model.safetensors').st_ino,os.stat(self.base/'model.safetensors').st_ino)

    def test_scale_and_duplicate_order(self):
        for scale in [0,-1,0.5,2]:
            out,_=self.run_fold(str(self.adapter)+':'+str(scale))
            self.assertEqual(values(out/'model.safetensors',self.name),[bf16(1+0.5*scale),bf16(1+scale)]*6)
        out,_=self.run_fold(str(self.adapter)+':0.5',str(self.adapter)+':-0.5')
        self.assertEqual((out/'model.safetensors').read_bytes(),self.original)

    def test_mappings_and_dtypes(self):
        for module in ['base_model.model.blocks.0.attn.qkv_proj','model.diffusion_model.blocks.0.attn.qkv_proj','transformer.transformer_blocks.0.attn.qkv_proj','lora_unet_blocks_0_attn_qkv_proj']:
            for dtype in ['BF16','F16','F32']:
                self.make_adapter(module=module,dtype=dtype,parts=('lora_down.default','lora_up.default'))
                out,_=self.run_fold();self.assertEqual(values(out/'model.safetensors',self.name),[bf16(1.5),bf16(2)]*6)

    def test_split_qkv(self):
        for part,start in [('q',0),('k',2),('v',4)]:
            self.make_adapter(module='transformer_blocks.0.attn.to_'+part,rows=2)
            out,_=self.run_fold();expected=[bf16(1)]*12;expected[start*2:start*2+4]=[bf16(1.5),bf16(2)]*2
            self.assertEqual(values(out/'model.safetensors',self.name),expected)

    def test_metadata_alpha(self):
        for metadata in [{'alpha':'0.5'},{'adapter_config':'{"r":1,"lora_alpha":0.5,"use_rslora":false,"use_dora":false,"peft_type":"LORA"}'}]:
            self.make_adapter(metadata=metadata);out,_=self.run_fold()
            self.assertEqual(values(out/'model.safetensors',self.name),[bf16(1.25),bf16(1.5)]*6)

    def test_reject_metadata(self):
        for metadata in [{'alpha':'-1'},{'rank':'2'},{'alpha':'1','lora_alpha':'2'},{'use_rslora':'true'},{'peft_type':'DORA'},{'alpha_pattern':'{}'},{'adapter_config':'{"r":1,"r":2}'}]:
            self.make_adapter(metadata=metadata);self.run_fold(good=False)

    def test_invalid_arguments(self):
        for arg in [str(self.adapter)+':nan',str(self.adapter)+':inf',str(self.adapter)+':1e99',str(self.adapter)+':',str(self.adapter)+':0x1p0','missing:1']:
            self.run_fold(arg,good=False)

    def test_literal_path_spaces_colon(self):
        p=self.root/'adapter path:2';shutil.copyfile(self.adapter,p)
        a,_=self.run_fold(p);b,_=self.run_fold(self.adapter);self.assertEqual(a,b)

    def test_content_identity(self):
        a,_=self.run_fold(str(self.adapter)+':0.50')
        b,_=self.run_fold(str(self.adapter)+':5e-1',memory=1);self.assertEqual(a,b)
        c,_=self.run_fold(str(self.adapter)+':0.5',mode=True);self.assertNotEqual(a,c)
        moved=self.root/'moved';shutil.copytree(self.base,moved)
        d,_=self.run_fold(str(self.adapter)+':0.5',base=moved);self.assertEqual(a,d)
        (moved/'config.json').write_text('{"num_attention_heads":2,"attention_head_dim":1}')
        e,_=self.run_fold(str(self.adapter)+':0.5',base=moved);self.assertNotEqual(a,e)

    def test_repair(self):
        out,_=self.run_fold();expected=(out/'model.safetensors').read_bytes()
        with (out/'model.safetensors').open('r+b') as f:f.seek(-2,2);f.write(b'xx')
        again,r=self.run_fold();self.assertEqual(out,again);self.assertEqual((again/'model.safetensors').read_bytes(),expected)
        self.assertIn('quarantined',r.stderr)

    def test_cancel(self):
        for phase in ['planning','copying','folding','validating','publishing']:
            self.run_fold(good=False,env={'H3_TEST_LORA_CANCEL_PHASE':phase})
            self.assertFalse(list(self.cache.glob('staging/*')))

    def test_nonfinite_zero_scale(self):
        self.make_adapter(a=[float('nan'),1]);self.run_fold(str(self.adapter)+':0',good=False)

    def test_faults_and_short_writes(self):
        for fault in ['selection-allocation','allocation','copy','permission','lock','write','enospc','fsync','rename']:
            self.run_fold(good=False,env={'H3_TEST_LORA_FAULT':fault})
            self.assertFalse(list(self.cache.glob('staging/*')))
        out,_=self.run_fold(env={'H3_TEST_LORA_SHORT_WRITE':'1'})
        self.assertEqual(values(out/'model.safetensors',self.name),[bf16(1.5),bf16(2)]*6)

    def test_low_cow_headroom_falls_back_without_changing_bytes(self):
        out,_=self.run_fold()
        expected={p.name:p.read_bytes() for p in out.iterdir()}
        shutil.rmtree(out.parent)
        replacement,_=self.run_fold(env={'H3_TEST_LORA_LOW_COW_SPACE':'1','H3_TEST_LORA_DELAYED_QUOTA':'1'})
        self.assertEqual(out,replacement)
        self.assertEqual(expected,{p.name:p.read_bytes() for p in replacement.iterdir()})
        manifest=json.loads((replacement.parent/'manifest.json').read_text())
        self.assertGreater(manifest['copy_files'],0)

    def test_cancel_waiting_for_released_clone_quota(self):
        out,_=self.run_fold()
        manifest=json.loads((out.parent/'manifest.json').read_text())
        if not manifest['clone_files']:
            self.skipTest('filesystem does not support cloned files')
        shutil.rmtree(out.parent)
        self.run_fold(good=False,env={'H3_TEST_LORA_LOW_COW_SPACE':'1',
            'H3_TEST_LORA_DELAYED_QUOTA':'1','H3_TEST_LORA_CANCEL_PHASE':'released disk quota'})
        self.assertFalse(list(self.cache.glob('staging/*')))

    def test_manifest_and_symlink_repair(self):
        out,_=self.run_fold();manifest=out.parent/'manifest.json'
        for mutation in ['manifest','truncated','missing','symlink','extra','traversal']:
            if mutation=='manifest':manifest.write_text('{}')
            if mutation=='truncated':(out/'model.safetensors').write_bytes(b'bad')
            if mutation=='missing':(out/'model.safetensors').unlink()
            if mutation=='symlink':
                (out/'model.safetensors').unlink();(out/'model.safetensors').symlink_to(self.base/'model.safetensors')
            if mutation=='extra':(out/'extra').write_text('unexpected')
            if mutation=='traversal':
                m=json.loads(manifest.read_text());m['files'][0]['name']='../outside';manifest.write_text(json.dumps(m))
            again,r=self.run_fold();self.assertEqual(again,out);self.assertIn('quarantined',r.stderr)

    def test_same_stat_adapter_mutation(self):
        out,_=self.run_fold();st=self.adapter.stat();data=bytearray(self.adapter.read_bytes())
        data[-4:]=struct.pack('<f',0.75);self.adapter.write_bytes(data);os.utime(self.adapter,ns=(st.st_atime_ns,st.st_mtime_ns))
        again,_=self.run_fold();self.assertNotEqual(out,again)

    def test_readonly_hit(self):
        out,_=self.run_fold();manifest=out.parent/'manifest.json';stamp=manifest.stat().st_mtime_ns
        paths=list(self.cache.rglob('*'))+[self.cache]
        for p in paths:p.chmod(0o500 if p.is_dir() else 0o400)
        try:
            again,r=self.run_fold();self.assertEqual(out,again);self.assertIn('verified cache hit',r.stderr)
            self.assertEqual(manifest.stat().st_mtime_ns,stamp)
        finally:
            for p in reversed(paths):p.chmod(0o700 if p.is_dir() else 0o600)

    def test_concurrent_writers(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            outs=list(pool.map(lambda _:self.run_fold()[0],range(4)))
        self.assertEqual(len(set(outs)),1);self.assertEqual(len(list(self.cache.glob('variants/*'))),1)
        self.assertFalse(list(self.cache.glob('staging/*')))

    def test_different_key_writers(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            outs=list(pool.map(lambda scale:self.run_fold(str(self.adapter)+':'+str(scale))[0],[0.25,0.5,0.75]))
        self.assertEqual(len(set(outs)),3)

    def start_paused(self,phase):
        p=subprocess.Popen([HOST,str(self.base),str(self.cache),'512',str(self.adapter)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            env={**os.environ,'H3_TEST_MIN_AVAILABLE_MEMORY_BYTES':'0','H3_TEST_LORA_PAUSE_PHASE':phase})
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            line=p.stderr.readline()
            if 'PAUSED' in line:return p
            if p.poll() is not None:self.fail('writer failed before requested phase: '+line)
        p.kill();self.fail('writer did not pause')

    def test_interrupted_writers(self):
        for phase in ['hashing','copying','folding','validating','publishing']:
            p=self.start_paused(phase);p.kill();p.communicate(timeout=5)
            self.assertFalse(list(self.cache.glob('variants/*/manifest.json')))
            out,_=self.run_fold();shutil.rmtree(out.parent)
            self.assertFalse(list(self.cache.glob('staging/*')))

    def test_source_mutation_during_fold(self):
        for phase in ['folding','publishing']:
            p=self.start_paused(phase)
            data=bytearray(self.adapter.read_bytes());data[-4:]=struct.pack('<f',0.75 if phase=='folding' else 0.625);self.adapter.write_bytes(data)
            os.kill(p.pid,signal.SIGCONT)
            stdout,stderr=p.communicate(timeout=10)
            self.assertNotEqual(p.returncode,0,stderr)
            self.assertFalse(list(self.cache.glob('variants/*/manifest.json')))
            self.assertFalse(list(self.cache.glob('staging/*')))

    def test_rounding_and_budget_invariance(self):
        a,_=self.run_fold(memory=1,env={'H3_TEST_LORA_ROUND':'1'})
        original=(a/'model.safetensors').read_bytes();identity=(a/'h3_lora_manifest.json').read_bytes()
        shutil.rmtree(a.parent)
        b,_=self.run_fold(memory=512)
        self.assertEqual(a,b);self.assertEqual(original,(b/'model.safetensors').read_bytes());self.assertEqual(identity,(b/'h3_lora_manifest.json').read_bytes())

    def test_subnormal_scale_is_independent_of_caller_flush_mode(self):
        for scale in ['1e-310','-1e-310','5e-324','-0']:
            a,_=self.run_fold(str(self.adapter)+':'+scale)
            b,_=self.run_fold(str(self.adapter)+':'+scale,
                env={'H3_TEST_LORA_FLUSH':'1','H3_TEST_LORA_ROUND':'1'})
            self.assertEqual(a,b)

    def test_symlinked_sources_produce_independent_outputs(self):
        original=self.root/'original.safetensors'
        (self.base/'model.safetensors').rename(original)
        (self.base/'model.safetensors').symlink_to(original)
        alias=self.root/'adapter-link.safetensors';alias.symlink_to(self.adapter)
        out,_=self.run_fold(alias)
        self.assertFalse((out/'model.safetensors').is_symlink())
        self.assertEqual(original.read_bytes(),self.original)
        self.assertEqual(values(out/'model.safetensors',self.name),[bf16(1.5),bf16(2.0)]*6)

    def test_sharded_index(self):
        tensor_file(self.base/'extra.safetensors',[('other.weight',[1,1],[1],'BF16')])
        index={'weight_map':{self.name:'model.safetensors','untouched':'model.safetensors','other.weight':'extra.safetensors'}}
        path=self.base/'model.safetensors.index.json';path.write_text(json.dumps(index))
        self.run_fold();shutil.rmtree(self.cache)
        index['weight_map']['other.weight']='model.safetensors';path.write_text(json.dumps(index))
        self.run_fold(good=False)

    def test_orphans_extras_overlap_and_alpha(self):
        a=('blocks.0.attn.qkv_proj.lora_A.weight',[1,2],[1,2],'F32')
        b=('blocks.0.attn.qkv_proj.lora_B.weight',[6,1],[0.5]*6,'F32')
        alpha=('blocks.0.attn.qkv_proj.alpha',[],[0.5],'F32')
        tensor_file(self.adapter,[a,b,alpha]);out,_=self.run_fold()
        self.assertEqual(values(out/'model.safetensors',self.name),[bf16(1.25),bf16(1.5)]*6);shutil.rmtree(self.cache)
        for tensors in [[a],[alpha],[a,b,('unhandled',[1],[1],'F32')],[a,b,('blocks.0.attn.to_q.lora_A.weight',[1,2],[1,2],'F32'),('blocks.0.attn.to_q.lora_B.weight',[2,1],[1,1],'F32')]]:
            tensor_file(self.adapter,tensors);self.run_fold(good=False)

    def test_single_rounding_and_order(self):
        self.make_adapter(a=[0.002,0.002],b=[1]*6)
        out,_=self.run_fold(self.adapter,self.adapter,self.adapter)
        expected=f32(f32(f32(1+f32(.002))+f32(.002))+f32(.002))
        self.assertEqual(values(out/'model.safetensors',self.name),[bf16(expected)]*12)
        other=self.root/'negative.safetensors'
        self.make_adapter(a=[16777216,16777216],b=[1]*6)
        self.make_adapter(path=other,a=[-16777216,-16777216],b=[1]*6)
        first,_=self.run_fold(self.adapter,other);second,_=self.run_fold(other,self.adapter)
        self.assertNotEqual(first,second)
        self.assertEqual(values(first/'model.safetensors',self.name),[bf16(0)]*12)
        self.assertEqual(values(second/'model.safetensors',self.name),[bf16(1)]*12)

    def test_ties_subnormal_and_overflow(self):
        self.make_adapter(a=[2**-8,3*2**-8],b=[1]*6)
        out,_=self.run_fold();self.assertEqual(values(out/'model.safetensors',self.name),[bf16(1),bf16(1+2**-6)]*6)
        # Cancellation through the base plus a subnormal update checks gradual underflow.
        self.make_adapter(a=[-1,-1],b=[1]*6)
        sub=self.root/'sub.safetensors';self.make_adapter(path=sub,a=[2**-134,3*2**-134],b=[1]*6)
        out,_=self.run_fold(self.adapter,sub,env={"H3_TEST_LORA_FLUSH":"1"})
        self.assertEqual(values(out/'model.safetensors',self.name),[0,2]*6)
        shutil.rmtree(self.cache)
        self.make_adapter(a=[3.402823466e38,3.402823466e38],b=[1]*6)
        self.run_fold(good=False)

    def test_rank_column_row_tiles(self):
        # Crossing all three tile boundaries also catches rank-block resets.
        rng=random.Random(101);r,c,k=35,517,131
        A=[f32(rng.uniform(-.125,.125)) for _ in range(k*c)]
        B=[f32(rng.uniform(-.125,.125)) for _ in range(r*k)]
        base=self.root/'matrix';base.mkdir();(base/'config.json').write_text('{}')
        name='projection.weight';tensor_file(base/'model.safetensors',[(name,[r,c],[1]*(r*c),'BF16')])
        adapter=self.root/'matrix.safetensors'
        tensor_file(adapter,[('projection.lora_A.weight',[k,c],A,'F32'),('projection.lora_B.weight',[r,k],B,'F32')])
        first,_=self.run_fold(adapter,base=base,memory=1);raw=(first/'model.safetensors').read_bytes();shutil.rmtree(first.parent)
        second,_=self.run_fold(adapter,base=base,memory=512)
        self.assertEqual(raw,(second/'model.safetensors').read_bytes())
        got=values(second/'model.safetensors',name)
        for row,col in [(0,0),(31,511),(32,512),(34,516),(3,228)]:
            dot=0.0
            for rank in range(k):dot=f32(dot+f32(B[row*k+rank]*A[rank*c+col]))
            self.assertEqual(got[row*c+col],bf16(f32(1+dot)))

    def test_lease_blocks_repair(self):
        out,_=self.run_fold()
        env={**os.environ,'H3_TEST_MIN_AVAILABLE_MEMORY_BYTES':'0','H3_TEST_LORA_LEASE_SECONDS':'3'}
        holder=subprocess.Popen([HOST,str(self.base),str(self.cache),'512',str(self.adapter)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env)
        self.assertEqual(Path(holder.stdout.readline().strip()),out)
        (out/'model.safetensors').write_bytes(b'broken')
        writer=subprocess.Popen([HOST,str(self.base),str(self.cache),'512',str(self.adapter)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            env={**os.environ,'H3_TEST_MIN_AVAILABLE_MEMORY_BYTES':'0'})
        time.sleep(.25);self.assertIsNone(writer.poll());self.assertFalse(list(self.cache.glob('invalid/*')))
        holder.terminate();holder.communicate(timeout=5)
        stdout,stderr=writer.communicate(timeout=5);self.assertEqual(writer.returncode,0,stderr)
        self.assertEqual(Path(stdout.strip()),out)

    def test_folded_parent(self):
        parent,_=self.run_fold();child_cache=self.root/'child-cache'
        child,_=self.run_fold(base=parent,cache=child_cache)
        self.assertEqual(values(child/'model.safetensors',self.name),[bf16(2),bf16(3)]*6)
        self.assertNotEqual(parent.parent.name,child.parent.name)

    def test_cache_overlap_rejected_without_writes(self):
        self.run_fold(cache=self.base/'nested-cache',good=False)
        self.assertFalse((self.base/'nested-cache').exists())
        link=self.root/'cache-link';link.symlink_to(self.base,target_is_directory=True)
        self.run_fold(cache=link/'nested',good=False);self.assertFalse((self.base/'nested').exists())

    def test_strict_safetensors_and_metadata(self):
        original=self.adapter.read_bytes()
        n,=struct.unpack('<Q',original[:8]);head=json.loads(original[8:8+n]);payload=original[8+n:]
        cases=[]
        dup=original[8:8+n].decode().strip();dup=dup[:-1]+',"__metadata__":{"alpha":"1","alpha":"2"}}';cases.append((dup.encode(),payload))
        bad=json.loads(json.dumps(head));key=next(iter(bad));bad[key]['data_offsets']=[1,bad[key]['data_offsets'][1]+1];cases.append((json.dumps(bad).encode(),payload))
        bad=json.loads(json.dumps(head));bad[key]['shape']=[2**63,2**63];cases.append((json.dumps(bad).encode(),payload))
        bad=json.loads(json.dumps(head));bad[key]['dtype']='F8';cases.append((json.dumps(bad).encode(),payload))
        cases.extend([(original[8:8+n],payload[:-1]),(original[8:8+n],payload+b'x')])
        for h,data in cases:
            self.adapter.write_bytes(struct.pack('<Q',len(h))+h+data);self.run_fold(good=False)

    def test_cancel_and_failed_repair_preserve_old_entry(self):
        out,_=self.run_fold();(out/'model.safetensors').write_bytes(b'bad')
        r=subprocess.run([HOST,str(self.base),str(self.cache),'512',str(self.adapter)],text=True,capture_output=True,
            env={**os.environ,'H3_TEST_MIN_AVAILABLE_MEMORY_BYTES':'0','H3_TEST_LORA_FAULT':'repair'})
        self.assertNotEqual(r.returncode,0);self.assertTrue(out.exists());self.assertFalse(list(self.cache.glob('staging/*')))
        self.run_fold()

    def test_zero_strength_base_nonfinite(self):
        base=self.root/'bad-base';shutil.copytree(self.base,base)
        tensor_file(base/'model.safetensors',[(self.name,[6,2],[float('nan')]+[1]*11,'BF16')])
        self.run_fold(str(self.adapter)+':0',base=base,good=False)

    def test_canonical_key_recipe_and_negative_zero(self):
        def expected(recipe,scale):
            h=hashlib.sha256()
            def number(n):h.update(struct.pack('<Q',n))
            def string(s):
                raw=s.encode();number(len(raw));h.update(raw)
            string(recipe);number(0)
            files=sorted(self.base.iterdir());number(len(files))
            for file in files:
                string(file.name);data=file.read_bytes();number(len(data));h.update(hashlib.sha256(data).digest())
            number(1);h.update(hashlib.sha256(self.adapter.read_bytes()).digest());h.update(struct.pack('<d',scale))
            return h.hexdigest()
        out,_=self.run_fold(str(self.adapter)+':0')
        self.assertEqual(out.parent.name,expected('h3-runtime-lora/schema=1/mapping=1/recipe=1',0.0))
        self.assertNotEqual(out.parent.name,expected('h3-runtime-lora/schema=1/mapping=1/recipe=2',0.0))
        other,_=self.run_fold(str(self.adapter)+':-0');self.assertEqual(other,out)


if __name__=='__main__':unittest.main()
