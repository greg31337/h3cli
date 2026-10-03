#!/usr/bin/env python3
"""Adversarial, checksummed container tests independent of the C decoder."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get('H3_SAMPLER_TEST_BINARY', str(ROOT/'bin/sampler_tests')))

def entries(data):
    result=[]
    for i in range(struct.unpack_from('<I',data,20)[0]):
        kind,version,flags,dtype,count,offset,size=struct.unpack_from('<4I3Q',data,96+i*72)
        result.append([kind,version,flags,dtype,count,bytearray(data[offset:offset+size])])
    return result

def build(parts):
    header=bytearray(96); header[:8]=b'H3SAMPLE'
    struct.pack_into('<4IQI',header,8,2,96,0,len(parts),0,0x01020304)
    table=bytearray(); payload=bytearray(); offset=96+72*len(parts)
    for kind,version,flags,dtype,count,data in parts:
        table += struct.pack('<4I3Q',kind,version,flags,dtype,count,offset,len(data))+hashlib.sha256(data).digest()
        payload += data; offset += len(data)
    struct.pack_into('<Q',header,24,offset)
    result=header+table+payload
    result[64:96]=hashlib.sha256(result).digest()
    return result

def global_checksum(data):
    data[64:96]=bytes(32); data[64:96]=hashlib.sha256(data).digest(); return data

class ContainerTests(unittest.TestCase):
    total_cases=0
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='h3-sample-corruption-')
        cls.path=Path(cls.temp.name)/'state.h3sample'
        subprocess.run([str(BINARY),'--fixture',str(cls.path)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        cls.original=cls.path.read_bytes()
    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        print(f'ok: {cls.total_cases} adversarial container cases')
    def check(self,data,accepted=False,error=None):
        type(self).total_cases+=1
        self.path.write_bytes(data)
        result=subprocess.run([str(BINARY),'--load',str(self.path)],capture_output=True,text=True)
        self.assertEqual(result.returncode==0,accepted,result.stderr)
        if error: self.assertIn(error,result.stderr)
    def test_adaptive_budget_format(self):
        fixture=Path(self.temp.name)/'adaptive.h3sample'
        subprocess.run([str(BINARY),'--adaptive-fixture',str(fixture)],cwd=ROOT,check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        original=fixture.read_bytes();parts=entries(original)
        policy=next(p for p in parts if p[0]==40)
        self.assertEqual(policy[1:3],[3,1]);self.assertEqual(len(policy[-1]),52)
        self.assertEqual(struct.unpack_from('<Q',policy[-1],36)[0],4096*1048576)
        self.check(original,True)
        legacy=entries(original);p=next(p for p in legacy if p[0]==40)
        p[1]=1;p[-1]=p[-1][:36];p[4]=36;self.check(build(legacy),False)
        for version in (0,2,99):
            altered=entries(original);next(p for p in altered if p[0]==40)[1]=version
            self.check(build(altered),error='unsupported required')
        for kind in (40,41,42):
            self.check(build([p for p in parts if p[0]!=kind]))
            altered=entries(original);next(p for p in altered if p[0]==kind)[2]=0;self.check(build(altered))
        self.assertEqual(struct.unpack_from('<fi',policy[-1],44),(struct.unpack('<f',struct.pack('<f',.04))[0],1))
        for offset,fmt,values in [(44,'f',[-.01,1.01,float('nan'),float('inf')]),(48,'i',[0,-1,17,2**31-1])]:
            for value in values:
                altered=entries(original);p=next(p for p in altered if p[0]==40)
                struct.pack_into('<'+fmt,p[-1],offset,value);self.check(build(altered))
        for threshold in (0.,.04,1.):
            for hits in (1,2,3,16):
                altered=entries(original);p=next(p for p in altered if p[0]==40)
                struct.pack_into('<fi',p[-1],44,threshold,hits);self.check(build(altered),True)
        for budget in (0,1,1048576):
            altered=entries(original);p=next(p for p in altered if p[0]==40)
            struct.pack_into('<Q',p[-1],36,budget);self.check(build(altered))
        for count in (0,1,2**64-1):
            altered=entries(original);p=next(p for p in altered if p[0]==40)
            struct.pack_into('<Q',p[-1],28,count);self.check(build(altered))
        for kind in (41,42):
            altered=entries(original);p=next(p for p in altered if p[0]==kind)
            del p[-1][-2:];p[4]-=1;self.check(build(altered),error='before allocation')
        for offset,value in ((8,2),(12,17),(16,2),(20,999),(24,999)):
            altered=entries(original);p=next(p for p in altered if p[0]==40)
            struct.pack_into('<I',p[-1],offset,value)
            self.check(build(altered),error='before allocation')
        altered=entries(original);p=next(p for p in altered if p[0]==40)
        struct.pack_into('<Q',p[-1],36,8192*1048576);self.check(build(altered),True)
        altered=entries(original);p=next(p for p in altered if p[0]==41)
        p[-1][:2]=b'\x80\x7f';self.check(build(altered),error='nonfinite')
        # Sparse file: bound is checked by fstat, before mmap or hashing.
        with self.path.open('wb') as f:f.truncate(16*1024**3+1)
        r=subprocess.run([str(BINARY),'--load',str(self.path)],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)

    def test_roundtrip(self): self.check(self.original,True)
    def test_quant_execution_extension(self):
        for mode in (1,2):
            parts=entries(self.original)
            parts.append([34,1,1,1,8,bytearray(struct.pack('<II',mode,2))])
            self.check(build(parts),True)
            parts[-1][2]=0;self.check(build(parts),error='quantization policy must be required')
            parts[-1][2]=1
            for version in (1,99):
                struct.pack_into('<I',parts[-1][-1],4,version)
                self.check(build(parts),error='quantization')
    def test_attention_execution_extension(self):
        self.assertNotIn(35,[p[0] for p in entries(self.original)])
        for mode in (1,2):
            parts=entries(self.original)+[[35,1,1,1,12,bytearray(struct.pack('<3I',mode,1,1))]]
            self.check(build(parts),True)
            parts[-1][2]=0;self.check(build(parts),error='attention policy must be required')
            parts[-1][2]=1;parts[-1][1]=2;self.check(build(parts),error='unsupported required')
            parts[-1][1]=1
            for values in [(0,1,1),(3,1,1),(mode,0,1),(mode,2,1),(mode,1,0),(mode,1,2)]:
                parts[-1][-1]=bytearray(struct.pack('<3I',*values));self.check(build(parts))
            for length in [0,4,8,11,13,16]:
                parts[-1][-1]=bytearray(length);parts[-1][4]=length;self.check(build(parts))
    def test_cuda_sol_required_policy(self):
        parts=[x for x in entries(self.original) if x[0] not in (17,18,19,20)]
        identity=next(x[-1] for x in parts if x[0]==1)
        struct.pack_into('<i',identity,12,1);struct.pack_into('<i',identity,16+5*4,1)
        history=next(x[-1] for x in parts if x[0]==16)
        struct.pack_into('<ii',history,0,-1,-1);history[8:]=bytes([1]*20)
        parts.append([35,1,1,1,12,bytearray(struct.pack('<3I',3,1,1))])
        self.check(build(parts),error='missing required')
        policy=bytearray(struct.pack('<2I5i3f',1,1,32,64,1,1,1,1.,.75,-1.))
        parts.append([38,1,1,1,len(policy),policy])
        self.check(build(parts),True)
        parts[-1][2]=0;self.check(build(parts));parts[-1][2]=1
        for off,fmt,value in [(0,'I',2),(4,'I',2),(8,'i',16),(12,'i',128),(28,'f',float('nan')),(32,'f',1.5)]:
            altered=entries(build(parts));struct.pack_into('<'+fmt,altered[-1][-1],off,value);self.check(build(altered))
        for length in [0,12,36,39,41]:
            altered=entries(build(parts));altered[-1][-1]=bytearray(length);altered[-1][4]=length;self.check(build(altered))
        altered=entries(build(parts));next(x for x in altered if x[0]==35)[-1]=bytearray(struct.pack('<3I',1,1,1));self.check(build(altered))

    def test_q8_weight_extension(self):
        native=bytearray(struct.pack('<9I2f3If4IQ50I',1,0,6,1,1,32,64,1,1,
                                    1.,.1,0,0,0,-1.,0,0,4096,512,0,*([0]*50)))
        original=entries(self.original)+[[36,1,1,1,len(native),native]]
        self.check(build(original),True)
        # The version-6 native marker remains fixed in the saved wire format.
        for marker in (0,2):
            altered=entries(build(original))
            struct.pack_into('<I',altered[-1][-1],12,marker)
            self.check(build(altered))
        for kernel in (0,1):
            parts=original+[[37,1,1,1,16,bytearray(struct.pack('<4I',1,1,64,kernel))]]
            self.check(build(parts),True)
            parts[-1][2]=0;self.check(build(parts),error='Q8 weight policy must be required')
            parts[-1][2]=1
            for index,value in ((0,0),(0,2),(1,99),(2,32),(3,2)):
                parts[-1][-1]=bytearray(struct.pack('<4I',1,1,64,kernel))
                struct.pack_into('<I',parts[-1][-1],4*index,value);self.check(build(parts))
            for length in (0,4,12,15,17,20):
                parts[-1][4]=length;parts[-1][-1]=bytearray(length);self.check(build(parts))
        # A required Q8 policy cannot be attached to the BF16 oracle backend.
        self.check(build(entries(self.original)+[[37,1,1,1,16,bytearray(struct.pack('<4I',1,1,64,0))]]))
    def test_truncations(self):
        for n in [0,7,8,95,96,100,1024,len(self.original)-1]:
            with self.subTest(length=n): self.check(self.original[:n])
    def test_checksums(self):
        for n in [64,95,100,170,3000,len(self.original)-1]:
            data=bytearray(self.original); data[n]^=128
            with self.subTest(offset=n): self.check(data,error='whole-file checksum')
        data=bytearray(self.original); data[-1]^=128; self.check(global_checksum(data),error='section checksum')
    def test_missing_required_sections(self):
        for kind in list(range(1,21))+[24,32]:
            with self.subTest(section=kind): self.check(build([p for p in entries(self.original) if p[0]!=kind]))
    def test_optional_forward_compatibility(self):
        parts=[p for p in entries(self.original) if p[0]<21 or p[0] in [24,32]]
        self.check(build(parts),True) # original noise and diagnostics are optional
        parts.append([999,42,0,99,7,bytearray(b'unknown')]); self.check(build(parts),True)
        parts[-1][2]=1; self.check(build(parts),error='unsupported required')
        parts=entries(self.original); next(p for p in parts if p[0]==23)[1]=42; self.check(build(parts),True)
    def test_duplicate_required_and_optional(self):
        for kind in [1,12,23,999]:
            parts=entries(self.original)
            p=next((p.copy() for p in parts if p[0]==kind),[999,1,0,1,0,bytearray()])
            if kind==999: parts.append(p.copy())
            parts.append(p)
            self.check(build(parts),error='duplicate')
    def test_counts_offsets_flags_versions(self):
        for offset,value,width in [(8,1,4),(16,1,4),(20,257,4),(24,2**64-1,8),(32,0,4),(36,1,4),
                                    (96+8,2,4),(96+24,0,8),(96+32,2**64-1,8),(96+16,2**64-1,8)]:
            data=bytearray(self.original); struct.pack_into('<I' if width==4 else '<Q',data,offset,value)
            with self.subTest(offset=offset): self.check(global_checksum(data))
        for kind in [1,5,11,12,16,20]:
            parts=entries(self.original); next(p for p in parts if p[0]==kind)[1]=2
            self.check(build(parts),error='unsupported required')
        parts=entries(self.original); parts[0][2]=0; self.check(build(parts),error='required section marked optional')
    def test_typed_payloads(self):
        for kind,offset,fmt,value in [(1,0,'I',1),(1,8,'i',-1),(1,8,'i',21),(1,4,'i',1001),
                                       (10,0,'Q',2**64-1),(11,0,'i',1001),(11,4,'f',float('nan')),
                                       (12,0,'f',float('inf')),(16,0,'i',20),(14,0,'I',42),
                                       (4,0,'Q',2**64-1),(23,0,'Q',2**64-1),(24,0,'I',42),(24,40,'I',1),(24,12,'Q',2**64-1)]:
            parts=entries(self.original); p=next(p for p in parts if p[0]==kind)
            struct.pack_into('<'+fmt,p[-1],offset,value)
            with self.subTest(section=kind,offset=offset,value=value): self.check(build(parts))
        for kind in [5,7,8,9,10,11,12,13,17,18,19,20]:
            parts=entries(self.original); p=next(p for p in parts if p[0]==kind)
            del p[-1][-4:]; p[4]=len(p[-1])//({2:4,3:2}.get(p[3],1))
            self.check(build(parts))
    def test_reuse_one_rejects_skipped_evaluations(self):
        parts=entries(self.original)
        identity=next(p for p in parts if p[0]==1)
        struct.pack_into('<i',identity[-1],12,1) # explicit reuse interval
        struct.pack_into('<i',identity[-1],16+5*4,1) # params.denoise_reuse
        reuse=next(p for p in parts if p[0]==16)
        struct.pack_into('<ii',reuse[-1],0,-1,-1)
        reuse[-1][8:]=bytes([1]*20)
        self.check(build(parts),True)
        reuse[-1][8+4]=0
        self.check(build(parts),error='sigma/reuse schedule')
    def test_native_gpu_history_sections(self):
        parts=entries(self.original)
        identity=next(p[-1] for p in parts if p[0]==1)
        struct.pack_into('<I',identity,len(identity)-4,1)
        for kind,latent in [(25,12),(26,12),(27,13),(28,13)]:
            count=len(next(p[-1] for p in parts if p[0]==latent))//4
            parts.append([kind,1,1,3,count,bytearray(b'\xc1\x7f'*count)])
        parts=[p for p in parts if p[0] not in range(17,21)]
        self.check(build(parts),True)
        for kind in range(25,29):
            self.check(build([p for p in parts if p[0]!=kind]),error='missing required')
            altered=entries(build(parts)); next(p for p in altered if p[0]==kind)[1]=2
            self.check(build(altered),error='unsupported required')
        altered=entries(build(parts)); p=next(p for p in altered if p[0]==25)
        del p[-1][-2:]; p[4]-=1; self.check(build(altered))
    def test_core_residual_shape_and_count(self):
        parts=entries(self.original)
        identity=next(p[-1] for p in parts if p[0]==1)
        struct.pack_into('<i',identity,12,1)
        struct.pack_into('<i',identity,16+5*4,1)
        struct.pack_into('<i',identity,16+7*4,4)
        history=next(p[-1] for p in parts if p[0]==16)
        struct.pack_into('<ii',history,0,-1,-1); history[8:]=bytes([1]*20)
        rows=struct.unpack_from('<Q',next(p[-1] for p in parts if p[0]==10))[0]
        execution=next(p[-1] for p in parts if p[0]==24)
        struct.pack_into('<IIIQQQ',execution,0,1,4,1,rows,5376,rows*5376)
        parts.append([29,1,1,3,rows*5376,bytearray(rows*5376*2)])
        self.check(build(parts),True)
        self.check(build([p for p in parts if p[0]!=29]),error='missing required')
        for offset,fmt,value in [(4,'I',0),(8,'I',0),(12,'Q',rows+1),(20,'Q',5375),(28,'Q',rows*5376+1)]:
            altered=entries(build(parts)); payload=next(p[-1] for p in altered if p[0]==24)
            struct.pack_into('<'+fmt,payload,offset,value); self.check(build(altered))
    def test_optional_prepared_cache_structure(self):
        parts=entries(self.original)
        payload=struct.pack('<I32sQ',1,bytes(32),2)
        payload+=struct.pack('<IQ4H',1,4,0,0x8000,0x7fc1,0xffff)
        payload+=struct.pack('<IQ2H',2,2,0x3f80,0xbf80)
        parts.append([30,1,0,1,len(payload),bytearray(payload)])
        self.check(build(parts),True)
        altered=entries(build(parts)); next(p for p in altered if p[0]==30)[1]=2
        self.check(build(altered),True)
        for offset,fmt,value in [(36,'Q',53),(44+4,'Q',2**64-1),(64,'I',1)]:
            altered=entries(build(parts)); payload=next(p[-1] for p in altered if p[0]==30)
            struct.pack_into('<'+fmt,payload,offset,value); self.check(build(altered))
    def test_trailing_bytes(self): self.check(self.original+b'garbage')

    def test_refvideo_pipeline_provenance(self):
        for recipe in (1, 99):
            parts=entries(self.original); payload=next(p[-1] for p in parts if p[0]==32)
            struct.pack_into('<I',payload,0,recipe);self.check(build(parts))
        for flags,version in [(0,1),(1,2)]:
            parts=entries(self.original);p=next(p for p in parts if p[0]==32)
            p[2]=flags;p[1]=version;self.check(build(parts))
        parts=entries(self.original);p=next(p for p in parts if p[0]==32)
        p[-1]=bytearray();p[4]=0;self.check(build(parts))

if __name__=='__main__': unittest.main()
