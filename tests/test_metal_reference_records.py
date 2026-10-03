"""Reject broken evidence before expensive Reference qualification."""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from array import array
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metal_reference_qualify import av_payload, validate_run, check_pair, quality_acceptance, check_hashes


class Records(unittest.TestCase):
    def test_acceptance_is_not_transferred_to_changed_binary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);binary=root/'bin/h3cli'
            binary.parent.mkdir()
            binary.write_bytes(b'changed implementation')
            contract=root/'contract.json';contract.write_text('{}')
            acceptance=root/'acceptance.json';acceptance.write_text(json.dumps({
                'version':1,'binary_sha256':'old executable','geometry':[640,480,243],
                'maximum_evaluations':6,'mixed_recipe':4,'evidence_sha256':{}}))
            with self.assertRaisesRegex(ValueError,'does not apply'):
                quality_acceptance(acceptance,binary,contract)
    def test_changed_accepted_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'evidence';path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Changed frozen evidence'):
                check_hashes({str(path):'previous checksum'})

    def state(self,value=1.):
        b=bytearray(160);b[:8]=b'H3AV\r\n\x1a\n';struct.pack_into('<QQ',b,72,4,4)
        b+=struct.pack('<ff',value,2.);b[128:160]=hashlib.sha256(b[:128]+b[160:]).digest();return b
    def test_av_rejects_nonfinite_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.h3av';p.write_bytes(self.state());av_payload(p)
            for value in (float('nan'),float('inf')):
                p.write_bytes(self.state(value))
                with self.assertRaises(ValueError):av_payload(p)
            b=self.state();b[-1]^=1;p.write_bytes(b)
            with self.assertRaisesRegex(ValueError,'checksum'):av_payload(p)
            p.write_bytes(self.state()[:-1])
            with self.assertRaisesRegex(ValueError,'lengths'):av_payload(p)
    def test_teacher_is_never_an_independent_clip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'record.json').write_text(json.dumps({'teacher_from':'reference'}))
            with self.assertRaisesRegex(ValueError,'nonindependent'):validate_run(p,True)
    def test_continuation_requires_source_tail_not_only_equal_outputs(self):
        def write(path,context=None,wrong=False):
            vt,at,area=72,405,4;values=[]
            for channels,time,spatial,prefix in ((24,vt,area,12+15*((context-39)//51) if context else 0),
                                                (64,at,1,65+85*((context-39)//51) if context else 0)):
                for channel in range(channels):
                    for t in range(time):
                        source_t=time-prefix+t if t<prefix else t
                        values.extend([float(channel*1000+source_t)]*spatial)
            if wrong:values[0]+=1
            body=array('f',values)
            if sys.byteorder!='little':body.byteswap()
            h=bytearray(160);h[:8]=b'H3AV\r\n\x1a\n'
            struct.pack_into('<10I',h,24,32,32,243,vt,2,2,at,24,32,2)
            struct.pack_into('<QQ',h,72,24*vt*area*4,64*at*4)
            b=h+body.tobytes();b[128:160]=hashlib.sha256(b[:128]+b[160:]).digest();path.write_bytes(b)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.h3av';write(source)
            r=root/'reference';c=root/'candidate';r.mkdir();c.mkdir()
            for context in (39,90,141,192):
                record={'continuation_context':context,'continuation_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
                with patch('metal_reference_qualify.validate_run',return_value=record):
                    write(r/'result.h3av',context);write(c/'result.h3av',context)
                    self.assertTrue(check_pair(r,c,source)['prefix']['video']['byte_identical_to_source'])
                    # Equal reference/candidate prefixes are insufficient when
                    # both are wrong relative to the inherited source.
                    write(r/'result.h3av',context,True);write(c/'result.h3av',context,True)
                    with self.assertRaisesRegex(ValueError,'Inherited video'):check_pair(r,c,source)


if __name__=='__main__':unittest.main()
