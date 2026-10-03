#!/usr/bin/env python3
"""Verify released FL2VA/Ref2VA text shards are identical and record hashes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/scalingfix-validation'
def main():
    a=ROOT/'models/MiniMax-H3/FL2VA/text_encoder';b=ROOT/'models/MiniMax-H3/Ref2VA/text_encoder'
    names=sorted(set(json.loads((a/'model.safetensors.index.json').read_text())['weight_map'].values()));rows={}
    for name in names:
        h=hashlib.sha256();same=True
        with (a/name).open('rb') as x,(b/name).open('rb') as y:
            while True:
                u=x.read(8*1024*1024);v=y.read(8*1024*1024)
                if u!=v:same=False;break
                if not u:break
                h.update(u)
        rows[name]={'identical':same,'sha256':h.hexdigest() if same else None}
    (OUT/'text-weight-identity.json').write_text(json.dumps(rows,indent=2)+'\n')
    assert all(x['identical'] for x in rows.values()),'Use the matching checkpoint for each presentation; weights differ'
if __name__=='__main__':main()
