#!/usr/bin/env python3
"""Compile a fresh immutable source snapshot for serial Metal validation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from metal_native_bench import ROOT,sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():p.error('use a fresh snapshot directory')
    source=out/'source';source.mkdir(parents=True)
    files=subprocess.run(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT,capture_output=True,check=True).stdout.split(b'\0')
    hashes={}
    for raw in files:
        if not raw:continue
        name=raw.decode();path=ROOT/name
        if not path.is_file() or path.is_symlink():continue
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);hashes[name]=sha(target)
    with (out/'build.log').open('w') as log:
        subprocess.run(['make','-j4','bin/h3cli','bin/metal_fp16','bin/metal_diagnostics','bin/metal_sol','bin/metal_layout','bin/ane_tests','bin/ane_probe','bin/metal_lifetime','bin/metal_cache_lifetime','bin/metal_q8'],cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'bin').mkdir()
    for name in ('bin/h3cli','bin/metal_fp16','bin/metal_diagnostics','bin/metal_sol','bin/metal_layout','bin/ane_tests','bin/ane_probe','bin/metal_lifetime','bin/metal_cache_lifetime','bin/metal_q8'):shutil.copy2(source/name,out/name)
    (out/'build-provenance.json').write_text(json.dumps({'binary_sha256':sha(out/'bin/h3cli'),
        'source_snapshot':str(source),'source_sha256':hashes,'build_command':['make','-j4','bin/h3cli','bin/metal_fp16','bin/metal_diagnostics','bin/metal_sol','bin/metal_layout','bin/ane_tests','bin/ane_probe','bin/metal_lifetime','bin/metal_cache_lifetime','bin/metal_q8']},indent=2)+'\n')
    print(out/'bin/h3cli')
if __name__=='__main__':main()
