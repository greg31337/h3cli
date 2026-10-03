#!/usr/bin/env python3
"""Qualify a prebuilt Linux release without installing application dependencies.

Runs the immutable 204-output gate in each pinned userland. A failed golden
comparison remains a failure; optional baseline comparison is diagnostic only.
The Python harness stays on the host. Only child programs enter the clean OCI
filesystem, with explicit driver/model/artifact/fixture bindings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/linux'))
import environment
import package
import cuda_reference_regression as gate

class Maps:
    def __init__(self,roots):self.roots=[str(r) for r in roots];self.records={};self.stopped=threading.Event()
    def program_mapping(self,path):
        # A harness Python process may load bundled libcrypto via LD_LIBRARY_PATH.
        # That does not make its Python modules application dependencies.
        for root in self.roots:
            if path.startswith(root+'/'):
                parts=Path(path[len(root)+1:]).parts
                if 'bin' in parts[:-1] or 'tools' in parts[:-1]:return True
        return False
    def sample(self):
        while not self.stopped.wait(.05):
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    executable=os.readlink(p/'exe')
                    mappings=set()
                    for line in (p/'maps').read_text().splitlines():
                        fields=line.split(maxsplit=5)
                        if len(fields)==6 and fields[5].startswith('/'):mappings.add(fields[5])
                    # Under PRoot /proc/PID/exe can identify its temporary ELF
                    # loader. The real core/media object still appears in maps.
                    if not any(executable.startswith(r+'/') for r in self.roots) and not any(self.program_mapping(path) for path in mappings):continue
                    record=self.records.setdefault(p.name,{'exe':executable,'mappings':set()})
                    record['mappings'].update(mappings)
                except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    def __enter__(self):self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stopped.set();self.thread.join()
    def save(self,path):package.write_json(path,{pid:{**row,'mappings':sorted(row['mappings'])} for pid,row in self.records.items()})

    def audit(self):
        allowed=re.compile(r'^(?:ld-linux-x86-64.so.2|lib(?:c|m|pthread|dl|rt|resolv|util|anl)\.so\.[0-9]+|libnss_[^.]+\.so\.[0-9]+|libcuda\.so.*|libnvidia-.*|libcudadebugger\.so.*)$')
        paths=sorted({p for row in self.records.values() for p in row['mappings'] if re.search(r'\.so(?:\.[0-9]+)*$',Path(p).name)})
        unexpected=[p for p in paths if not any(p.startswith(r+'/') for r in self.roots) and not allowed.fullmatch(Path(p).name)]
        math=any(Path(p).name.startswith('libcublas') for p in paths)
        return {'passed':bool(self.records) and math and not unexpected,
                'processes':len(self.records),'shared_objects':len(paths),
                'math_observed':math,'unexpected':unexpected,'libraries':paths}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Exact source snapshot used for the build')
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True,help='Pinned OCI/PRoot build cache')
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path,help='Unchanged-source same-device result.json; never replaces goldens')
    a=p.parse_args();source=a.source.resolve();release=a.release.resolve();cache=a.cache.resolve();model=a.model.resolve();out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    runtime=release/'linux-runtime';validation=release/'linux-validation';binary=release/'h3cli-linux-x86_64'
    package.seal(runtime)
    baseline=json.loads(a.baseline.read_text()) if a.baseline else None
    records=[];env=dict(os.environ,H3_TEST_MAX_EVALUATIONS='6',H3CLI_RUNTIME_CACHE=str(out/'cache'))
    for key in ('LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH','H3CLI_RUNTIME_ROOT','H3CLI_BUNDLE_INFO'):env.pop(key,None)
    for distro in ('ubuntu2204','ubuntu2404','debian12'):
        root=environment.prepare(cache,distro)
        # Bind only immutable artifacts/source fixtures, models, and evidence.
        # The exact source snapshot supplies provenance; its bin/ is masked.
        empty=out/'empty-bin';empty.mkdir(exist_ok=True)
        bindings=[str(source)+':'+str(source),str(empty)+':'+str(source/'bin'),str(release)+':'+str(release),str(model.parent)+':'+str(model.parent),str(out)+':'+str(out)]
        executor=environment.command(cache,root,bindings,gpu=True,cwd=str(source))
        config=out/(distro+'-executor.json');package.write_json(config,executor)
        modes=('unpacked','single-file') if distro=='ubuntu2204' else ('single-file',)
        for mode in modes:
            folder=out/(distro+'-'+mode)
            cmd=[sys.executable,str(source/'tests/cuda_reference_regression.py'),'--source',str(source),'--runtime',str(runtime),'--validation',str(validation),'--executor-config',str(config),'--model',str(model),'--out',str(folder)]
            if mode=='single-file':cmd+=['--artifact',str(binary)]
            start=time.monotonic()
            with (out/(folder.name+'.log')).open('w') as log,Maps([release,out/'cache']) as maps:
                process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
            maps.save(out/(folder.name+'-maps.json'))
            loaded=maps.audit();package.write_json(out/(folder.name+'-loaded-paths.json'),loaded)
            result=json.loads((folder/'result.json').read_text())
            row={'distro':distro,'mode':mode,'passed':result['passed'] and loaded['passed'],'goldens_passed':result['passed'],'loaded_paths':loaded,'returncode':process.returncode,'wall_seconds':time.monotonic()-start,'output_count':len(result.get('files',{})),'artifact_sha256':package.sha(binary)}
            if baseline and 'files' in result:
                actual=result['files'];old=baseline['files'];row['same_device_baseline']={'complete_set':set(actual)==set(old),'different':[name for name in old if old[name]!=actual.get(name)],'baseline_source_sha256':baseline['source_sha256']}
            row['error']=result.get('error');records.append(row);package.write_json(out/'result.json',{'complete':False,'passed':False,'checks':records})
            print(distro,mode,'goldens',row['passed'],'outputs',row['output_count'],'baseline differences',len(row.get('same_device_baseline',{}).get('different',[])),flush=True)
    package.write_json(out/'result.json',{'complete':True,'passed':all(r['passed'] for r in records),'checks':records})
    return 0 if all(r['passed'] for r in records) else 1

if __name__=='__main__':raise SystemExit(main())
