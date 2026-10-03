#!/usr/bin/env python3
"""Deterministic regular-file bundle writer and runtime dependency audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import zlib

MAGIC=b'H3CLI_BUNDLE_V1\0'
ENVIRONMENT={
 'H3_SGLANG_CUBLAS_LIBRARY':'lib/sglang/libcublas.so.13',
 'H3_SGLANG_CUDNN_LIBRARY':'lib/nvidia/cudnn/lib/libcudnn.so.9',
 'H3_SGLANG_JPEG_LIBRARY':'lib/libturbojpeg.so.0',
 'H3_SGLANG_INPUT_FFMPEG':'tools/ffmpeg',
 'H3_FFPROBE':'tools/ffprobe',
 'H3_FFMPEG':'tools/ffmpeg'}
HOST=re.compile(r'^(?:ld-linux-x86-64\.so\.2|lib(?:c|m|pthread|dl|rt|resolv|util|anl)\.so\.[0-9]+|libnss_[^.]+\.so\.[0-9]+|libcuda\.so(?:\..*)?|libnvidia-[^.]+\.so(?:\..*)?)$')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def write_json(path,record):
 Path(path).write_text(json.dumps(record,sort_keys=True,indent=2,ensure_ascii=True)+'\n')

def seal(root):
 """Verify an unpacked runtime before creating its local readiness metadata.

 The marker is machine-local, excluded from release bytes, and cannot bypass a
 content check. A copied runtime must be resealed before direct-core execution.
 """
 manifest_path=root/'share/h3cli/runtime.json'
 manifest=json.loads(manifest_path.read_text());records={}
 for name,item in manifest['files'].items():
  p=root/name
  if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or not p.is_file() or p.stat().st_size!=item['size'] or sha(p)!=item['sha256']:
   raise ValueError('Cannot seal changed runtime file: '+name)
 for p in sorted(root.rglob('*')):
  if p.name=='.ready.json' or p.is_dir():continue
  if p.is_symlink() or not p.is_file():raise ValueError('Unsafe runtime member: '+str(p))
  name=str(p.relative_to(root))
  if name!='share/h3cli/runtime.json' and name not in manifest['files']:raise ValueError('Unrecorded runtime member: '+name)
  s=p.stat()
  records[name]=[s.st_size,s.st_mode&0o7777,s.st_ino,s.st_dev,s.st_mtime_ns//10**9,s.st_mtime_ns%10**9,s.st_ctime_ns//10**9,s.st_ctime_ns%10**9]
 write_json(root/'.ready.json',{'id':sha(manifest_path),'files':records})
 (root/'.ready.json').chmod(0o600)
 return {'sealed_files':len(records)}

def elf(path):
 with Path(path).open('rb') as f:return f.read(4)==b'\x7fELF'

def dynamic(path):
 return subprocess.check_output(['readelf','-d',str(path)],text=True)

def requirements(path):
 text=subprocess.check_output(['readelf','--version-info',str(path)],text=True)
 # Only imports, not versions exported by libstdc++/other providers.
 text=text.split('Version needs section',1)[-1] if 'Version needs section' in text else ''
 return sorted(set(re.findall(r'Name: (GLIBC(?:XX)?_[0-9.]+|CXXABI_[0-9.]+)',text)))

def nvidia_library(name):
 return name.startswith(('libcublas', 'libcudart.', 'libcudnn', 'libnvrtc'))

def library_destination(name):
 return ('lib/nvidia/cudnn/lib/' if name.startswith('libcudnn') else 'lib/')+name

def library_rpath(root,destination):
 return ':'.join('$ORIGIN/'+os.path.relpath(root/folder,destination.parent)
                 for folder in ('lib','lib/nvidia/cudnn/lib'))

def check_rpaths(root,path,paths):
 for entry in (entry for value in paths for entry in value.split(':')):
  if entry!='$ORIGIN' and not entry.startswith('$ORIGIN/'):
   raise ValueError(f'Nonrelative runtime search path: {path}: {paths}')
  resolved=(path.parent/entry.removeprefix('$ORIGIN').lstrip('/')).resolve()
  if not resolved.is_relative_to(root.resolve()):
   raise ValueError(f'Runtime search path escapes payload: {path}: {entry}')

def audit(root):
 report={}
 for p in sorted(root.rglob('*')):
  if not p.is_file() or not elf(p):continue
  header=subprocess.check_output(['readelf','-h',str(p)],text=True)
  if 'Advanced Micro Devices X86-64' not in header:raise ValueError(f'Non-x86-64 ELF: {p}')
  notes=subprocess.check_output(['readelf','-n',str(p)],text=True)
  isa=[line.strip() for line in notes.splitlines() if 'ISA needed:' in line]
  if any(re.search(r'x86-64-v[234]',line) for line in isa):raise ValueError(f'Nonbaseline CPU requirement: {p}: {isa}')
  versions=requirements(p)
  if any(tuple(map(int,v[6:].split('.')))>(2,35) for v in versions if v.startswith('GLIBC_')):
   raise ValueError(f'glibc >2.35 dependency: {p}: {versions}')
  table=dynamic(p)
  paths=re.findall(r'\((?:RPATH|RUNPATH)\).*?\[(.*?)\]',table)
  check_rpaths(root,p,paths)
  needed=re.findall(r'\(NEEDED\).*?\[(.*?)\]',table)
  for name in needed:
   if not HOST.fullmatch(name) and not (p.parent/name).is_file() and not (root/library_destination(name)).is_file():raise ValueError(f'Missing dependency {name} of {p}')
  report[str(p.relative_to(root))]={'requires':versions,'needed':needed,'rpath':paths,'cpu_isa':isa,'host_dependencies':[n for n in needed if HOST.fullmatch(n)]}
 return report

def assemble(root,core,search,input_ffmpeg,input_ffprobe,output_ffmpeg,provenance,reference):
 if root.exists():raise ValueError(f'Refusing to overwrite runtime: {root}')
 if Path(input_ffmpeg).resolve()!=Path(output_ffmpeg).resolve():raise ValueError('Production must use one FFmpeg for input and output')
 root.mkdir(parents=True)
 files={Path(core):'bin/h3cli',Path(input_ffmpeg):'tools/ffmpeg',Path(input_ffprobe):'tools/ffprobe',Path(output_ffmpeg):'tools/ffmpeg'}
 def locate(name):
  if '/' in name:raise ValueError(f'Non-SONAME dependency: {name}')
  for folder in search:
   p=folder/name
   if p.is_file():return p.resolve()
  raise ValueError(f'Cannot locate pinned library: {name}')
 for name in ('libcublas.so.13','libcublasLt.so.13','libcudnn.so.9','libturbojpeg.so.0'):
  files[locate(name)]=library_destination(name)
 files[(reference/'libnvrtc.so.13').resolve(strict=True)]='lib/libnvrtc.so.13'
 # Ordinary CUDA and the explicitly loaded SGLang cuBLAS retain their distinct
 # versions. RTLD_DEEPBIND plus each provider's original relative RUNPATH keeps the
 # matching cuBLASLt beside it; combining these SONAMEs changes arithmetic.
 for name in ('libcublas.so.13','libcublasLt.so.13'):
  files[(reference/name).resolve(strict=True)]='lib/sglang/'+name
 for folder in search:
  for pattern in ('libcudnn_*.so.9',):
   for p in sorted(folder.glob(pattern)):
    files.setdefault(p.resolve(),library_destination(p.name))
 for p in sorted(reference.glob('libnvrtc-builtins*.so.*')):
  files[p.resolve()]='lib/'+p.name
 pending=list(files.items());destinations=set();origins={};providers={}
 while pending:
  source,relative=pending.pop(0)
  if relative in destinations:
   if sha(source)!=origins[relative]:raise ValueError(f'Conflicting runtime provider: {relative}')
   continue
  destinations.add(relative);origins[relative]=sha(source)
  providers[relative]={'input_sha256':origins[relative],'builder_path':str(source)}
  dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);dest.chmod(0o755 if relative.startswith(('bin/','tools/')) else 0o644)
  if not elf(source):continue
  for name in re.findall(r'\(NEEDED\).*?\[(.*?)\]',dynamic(source)):
   if not HOST.fullmatch(name):
    if relative.startswith('lib/sglang/') and (reference/name).is_file():
     pending.append(((reference/name).resolve(),'lib/sglang/'+name))
    else:pending.append((locate(name),library_destination(name)))
  # NVIDIA redistributables remain byte-for-byte original. Preserve cuDNN's
  # wheel layout so every original $ORIGIN search path stays inside the payload.
  if not nvidia_library(source.name) and '(NEEDED)' in dynamic(dest):
   subprocess.run(['patchelf','--force-rpath','--set-rpath',library_rpath(root,dest),str(dest)],check=True)
  if nvidia_library(source.name) and sha(dest)!=origins[relative]:
   raise ValueError('Modified NVIDIA redistributable: '+relative)
 report=audit(root)
 inventory={}
 for p in sorted(root.rglob('*')):
  if p.is_file():
   name=str(p.relative_to(root));inventory[name]={'sha256':sha(p),'size':p.stat().st_size,'mode':p.stat().st_mode&0o777,**providers[name]}
 for name,path in ENVIRONMENT.items():
  if path not in inventory:raise ValueError(f'Missing runtime resource {name}: {path}')
 metadata=root/'share/h3cli';metadata.mkdir(parents=True,exist_ok=True)
 write_json(metadata/'runtime.json',{'schema':1,'environment':ENVIRONMENT,'files':inventory,'build':provenance,'abi':report})
 return report

def package(root,launcher,output):
 records=[];total=0;payload_hash=hashlib.sha256()
 output.parent.mkdir(parents=True,exist_ok=True)
 # A temporary output is never mistaken for a complete executable.
 with tempfile.NamedTemporaryFile(dir=output.parent,prefix='.'+output.name+'.',delete=False) as f:
  temp=Path(f.name)
  try:
   with Path(launcher).open('rb') as src:shutil.copyfileobj(src,f)
   start=f.tell()
   for p in sorted(root.rglob('*')):
    if p==root/'.ready.json':continue
    if p.is_symlink():raise ValueError(f'Bundle files must be regular (flatten library symlinks): {p}')
    if p.is_dir():continue
    if not p.is_file():raise ValueError(f'Special file in runtime: {p}')
    name=str(p.relative_to(root))
    if not name or name.startswith('/') or any(x in ('','.','..') for x in name.split('/')) or any(ord(c)<32 for c in name) or '\\' in name:
     raise ValueError(f'Unsafe bundle path: {name}')
    offset=f.tell()-start;h=hashlib.sha256();size=0;compress=zlib.compressobj(9)
    with p.open('rb') as src:
     for data in iter(lambda:src.read(1<<20),b''):
      h.update(data);size+=len(data);chunk=compress.compress(data);f.write(chunk);payload_hash.update(chunk)
    chunk=compress.flush();f.write(chunk);payload_hash.update(chunk)
    records.append({'path':name,'offset':offset,'packed':f.tell()-start-offset,'size':size,'sha256':h.hexdigest(),'mode':0o755 if p.stat().st_mode&0o111 else 0o644})
    total+=size
   length=f.tell()-start
   manifest=json.dumps({'schema':1,'files':records,'requirements':{'architecture':'x86_64','glibc':'2.35','driver':'580.126.20'}},sort_keys=True,separators=(',',':')).encode()
   manifest_offset=f.tell();f.write(manifest)
   footer=struct.pack('<16sQQQQ32s32sQQ',MAGIC,manifest_offset,len(manifest),start,length,hashlib.sha256(manifest).digest(),payload_hash.digest(),total,0)
   assert len(footer)==128
   f.write(footer);f.flush();os.fsync(f.fileno());temp.chmod(0o755);temp.replace(output)
  finally:temp.unlink(missing_ok=True)
 result={'sha256':sha(output),'bytes':output.stat().st_size,'payload_sha256':payload_hash.hexdigest(),'unpacked_bytes':total,'files':len(records)}
 write_json(output.with_name(output.name+'.json'),result)
 output.with_name(output.name+'.sha256').write_text(result['sha256']+'  '+output.name+'\n')
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 b=sub.add_parser('bundle');b.add_argument('--root',type=Path,required=True);b.add_argument('--launcher',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
 a=sub.add_parser('assemble');a.add_argument('--root',type=Path,required=True);a.add_argument('--core',type=Path,required=True);a.add_argument('--search',type=Path,action='append',required=True)
 for name in ('input-ffmpeg','input-ffprobe','output-ffmpeg'):a.add_argument('--'+name,type=Path,required=True)
 a.add_argument('--provenance',type=Path,required=True)
 a.add_argument('--reference',type=Path,required=True,help='Pinned SGLang cuBLAS directory, separate from toolkit cuBLAS')
 c=sub.add_parser('audit');c.add_argument('root',type=Path)
 c=sub.add_parser('seal');c.add_argument('root',type=Path)
 args=p.parse_args()
 if args.action=='bundle':result=package(args.root,args.launcher,args.output)
 elif args.action=='audit':result=audit(args.root)
 elif args.action=='seal':result=seal(args.root)
 else:result=assemble(args.root,args.core,args.search,args.input_ffmpeg,args.input_ffprobe,args.output_ffmpeg,json.loads(args.provenance.read_text()),args.reference)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
