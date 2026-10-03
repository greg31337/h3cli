#!/usr/bin/env python3
"""Retain model identities, reusing validated sampler hash-cache entries."""
import hashlib, json, os, pathlib, struct, time

out=pathlib.Path('outputs/fast-vae/cuda/model-files.json')
roots=[pathlib.Path(os.environ.get('H3_MODEL_DIR','models/MiniMax-H3')).expanduser(),
       pathlib.Path(os.environ.get('H3_TEST_IMAGE_VAE_DIR','models/image-vae')).expanduser()]
entries=[];digests={};start=time.time()
for root in roots:
    for directory,dirs,files in os.walk(root,followlinks=True):
        dirs[:]=sorted(d for d in dirs if not d.startswith('.'))
        for name in sorted(files):
            if name.startswith('.'):continue
            path=pathlib.Path(directory)/name;st=path.stat()
            stamp=struct.pack('<QQQqqqq',st.st_dev,st.st_ino,st.st_size,
                st.st_mtime_ns//10**9,st.st_mtime_ns%10**9,
                st.st_ctime_ns//10**9,st.st_ctime_ns%10**9)
            if stamp not in digests:
                cache=pathlib.Path('outputs/.h3-model-hashes')/hashlib.sha256(stamp).hexdigest()
                cached=cache.read_bytes() if cache.is_file() else b''
                if len(cached)==120 and cached[:56]==stamp and hashlib.sha256(cached[:88]).digest()==cached[88:]:
                    digest=cached[56:88].hex();method='validated sampler cache: file identity, size, mtime, ctime, entry checksum'
                else:
                    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                    after=path.stat();assert (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino),path
                    method='fresh SHA256'
                digests[stamp]=(digest,method)
            digest,method=digests[stamp]
            entries.append(dict(path=str(path),bytes=st.st_size,sha256=digest,method=method,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns))
with pathlib.Path('taeh3.safetensors').open('rb') as stream:tiny=hashlib.file_digest(stream,'sha256').hexdigest()
out.write_text(json.dumps(dict(created_unix=time.time(),seconds=time.time()-start,files=entries,taeh3_sha256=tiny),indent=2)+'\n')
print('Retained',len(entries),'model file identities in',time.time()-start,'seconds')
