#!/usr/bin/env python3
"""Explicit two-stage Apple notarization; never invoked by the ordinary builder."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import package


def notarize(container, profile, output, phase):
    # Only a Keychain profile name crosses the process boundary, never secrets.
    result = subprocess.run(['xcrun','notarytool','submit',str(container),'--keychain-profile',profile,
                             '--wait','--timeout','30m','--output-format','json'],capture_output=True,text=True)
    try: record=json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ValueError('Notary submission did not return JSON; verify the installed Keychain profile') from None
    package.write(output/(phase+'-submission.json'),record)
    if record.get('id'):
        subprocess.run(['xcrun','notarytool','log',record['id'],'--keychain-profile',profile,
                        str(output/(phase+'-notary-log.json'))],check=True)
    if result.returncode or record.get('status')!='Accepted':
        raise ValueError(phase+' notarization was not accepted; inspect retained submission/log records')
    log=json.loads((output/(phase+'-notary-log.json')).read_text())
    if log.get('status')!='Accepted' or any(i.get('severity')=='error' for i in log.get('issues') or []):
        raise ValueError('Notary log contains an error')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--identity',required=True);p.add_argument('--keychain-profile',required=True)
    args=p.parse_args()
    if not args.identity.startswith('Developer ID Application: ') or not args.keychain_profile.strip():
        p.error('An installed Developer ID Application identity and a nonempty Keychain profile are required')
    if args.output.exists():p.error('Use a fresh output directory; failed runs retain evidence')
    source=args.input.resolve();output=args.output.resolve()
    record=json.loads((source/'build.json').read_text())
    for name,item in record['artifacts'].items():
        if package.sha(source/name)!=item['sha256']:raise ValueError('Input release changed: '+name)
    # Reconstruct from exact source materials, not the caller's potentially newer checkout.
    import tarfile
    output.mkdir(parents=True)
    with tarfile.open(source/'h3cli-macos-arm64-sources.tar.gz') as archive:
        members=[m for m in archive.getmembers() if m.name.startswith('h3cli/')]
        if any(m.issym() or m.islnk() or m.isdev() or '..' in Path(m.name).parts or Path(m.name).is_absolute() for m in members):
            raise ValueError('Unsafe source material archive')
        archive.extractall(output/'work',members=members)
    build_source=output/'work/h3cli'
    runtime=output/'macos-runtime';shutil.copytree(source/'macos-runtime',runtime)
    (runtime/'.ready.json').unlink(missing_ok=True)
    package.audit(runtime)
    manifest=json.loads((runtime/'share/h3cli/runtime.json').read_text())
    if package.inventory(runtime)!=manifest['files']:raise ValueError('Input runtime changed')
    for name in ('bin/h3cli','tools/ffmpeg','tools/ffprobe'):package.sign(runtime/name,args.identity)
    # Sign all code first, then record the exact accepted code bytes in the payload.
    manifest['files']=package.inventory(runtime)
    package.write(runtime/'share/h3cli/runtime.json',manifest)
    inner=output/'inner-runtime.zip'
    subprocess.run(['/usr/bin/ditto','-c','-k','--keepParent',str(runtime),str(inner)],check=True)
    first=notarize(inner,args.keychain_profile,output,'inner')
    accepted=package.inventory(runtime)
    bundle=package.payload(runtime,output/'work/payload.bin',build_source)
    if accepted!=package.inventory(runtime):raise ValueError('Inner bytes changed after notarization')
    executable=output/'h3cli-macos-arm64'
    package.embed(build_source,output/'work/payload.bin',executable)
    package.sign(executable,args.identity,identifier='org.h3cli.launcher')
    image=package.dmg(output,executable,'h3cli Developer ID distribution')
    package.sign(image,args.identity,identifier='org.h3cli.distribution')
    second=notarize(image,args.keychain_profile,output,'outer')
    subprocess.run(['xcrun','stapler','staple',str(image)],check=True)
    subprocess.run(['xcrun','stapler','validate',str(image)],check=True)
    subprocess.run(['/usr/sbin/spctl','--assess','--type','open','--context','context:primary-signature','--verbose=2',str(image)],check=True)
    for name in ('h3cli-macos-arm64-sources.tar.gz','libh3.a'):shutil.copyfile(source/name,output/name)
    package.seal(runtime)
    artifacts={p.name:{'sha256':package.sha(p),'size':p.stat().st_size} for p in (executable,image,output/'h3cli-macos-arm64-sources.tar.gz',output/'libh3.a')}
    package.write(output/'build.json',{'schema':1,'signing':'Developer ID; inner and outer notarized; DMG stapled',
        'input_build_sha256':package.sha(source/'build.json'),'inner':first,'outer':second,
        'inner_files':accepted,'bundle':bundle,'artifacts':artifacts,
        'qualification':'Fresh quarantine/trust, cold offline extraction and GPU qualification remain required.'})
    (output/'SHA256SUMS').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in artifacts.items()))
    print('Signed artifact built. Complete fresh-trust qualification before distributing.')


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as e:sys.exit(str(e))
