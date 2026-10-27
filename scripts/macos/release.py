#!/usr/bin/env python3
"""Explicit two-stage Apple notarization; never invoked by the ordinary builder."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_process
import package


def notarize(container, profile, output, phase):
    # Persist the submission ID before waiting. A retry waits for the same
    # upload; an interrupted acknowledgement is reconciled by its unique name.
    path=output/(phase+'-submission.json')
    identity=output/(phase+'-upload.json')
    if identity.exists():
        upload=json.loads(identity.read_text())
        if upload['sha256']!=package.sha(container):raise ValueError('Notary upload bytes changed')
    else:
        suffix=container.suffix
        upload={'name':phase+'-'+uuid.uuid4().hex+suffix,'sha256':package.sha(container)}
        package.write(identity,upload)
    archive=output/upload['name']
    if not archive.exists():shutil.copyfile(container,archive)
    record=json.loads(path.read_text()) if path.exists() else None
    if record is None:
        history=build_process.run(['xcrun','notarytool','history','--keychain-profile',profile,
                                  '--output-format','json'],capture_output=True,text=True,check=True)
        matches=[r for r in json.loads(history.stdout)['history'] if r['name']==upload['name']]
        if len(matches)>1:raise ValueError('Ambiguous notarization history for this upload')
        if matches:record=matches[0]
        else:
            result=build_process.run(['xcrun','notarytool','submit',str(archive),'--keychain-profile',profile,
                                     '--no-wait','--output-format','json'],capture_output=True,text=True,check=True)
            record=json.loads(result.stdout)
        if not record.get('id'):raise ValueError('Notary service did not return a submission ID')
        package.write(path,record)
    if record.get('status')!='Accepted':
        print('Waiting for '+phase+' notarization: '+record['id'],flush=True)
        result=build_process.run(['xcrun','notarytool','wait',record['id'],'--keychain-profile',profile,
                                 '--timeout','30m','--output-format','json'],capture_output=True,text=True,check=True)
        record=json.loads(result.stdout);package.write(path,record)
    build_process.run(['xcrun','notarytool','log',record['id'],'--keychain-profile',profile,
                      str(output/(phase+'-notary-log.json'))],check=True)
    if record.get('status')!='Accepted':
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
    source=args.input.resolve();output=args.output.resolve()
    record=json.loads((source/'build.json').read_text())
    for name,item in record['artifacts'].items():
        if package.sha(source/name)!=item['sha256']:raise ValueError('Input release changed: '+name)
    # Reconstruct from exact source materials, not the caller's potentially newer checkout.
    import tarfile
    identity={'input':package.sha(source/'build.json'),'identity':args.identity,'profile':args.keychain_profile}
    if output.exists():
        if not (output/'identity.json').is_file() or json.loads((output/'identity.json').read_text())!=identity:
            raise ValueError('Signing output belongs to different inputs')
    else:
        output.mkdir(parents=True);package.write(output/'identity.json',identity)
    if (output/'build.json').is_file():
        finished=json.loads((output/'build.json').read_text())
        if any(package.sha(output/n)!=v['sha256'] for n,v in finished['artifacts'].items()):raise ValueError('Signed artifact changed')
        print('Reusing verified completed signing output.');return
    build_source=output/'work/h3cli'
    runtime=output/'macos-runtime'
    inner=output/'inner-runtime.zip'
    if not (output/'inner-ready.json').exists():
        for directory in (output/'work',runtime):
            if directory.exists():shutil.rmtree(directory)
        with tarfile.open(source/'h3cli-macos-arm64-sources.tar.gz') as archive:
            members=[m for m in archive.getmembers() if m.name.startswith('h3cli/')]
            if any(m.issym() or m.islnk() or m.isdev() or '..' in Path(m.name).parts or Path(m.name).is_absolute() for m in members):
                raise ValueError('Unsafe source material archive')
            archive.extractall(output/'work',members=members)
        shutil.copytree(source/'macos-runtime',runtime)
        (runtime/'.ready.json').unlink(missing_ok=True)
        package.audit(runtime)
        manifest=json.loads((runtime/'share/h3cli/runtime.json').read_text())
        if package.inventory(runtime)!=manifest['files']:raise ValueError('Input runtime changed')
        for name in ('bin/h3cli','tools/ffmpeg','tools/ffprobe'):package.sign(runtime/name,args.identity)
        manifest['files']=package.inventory(runtime)
        package.write(runtime/'share/h3cli/runtime.json',manifest)
        inner.unlink(missing_ok=True)
        build_process.run(['/usr/bin/ditto','-c','-k','--keepParent',str(runtime),str(inner)],check=True)
        package.write(output/'inner-ready.json',{'sha256':package.sha(inner),'files':package.inventory(runtime)})
    ready=json.loads((output/'inner-ready.json').read_text())
    if ready['sha256']!=package.sha(inner) or ready['files']!=package.inventory(runtime):raise ValueError('Prepared inner runtime changed')
    first=notarize(inner,args.keychain_profile,output,'inner')
    accepted=package.inventory(runtime)
    executable=output/'h3cli-macos-arm64'
    image=output/'h3cli-macos-arm64.dmg'
    if not (output/'outer-ready.json').exists():
        bundle=package.payload(runtime,output/'work/payload.bin',build_source)
        if accepted!=package.inventory(runtime):raise ValueError('Inner bytes changed after notarization')
        package.embed(build_source,output/'work/payload.bin',executable)
        package.sign(executable,args.identity,identifier='org.h3cli.launcher')
        image.unlink(missing_ok=True)
        if (output/'dmg-contents').exists():shutil.rmtree(output/'dmg-contents')
        package.dmg(output,executable,'h3cli Developer ID distribution')
        package.sign(image,args.identity,identifier='org.h3cli.distribution')
        # Keep the submitted DMG separate: stapling changes the deliverable.
        shutil.copyfile(image,output/'outer-unstapled.dmg')
        package.write(output/'outer-ready.json',{'bundle':bundle,'executable_sha256':package.sha(executable),
                                                'image_sha256':package.sha(output/'outer-unstapled.dmg')})
    ready=json.loads((output/'outer-ready.json').read_text());bundle=ready['bundle']
    if package.sha(executable)!=ready['executable_sha256'] or package.sha(output/'outer-unstapled.dmg')!=ready['image_sha256']:
        raise ValueError('Prepared outer code changed')
    second=notarize(output/'outer-unstapled.dmg',args.keychain_profile,output,'outer')
    shutil.copyfile(output/'outer-unstapled.dmg',image)
    subprocess.run(['xcrun','stapler','staple',str(image)],check=True)
    subprocess.run(['xcrun','stapler','validate',str(image)],check=True)
    subprocess.run(['/usr/sbin/spctl','--assess','--type','open','--context','context:primary-signature','--verbose=2',str(image)],check=True)
    for name in ('h3cli-macos-arm64-sources.tar.gz','libh3.a'):shutil.copyfile(source/name,output/name)
    package.seal(runtime)
    artifacts={p.name:{'sha256':package.sha(p),'size':p.stat().st_size} for p in (executable,image,output/'h3cli-macos-arm64-sources.tar.gz',output/'libh3.a')}
    package.write(output/'build.json',{'schema':1,'signing':'Developer ID; inner and outer notarized; DMG stapled',
        'input_build_sha256':package.sha(source/'build.json'),'inner':first,'outer':second,
        'dependencies':record.get('dependencies',{'mode':'source'}),
        'inner_files':accepted,'bundle':bundle,'artifacts':artifacts,
        'qualification':'Run test-packages against these exact final artifacts; fresh-account trust research is optional.'})
    (output/'SHA256SUMS').write_text(''.join(v['sha256']+'  '+n+'\n' for n,v in artifacts.items()))
    print('Signed artifact built. Run test-packages before publication.')


if __name__=='__main__':
    try:
        with build_process.cancellation():main()
    except KeyboardInterrupt:sys.exit(130)
    except (ValueError,OSError,subprocess.CalledProcessError) as e:sys.exit(str(e))
