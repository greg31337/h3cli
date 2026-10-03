#!/usr/bin/env python3
"""Audit retained multi-reference evidence without launching the renderer."""
import argparse
import gzip
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from cuda_multi_reference import ROOT, digest, load_manifest, save, sha


def transfer_manifest(write=False):
    """Freeze remote artifacts after rendering, or verify their local copies."""
    import fcntl
    excluded={'transfer-manifest.json','transfer-verification.json','runner.lock'}
    path=ROOT/'transfer-manifest.json'
    if write:
        # The renderer runner holds this lock for the complete serial batch.
        # A snapshot taken while it writes logs or telemetry would be invalid.
        with (ROOT/'runner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
            report=json.loads((ROOT/'results.json').read_text())
            assert not report['coverage']['pending'],'Finish the campaign before freezing transfers'
            files={}
            for item in sorted(ROOT.rglob('*')):
                relative=item.relative_to(ROOT).as_posix()
                if relative in excluded:continue
                assert not item.is_symlink(),('Unexpected artifact symlink',relative)
                if item.is_file():
                    files[relative]={'bytes':item.stat().st_size,'sha256':sha(item)}
            document={'manifest_identity':report['manifest_identity'],
                      'final_identity':report['final_identity'],
                      'sampling_identity':report.get('sampling_identity'),'files':files}
            document['identity']=digest(document)
            save(path,document)
        print(json.dumps({'transfer_manifest':'written','files':len(files)}))
        return
    document=json.loads(path.read_text())
    payload={k:v for k,v in document.items() if k!='identity'}
    assert digest(payload)==document['identity'],'Transfer manifest checksum mismatch'
    manifest=load_manifest()
    assert document['manifest_identity']==manifest['identity']
    assert document['final_identity']==manifest['final_identity']
    assert document.get('sampling_identity')==manifest.get('sampling_identity')
    total=0
    for relative,expected in document['files'].items():
        item=ROOT/relative
        assert not item.is_symlink() and item.resolve().is_relative_to(ROOT.resolve()),relative
        assert item.is_file(),('Missing downloaded artifact',relative)
        assert item.stat().st_size==expected['bytes'],('Downloaded size mismatch',relative)
        assert sha(item)==expected['sha256'],('Downloaded checksum mismatch',relative)
        total+=expected['bytes']
    result={'status':'pass','transfer_manifest_identity':document['identity'],
            'files':len(document['files']),'bytes':total}
    save(ROOT/'transfer-verification.json',result)
    print(json.dumps(result))


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets=[]

    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ('href','src','poster') and value:
                self.targets.append(value)


def audit(final=False):
    manifest=load_manifest()
    report=json.loads((ROOT/'results.json').read_text())
    cases={c['id']:c for c in manifest['cases']}
    assert report['manifest_identity']==manifest['identity']
    for asset in manifest['assets'].values():
        assert sha(asset['path'])==asset['sha256'],('Input asset changed',asset['id'])
    rows=report['records']
    results=[]
    for row in rows+report.get('superseded_attempts',[]):
        name=row['case']['id']
        assert row['case']==cases[name],name
        directory=ROOT/row['artifact_directory']
        request=json.loads((directory/'request.json').read_text())
        assert request['case']==row['case'] and request['argv']==row['argv'],name
        # Early smoke requests predate the redundant per-case digest; their
        # complete case definition and frozen base identity are still checked.
        assert request.get('case_identity',digest(row['case']))==digest(row['case']),name
        assert request['manifest_identity']==manifest['identity'],name
        assert row.get('completed_steps',0)<=10,name
        if row['case']['phase']=='main':
            assert row['case']['steps']==10 and row['final_identity']==manifest['final_identity'],name
        if row['case']['phase']=='sample':
            assert 1<=row['case']['steps']<=2 and row['sampling_identity']==manifest['sampling_identity'],name
        if row['status']=='interrupted':
            assert name in report['interruptions'],name
        samples=0
        peak=0
        maximum_gap=0
        gaps=0
        previous=None
        with gzip.open(directory/'telemetry.jsonl.gz','rt') as stream:
            for line in stream:
                point=json.loads(line)
                samples+=1
                peak=max(peak,point['device_bytes'])
                if previous is not None:
                    interval=point['monotonic']-previous
                    maximum_gap=max(maximum_gap,interval)
                    gaps+=interval>.1
                previous=point['monotonic']
        telemetry=row['telemetry']
        assert samples==telemetry['samples'],name
        assert peak==telemetry['peaks_bytes']['device'],name
        assert maximum_gap==telemetry['max_gap_seconds'],name
        assert gaps==telemetry['gaps_over_100ms'],name
        if row.get('output'):
            assert sha(ROOT/row['output'])==row['validation']['sha256'],name
        if row['status']=='complete':
            assert row['validation']['valid'] and row['gpu_released'],name
            assert row['completed_steps']==row['case']['steps'],name
            assert row['validation']['frames']==row['case']['frames'],name
            assert row['decoded_frame_identity']['frames']==row['case']['frames'],name
            assert (directory/'decode.log').exists(),name
            blank=row['black_frame_check']
            assert blank['input_sha256']==row['validation']['sha256'],name
            assert (ROOT/blank['log']).exists(),name
            assert json.loads((directory/'black-frame-check.json').read_text())==blank,name
        results.append({'case':name,'samples':samples,'peak_bytes':peak,
                        'gaps_over_100ms':gaps,'status':row['status']})
    links=0
    for page in ROOT.glob('review*.html'):
        parser=Links()
        parser.feed(page.read_text())
        for target in parser.targets:
            u=urlsplit(target)
            assert not u.scheme and not u.netloc,('Remote gallery dependency',target)
            if not u.path:continue
            path=(page.parent/unquote(u.path)).resolve()
            assert path.is_relative_to(ROOT.resolve()),target
            assert path.exists(),('Missing gallery asset',target)
            links+=1
    coverage=report['coverage']
    assert len(rows)+len(coverage['pending'])+len(coverage['not_run_with_reason'])==len(cases)
    if final:assert not coverage['pending'],'Campaign still has pending cases'
    result={'final':final,'manifest_identity':manifest['identity'],
            'extension_identity':manifest.get('extension_identity'),
            'final_identity':manifest.get('final_identity'),
            'sampling_identity':manifest.get('sampling_identity'),
            'coverage':coverage,'checked_gallery_links':links,'cases':results}
    result['verified_input_assets']=len(manifest['assets'])
    save(ROOT/'evidence-audit.json',result)
    print(json.dumps({'measured':len(rows),'pending':len(coverage['pending']),
                      'checked_gallery_links':links,'audit':'pass'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final',action='store_true')
    transfer=parser.add_mutually_exclusive_group()
    transfer.add_argument('--write-transfer-manifest',action='store_true')
    transfer.add_argument('--verify-transfer',action='store_true')
    args=parser.parse_args()
    if args.write_transfer_manifest:transfer_manifest(write=True)
    elif args.verify_transfer:transfer_manifest()
    else:audit(args.final)
