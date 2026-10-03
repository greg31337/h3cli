#!/usr/bin/env python3
"""Verify downloaded playback assets without model, checkpoint or GPU access."""
import argparse,hashlib,json,time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('src','href') and v)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',nargs='?',type=Path,default=Path('outputs/cuda-sol'));a=p.parse_args();root=a.root.resolve()
    manifest=json.loads((root/'checksums.json').read_text());bad=[];linked=[]
    for rel,expected in manifest.items():
        path=(root/rel).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path)!=expected:bad.append(rel)
    for name in ('index.html','640.html','1344.html'):
        page=root/name;parser=Links();parser.feed(page.read_text())
        for link in parser.links:
            url=urlsplit(link)
            if url.scheme or url.netloc:continue
            path=(page.parent/unquote(url.path)).resolve()
            if not path.is_relative_to(root) or not path.exists():bad.append(name+': '+link)
            linked.append(str(path.relative_to(root)))
    clock=json.loads((root/'clock.json').read_text());elapsed=time.time()-clock['start_epoch']
    closing=json.loads((root/'closing.json').read_text());closing_elapsed=time.time()-closing['start_epoch']
    coverage=json.loads((root/'coverage.json').read_text())
    overhead=max(0.,elapsed-coverage['recorded_process_interval_union_seconds'])
    contingency=overhead+coverage['buckets_child_wall_seconds']['contingency']
    result=dict(passed=not bad and elapsed<=480*60 and closing_elapsed<=1800 and contingency<=2100,checked_files=len(manifest),checked_gallery_links=len(linked),
                failures=bad,manifest_sha256=sha(root/'checksums.json'),elapsed_seconds_at_delivery=elapsed,
                closing_seconds_through_delivery=closing_elapsed,
                conservative_contingency_seconds=contingency,
                budget_accounting='All time outside recorded child-process intervals is conservatively charged to the 35-minute contingency, including overlapping report/delivery work.',
                completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                raw_diagnostics='Retained on node; model weights, QKV, activation buffers, AV and sampler states were not downloaded')
    (root/'delivery-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
