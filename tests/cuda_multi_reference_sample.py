#!/usr/bin/env python3
"""Run the user-authorized one-hour sample, preserving the original matrix."""
import datetime
import fcntl
import json
import time
from cuda_multi_reference import ROOT, CONTRACT, digest, freeze_check, load_manifest, records, run_case, sample_cases, save, validate_case


def main():
    lock=(ROOT/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=ROOT/'sample-one-hour.json'
    if not path.exists():
        m=load_manifest()
        document={'base_identity':m['identity'],'final_identity':m['final_identity'],
            'authorization':'User requested sampling of remaining tests within one hour, with at most two steps',
            'selection':'Matched one-step 362-frame no-reference/nine-image-max pairs at both resolutions; existing two-step smoke covers maximal video mixtures',
            'maximum_steps':2,'render_deadline_utc':'2026-09-23T15:45:00Z',
            'cases':sample_cases()}
        document['identity']=digest(document)
        save(path,document)
        save(CONTRACT.with_name('multi-reference-sample.json'),document)
    document=json.loads(path.read_text())
    deadline=datetime.datetime.fromisoformat(document['render_deadline_utc'].replace('Z','+00:00')).timestamp()
    m=load_manifest()
    freeze_check(m)
    done={r['case']['id'] for r in records()}
    blocked_path=ROOT/'blocked.json'
    blocked=json.loads(blocked_path.read_text()) if blocked_path.exists() else {}
    for c in m['cases']:
        if c['phase']=='main' and c['id'] not in done:
            blocked[c['id']]='Superseded by the user-authorized one-hour sample with at most two steps; original definition retained'
    save(blocked_path,blocked)
    # Keep the two user-requested nine-image outputs ahead of the final small
    # control if the remaining wall budget becomes tight. Definitions and
    # matched baseline keys remain frozen; only serial execution order changes.
    order=sorted(document['cases'],key=lambda c:
        (0 if c['width']==1344 and c.get('baseline') else
         1 if c['width']==1344 else 3 if c.get('baseline') else 2))
    save(ROOT/'sample-execution-order.json',{'case_ids':[c['id'] for c in order],
         'reason':'Prioritize both requested nine-image/362-frame outputs within the hour'})
    for c in order:
        if c['id'] in done:continue
        validate_case(c,m['assets'])
        remaining=deadline-time.time()-30
        required=250 if c['width']==640 and c.get('baseline') else 150
        if remaining<required:
            blocked[c['id']]='One-hour campaign budget exhausted before launch'
            save(blocked_path,blocked)
            continue
        print('SAMPLE',c['id'],'steps',c['steps'],'remaining seconds',round(remaining),flush=True)
        row=run_case(c,m,remaining)
        if row.get('termination_reason') or not row.get('gpu_released',False):
            for other in document['cases']:
                if other['id'] not in {r['case']['id'] for r in records()}:
                    blocked[other['id']]='Sample stopped at the wall-time/resource boundary; see the preceding attempt'
            save(blocked_path,blocked)
            break


if __name__=='__main__':main()
