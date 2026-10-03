#!/usr/bin/env python3
"""Intersect actual ANE/GPU activity inside h3cli's traced QKV worker windows."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from metal_native_bench import sha

def merge(intervals):
    result=[]
    for begin,end in sorted(intervals):
        if end<=begin:continue
        if result and begin<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([begin,end])
    return result

def intersect(left,right):
    left=merge(left);right=merge(right);i=j=0;result=[]
    while i<len(left) and j<len(right):
        a,b=left[i];c,d=right[j]
        if min(b,d)>max(a,c):result.append([max(a,c),min(b,d)])
        if b<d:i+=1
        else:j+=1
    return result

def values(root):
    ids={e.attrib['id']:e for e in root.iter() if 'id' in e.attrib}
    def resolve(e):
        if e is None:return ''
        if 'ref' in e.attrib:e=ids[e.attrib['ref']]
        return e.text or e.attrib.get('fmt','')
    return resolve

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    a=p.parse_args();base=a.directory.resolve()
    root=ET.parse(base/'device-intervals.xml').getroot();value=values(root)
    ane=[];gpu=[];gpu_processes=set()
    for node in root.findall('node'):
        for row in node.findall('row'):
            if value(row.find('gpu-state'))!='Active':continue
            start=int(value(row.find('start-time')));duration=int(value(row.find('duration')))
            if row.find('ane-event-name') is not None:ane.append([start,start+duration])
            elif value(row.find('gpu-channel-name'))=='Compute' and value(row.find('metal-nesting-level'))=='0':
                process=value(row.find('process'))
                if process.startswith('h3cli ('):
                    gpu_processes.add(process);gpu.append([start,start+duration])
    root=ET.parse(base/'signposts.xml').getroot();value=values(root);opened={};windows=[];owners=set()
    for row in root.iter('row'):
        if value(row.find('signpost-name'))!='ANE QKV' or value(row.find('subsystem'))!='org.h3.ane':continue
        process=value(row.find('process'));owners.add(process)
        key=(process,value(row.find('os-signpost-identifier')))
        time=int(value(row.find('event-time')))
        if value(row.find('event-type'))=='Begin':opened[key]=time
        elif value(row.find('event-type'))=='End' and key in opened:windows.append([opened.pop(key),time])
    if len(gpu_processes)!=1 or owners!=gpu_processes:raise ValueError('ambiguous process attribution')
    attributed=intersect(ane,windows);overlap=intersect(attributed,gpu)
    total=lambda rows:sum(b-a for a,b in merge(rows))*1e-9
    if not windows or not ane or not gpu:raise ValueError('missing hardware activity')
    report={'schema':1,'script_sha256':sha(__file__),
        'input_sha256':{name:sha(base/name) for name in ('trace-toc.xml','device-intervals.xml','signposts.xml')},
        'process':next(iter(owners)),'ane_hardware_intervals':len(ane),'gpu_compute_intervals':len(gpu),
        'complete_qkv_worker_windows':len(windows),'ane_active_seconds':total(ane),
        'ane_active_inside_worker_seconds':total(attributed),'gpu_active_seconds':total(gpu),
        'hardware_overlap_seconds':total(overlap),'ane_worker_time_overlapping_gpu_fraction':total(overlap)/total(attributed),
        'hardware_overlap_observed':total(overlap)>0,
        'attribution':'ANE hardware intervals have no PID. Restrict them to complete org.h3.ane QKV signpost windows; GPU compute intervals carry the same h3cli PID. The public compute plan assigns every matmul to ANE. No separate Core ML model interval was exported.',
        'scope':'30-second diagnostic window, not the unfenced B5 speed measurement',
        'worker_windows_ns':windows,'ane_intervals_in_worker_ns':attributed,'overlap_intervals_ns':overlap}
    (base/'trace-overlap.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_ns')},indent=2))

if __name__=='__main__':main()
