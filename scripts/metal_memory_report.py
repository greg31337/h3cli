#!/usr/bin/env python3
"""Verify retained M5 records and export memory plots and local playback HTML."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import statistics


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--run',action='append',required=True)
    a=p.parse_args();root=a.root.resolve();records=[];sections=[]
    os.environ.setdefault('MPLCONFIGDIR',str(root/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in a.run:
        run=root/name;r=json.loads((run/'record.json').read_text())
        for key,file in [('log_sha256','run.log'),('memory_sha256','memory.jsonl')]:
            if r[key]!=sha(run/file):raise ValueError('changed record: '+str(run/file))
        binary=Path(r['command'][0])
        if r['binary_sha256']!=sha(binary):raise ValueError('changed executable: '+str(binary))
        provenance=binary.parent/'build-provenance.json'
        if not provenance.is_file():raise ValueError('missing frozen build: '+str(binary))
        build=json.loads(provenance.read_text())
        for file,digest in build['source_sha256'].items():
            if sha(Path(build['source_snapshot'])/file)!=digest:raise ValueError('changed frozen source: '+file)
        shader=r.get('environment',{}).get('H3_SHADER_PATH')
        if shader and sha(shader)!=build['source_sha256']['src/metal/shaders.metal']:
            raise ValueError('runtime shader differs from frozen source: '+shader)
        groups={x:[] for x in ['h3_step','h3_memory','h3_lifetime','h3_execution']}
        log=(run/'run.log').read_text()
        for line in log.splitlines():
            for key,items in groups.items():
                marker=key+' {'
                if marker in line:items.append(json.loads(line[line.index(marker)+len(key)+1:]))
        samples=[json.loads(x) for x in (run/'memory.jsonl').read_text().splitlines()]
        steps=groups['h3_step'];memory=groups['h3_memory'];life=groups['h3_lifetime']
        record={'name':name,'run':r,'build_provenance_sha256':sha(provenance),**groups,
                'artifacts':{file:sha(run/file) for file in ('result.mp4','result.h3av','pause.h3sample','conditioning.h3cond') if (run/file).is_file()},
                'median_step_seconds':statistics.median(x['wall_seconds'] for x in steps) if steps else None,
                'peak_sampled_footprint_bytes':max(x['footprint_bytes'] for x in samples),
                'peak_boundary_metal_bytes':max((x['metal_allocated_bytes'] for x in memory),default=0),
                'last_lifetime':life[-1] if life else None}
        records.append(record)
        fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
        stride=max(1,len(samples)//2400)
        reduced=[max(samples[i:i+stride],key=lambda x:x['footprint_bytes']) for i in range(0,len(samples),stride)]
        axes[0].plot([x['seconds']/60 for x in reduced],[x['footprint_bytes']/1e9 for x in reduced],label='Process footprint')
        limit=(memory[0]['limit_bytes'] if memory else 110000000000)/1e9
        axes[0].axhline(limit,color='firebrick',ls='--',label=f'Renderer cap ({limit:g} GB)')
        axes[0].set(xlabel='Wall time (minutes)',ylabel='GB, decimal',ylim=(0,115),title='Independent 50 ms samples')
        blocks=[x for x in memory if x['phase']=='DiT block']
        for key,label in [('footprint_bytes','Footprint'),('metal_allocated_bytes','Metal reservations'),('tensor_live_bytes','Tensor storage')]:
            axes[1].plot([x['step']*50+x['block'] for x in blocks],[x[key]/1e9 for x in blocks],label=label)
        axes[1].set(xlabel='Step × 50 + block (encoding boundary)',ylabel='GB, decimal',ylim=(0,115),title='Overlapping measures; do not sum')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(name);fig.savefig(run/'memory.svg');fig.savefig(run/'memory.png',dpi=130);plt.close(fig)
        if r.get('watchdog_reason'):status='External watchdog stop: '+r['watchdog_reason']
        elif r['returncode'] and 'memory safety:' in log:status='Renderer stopped at its memory budget (safety test)'
        elif r['returncode']:status='Renderer error; see log'
        elif (run/'pause.h3sample').is_file() and not (run/'result.mp4').is_file():status='Paused after the configured step; checkpoint saved'
        else:status='Completed'
        def option(flag):
            argv=r['command']
            return argv[argv.index(flag)+1] if flag in argv else '?'
        geometry=f"{option('--width')}×{option('--height')}, {option('--frames')} frames"
        video=f'<video controls preload="metadata" src="{html.escape(name)}/result.mp4"></video>' if (run/'result.mp4').is_file() else '<p>No completed playback artifact.</p>'
        rows=''.join(f'<tr><td>{x["step"]}</td><td>{x["wall_seconds"]:.2f}</td><td>{x["physical_footprint_bytes"]/1e9:.2f}</td><td>{x["metal_current_allocated_bytes"]/1e9:.2f}</td><td>{x["swap_used_bytes"]/1e9:.2f}</td></tr>' for x in steps)
        sections.append(f'''<section><h2>{html.escape(name)}</h2><p>{html.escape(status)}. Peak sampled footprint: {record['peak_sampled_footprint_bytes']/1e9:.2f} GB.</p>
<p>{html.escape(geometry)} · build: {html.escape(binary.parent.name)}</p>
<img src="{html.escape(name)}/memory.svg" alt="Footprint over time and per-block memory reservations">
<table><thead><tr><th>Step</th><th>Seconds</th><th>Footprint GB</th><th>Metal GB</th><th>System swap GB</th></tr></thead><tbody>{rows}</tbody></table>
{video}<p><a href="{html.escape(name)}/run.log">Execution log</a> · <a href="{html.escape(name)}/record.json">Command and checksums</a></p></section>''')
    (root/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    (root/'review.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>M5 memory validation · 362 frames</title>
<style>body{font:16px system-ui;background:#f6f7fa;color:#202936;margin:2rem auto;max-width:1200px;padding:0 1rem}section{background:white;padding:1.5rem;margin:1.5rem 0;border:1px solid #ccd4e0;border-radius:8px}img{width:100%}video{display:block;max-width:100%;width:640px;margin:1rem 0}table{border-collapse:collapse}td,th{padding:.45rem 1rem;text-align:right;border-bottom:1px solid #ddd}a{color:#145da0}</style>
<h1>M5 memory validation · 362 frames</h1><p>640×480, reference inputs/2.jpg at maximum size. Large-resolution tests use at most two steps. Clips are reduced-step memory diagnostics, not a new visual-quality qualification.</p><p>Process footprint includes compressed memory. Metal reservations overlap it. The default cap is 110 GB decimal; lower-cap guard tests identify their override in the command record. A separate watchdog stays below the default cap.</p>
'''+''.join(sections)+'<p><a href="summary.json">Verified metrics and source provenance</a></p></html>')
    print(root/'review.html')

if __name__=='__main__':main()
