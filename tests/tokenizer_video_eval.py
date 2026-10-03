#!/usr/bin/env python3
"""Extract review frames and coarse Apple Vision lip measurements, locally."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', nargs='?', type=Path, default=ROOT/'outputs/tokenfix-validation/generation')
    args = p.parse_args()
    out = args.directory.resolve()
    tool = ROOT/'bin/tokenizer_lips'
    tool.parent.mkdir(parents=True,exist_ok=True)
    source = ROOT/'tests/tokenizer_lips.m'
    if not tool.exists() or tool.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(['clang', '-O2', '-fobjc-arc', str(source), '-framework', 'Foundation',
            '-framework', 'Vision', '-framework', 'ImageIO', '-framework', 'CoreGraphics', '-o', str(tool)], check=True)
    record_path = out/'video-observations.json'
    records = json.loads(record_path.read_text()) if record_path.exists() else {}
    for name, run in json.loads((out/'results.json').read_text()).items():
        if (records.get(name, {}).get('mp4_sha256') == run['hashes']['mp4']
                and (out/(name+'.contact.png')).exists()):
            continue
        video = out/(name+'.mp4')
        frames = out/(name+'.frames')
        frames.mkdir(exist_ok=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(video), '-vf', 'fps=12',
            str(frames/'%04d.png')], check=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(video), '-vf',
            r'select=not(mod(n\,24)),tile=6x1', '-frames:v', '1', str(out/(name+'.contact.png'))], check=True)
        lip_file = out/(name+'.lips.json')
        if (not lip_file.exists() or lip_file.stat().st_size == 0 or
                records.get(name, {}).get('mp4_sha256') != run['hashes']['mp4']):
            with lip_file.open('w') as f:
                subprocess.run([str(tool), str(frames)], stdout=f, check=True)
        lips = json.loads(lip_file.read_text())
        detected = [r for r in lips if r.get('inner_lip_aspect') is not None]
        records[name] = {'mp4_sha256': run['hashes']['mp4'],
            'frames': len(lips), 'frames_with_lips': len(detected),
            'inner_lip_aspect_min': min((r['inner_lip_aspect'] for r in detected), default=None),
            'inner_lip_aspect_max': max((r['inner_lip_aspect'] for r in detected), default=None),
            'contact_sheet_times_seconds': [0, 1, 2, 3, 4, 5]}
        print(name, records[name], flush=True)
    record_path.write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
