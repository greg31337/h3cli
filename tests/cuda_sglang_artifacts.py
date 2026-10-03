#!/usr/bin/env python3
"""Validate downloaded playback pages, checksums, and complete local media."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlsplit


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value:
                url = urlsplit(value)
                if not url.scheme and not url.netloc and url.path:
                    self.paths.add(unquote(url.path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    records = []
    checksums = json.loads((root / 'report-checksums.json').read_text())
    for relative, expected in checksums.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.parent):
            raise ValueError('asset escapes the retained report tree')
        records.append(dict(check='checksum', path=relative,
                            passed=path.is_file() and digest(path) == expected))
    pages = ['review.html', 'fast-regression.html']
    if (root / 'final-review.html').exists():
        pages.append('final-review.html')
    for name in pages:
        page = root / name
        parser = Links()
        parser.feed(page.read_text())
        for target in sorted(parser.paths):
            path = (page.parent / target).resolve()
            records.append(dict(check='local_link', page=name, path=target,
                                passed=path.is_relative_to(root.parent) and path.is_file()))
    media_cases={}
    for path in sorted(root.rglob('result.json')):
        result = json.loads(path.read_text())
        if result.get('comparison')=='native reference interruption and cache replay':
            for row in result.get('records',[]):
                if row.get('validation',{}).get('media_valid'):
                    media_cases[path.parent/row['name']/'video.mp4']=row['validation']['frames']
        if result.get('status') != 'media_valid' or 'spec' not in result:
            continue
        media_cases[path.parent/'video.mp4']=result['spec']['frames']
    for media,expected_frames in sorted(media_cases.items()):
        row = dict(check='complete_video', path=str(media.relative_to(root)), passed=False)
        try:
            probe = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(media)],
                stdin=subprocess.DEVNULL, stderr=subprocess.PIPE))
            video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            row.update(width=video['width'], height=video['height'], frames=int(video['nb_read_frames']), fps=video['avg_frame_rate'])
            decoded = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(media), '-f', 'null', '-'],
                                     stdin=subprocess.DEVNULL, capture_output=True, text=True)
            row['passed'] = ((row['width'], row['height'], row['frames'], row['fps']) ==
                             (640, 480, expected_frames, '24/1') and
                             any(s['codec_type'] == 'audio' for s in probe['streams']) and
                             decoded.returncode == 0 and not decoded.stderr.strip())
            if decoded.stderr.strip():
                row['decode_error'] = decoded.stderr
        except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError) as exc:
            row['error'] = str(exc)
        records.append(row)
    report = dict(kind='local playback artifact verification', root=str(root), records=records,
                  passed=all(r['passed'] for r in records),
                  failed=[r for r in records if not r['passed']])
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(checks=len(records), passed=report['passed'], failed=report['failed'])), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
