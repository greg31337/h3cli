#!/usr/bin/env python3
"""Render a cabinet of miniature worlds with native 2x latent upscaling.

Usage: python3 test.py [miniatures.txt] [--steps N] [--no-overlay] [-o miniatures.mp4]

Forty independent text-to-video scenes; no continuation or image handoffs.
Each scene renders 960x544, 243 native frames at 24 fps, then uses the learned
latent upscaler to produce 1920x1088 with four refinement steps by default.
Assembly keeps exactly 240 frames (10 seconds), center-crops to 1920x1080,
and softens the quiet audio with half gain and half-second edge fades.
The full sample delivers 9,600 frames: 6 minutes 40 seconds.

Adapted from samples/flight and samples/panda. Requires Python 3.10+, h3cli,
FFmpeg and FFprobe. States, original clips, prepared clips and logs are retained.
--dry-run prints the command plan without tools, downloads or output creation.
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT, FRAMES, FPS = 960, 544, 243, 24
UPSCALE_WIDTH, UPSCALE_HEIGHT = WIDTH*2, HEIGHT*2
OUTPUT_WIDTH, OUTPUT_HEIGHT, DELIVERED_FRAMES = 1920, 1080, 240
AUDIO_RATE, AUDIO_GAIN, AUDIO_FADE_SECONDS = 32000, 0.5, 0.5
HEADER = re.compile(r'^\s*###\s*([0-9]+)\s*###\s*$')
DIGITS = (
    '01110 10001 10011 10101 11001 10001 01110',
    '00100 01100 00100 00100 00100 00100 01110',
    '01110 10001 00001 00010 00100 01000 11111',
    '11110 00001 00001 01110 00001 00001 11110',
    '00010 00110 01010 10010 11111 00010 00010',
    '11111 10000 10000 11110 00001 00001 11110',
    '01110 10000 10000 11110 10001 10001 01110',
    '11111 00001 00010 00100 01000 01000 01000',
    '01110 10001 10001 01110 10001 10001 01110',
    '01110 10001 10001 01111 00001 00001 01110',
)


def read_prompts(path):
    sections, seen, number, lines = [], set(), None, []

    def finish():
        prompt = '\n'.join(lines).strip()
        if not prompt:
            raise ValueError(f'Segment {number} has an empty prompt')
        sections.append((number, prompt))

    for line in path.read_text(encoding='utf-8-sig').splitlines():
        match = HEADER.fullmatch(line)
        if match:
            if number is not None:
                finish()
            number = match[1]
            if len(number) > 12:
                raise ValueError('Segment numbers must contain at most 12 digits')
            if int(number) in seen:
                raise ValueError(f'Duplicate segment number: {number}')
            seen.add(int(number))
            lines = []
        elif number is not None:
            lines.append(line)
    if number is None:
        raise ValueError('No prompts found; start each section with ### NUMBER ###')
    finish()
    return sections


def executable(value):
    value = os.path.expanduser(str(value))
    found = shutil.which(value)
    if not found:
        raise ValueError(f'Executable not found: {value}')
    return str(Path(found).resolve())


def run(command, log):
    """Keep a log and show live progress; stop the child process tree on Ctrl-C."""
    with log.open('xb') as stream:
        child = subprocess.Popen(list(map(str, command)), stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            while chunk := os.read(child.stdout.fileno(), 8192):
                stream.write(chunk)
                stream.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            code = child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        finally:
            child.stdout.close()
    if code:
        raise RuntimeError(f'Command exited {code}; see {log}')


def badge(path, number):
    scale, pad = 4, 8
    width, height = (len(number)*6-1)*scale+2*pad, 7*scale+2*pad
    pixels = bytearray(bytes((24, 24, 24)) * (width*height))
    for index, digit in enumerate(number):
        for row, pattern in enumerate(DIGITS[int(digit)].split()):
            for col, bit in enumerate(pattern):
                if bit == '1':
                    for dy in range(scale):
                        at = ((pad+row*scale+dy)*width + pad+(index*6+col)*scale)*3
                        pixels[at:at+scale*3] = b'\xff'*(scale*3)
    path.write_bytes(f'P6\n{width} {height}\n255\n'.encode()+pixels)


def inspect_video(ffprobe, path, frames, width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT):
    result = subprocess.run([ffprobe, '-v', 'error', '-count_frames', '-show_streams',
                             '-of', 'json', str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)['streams']
    video = next((s for s in streams if s['codec_type'] == 'video'), None)
    audio = next((s for s in streams if s['codec_type'] == 'audio'), None)
    if not video or not audio:
        raise ValueError(f'{path}: both video and audio streams are required')
    if ((video['width'], video['height']) != (width, height) or
            int(video.get('nb_read_frames', 0)) != frames or
            Fraction(video['avg_frame_rate']) != FPS):
        raise ValueError(f'{path}: expected {width}x{height}, {frames} frames at {FPS} fps')


def prepare_command(ffmpeg, source, image, destination, frames=DELIVERED_FRAMES):
    samples, duration = frames*AUDIO_RATE//FPS, frames/FPS
    fade = min(AUDIO_FADE_SECONDS, duration/2)
    command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'warning', '-n',
               '-i', str(source)]
    filters = (f'[0:v]trim=end_frame={frames},setpts=PTS-STARTPTS,'
               f'crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:0:{(UPSCALE_HEIGHT-OUTPUT_HEIGHT)//2},setsar=1')
    if image is not None:
        command += ['-i', str(image)]
        filters += '[base];[base][1:v]overlay=x=16:y=H-h-16:eof_action=repeat'
    command += ['-filter_complex', filters+'[v]', '-map', '[v]',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-pix_fmt', 'yuv420p', '-r', str(FPS), '-map', '0:a:0',
                '-map_metadata', '-1', '-af',
                f'aresample={AUDIO_RATE},apad=whole_len={samples},atrim=end_sample={samples},'
                f'asetpts=PTS-STARTPTS,volume={AUDIO_GAIN},'
                f'afade=t=in:st=0:d={fade},afade=t=out:st={duration-fade}:d={fade}',
                '-c:a', 'pcm_s16le', '-ar', str(AUDIO_RATE), '-ac', '2',
                '-video_track_timescale', '24000', str(destination)]
    return command


def merge_command(ffmpeg, listing, destination):
    # PCM intermediates avoid introducing another AAC encoder delay at every join.
    return [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'warning', '-n',
            '-f', 'concat', '-safe', '1', '-i', str(listing),
            '-map', '0:v:0', '-map', '0:a:0', '-map_metadata', '-1',
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart', str(destination)]


def preflight(ffmpeg, ffprobe, work, overlay=True):
    """Check the actual crop, optional badge, audio fades and codecs before inference."""
    source, prepared = work/'probe.mov', work/'probe-ready.mov'
    image = work/'probe.ppm' if overlay else None
    if image is not None:
        badge(image, '0')
    run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
         '-f', 'lavfi', '-i', f'color=c=gray:s={UPSCALE_WIDTH}x{UPSCALE_HEIGHT}:r={FPS}:d=0.25',
         '-f', 'lavfi', '-i', f'anullsrc=r={AUDIO_RATE}:cl=stereo', '-t', '0.25',
         '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le',
         '-video_track_timescale', '24000', str(source)], work/'probe-source.log')
    run(prepare_command(ffmpeg, source, image, prepared, 6), work/'probe-prepare.log')
    listing = work/'probe-concat.txt'
    listing.write_text("ffconcat version 1.0\nfile 'probe-ready.mov'\nduration 0.25\n")
    run(merge_command(ffmpeg, listing, work/'probe.mp4'), work/'probe-merge.log')
    inspect_video(ffprobe, work/'probe.mp4', 6)


def model_options(args):
    options = []
    for option, value in [('-d', args.model_dir), ('--models-path', args.models_path)]:
        if value is not None:
            options += [option, str(value.expanduser().resolve())]
    if args.offline:
        options += ['--offline']
    return options


def render_command(args, prompt, source, index):
    return [args.h3cli, '-p', prompt, '--width', str(WIDTH), '--height', str(HEIGHT),
            '--frames', str(FRAMES), '--steps', str(args.steps),
            '--seed', str(args.seed+index), '--no-preview-vae',
            '--save-upscale-state', str(source), '--state-only'] + model_options(args)


def upscale_command(args, source, state, video, index):
    command = [args.h3cli, '--upscale-state', str(source),
               '--upscale-refine-steps', str(args.upscale_refine_steps),
               '--upscale-seed', str(args.seed+index),
               '--save-av-state', str(state), '-o', str(video)] + model_options(args)
    if args.upscale_refine_steps:
        command += ['--upscale-noise', '0.25']
    if args.upscale_model is not None:
        command += ['--upscale-model', str(args.upscale_model.expanduser().resolve())]
    return command


def segment_plan(args, sections, work):
    for index, (number, prompt) in enumerate(sections):
        stem = f'segment-{index+1:04d}'
        source, state = work/(stem+'.h3up'), work/(stem+'.h3av')
        video, prepared = work/(stem+'.mp4'), work/(stem+'-prepared.mov')
        image_badge = None if args.no_overlay else work/(stem+'.ppm')
        yield dict(number=number, prompt=prompt, stem=stem, frames=DELIVERED_FRAMES,
                   source=str(source), state=str(state), video=str(video), prepared=str(prepared),
                   badge=None if image_badge is None else str(image_badge),
                   render_command=render_command(args, prompt, source, index),
                   upscale_command=upscale_command(args, source, state, video, index),
                   prepare_command=prepare_command(args.ffmpeg, video, image_badge, prepared))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('prompts', type=Path, nargs='?', default=ROOT/'miniatures.txt')
    parser.add_argument('-o', '--output', type=Path, help='Default: PROMPTS_STEM.mp4 in the launch directory')
    parser.add_argument('--work-dir', type=Path, help='Fresh directory for segments and states (default: outputs/...)')
    repo_binary = ROOT.parents[1]/'bin/h3cli'
    default_binary = str(repo_binary) if repo_binary.is_file() else 'h3cli'
    parser.add_argument('--h3cli', default=os.environ.get('H3CLI_BIN', default_binary))
    parser.add_argument('--ffmpeg', default=os.environ.get('H3_FFMPEG', 'ffmpeg'))
    parser.add_argument('--ffprobe', default=os.environ.get('H3_FFPROBE', 'ffprobe'))
    parser.add_argument('-d', '--model-dir', type=Path)
    parser.add_argument('--models-path', type=Path)
    parser.add_argument('--steps', type=int, default=50, help='Denoising steps (default: 50)')
    parser.add_argument('--upscale-refine-steps', type=int, choices=(0, 2, 3, 4), default=4,
                        help='Native upscale refinement steps (default: 4; 0 is learned transfer only)')
    parser.add_argument('--upscale-model', type=Path, help='Override the native latent upscaler weights')
    parser.add_argument('--offline', action='store_true', help='Require existing local models')
    parser.add_argument('--dry-run', action='store_true', help='Print a JSON command plan without running tools')
    parser.add_argument('--no-overlay', action='store_true', help='Do not overlay segment numbers on the video')
    parser.add_argument('--seed', type=int, default=42, help='First seed; increments per segment (default: 42)')
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 1000:
        parser.error('--steps must be between 1 and 1000')
    started, work, report = time.monotonic(), None, {'status': 'failed', 'segments': []}
    try:
        sections = read_prompts(args.prompts.expanduser().resolve())
        if not 0 <= args.seed <= (1 << 64)-len(sections):
            raise ValueError('Seeds must fit unsigned 64-bit integers')
        output = (args.output or Path(args.prompts.stem+'.mp4')).expanduser().absolute()
        if output.suffix.lower() != '.mp4':
            raise ValueError('Output filename must end in .mp4')
        if output.exists():
            raise ValueError(f'Output already exists: {output}; choose another -o path')
        if args.dry_run:
            planned_work = (args.work_dir or Path('outputs')/(args.prompts.stem+'-plan')).expanduser().resolve()
            rows = list(segment_plan(args, sections, planned_work))
            total_frames = sum(row['frames'] for row in rows)
            print(json.dumps(dict(output=str(output), width=WIDTH, height=HEIGHT,
                                  upscale_width=UPSCALE_WIDTH, upscale_height=UPSCALE_HEIGHT,
                                  output_width=OUTPUT_WIDTH, output_height=OUTPUT_HEIGHT,
                                  fps=FPS, source_frames=FRAMES, independent_segments=True,
                                  audio_gain=AUDIO_GAIN, audio_fade_seconds=AUDIO_FADE_SECONDS,
                                  delivered_frames=total_frames,
                                  duration_seconds=total_frames/FPS, segments=rows,
                                  merge_command=merge_command(args.ffmpeg, planned_work/'concat.txt', output)),
                             indent=2))
            return 0
        args.h3cli, args.ffmpeg, args.ffprobe = map(executable, (args.h3cli, args.ffmpeg, args.ffprobe))
        ffmpeg, ffprobe = args.ffmpeg, args.ffprobe
        if args.work_dir:
            candidate = args.work_dir.expanduser().resolve()
            candidate.mkdir(parents=True, exist_ok=False)
            work = candidate
        else:
            Path('outputs').mkdir(exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix=args.prompts.stem+'-upscale-', dir='outputs')).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        report.update(output=str(output), width=WIDTH, height=HEIGHT,
                      upscale_width=UPSCALE_WIDTH, upscale_height=UPSCALE_HEIGHT,
                      output_width=OUTPUT_WIDTH, output_height=OUTPUT_HEIGHT,
                      frames=FRAMES, steps=args.steps, upscale_refine_steps=args.upscale_refine_steps,
                      overlay=not args.no_overlay, independent_segments=True, fps=FPS,
                      audio_gain=AUDIO_GAIN, audio_fade_seconds=AUDIO_FADE_SECONDS)
        print(f'{len(sections)} segments; intermediate files: {work}', flush=True)
        print(f'Native upscale: {WIDTH}x{HEIGHT} -> {UPSCALE_WIDTH}x{UPSCALE_HEIGHT}; '
              f'final crop: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', flush=True)
        print(f'Each independent clip: {DELIVERED_FRAMES} frames, '
              f'{DELIVERED_FRAMES/FPS:.0f} seconds; quiet audio with edge fades', flush=True)
        preflight(ffmpeg, ffprobe, work, overlay=not args.no_overlay)
        entries, total_frames = ['ffconcat version 1.0'], 0
        for index, row in enumerate(segment_plan(args, sections, work)):
            segment_start = time.monotonic()
            number, stem, frames = row['number'], row['stem'], row['frames']
            state, video, prepared = map(Path, (row['state'], row['video'], row['prepared']))
            row['status'] = 'rendering'
            report['segments'].append(row)
            (work/'run.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'\nRendering segment {number} ({index+1}/{len(sections)})', flush=True)
            run(row['render_command'], work/(stem+'-render.log'))
            source = Path(row['source'])
            if not source.is_file() or not source.stat().st_size:
                raise ValueError(f'Missing native upscale source: {source}')
            row['status'] = 'upscaling'
            (work/'run.json').write_text(json.dumps(report, indent=2)+'\n')
            run(row['upscale_command'], work/(stem+'-upscale.log'))
            for required in (state, Path(str(state)+'.presentation')):
                if not required.is_file() or not required.stat().st_size:
                    raise ValueError(f'Missing upscaled state or presentation sidecar: {required}')
            inspect_video(ffprobe, video, FRAMES, UPSCALE_WIDTH, UPSCALE_HEIGHT)
            if row['badge'] is not None:
                badge(Path(row['badge']), number)
            run(row['prepare_command'], work/(stem+'-prepare.log'))
            inspect_video(ffprobe, prepared, frames)
            entries += [f"file '{prepared.name}'", f'duration {frames/FPS:.9f}']
            total_frames += frames
            row.update(status='complete', wall_seconds=time.monotonic()-segment_start)
            (work/'run.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'Segment {number} complete: {row["wall_seconds"]:.2f} s', flush=True)
        listing = work/'concat.txt'
        listing.write_text('\n'.join(entries)+'\n')
        # Keep partial output separate; publish only after validating the merged file.
        fd, partial = tempfile.mkstemp(prefix='.'+output.stem+'-', suffix='.mp4', dir=output.parent)
        os.close(fd)
        Path(partial).unlink()
        try:
            run(merge_command(ffmpeg, listing, partial), work/'merge.log')
            inspect_video(ffprobe, partial, total_frames)
            os.link(partial, output)  # Exclusive publication; never overwrite an existing movie.
        finally:
            Path(partial).unlink(missing_ok=True)
        report.update(status='complete', delivered_frames=total_frames, duration_seconds=total_frames/FPS)
        print(f'\nCombined video: {output}', flush=True)
        return 0
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        print('\nInterrupted; intermediate files were preserved.', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        report['error'] = str(error)
        print(f'Error: {error}', file=sys.stderr)
        return 1
    finally:
        elapsed = time.monotonic()-started
        report['total_wall_seconds'] = elapsed
        if work is not None:
            (work/'run.json').write_text(json.dumps(report, indent=2)+'\n')
        if not args.dry_run:
            print(f'Total wall time: {elapsed:.2f} s ({elapsed/60:.2f} min)', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
