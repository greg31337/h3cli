#!/usr/bin/env python3
"""Local Whisper and waveform observations for token-fix generation artifacts.

ASR is an observation, not proof of intelligibility or lip synchronization.
No audio leaves this machine. Run after tokenizer_generation.py (or on its
completed clips while remaining GPU tests run).
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, default=Path('outputs/tokenfix-validation/generation'))
    p.add_argument('--model', type=Path, default=Path('outputs/tokenfix-validation/whisper-base.en'))
    args = p.parse_args()
    torch.set_num_threads(4)
    processor = WhisperProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(args.model), local_files_only=True).eval()
    result_path = args.directory/'audio-observations.json'
    records = json.loads(result_path.read_text()) if result_path.exists() else {}
    runs = json.loads((args.directory/'results.json').read_text())
    for name, run in runs.items():
        if (name in records and records[name]['mp4_sha256'] == run['hashes']['mp4']
                and 'asr_timestamp_outside_clip' in records[name]):
            continue
        path = args.directory/(name+'.mp4')
        samples = np.frombuffer(subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
            '-ac', '1', '-ar', '16000', '-f', 'f32le', '-']), dtype=np.float32).copy()
        assert np.isfinite(samples).all(), name
        inputs = processor(samples, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
        with torch.inference_mode():
            tokens = model.generate(**inputs, return_timestamps=True)
        decoded = processor.tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True, output_offsets=True)
        # RMS is measured in 20-ms windows. These are sound/activity measurements,
        # deliberately not labelled as speech in ambient/music-containing clips.
        frames = samples[:len(samples)//320*320].reshape(-1, 320)
        rms = np.sqrt((frames**2).mean(axis=1))
        db = 20*np.log10(np.maximum(rms, 1e-12))
        active = np.flatnonzero(db > max(-55, float(db.max())-30))
        records[name] = {'mp4_sha256': run['hashes']['mp4'], 'asr_model': str(args.model),
            'asr': decoded, 'duration_seconds': len(samples)/16000,
            'asr_timestamp_outside_clip': any(
                t > len(samples)/16000 + 0.04 for segment in decoded['offsets']
                for t in segment['timestamp']),
            'rms': float(np.sqrt((samples**2).mean())), 'peak': float(np.abs(samples).max()),
            'activity_first_seconds': float(active[0]*0.02) if len(active) else None,
            'activity_last_seconds': float((active[-1]+1)*0.02) if len(active) else None,
            'rms_db_20ms': db.tolist()}
        result_path.write_text(json.dumps(records, indent=2)+'\n')
        print(name, decoded, flush=True)


if __name__ == '__main__':
    main()
