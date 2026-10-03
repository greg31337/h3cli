#!/usr/bin/env python3
"""Local Whisper transcription as supporting evidence; not a lip-sync oracle."""
import json,subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import WhisperProcessor,WhisperForConditionalGeneration
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/scalingfix-validation'
def main():
    torch.set_num_threads(8);model_dir=OUT/'whisper-tiny.en';processor=WhisperProcessor.from_pretrained(model_dir);model=WhisperForConditionalGeneration.from_pretrained(model_dir).eval()
    results={}
    for mode in ('legacy','scaled-q','reference'):
        path=OUT/'generation'/f'fl2va-{mode}.mp4'
        if not path.exists():continue
        data=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-vn','-ar','16000','-ac','1','-f','f32le','-'])
        samples=np.frombuffer(data,np.float32).copy();features=processor(samples,sampling_rate=16000,return_tensors='pt')
        with torch.no_grad():ids=model.generate(**features,return_timestamps=True)
        results[mode]={'transcription':processor.batch_decode(ids,skip_special_tokens=True)[0],'timestamped':processor.batch_decode(ids,skip_special_tokens=True,decode_with_timestamps=True)[0]}
        print(mode,results[mode],flush=True)
    (OUT/'generation/speech.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
