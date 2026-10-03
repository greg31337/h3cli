"""Validate delivery geometry and the exact AV latent-to-audio trim contract."""
import json,re,subprocess
from pathlib import Path
from continuation_metrics import state
from quant_qualify import inspect

def inspect_state_media(media,av,width,height,frames):
    row=inspect(media,width,height,frames)
    geometry=state(Path(av))[0]
    sidecar=Path(str(av)+'.presentation').read_text()
    trim_frames,trim_samples=map(int,re.search(r'^trim (\d+) (\d+)$',sidecar,re.M).groups())
    assert geometry[2]-trim_frames==frames,'video trim differs from completed state'
    # H3 has 40 audio latents/s and 24 video frames/s; round to nearest
    # latent before multiplying by its 800-sample decoder stride.
    assert geometry[6]==(5*geometry[2]+1)//3,'audio latent duration differs from frame contract'
    expected_samples=geometry[6]*800-trim_samples
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(media)]))
    streams={s['codec_type']:s for s in probe['streams']}
    assert abs(float(streams['video']['start_time']))<1e-6
    assert abs(float(streams['audio']['start_time']))<1/32000
    assert abs(float(streams['video']['duration'])-frames/24)<1e-5
    assert abs(float(streams['audio']['duration'])-expected_samples/32000)<1/32000+1e-6,'audio delivery trim differs from latent contract'
    row.update(audio_samples=expected_samples,audio_duration=expected_samples/32000,
        trim_frames=trim_frames,trim_samples=trim_samples,exact_trim_contract=True)
    return row
