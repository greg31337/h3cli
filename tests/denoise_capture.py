#!/usr/bin/env python3
"""Build a test-only public-API driver that records preview latents/velocities."""
import subprocess
from pathlib import Path
from scalingfix_prepare import ROOT, FLAGS


def main():
    out = ROOT / 'outputs/denoise-validation/capture-build'
    out.mkdir(parents=True, exist_ok=True)
    helper = '''
#include <stdio.h>
#include <stdlib.h>
static void preview_fixture_write(const char *kind,int step,const float *v,size_t n) {
    const char *base=getenv("H3_TEST_PREVIEW_CAPTURE");if(!base)return;
    char path[4096];snprintf(path,sizeof(path),"%s.%s-step-%02d.f32",base,kind,step);
    FILE *f=fopen(path,"wb");if(!f||fwrite(v,sizeof(float),n,f)!=n||fclose(f))abort();
}
'''
    source = (ROOT / 'src/engine.c').read_text()
    needle = '    char detail[512];\n    h3_video_frames decoded;'
    assert source.count(needle) == 1
    source = source.replace(needle, '    preview_fixture_write("x0",completed_steps,video_latent,video_elements);\n' + needle)
    (out / 'api.c').write_text(helper + source)
    source = (ROOT / 'src/denoise/dit.c').read_text()
    needle = '            h3_preview_denoised_f32(video_velocity, video_latent, video_count,'
    assert source.count(needle) == 1
    source = source.replace(needle, '            preview_fixture_write("velocity",step+1,video_velocity,video_count);\n' + needle)
    (out / 'dit.c').write_text(helper + source)
    (ROOT/'bin').mkdir(exist_ok=True)
    subprocess.run(['clang', '-O3', '-std=c11', '-D_DARWIN_C_SOURCE', '-I', str(ROOT),
                    str(ROOT / 'tests/denoise_generate.c'), str(out / 'api.c'), str(out / 'dit.c'),
                    str(ROOT / 'bin/libh3.a'), *FLAGS, '-o', str(ROOT / 'bin/preview_capture_generate')], check=True)


if __name__ == '__main__':
    main()
