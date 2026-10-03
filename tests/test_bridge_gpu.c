#include "src/sampling/bridge.h"
#include "src/denoise/dit.h"
#include "src/gpu.h"
#include "src/sampling/av_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char error[512];
static size_t checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #x, error); exit(1); } } while (0)
static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result; memcpy(&result, &bits, sizeof(result)); return result;
}

/* Exercise actual packed video patches and both noninterleaved audio streams.
 * An independent CPU oracle uses the public unpack/Euler/repack path. */
static void trajectory(h3_gpu *gpu, int length, float maximum,
                       h3_bridge_profile_type type, int reuse, const h3_av_state *fixture) {
    enum { VT=27, H=4, W=6, AT=150, VC=24*VT*H*W, AC=32*2*AT,
           VR=VC/96, AR=AC/32, COUNT=VC>AC?VC:AC,
           OFFSET=193, EXTRA=7, STEPS=20 };
    h3_bridge_profile p;
    CHECK(h3_bridge_profile_build(39,length,maximum,type,&p,NULL,0,error,sizeof(error)));
    float video[VC], audio[AC], vv[VC], av[AC];
    float packed[2][COUNT+OFFSET+EXTRA], want[COUNT], got[COUNT+OFFSET+EXTRA];
    uint16_t last[2][COUNT], previous[2][COUNT];
    uint32_t classes[2][AR];
    h3_gpu_tensor *sample[2], *initial[2], *last_gpu[2], *previous_gpu[2], *map[2];
    const int counts[2]={VC,AC}, rows[2]={VR,AR}, widths[2]={96,32};
    for (int stream=0;stream<2;stream++) {
        for (int r=0;r<rows[stream];r++)
            classes[stream][r]=(uint32_t)h3_bridge_row_class(&p,stream,(size_t)r,H,W,AT);
        for (int i=0;i<counts[stream]+OFFSET+EXTRA;i++)
            packed[stream][i]=(float)((i*31)%97-48)/23.0f;
        for (int i=0;i<counts[stream];i++) {
            if (fixture) {
                const float *values=stream ? fixture->audio : fixture->video;
                size_t count=stream ? fixture->info.audio_elements : fixture->info.video_elements;
                packed[stream][OFFSET+i]=values[(size_t)i%count];
            }
            if (p.class_mask[classes[stream][i/widths[stream]]]==0) {
                // Exact preservation is bitwise, including NaN payloads.
                const uint32_t bits[]={0x80000000,0,0x7fc01234,0xffc05678};
                memcpy(packed[stream]+OFFSET+i,bits+i%4,sizeof(float));
            }
            last[stream][i]=previous[stream][i]=0;
        }
        sample[stream]=h3_gpu_tensor_from_f32(gpu,packed[stream],(size_t)counts[stream]+OFFSET+EXTRA);
        initial[stream]=h3_gpu_tensor_from_f32(gpu,packed[stream]+OFFSET,(size_t)counts[stream]);
        last_gpu[stream]=h3_gpu_tensor_new_bf16(gpu,(size_t)counts[stream]);
        previous_gpu[stream]=h3_gpu_tensor_new_bf16(gpu,(size_t)counts[stream]);
        map[stream]=h3_gpu_tensor_from_u32(gpu,classes[stream],(size_t)rows[stream]);
        CHECK(sample[stream] && initial[stream] && last_gpu[stream] && previous_gpu[stream] && map[stream]);
    }
    CHECK(h3_dit_unpatchify_video(packed[0]+OFFSET,24,VT,H,W,video,VC));
    CHECK(h3_dit_unpack_audio(packed[1]+OFFSET,32,AT,audio,AC));
    h3_gpu_tensor *strengths=h3_gpu_tensor_from_f32(gpu,p.class_mask,H3_TARGET_ROW_CLASSES);
    uint32_t zero=0;
    h3_gpu_tensor *flag=h3_gpu_tensor_from_u32(gpu,&zero,1);
    CHECK(strengths && flag);
    uint8_t selected[STEPS];
    CHECK(h3_dit_reuse_schedule(STEPS,reuse,selected,STEPS)>0);
    int last_step=-1, previous_step=-1;
    for (int step=0;step<STEPS;step++) {
        if (selected[step]) {
            previous_step=last_step; last_step=step;
            for (int stream=0;stream<2;stream++) {
                memcpy(previous[stream],last[stream],(size_t)counts[stream]*sizeof(uint16_t));
                for (int i=0;i<counts[stream];i++) {
                    // BF16 powers/mantissas, signs and nonfinite unused exact velocities.
                    last[stream][i]=(uint16_t)(0x3e00+(i*13+step*37)%768+((i+step)%2 ? 0x8000 : 0));
                    if (p.class_mask[classes[stream][i/widths[stream]]]==0)
                        last[stream][i]=0x7fc1;
                }
                CHECK(h3_gpu_tensor_write_bf16(last_gpu[stream],last[stream],(size_t)counts[stream]));
                CHECK(h3_gpu_tensor_write_bf16(previous_gpu[stream],previous[stream],(size_t)counts[stream]));
            }
        }
        float sv=(float)(STEPS-step)/(float)STEPS, nv=(float)(STEPS-step-1)/(float)STEPS;
        float sa=sv*sv, na=nv*nv;
        float ratios[2]={0,0};
        if (!selected[step] && previous_step>=0) {
            float ls=(float)(STEPS-last_step)/(float)STEPS, ps=(float)(STEPS-previous_step)/(float)STEPS;
            ratios[0]=(sv-ls)/(ls-ps); ratios[1]=(sa-ls*ls)/(ls*ls-ps*ps);
        }
        for (int stream=0;stream<2;stream++) {
            for (int i=0;i<counts[stream];i++) {
                float value=from_bf16(last[stream][i]);
                want[i]=ratios[stream]!=0 ? fmaf(ratios[stream],value-from_bf16(previous[stream][i]),value) : value;
            }
            if (stream) CHECK(h3_dit_unpack_audio(want,32,AT,av,AC));
            else CHECK(h3_dit_unpatchify_video(want,24,VT,H,W,vv,VC));
        }
        CHECK(h3_bridge_euler_step(&p,VT,H,W,AT,video,audio,vv,av,sv,nv,sa,na,NULL));
        CHECK(h3_gpu_begin(gpu));
        for (int stream=0;stream<2;stream++) {
            CHECK(h3_gpu_bridge_euler_bf16(gpu,sample[stream],OFFSET,last_gpu[stream],previous_gpu[stream],
                map[stream],strengths,(uint32_t)rows[stream],(uint32_t)widths[stream],stream ? sa-na : sv-nv,ratios[stream]));
            CHECK(h3_gpu_bridge_check_exact(gpu,sample[stream],OFFSET,initial[stream],map[stream],strengths,flag,
                (uint32_t)rows[stream],(uint32_t)widths[stream]));
        }
        CHECK(h3_gpu_submit(gpu));
        uint32_t changed=99;
        CHECK(h3_gpu_tensor_read_u32(flag,&changed,1) && changed==0);
        for (int stream=0;stream<2;stream++) {
            if (stream) CHECK(h3_dit_pack_audio(audio,32,AT,want,AC));
            else CHECK(h3_dit_patchify_video(video,24,VT,H,W,want,VC));
            CHECK(h3_gpu_tensor_read_f32(sample[stream],got,(size_t)counts[stream]+OFFSET+EXTRA));
            CHECK(!memcmp(got,packed[stream],OFFSET*sizeof(float)));
            CHECK(!memcmp(got+OFFSET+counts[stream],packed[stream]+OFFSET+counts[stream],EXTRA*sizeof(float)));
            for (int i=0;i<counts[stream];i++) CHECK(!memcmp(got+OFFSET+i,want+i,sizeof(float)));
        }
    }
    // The audit must detect a single changed bit in an exact sample.
    size_t index=0;
    while (p.class_mask[classes[0][index/96]]!=0) index++;
    float corrupt=123.0f;
    CHECK(h3_gpu_tensor_write_f32_range(sample[0],OFFSET+index,&corrupt,1));
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_bridge_check_exact(gpu,sample[0],OFFSET,initial[0],map[0],strengths,flag,VR,96));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_u32(flag,&zero,1) && zero==1);
    // A malformed class must be reported by the device audit, without reading
    // beyond the strength table or writing outside the target range.
    classes[0][0]=UINT32_MAX;
    h3_gpu_tensor *bad_map=h3_gpu_tensor_from_u32(gpu,classes[0],VR); CHECK(bad_map);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_bridge_euler_bf16(gpu,sample[0],OFFSET,last_gpu[0],previous_gpu[0],bad_map,strengths,VR,96,.1f,0));
    CHECK(h3_gpu_bridge_check_exact(gpu,sample[0],OFFSET,initial[0],bad_map,strengths,flag,VR,96));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_u32(flag,&zero,1) && zero==3);
    h3_gpu_tensor_free(bad_map);
    CHECK(!h3_gpu_bridge_euler_bf16(gpu,sample[0],SIZE_MAX,last_gpu[0],previous_gpu[0],map[0],strengths,VR,96,.1f,0));
    CHECK(!h3_gpu_bridge_euler_bf16(gpu,sample[0],0,last_gpu[0],previous_gpu[0],map[0],strengths,UINT32_MAX,96,.1f,0));
    CHECK(!h3_gpu_bridge_euler_bf16(gpu,sample[0],0,last_gpu[0],previous_gpu[0],map[0],strengths,VR,0,.1f,0));
    CHECK(!h3_gpu_bridge_euler_bf16(gpu,sample[0],OFFSET,last_gpu[0],previous_gpu[0],map[0],strengths,VR,96,0,0));
    CHECK(!h3_gpu_bridge_euler_bf16(gpu,sample[0],OFFSET,last_gpu[0],previous_gpu[0],map[0],strengths,VR,96,-.1f,0));
    CHECK(!h3_gpu_bridge_euler_bf16(gpu,sample[0],OFFSET,last_gpu[0],previous_gpu[0],map[0],strengths,VR,96,NAN,0));
    for (int s=0;s<2;s++) {
        h3_gpu_tensor_free(sample[s]); h3_gpu_tensor_free(initial[s]);
        h3_gpu_tensor_free(last_gpu[s]); h3_gpu_tensor_free(previous_gpu[s]); h3_gpu_tensor_free(map[s]);
    }
    h3_gpu_tensor_free(strengths); h3_gpu_tensor_free(flag);
}

int main(int argc, char **argv) {
    h3_gpu *gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error)); CHECK(gpu);
    const int lengths[]={4,6,8,9,10};
    const float maxima[]={0,.25f,.4f,.5f,.65f};
    for (int type=0;type<3;type++) for (int i=0;i<5;i++) for (int reuse=1;reuse<=3;reuse++)
        trajectory(gpu,lengths[i],maxima[i],(h3_bridge_profile_type)type,reuse,NULL);
    int trajectories=45;
    if (argc>1) {
        const char *names[]={"face1","body1","face2","body2","12"};
        for (size_t i=0;i<sizeof(names)/sizeof(*names);i++) {
            char path[4096];
            CHECK(snprintf(path,sizeof(path),"%s/%s.h3av",argv[1],names[i])<(int)sizeof(path));
            h3_av_state *fixture=h3_av_state_load(path,error,sizeof(error)); CHECK(fixture);
            for (int reuse=1;reuse<=3;reuse++) {
                trajectory(gpu,8,.5f,H3_BRIDGE_STEPPED,reuse,fixture); trajectories++;
            }
            h3_av_state_free(fixture);
            printf("bridge GPU encoded numerical fixture: %s\n",names[i]);
        }
    }
    h3_gpu_free(gpu);
    printf("bridge GPU: %zu checks; %d twenty-step CPU/GPU trajectories bit-identical\n",checks,trajectories);
    return 0;
}
