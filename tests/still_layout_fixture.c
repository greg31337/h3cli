#include "src/host.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv) {
    if (argc != 2)
        return 2;
    int ref = atoi(argv[1]);
    char error[512];
    h3_layout l;
    h3_layout_ref r[2]={{.kind=H3_LAYOUT_REF_IMAGE,.latent_t=1,.latent_h=16,.latent_w=16},{.kind=H3_LAYOUT_REF_IMAGE,.latent_t=1,.latent_h=20,.latent_w=32}};
    h3_layout_spec s={.text_len=11,.latent_t=1,.latent_h=30,.latent_w=40,.audio_t=2,.frame_count=1,.reference_count=(size_t)ref,.references=r};
    if(ref<0||ref>2||!h3_still_layout_build(&s,&l,error,sizeof(error)))return 1;
    printf("{\"positions\":[");for(size_t i=0;i<l.seq_len;i++)printf("%s[%.17g,%.17g,%.17g]",i?",":"",l.positions[i].t,l.positions[i].h,l.positions[i].w);
    printf("],\"segments\":[");for(size_t i=0;i<l.segment_count;i++)printf("%s[\"%s\",%zu,%zu]",i?",":"",h3_segment_name(l.segments[i].kind),l.segments[i].start,l.segments[i].stop);
    h3_sigma_schedule sig;h3_schedule_build(6,&sig);printf("],\"video_sigmas\":[");for(int i=0;i<=6;i++)printf("%s%.9g",i?",":"",sig.video[i]);
    printf("],\"audio_sigmas\":[");for(int i=0;i<=6;i++)printf("%s%.9g",i?",":"",sig.audio[i]);
    h3_rng rng;h3_rng_seed(&rng,42);printf("],\"noise\":[");for(int i=0;i<128;i++)printf("%s%.9g",i?",":"",h3_rng_normal(&rng));puts("]}");h3_layout_free(&l);return 0;
}
