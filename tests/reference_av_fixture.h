#ifndef H3_REFERENCE_AV_FIXTURE_H
#define H3_REFERENCE_AV_FIXTURE_H
/* Test-only adapter for the immutable recorded AV fixture. Production loads
 * current schema only. The whole fixture hash is checked before interpretation. */
#include "src/sampling/av_state.h"
#include "src/digest.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t reference_av_u64(const uint8_t *p,unsigned n) {
    uint64_t v=0;for(unsigned i=0;i<n;i++)v|=(uint64_t)p[i]<<(8*i);return v;
}
static h3_av_state *reference_av_load(const char *path,char *error,size_t size) {
    FILE *f=fopen(path,"rb");if(!f){snprintf(error,size,"cannot open recorded AV fixture");return NULL;}
    uint8_t header[160],buffer[65536],digest[32];h3_sha256_ctx hash;h3_sha256_init(&hash);
    size_t n;while((n=fread(buffer,1,sizeof(buffer),f)))h3_sha256_update(&hash,buffer,(h3_sha256_size)n);
    int bad=ferror(f);h3_sha256_final(digest,&hash);rewind(f);
    char hex[65];for(int i=0;i<32;i++)snprintf(hex+2*i,3,"%02x",digest[i]);
    if(bad||strcmp(hex,"c309b366a3c717ac7ae2090bae36d9150daa54ecea7621a2a53c4a47edefe80b")||fread(header,1,160,f)!=160) {
        fclose(f);return h3_av_state_load(path,error,size);
    }
    h3_av_state *s=h3_av_state_new((int)reference_av_u64(header+24,4),(int)reference_av_u64(header+28,4),
        (int)reference_av_u64(header+32,4),reference_av_u64(header+64,8),header+88);
    int ok=s&&reference_av_u64(header+8,4)==1&&reference_av_u64(header+72,8)==s->info.video_elements*4&&
        reference_av_u64(header+80,8)==s->info.audio_elements*4&&
        fread(s->video,4,s->info.video_elements,f)==s->info.video_elements&&
        fread(s->audio,4,s->info.audio_elements,f)==s->info.audio_elements&&fgetc(f)==EOF&&!ferror(f);
    fclose(f);if(!ok){h3_av_state_free(s);snprintf(error,size,"invalid recorded AV fixture");return NULL;}return s;
}
#endif
