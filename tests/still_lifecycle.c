#include "src/h3.h"
#include "src/vae/image_vae.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static int count;
static int cancel_frame(const h3_frame *f,void *p){(void)p;count++;if(f->frame_count!=1||f->frame_index||f->denoise_step!=-1)abort();return 1;}
static int cancel_load(const char *phase,int a,int b,void *p){(void)phase;(void)a;(void)b;(void)p;return 1;}
int main(void) {
    char dir[]="/tmp/h3-still-delivery-XXXXXX";if(!mkdtemp(dir))return 1;
    char out[256];snprintf(out,sizeof(out),"%s/output.png",dir);FILE *f=fopen(out,"wb");if(!f)return 1;fwrite("sentinel",1,8,f);fclose(f);
    h3_decode_options options={.output_path=out,.on_frame=cancel_frame};char error[512];
    const char *z="outputs/single-still/fixtures/256/latent.safetensors",*vae="models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors";
    h3_result *r=h3_decode_still_latent(z,vae,&options,error,sizeof(error));
    int ok=!r && count==1 && strstr(error,"cancel");h3_result_free(r);
    f=fopen(out,"rb");char bytes[9]={0};ok=ok && f && fread(bytes,1,8,f)==8 && !strcmp(bytes,"sentinel");if(f)fclose(f);
    options.on_progress=cancel_load;count=0;r=h3_decode_still_latent(z,vae,&options,error,sizeof(error));
    ok=ok && !r && !count && strstr(error,"cancel");h3_result_free(r);unlink(out);rmdir(dir);
    if(!ok){fprintf(stderr,"lifecycle failed: %s\n",error);return 1;}puts("PASS load cancellation, exactly-one cancelled frame callback, existing PNG preserved");return 0;
}
