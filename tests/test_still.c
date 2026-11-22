/* Stock still geometry/artifacts and independent packing/frame-selection checks. */
#include "src/internal.h"
#include "src/sampling/av_state.h"
#include "src/vae/image_vae.h"
#include "src/vae/video_vae.c"
#include <unistd.h>
static unsigned checks;
static char error[512];
#define CHECK(x) do { checks++; if(!(x)){fprintf(stderr,"still check %u line %d: %s (%s)\n",checks,__LINE__,#x,error);exit(1);} } while(0)
static void host(void) {
    const int shapes[][3]={{640,480,300},{1344,768,1008},{2144,1216,2546},{3872,2176,8228},{4096,4096,16384},{8192,2048,16384}};
    for(size_t i=0;i<sizeof(shapes)/sizeof(*shapes);i++)for(int transpose=0;transpose<2;transpose++) {
        int w=shapes[i][transpose],h=shapes[i][1-transpose];
        CHECK(h3_still_geometry(h/16,w/16,error,sizeof(error)));
        h3_layout_spec spec={.text_len=11,.latent_t=1,.latent_h=h/16,.latent_w=w/16,.audio_t=2,.frame_count=1};h3_layout l={0};
        CHECK(h3_still_layout_build(&spec,&l,error,sizeof(error)));
        CHECK(l.img_target_rows==(uint32_t)shapes[i][2] && l.audio_target_rows==4 && l.seq_len==l.img_target_rows+15);h3_layout_free(&l);
        CHECK(!h3_layout_build(&spec,&l,error,sizeof(error)));
    }
    const int invalid[][2]={{INT32_MAX,INT32_MAX},{3,4},{0,2},{2,-2},{514,2},{258,256}};
    for(size_t i=0;i<sizeof(invalid)/sizeof(*invalid);i++)CHECK(!h3_still_geometry(invalid[i][0],invalid[i][1],error,sizeof(error)));
    CHECK(!h3_geometry_valid(3872,2176,0));CHECK(h3_geometry_valid(1344,768,0));
    char name[]="/tmp/h3-still-host-XXXXXX";int fd=mkstemp(name);CHECK(fd>=0);close(fd);
    float values[24*2*4];for(size_t i=0;i<sizeof(values)/4;i++)values[i]=(float)i/19;
    h3_still_latent z={.height=2,.width=4,.values=values};memset(z.compatibility_sha256,'a',64);
    for(int ref=0;ref<2;ref++) {
        z.reference_model=ref;CHECK(h3_still_latent_save(name,&z,error,sizeof(error)));h3_still_latent loaded={0};
        CHECK(h3_still_latent_load(name,&loaded,error,sizeof(error)));CHECK(loaded.reference_model==ref);
        CHECK(!memcmp(values,loaded.values,sizeof(values)));h3_still_latent_free(&loaded);
    }
    values[0]=NAN;CHECK(!h3_still_latent_save(name,&z,error,sizeof(error)));values[0]=0;
    FILE *file=fopen(name,"r+b");uint64_t hn;CHECK(file && fread(&hn,8,1,file)==1 && !fseek(file,(long)(8+hn),SEEK_SET));
    float invalid_value=NAN;CHECK(fwrite(&invalid_value,4,1,file)==1 && !fclose(file));h3_still_latent loaded={0};
    CHECK(!h3_still_latent_load(name,&loaded,error,sizeof(error)));CHECK(!truncate(name,12));
    CHECK(!h3_still_latent_load(name,&loaded,error,sizeof(error)));unlink(name);
    h3_params p=H3_PARAMS_DEFAULT;CHECK(h3_still_options(&p,error,sizeof(error)));
    p.image_vae="retired";CHECK(!h3_still_options(&p,error,sizeof(error)));CHECK(strstr(error,"retired"));p.image_vae=NULL;
    p.still=1;p.frames=1;p.width=3872;p.height=2176;CHECK(h3_still_options(&p,error,sizeof(error)));
    p.frames=5;CHECK(!h3_still_options(&p,error,sizeof(error)));p.frames=1;
    p.preview_vae=1;CHECK(!h3_still_options(&p,error,sizeof(error)));p.preview_vae=0;
    p.output_path="x.mp4";CHECK(!h3_still_options(&p,error,sizeof(error)));p.output_path="x.png";CHECK(h3_still_options(&p,error,sizeof(error)));
    h3_params video=H3_PARAMS_DEFAULT;char *vk=h3_conditioning_key("test",&video,640,480,0),*sk=h3_conditioning_key("test",&p,640,480,0);
    CHECK(vk && sk && strcmp(vk,sk));free(vk);free(sk);
    CHECK(h3_align_frame_count(1)==5 && h3_video_latent_t(243)==72);
    char directory[]="/tmp/h3-still-model-XXXXXX";CHECK(mkdtemp(directory));
    char weight[256],alias[256];snprintf(weight,sizeof(weight),"%s/model.safetensors",directory);
    snprintf(alias,sizeof(alias),"%s/image.png",directory);
    file=fopen(weight,"wb");CHECK(file && !fclose(file));CHECK(!link(weight,alias));
    CHECK(!h3_still_protect_vae(directory,NULL,weight,error,sizeof(error)));
    CHECK(!h3_still_protect_vae(directory,alias,NULL,error,sizeof(error)));
    CHECK(h3_still_protect_vae(directory,"unused.png","unused.safetensors",error,sizeof(error)));
    unlink(alias);unlink(weight);rmdir(directory);
}
static void gpu(void) {
    vae_context v={.latent_t=5,.latent_h=2,.latent_w=4,.patches=40,.sequence=45,.output_frames=1,.still=1};
    v.gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(v.gpu);
    CHECK(prepare_rope(&v,error,sizeof(error)));float cosines[45*24],sines[45*24];
    CHECK(h3_gpu_tensor_read_f32(v.rope_cos,cosines,45*24));CHECK(h3_gpu_tensor_read_f32(v.rope_sin,sines,45*24));
    for(int row=0;row<45;row++)for(int j=0;j<24;j++) {
        if(row>=40){CHECK(cosines[row*24+j]==1);CHECK(sines[row*24+j]==0);}
        else if(j<8){float angle=2*3.14159265358979323846f*(2*((float)(row/8)+.5f)/5-1)/powf(100,(float)j/8);
            CHECK(fabsf(cosines[row*24+j]-cosf(angle))<1e-6f);CHECK(fabsf(sines[row*24+j]-sinf(angle))<1e-6f);}
    }
    float input[24*40],mean[24],std[24];for(int c=0;c<24;c++){mean[c]=(float)c;std[c]=2;for(int r=0;r<40;r++)input[c*40+r]=(float)(r%8);}
    CHECK(prepare_input(&v,input,mean,std,error,sizeof(error)));float packed[24*40];CHECK(h3_gpu_tensor_read_f32(v.latent,packed,24*40));
    for(int r=0;r<40;r++)for(int c=0;c<24;c++)CHECK(packed[r*24+c]==(float)((r%8)*2+c));
    size_t n=45*OUTPUT_PATCH;float *projected=malloc(n*4);CHECK(projected);
    for(size_t i=0;i<n;i++)projected[i]=NAN;
    for(int p=0;p<40;p++)for(int c=0;c<3;c++)for(int t=0;t<4;t++)for(int y=0;y<16;y++)for(int x=0;x<16;x++)
        projected[p*OUTPUT_PATCH+((c*4+t)*16+y)*16+x]=(float)(t*100+p*3+c);
    v.projected=h3_gpu_tensor_from_f32(v.gpu,projected,n);CHECK(v.projected);
    h3_video_frames out={0};CHECK(unpack_frame_range(&v,0,1,&out,error,sizeof(error)));CHECK(out.frames==1 && out.width==64 && out.height==32);
    for(int y=0;y<32;y++)for(int x=0;x<64;x++)for(int c=0;c<3;c++)
        CHECK(out.rgb[(y*64+x)*3+c]==(float)(300+(y/16*4+x/16)*3+c));
    h3_video_frames_free(&out);free(projected);cleanup(&v);
}
int main(int argc,char **argv){if(argc>1 && !strcmp(argv[1],"gpu"))gpu();else host();printf("still: %u checks passed\n",checks);return 0;}
