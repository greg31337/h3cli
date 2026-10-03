/* Exercise the real T=1 tile core under ASan/UBSan, including all slice offsets. */
#include "../src/vae/video_vae.c"
#include "src/internal.h"
#include "src/media/ffmpeg.h"
#include <unistd.h>
static int checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
static const char *model="models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors";
static void host(void) {
    char error[512];h3_st_header h={0};h3_image_vae_info info;
    CHECK(h3_st_read_header(model,&h,error,sizeof(error)));
    CHECK(h3_image_vae_validate(&h,&info,error,sizeof(error)));CHECK(info.output_slice==3);
    const char *keys[]={"h3_t1_direct","h3_t1_format","h3_t1_output_slice","minimax_h3_video_vae"};
    const char *bad[][9]={{"false","True","yes","0","",NULL},{"full_decoder_v2","",NULL},{"-1","4","3x"," 3","3 ","3.0","03","999999999999999999",NULL},{"{}","{\"source_config\":null}","{\"a\":1,\"a\":2}","NaN",NULL}};
    for(int k=0;k<4;k++)for(size_t i=0;i<h.metadata_count;i++)if(!strcmp(h.metadata_keys[i],keys[k])) {
        char *saved=h.metadata_values[i];
        for(int b=0;bad[k][b];b++){h.metadata_values[i]=(char*)bad[k][b];CHECK(!h3_image_vae_validate(&h,&info,error,sizeof(error)));}
        h.metadata_values[i]=saved;
    }
    h3_dtype saved=h.tensors[0].dtype;h.tensors[0].dtype=H3_DTYPE_BF16;CHECK(!h3_image_vae_validate(&h,&info,error,sizeof(error)));h.tensors[0].dtype=saved;
    uint64_t shape=h.tensors[0].shape[0];h.tensors[0].shape[0]=UINT64_MAX;CHECK(!h3_image_vae_validate(&h,&info,error,sizeof(error)));h.tensors[0].shape[0]=shape;
    h3_st_free_header(&h);
    CHECK(h3_weight_f16_to_f32(0x3c00)==1);CHECK(h3_weight_f16_to_f32(0xc000)==-2);
    CHECK(h3_weight_f16_to_f32(1)==0x1p-24f);CHECK(h3_weight_f16_to_f32(0x7bff)==65504);
    CHECK(isnan(h3_weight_f16_to_f32(0x7e00)));CHECK(isinf(h3_weight_f16_to_f32(0x7c00)));
    CHECK(signbit(h3_weight_f16_to_f32(0x8000)));
    CHECK(!h3_still_geometry(INT32_MAX,INT32_MAX,error,sizeof(error)));CHECK(!h3_still_geometry(3,4,error,sizeof(error)));
    CHECK(h3_still_geometry(30,40,error,sizeof(error)));
    char name[]="/tmp/h3-still-host-XXXXXX";int fd=mkstemp(name);CHECK(fd>=0);close(fd);
    float values[24*2*4];for(size_t i=0;i<sizeof(values)/4;i++)values[i]=(float)i/19;
    h3_still_latent z={.height=2,.width=4,.values=values};memset(z.compatibility_sha256,'a',64);
    CHECK(h3_still_latent_save(name,&z,error,sizeof(error)));h3_still_latent loaded={0};
    CHECK(h3_still_latent_load(name,&loaded,error,sizeof(error)));CHECK(!memcmp(values,loaded.values,sizeof(values)));h3_still_latent_free(&loaded);
    values[0]=NAN;CHECK(!h3_still_latent_save(name,&z,error,sizeof(error)));values[0]=0;
    FILE *file=fopen(name,"r+b");uint64_t hn;CHECK(file && fread(&hn,8,1,file)==1 && !fseek(file,(long)(8+hn),SEEK_SET));
    float invalid=NAN;CHECK(fwrite(&invalid,4,1,file)==1 && !fclose(file));CHECK(!h3_still_latent_load(name,&loaded,error,sizeof(error)));
    CHECK(!truncate(name,12));CHECK(!h3_still_latent_load(name,&loaded,error,sizeof(error)));unlink(name);
    h3_params p=H3_PARAMS_DEFAULT;CHECK(h3_still_options(&p,error,sizeof(error)));p.image_vae=model;CHECK(!h3_still_options(&p,error,sizeof(error)));
    p.still=1;p.frames=1;CHECK(h3_still_options(&p,error,sizeof(error)));p.frames=5;CHECK(!h3_still_options(&p,error,sizeof(error)));p.frames=1;
    p.preview_vae=1;CHECK(!h3_still_options(&p,error,sizeof(error)));p.preview_vae=0;
    p.output_path="x.mp4";CHECK(!h3_still_options(&p,error,sizeof(error)));p.output_path="x.png";CHECK(h3_still_options(&p,error,sizeof(error)));
    h3_layout_spec spec={.text_len=11,.latent_t=1,.latent_h=30,.latent_w=40,.audio_t=2,.frame_count=1};h3_layout l={0};
    CHECK(!h3_layout_build(&spec,&l,error,sizeof(error)));CHECK(h3_still_layout_build(&spec,&l,error,sizeof(error)));
    CHECK(l.img_target_rows==300 && l.audio_target_rows==4 && l.seq_len==315);h3_layout_free(&l);
    spec.audio_t=0;CHECK(!h3_still_layout_build(&spec,&l,error,sizeof(error)));spec.audio_t=2;
    spec.reference_count=1;CHECK(!h3_still_layout_build(&spec,&l,error,sizeof(error)));spec.reference_count=0;
    h3_params video=H3_PARAMS_DEFAULT;char *vk=h3_conditioning_key("test",&video,640,480,0),*sk=h3_conditioning_key("test",&p,640,480,0);
    CHECK(vk && sk && strcmp(vk,sk));free(vk);free(sk);
    CHECK(h3_align_frame_count(1)==5 && h3_video_latent_t(243)==72);

}
static void gpu(void) {
    char error[512];vae_context v={.latent_t=1,.latent_h=2,.latent_w=4,.patches=8,.sequence=13,.output_frames=1,.image_weights=1,.image_slice=3};
    v.gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(v.gpu);
    char path[]="/tmp/h3-still-f16-XXXXXX";int fd=mkstemp(path);CHECK(fd>=0);FILE *f=fdopen(fd,"wb");CHECK(f);
    const char header[]="{\"weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
    uint64_t header_size=strlen(header);uint16_t half[]={0x3c00,0xc000,0x7bff,1};
    CHECK(fwrite(&header_size,8,1,f)==1 && fwrite(header,1,header_size,f)==header_size && fwrite(half,2,4,f)==4 && !fclose(f));
    h3_weight_store *weights=h3_weight_store_open_file(path,error,sizeof(error));CHECK(weights);
    uint64_t shape[]={4};h3_gpu_tensor *converted=h3_weight_load_f16_f32(weights,v.gpu,"weight",1,shape,error,sizeof(error));CHECK(converted);
    float converted_values[4];CHECK(h3_gpu_tensor_read_f32(converted,converted_values,4));
    for (int i = 0; i < 4; i++)
        CHECK(converted_values[i] == h3_weight_f16_to_f32(half[i]));
    h3_gpu_tensor_free(converted);
    f=fopen(path,"r+b");CHECK(f && !fseek(f,(long)(8+header_size),SEEK_SET));half[0]=0x7e00;CHECK(fwrite(half,2,4,f)==4 && !fclose(f));
    CHECK(!h3_weight_load_f16_f32(weights,v.gpu,"weight",1,shape,error,sizeof(error)));CHECK(strstr(error,"nonfinite"));
    h3_weight_store_free(weights);unlink(path);
    CHECK(prepare_rope(&v,error,sizeof(error)));float cosines[13*24],sines[13*24];
    CHECK(h3_gpu_tensor_read_f32(v.rope_cos,cosines,13*24));CHECK(h3_gpu_tensor_read_f32(v.rope_sin,sines,13*24));
    for(int i=0;i<13;i++)for(int j=0;j<24;j++)if(i>=8 || j<8){CHECK(cosines[i*24+j]==1);CHECK(sines[i*24+j]==0);}
    float input[24*8],mean[24],std[24];for(int c=0;c<24;c++){mean[c]=(float)c;std[c]=2;for(int r=0;r<8;r++)input[c*8+r]=(float)r;}
    CHECK(prepare_input(&v,input,mean,std,error,sizeof(error)));float packed[24*8];CHECK(h3_gpu_tensor_read_f32(v.latent,packed,24*8));
    for(int r=0;r<8;r++)for(int c=0;c<24;c++)CHECK(packed[r*24+c]==(float)(r*2+c));
    size_t n=13*OUTPUT_PATCH;float *projected=malloc(n*4);CHECK(projected);
    for(size_t i=0;i<n;i++)projected[i]=NAN;
    for(int p=0;p<8;p++)for(int c=0;c<3;c++)for(int t=0;t<4;t++)for(int y=0;y<16;y++)for(int x=0;x<16;x++)
        projected[p*OUTPUT_PATCH+((c*4+t)*16+y)*16+x]=(float)(t*100+p*3+c);
    v.projected=h3_gpu_tensor_from_f32(v.gpu,projected,n);CHECK(v.projected);
    for(int t=0;t<4;t++) {
        v.image_slice=t;h3_video_frames out={0};CHECK(unpack_frame_range(&v,0,1,&out,error,sizeof(error)));CHECK(out.frames==1 && out.width==64 && out.height==32);
        for(int y=0;y<32;y++)for(int x=0;x<64;x++)for(int c=0;c<3;c++){
            float expected=(float)(t*100+(y/16*4+x/16)*3+c)*(float[]){.229f,.224f,.225f}[c]+(float[]){.485f,.456f,.406f}[c];
            CHECK(out.rgb[(y*64+x)*3+c]==expected);
        }
        h3_video_frames_free(&out);
    }
    free(projected);cleanup(&v);
}
int main(int argc,char **argv) {if(argc>2)model=argv[2];if(argc>1 && !strcmp(argv[1],"gpu"))gpu();else host();printf("still: %d checks passed\n",checks);return 0;}
