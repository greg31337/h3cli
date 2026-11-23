#ifndef H3_GEOMETRY_H
#define H3_GEOMETRY_H
#include <stdint.h>
#include <stddef.h>

/* Serialized identities: never repurpose an existing profile. */
enum { H3_GEOMETRY_LEGACY=0, H3_GEOMETRY_UPSCALE=1,
       H3_GEOMETRY_CUDA_VIDEO=2, H3_GEOMETRY_CUDA_UPSCALE=3 };
#define H3_LEGACY_PIXELS INT64_C(1032192)
#define H3_CUDA_VIDEO_PIXELS INT64_C(2088960)
#define H3_CUDA_UPSCALE_PIXELS INT64_C(8355840)
/* Flash attention admits one million rows, below the 2D elementwise limit
 * (16 rows/CTA times 65535 Y blocks). Tensor offsets use size_t. */
#define H3_CUDA_SEQUENCE_ROWS UINT64_C(1000000)
static inline int h3_canvas_sequence_valid(uint64_t rows,int cuda) {
    return rows>0&&rows<=(cuda?H3_CUDA_SEQUENCE_ROWS:UINT32_MAX/(14336u*2u));
}
/* The text refiner's residual-add interface takes a U32 element count. */
static inline int h3_canvas_text_rows_valid(uint64_t rows) {
    return rows>0&&rows<=UINT32_MAX/5376u;
}
static inline uint64_t h3_canvas_dit_future(uint64_t rows) {
    return rows<=(UINT64_MAX-(UINT64_C(1)<<30))/358400?
        rows*358400+(UINT64_C(1)<<30):UINT64_MAX;
}
/* Preserve existing GEMM shapes for small requests. Larger MLPs use independent
 * row slabs; each dot product retains its complete K dimension. */
static inline uint32_t h3_canvas_mlp_rows(uint32_t rows,uint32_t hidden) {
    return hidden&&((uint64_t)rows*hidden*2>UINT32_MAX)?
        (rows<131072u?rows:131072u):rows;
}
#ifdef __APPLE__
#define H3_NATIVE_CUDA 0
#define H3_UPSCALE_LATENT_AXIS 128
#else
#define H3_NATIVE_CUDA 1
#define H3_UPSCALE_LATENT_AXIS 240
#endif
static inline int h3_canvas_delivery_valid(int w,int h,int profile) {
    if(w<32||h<32||w%2||h%2||profile<0||profile>3)return 0;
    int64_t pixels=(int64_t)w*h;
    if(!profile)return pixels<=H3_LEGACY_PIXELS;
    int axis=profile==3?3840:1920;
    return w<=axis&&h<=axis&&pixels<=(profile==3?H3_CUDA_UPSCALE_PIXELS:H3_CUDA_VIDEO_PIXELS);
}
static inline int h3_canvas_valid(int w,int h,int profile) {
    return w%32==0&&h%32==0&&h3_canvas_delivery_valid(w,h,profile);
}
static inline int h3_canvas_native_profile(int w,int h,int cuda) {
    if(h3_canvas_valid(w,h,0))return 0;
    return cuda&&h3_canvas_valid(w,h,2)?2:-1;
}
static inline int h3_canvas_backend(int profile,int cuda) {
    return profile>=0&&profile<=3&&(profile<2||cuda);
}
/* Host-only planning. Actual admission also uses current free memory and
 * backend scratch/conditioning allocations. Checked before multiplication. */
typedef struct {
    uint64_t sequence_rows,video_rows,video_bytes,audio_bytes;
    uint64_t dit_future_bytes,resident_core_bytes,stream_core_bytes;
    uint64_t upscale_reserve_bytes,decode_host_bytes;
    uint64_t sampler_host_bytes,decode_device_bytes,encoder_host_bytes;
    uint64_t streamed_host_cache_bytes;
} h3_canvas_memory;
static inline int h3_canvas_memory_estimate(int w,int h,int frames,int profile,
    uint64_t text_rows,uint64_t reference_rows,h3_canvas_memory *out) {
    if(!out||!h3_canvas_valid(w,h,profile)||frames<22||frames>362||(frames-5)%17)return 0;
    uint64_t t=2+5*(uint64_t)((frames-5)/17), hw=(uint64_t)w*h/256;
    uint64_t audio=((uint64_t)frames*40+12)/24,video=t*hw/4;
    if(text_rows>UINT64_MAX-reference_rows)return 0;
    uint64_t seq=text_rows+reference_rows;
    if(seq>UINT64_MAX-video-2*audio)return 0;
    seq+=video+2*audio;
    if(seq>(UINT64_MAX-(UINT64_C(1)<<30))/358400)return 0;
    uint64_t block=2*(4*UINT64_C(7168)*5376+3*UINT64_C(14336)*5376);
    h3_canvas_memory estimate={seq,video,t*hw*24*4,audio*64*4,
        h3_canvas_dit_future(seq),50*block,2*block,
        t*hw*512*2*4+(UINT64_C(2)<<30),0,0,UINT64_C(7)<<30,0,UINT64_C(40)<<30};
    *out=estimate;
    uint64_t nx=w<=256?1:((uint64_t)w-64+191)/192;
    uint64_t ny=h<=256?1:((uint64_t)h-64+191)/192;
    /* Worst case host stitching: retained 22-frame tiles, stitched chunk,
     * five overlap frames, and encoder/conversion buffers. No movie retention. */
    out->encoder_host_bytes=(uint64_t)w*h*3*4*2;
    out->decode_host_bytes=((uint64_t)w*h*27+nx*ny*(w<256?w:256)*(h<256?h:256)*22)*12+
        out->encoder_host_bytes+(UINT64_C(2)<<30);
    out->sampler_host_bytes=6*(out->video_bytes+out->audio_bytes)+
        reference_rows*96*4*2+text_rows*5120*2;
    return 1;
}
#endif
