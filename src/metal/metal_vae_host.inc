/* Original Metal full-video decoder primitives. Preview and image VAE are separate. */
int h3_gpu_video_vae_configure(h3_gpu *g,int reserved) {return g&&!reserved?0:-1;}
int h3_gpu_video_linear(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *w,const h3_gpu_tensor *b,uint32_t rows,uint32_t k,uint32_t n) {
    return h3_gpu_linear_f32(g,out,in,w,b,rows,k,n);
}
int h3_gpu_video_qkv(h3_gpu *g,h3_gpu_tensor *q,h3_gpu_tensor *k,h3_gpu_tensor *v,
    const h3_gpu_tensor *qkv,const h3_gpu_tensor *c,const h3_gpu_tensor *s,
    uint32_t rows,uint32_t heads,uint32_t dim,uint32_t half,float epsilon) {
    return h3_gpu_video_qkv_rope_f32(g,q,k,v,qkv,c,s,rows,heads,dim,half,epsilon);
}
int h3_gpu_video_sdpa(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,uint32_t rows,uint32_t heads,uint32_t dim,float scale) {
    return h3_gpu_sdpa_f32(g,out,q,k,v,rows,heads,dim,scale);
}
int h3_gpu_video_unpack(h3_gpu *g,const h3_gpu_tensor *p,float *rgb,int h,int w,int total,int first,int frames) {
    (void)g;(void)p;(void)rgb;(void)h;(void)w;(void)total;(void)first;(void)frames;return 0;
}
int h3_gpu_video_graph_begin(h3_gpu *g){(void)g;return 0;}
int h3_gpu_video_graph_end(h3_gpu *g){(void)g;return 1;}
int h3_gpu_video_graph_abort(h3_gpu *g){(void)g;return 0;}
