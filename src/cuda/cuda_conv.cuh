/* Direct dependency-free convolution reference paths. Output is channels last,
 * checkpoints are OIK/IOK and OIDHW exactly as on Metal. */
__global__ void conv1d_kernel(float*out,const float*in,const float*w,const float*b,
    uint batch,uint length,uint ci,uint co,uint kernel,uint stride,uint padding,uint dilation,uint output_length,bool transpose) {
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x,total=(size_t)batch*output_length*co;if(idx>=total)return;
    uint oc=idx%co,time=(idx/co)%output_length,ba=idx/co/output_length;float sum=b?b[oc]:0;
    for(uint tap=0;tap<kernel;tap++){
        int pos=transpose?(int)time+(int)padding-(int)tap:(int)(time*stride+tap*dilation)-(int)padding;
        if(transpose){if(pos<0||pos%(int)stride)continue;pos/=(int)stride;}
        if(pos<0||pos>=(int)length)continue;
        for(uint ic=0;ic<ci;ic++){
            size_t wi=transpose?((size_t)ic*co+oc)*kernel+tap:((size_t)oc*ci+ic)*kernel+tap;
            sum=fmaf(in[((size_t)ba*length+(uint)pos)*ci+ic],w[wi],sum);
        }
    }
    out[idx]=sum;
}
__global__ void conv3d_kernel(float*out,const float*in,const float*w,const float*b,
    uint batch,uint depth,uint height,uint width,uint ci,uint co,uint kd,uint kh,uint kw,uint sd,uint sh,uint sw,uint od,uint oh,uint ow){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x,total=(size_t)batch*od*oh*ow*co;if(idx>=total)return;
    size_t t=idx;uint oc=t%co;t/=co;uint x=t%ow;t/=ow;uint y=t%oh;t/=oh;uint z=t%od;t/=od;float sum=b?b[oc]:0;
    for(uint iz=0;iz<kd;iz++)for(uint iy=0;iy<kh;iy++)for(uint ix=0;ix<kw;ix++)for(uint ic=0;ic<ci;ic++){
        size_t input=((((t*depth+z*sd+iz)*height+y*sh+iy)*width+x*sw+ix)*ci+ic);
        size_t weight=((((size_t)oc*ci+ic)*kd+iz)*kh+iy)*kw+ix;
        sum=fmaf(in[input],w[weight],sum);
    }
    out[idx]=sum;
}

/* Bounded im2col for SGEMM. Columns match the checkpoint's I,D,H,W order;
 * rows match the shared NDHWC output layout. Only one position tile exists
 * at a time, so scratch does not grow with the full image/video volume. */
__global__ void conv3d_columns(float *columns,const float *input,size_t first,
    uint rows,uint depth,uint height,uint width,uint ci,uint kd,uint kh,uint kw,
    uint sd,uint sh,uint sw,uint od,uint oh,uint ow,uint filter) {
    size_t index=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=(size_t)rows*filter)return;
    size_t position=first+index/filter;
    uint column=index%filter,ix=column%kw;column/=kw;
    uint iy=column%kh;column/=kh;uint iz=column%kd,ic=column/kd;
    uint x=position%ow;position/=ow;uint y=position%oh;position/=oh;
    uint z=position%od;size_t batch=position/od;
    columns[index]=input[((((batch*depth+z*sd+iz)*height+y*sh+iy)*width+x*sw+ix)*ci+ic)];
}
