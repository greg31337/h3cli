// LBH recipe 1, full-volume BF16 operators. See THIRD_PARTY_NOTICES.md.
struct h3_up_args { uint time,height,width,channels,out_height,out_width,first,rows,kernel_time,kernel_space; };
inline float up_float(ushort b) { return as_type<float>(uint(b)<<16); }
inline ushort up_bf(float v) { uint b=as_type<uint>(v); return ushort((b+0x7fff+((b>>16)&1))>>16); }
kernel void h3_up_point(device ushort *out [[buffer(0)]],device const ushort *in [[buffer(1)]],
    device const ushort *a [[buffer(2)]],device const ushort *b [[buffer(3)]],
    constant h3_up_args &p [[buffer(4)]],constant uint &op [[buffer(5)]],uint index [[thread_position_in_grid]]) {

    uint i=index;
    uint spatial=p.time*p.height*p.width,c=p.channels;
    if(op==1) {
        uint filter=c*p.kernel_time*p.kernel_space*p.kernel_space;
        if(i>=p.rows*filter)return;
        uint pos=p.first+i/filter,q=i%filter;
        int dx=int(q%p.kernel_space)-int(p.kernel_space/2);q/=p.kernel_space;
        int dy=int(q%p.kernel_space)-int(p.kernel_space/2);q/=p.kernel_space;
        int dz=int(q%p.kernel_time)-int(p.kernel_time/2);uint ch=q/p.kernel_time;
        int x=int(pos%p.width)+dx,y=int((pos/p.width)%p.height)+dy,z=int(pos/p.width/p.height)+dz;
        out[i]=(pos<spatial&&x>=0&&x<int(p.width)&&y>=0&&y<int(p.height)&&z>=0&&z<int(p.time))?
            in[((uint(z)*p.height+uint(y))*p.width+uint(x))*c+ch]:ushort(0);return;
    }
    if(op==4) {
        if(i>=p.time*p.out_height*p.out_width*c)return;
        uint ch=i%c,pos=i/c,ox=pos%p.out_width,oy=(pos/p.out_width)%p.out_height,z=pos/p.out_width/p.out_height;
        float xf=max(0.f,(float(ox)+.5f)*float(p.width)/float(p.out_width)-.5f);
        float yf=max(0.f,(float(oy)+.5f)*float(p.height)/float(p.out_height)-.5f);
        uint x0=uint(xf),y0=uint(yf),x1=min(x0+1,p.width-1),y1=min(y0+1,p.height-1);
        float wx=xf-float(x0),wy=yf-float(y0);
        float v00=up_float(in[((z*p.height+y0)*p.width+x0)*c+ch]);
        float v01=up_float(in[((z*p.height+y0)*p.width+x1)*c+ch]);
        float v10=up_float(in[((z*p.height+y1)*p.width+x0)*c+ch]);
        float v11=up_float(in[((z*p.height+y1)*p.width+x1)*c+ch]);
        out[i]=up_bf((1.f-wy)*((1.f-wx)*v00+wx*v01)+wy*((1.f-wx)*v10+wx*v11));return;
    }
    if(i>=spatial*c)return;
    uint ch=i%c;
    if(op==2) {
        uint pos=i/c,z=pos/(p.height*p.width),xy=pos%(p.height*p.width);float value=0.f;
        for(uint k=0;k<p.kernel_time;k++) {
            int iz=int(z)+int(k)-int(p.kernel_time/2);
            if(iz>=0&&iz<int(p.time))value=fma(up_float(in[(uint(iz)*p.height*p.width+xy)*c+ch]),up_float(a[ch*p.kernel_time+k]),value);
        }
        out[i]=up_bf(value+up_float(b[ch]));return;
    }
    if(op==3) {
        float scale=up_float(up_bf(1.f+up_float(a[ch])));
        float value=up_float(up_bf(up_float(in[i])*scale));
        out[i]=up_bf(value+up_float(a[c+ch]));return;
    }
    if(op==5)out[i]=up_bf(up_float(in[i])+up_float(a[ch]));
}
kernel void h3_up_norm(device ushort *out [[buffer(0)]],device const ushort *in [[buffer(1)]],
    device const ushort *a [[buffer(2)]],device const ushort *b [[buffer(3)]],
    constant h3_up_args &p [[buffer(4)]],uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
    threadgroup float shared[256];
    uint width=p.channels/32,n=p.time*p.height*p.width*width;
    float sum=0;
    for(uint j=tid;j<n;j+=256)sum+=up_float(in[(j/width)*p.channels+group*width+j%width]);
    shared[tid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {if(tid<stride)shared[tid]+=shared[tid+stride];threadgroup_barrier(mem_flags::mem_threadgroup);}
    float mean=shared[0]/float(n);threadgroup_barrier(mem_flags::mem_threadgroup);sum=0;
    for(uint j=tid;j<n;j+=256) {float d=up_float(in[(j/width)*p.channels+group*width+j%width])-mean;sum=fma(d,d,sum);}
    shared[tid]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {if(tid<stride)shared[tid]+=shared[tid+stride];threadgroup_barrier(mem_flags::mem_threadgroup);}
    float inv=rsqrt(shared[0]/float(n)+1e-5f);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=tid;j<n;j+=256) {
        uint ch=group*width+j%width,i=(j/width)*p.channels+ch;
        out[i]=up_bf((up_float(in[i])-mean)*inv*up_float(a[ch])+up_float(b[ch]));
    }
}
