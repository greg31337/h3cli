#include "src/device.h"
#include <cuda_runtime_api.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysinfo.h>

int h3_device_configure(const char *index, const char *mode,
                        char *error, size_t error_size) {
    if (index) {
        char *end; errno=0;
        long n=strtol(index,&end,10);
        if (!*index || *end || errno || n<0 || n>INT_MAX) {
            snprintf(error,error_size,"invalid CUDA device index: %s",index); return 0;
        }
        if (setenv("H3_CUDA_DEVICE",index,1)) return 0;
    }
    if (mode) {
        if (strcmp(mode,"auto") && strcmp(mode,"resident") && strcmp(mode,"stream")) {
            snprintf(error,error_size,"CUDA weight mode must be auto, resident or stream"); return 0;
        }
        if (setenv("H3_CUDA_WEIGHT_MODE",mode,1)) return 0;
    }
    return 1;
}
int h3_device_query(h3_device_info *info,char *error,size_t error_size) {
    if (!info) return 0;
    memset(info,0,sizeof(*info));
    const char *index=getenv("H3_CUDA_DEVICE");
    const char *mode=getenv("H3_CUDA_WEIGHT_MODE");
    char selected_index[32];
    if (index) {
        if (strlen(index)>=sizeof(selected_index)) {snprintf(error,error_size,"invalid CUDA device index");return 0;}
        strcpy(selected_index,index);index=selected_index;
    }
    if (!h3_device_configure(index,mode,error,error_size)) return 0;
    int device=index ? (int)strtol(index,NULL,10):0;
    cudaError_t status=cudaSetDevice(device);
    if (status!=cudaSuccess) {
        snprintf(error,error_size,"CUDA device %d: %s",device,cudaGetErrorString(status)); return 0;
    }
    struct cudaDeviceProp prop;
    status=cudaGetDeviceProperties(&prop,device);
    if (status!=cudaSuccess) {
        snprintf(error,error_size,"CUDA device properties: %s",cudaGetErrorString(status));return 0;
    }
    size_t available=0,total=0;
    status=cudaMemGetInfo(&available,&total);
    if (status!=cudaSuccess) {
        snprintf(error,error_size,"CUDA memory query: %s",cudaGetErrorString(status));return 0;
    }
    if (prop.major<8 || (prop.major==8 && prop.minor<6)) {
        snprintf(error,error_size,"CUDA requires SM86 or newer; got SM%d%d",prop.major,prop.minor);return 0;
    }
    snprintf(info->name,sizeof(info->name),"%.*s",(int)sizeof(info->name)-1,prop.name);
    snprintf(info->backend,sizeof(info->backend),"cuda");
    snprintf(info->architecture,sizeof(info->architecture),"SM%d%d",prop.major,prop.minor);
    info->device_index=device;info->cuda_compute_major=prop.major;info->cuda_compute_minor=prop.minor;
    info->device_memory=total;info->free_device_memory=available;
    info->recommended_working_set=total;info->max_buffer_length=total;
    info->multiprocessor_count=prop.multiProcessorCount;
    memcpy(info->cuda_uuid,prop.uuid.bytes,sizeof(info->cuda_uuid));
    cudaDriverGetVersion(&info->cuda_driver_version);cudaRuntimeGetVersion(&info->cuda_runtime_version);
    struct sysinfo host;
    if (!sysinfo(&host)) info->physical_memory=(uint64_t)host.totalram*host.mem_unit;
    return 1;
}
int h3_device_memory_fits(uint64_t requested) {
    size_t available=0,total=0;
    if (cudaMemGetInfo(&available,&total)!=cudaSuccess) return 0;
    uint64_t reserve=total/10;
    return available>reserve && requested<=available-reserve;
}
