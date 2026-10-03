#include "src/memory.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

static _Thread_local h3_memory_query_fn test_query;
static _Thread_local void *test_opaque;
static _Thread_local h3_memory_snapshot_fn test_snapshot;
static _Thread_local void *test_snapshot_opaque;

void h3_memory_set_test_snapshot(h3_memory_snapshot_fn query, void *opaque) {
    test_snapshot=query;test_snapshot_opaque=opaque;
}

void h3_memory_sample(h3_memory_snapshot *s) {
    if (!s)
        return;
    memset(s, 0, sizeof(*s));
    if(test_snapshot){test_snapshot(s,test_snapshot_opaque);return;}
    s->available_valid=h3_memory_available_bytes(&s->available);
#ifdef __APPLE__
    task_vm_info_data_t task={0};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    if(task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&task,&count)==KERN_SUCCESS){
        s->resident=task.resident_size;s->physical_footprint=task.phys_footprint;
        s->process_compressed=task.compressed;s->process_valid=1;
    }
    mach_port_t host=mach_host_self();vm_statistics64_data_t vm={0};vm_size_t page=0;count=HOST_VM_INFO64_REV1_COUNT;
    if(host_page_size(host,&page)==KERN_SUCCESS&&host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&vm,&count)==KERN_SUCCESS){
        s->system_compressed=(uint64_t)vm.compressor_page_count*page;s->system_valid=1;
        s->system_wired=(uint64_t)vm.wire_count*page;
    }
    mach_port_deallocate(mach_task_self(),host);
    size_t bytes=sizeof(s->physical_total);sysctlbyname("hw.memsize",&s->physical_total,&bytes,NULL,0);
    struct xsw_usage swap={0};bytes=sizeof(swap);
    if(!sysctlbyname("vm.swapusage",&swap,&bytes,NULL,0)){s->swap_used=swap.xsu_used;s->swap_valid=1;}
#else
    FILE *file=fopen("/proc/meminfo","r");char line[256];unsigned long long kib;
    if(file) {
        while(fgets(line,sizeof(line),file)) {
            if(sscanf(line,"MemTotal: %llu kB",&kib)==1&&kib<=UINT64_MAX/1024)
                s->physical_total=(uint64_t)kib*1024;
        }
        fclose(file);s->system_valid=s->physical_total!=0;
    }
    file=fopen("/proc/self/status","r");uint64_t swapped=0;
    if(file){while(fgets(line,sizeof(line),file)) {
        if(sscanf(line,"VmRSS: %llu kB",&kib)==1&&kib<=UINT64_MAX/1024){s->resident=(uint64_t)kib*1024;s->process_valid=1;}
        if(sscanf(line,"VmSwap: %llu kB",&kib)==1&&kib<=UINT64_MAX/1024)swapped=(uint64_t)kib*1024;
    }fclose(file);
        s->physical_footprint=swapped>UINT64_MAX-s->resident?UINT64_MAX:s->resident+swapped;
    }
#endif
}

void h3_memory_set_test_query(h3_memory_query_fn query, void *opaque) {
    test_query = query;
    test_opaque = opaque;
}

int h3_memory_available_bytes(uint64_t *bytes) {
    if (!bytes) return 0;
    *bytes = 0;
    if (test_query) return test_query(bytes, test_opaque);
#ifdef __APPLE__
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    vm_statistics64_data_t stats = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_REV1_COUNT;
    kern_return_t page_result = host_page_size(host, &page_size);
    kern_return_t result = host_statistics64(host, HOST_VM_INFO64,
        (host_info64_t)&stats, &count);
    mach_port_deallocate(mach_task_self(), host);
    if (page_result != KERN_SUCCESS || result != KERN_SUCCESS || !page_size)
        return 0;
    /* Speculative pages are already included in free_count. */
    uint64_t pages = (uint64_t)stats.free_count + stats.inactive_count;
    if (pages > UINT64_MAX / page_size) return 0;
    *bytes = pages * page_size;
    return 1;
#else
    FILE *file = fopen("/proc/meminfo", "r");
    if (!file) return 0;
    char line[256];
    unsigned long long kib = 0;
    int found = 0;
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "MemAvailable: %llu kB", &kib) == 1) {
            found = 1; break;
        }
    }
    fclose(file);
    if (!found || kib > UINT64_MAX / 1024) return 0;
    *bytes = (uint64_t)kib * 1024;
    return 1;
#endif
}

uint64_t h3_memory_minimum_bytes(void) {
    const char *value = getenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES");
    if (value && *value >= '0' && *value <= '9') {
        char *end;
        errno = 0;
        unsigned long long parsed = strtoull(value, &end, 10);
        if (!errno && !*end && parsed <= UINT64_MAX) return (uint64_t)parsed;
    }
    return H3_MIN_AVAILABLE_MEMORY;
}

uint64_t h3_memory_limit_bytes(const h3_memory_snapshot *s) {
    uint64_t limit=H3_MAX_PROCESS_MEMORY;
    if(s&&s->physical_total) {
        uint64_t headroom=s->physical_total/8;
        if(headroom<H3_MIN_AVAILABLE_MEMORY)headroom=H3_MIN_AVAILABLE_MEMORY;
        uint64_t capacity=s->physical_total>headroom?s->physical_total-headroom:0;
        if(capacity<limit)limit=capacity;
    }
    /* A deployment/test may lower the cap; it cannot disable the safety cap. */
    const char *value=getenv("H3_MEMORY_LIMIT_BYTES");
    if(value&&*value>='0'&&*value<='9') {
        char *end;errno=0;unsigned long long parsed=strtoull(value,&end,10);
        if(!errno&&!*end&&parsed>0&&parsed<limit)limit=(uint64_t)parsed;
    }
    return limit;
}

int h3_memory_error(const char *error) {
    return error&&strstr(error,"memory safety:")!=NULL;
}

int h3_memory_check_gpu(uint64_t reserve_bytes,uint64_t allocated_bytes,
                        const char *phase,char *error,size_t error_size) {
    h3_memory_snapshot s;h3_memory_sample(&s);
    const char *where=phase?phase:"generation";
    if(!s.available_valid||!s.process_valid||!s.physical_total) {
        if(error&&error_size)snprintf(error,error_size,
            "memory safety: generation aborted during %s: cannot query physical headroom/process footprint",where);
        return 0;
    }
    uint64_t limit=h3_memory_limit_bytes(&s);
    uint64_t used=s.physical_footprint>allocated_bytes?s.physical_footprint:allocated_bytes;
    if(used>=limit||reserve_bytes>limit-used) {
        if(error&&error_size)snprintf(error,error_size,
            "memory safety: generation aborted during %s: process footprint=%.3f GB, Metal allocation=%.3f GB; "
            "memory limit=%.3f GB (%" PRIu64 " bytes), requested allocation reserve=%" PRIu64 " bytes",
            where,(double)s.physical_footprint/1e9,(double)allocated_bytes/1e9,(double)limit/1e9,limit,reserve_bytes);
        return 0;
    }
    uint64_t available=s.available;
    uint64_t floor = h3_memory_minimum_bytes();
    if (available >= floor && reserve_bytes <= available - floor) return 1;
    if (error && error_size)
        snprintf(error, error_size,
            "memory safety: generation aborted during %s: %.3f GiB reclaimable physical memory "
            "(available=%" PRIu64 " bytes); safety floor %.3f GiB "
            "(floor=%" PRIu64 " bytes), requested allocation reserve=%" PRIu64 " bytes",
            phase ? phase : "generation", (double)available / 1073741824.0,
            available, (double)floor / 1073741824.0, floor, reserve_bytes);
    return 0;
}

int h3_memory_check(uint64_t reserve_bytes,const char *phase,
                    char *error,size_t error_size) {
    return h3_memory_check_gpu(reserve_bytes,0,phase,error,error_size);
}

int h3_memory_checkpoint(int cancelled, const char *phase,
                         char *error, size_t error_size) {
    if (!h3_memory_check(0, phase, error, error_size)) return 0;
    if (!cancelled) return 1;
    if (error && error_size)
        snprintf(error, error_size, "generation cancelled during %s", phase);
    return 0;
}
