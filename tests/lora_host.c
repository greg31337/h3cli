#include "../src/weights/lora.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <fenv.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
static int calls;
static int progress(const char *phase,int done,int total,void *opaque) {
    (void)done;(void)total;(void)opaque;calls++;
    const char *pause=getenv("H3_TEST_LORA_PAUSE_PHASE");
    if(pause&&strstr(phase,pause)) { unsetenv("H3_TEST_LORA_PAUSE_PHASE");fprintf(stderr,"PAUSED %s\n",phase);fflush(stderr);raise(SIGSTOP); }
    const char *cancel=getenv("H3_TEST_LORA_CANCEL_PHASE");
    return cancel&&strstr(phase,cancel)!=NULL;
}
int main(int argc,char **argv) {
    if(argc<5){fprintf(stderr,"usage: lora_host transformer cache memory_mib [--ref2va] adapter...\n");return 2;}
    int first=4,mode=0;if(!strcmp(argv[first],"--ref2va")){mode=1;first++;}
    h3_lora_options options={.adapters=(const char *const *)(argv+first),.count=(size_t)(argc-first),.cache_dir=argv[2],.memory_mib=(size_t)strtoull(argv[3],NULL,10)};
    h3_lora_selection *s=NULL;h3_lora_variant v={.lease=-1};char error[512];
    const char *round=getenv("H3_TEST_LORA_ROUND");
    if(round)fesetround(FE_DOWNWARD);
#if defined(__aarch64__)
    uint64_t before_fpcr;
    __asm__ volatile("mrs %0, fpcr":"=r"(before_fpcr));
    if(getenv("H3_TEST_LORA_FLUSH")){before_fpcr|=UINT64_C(1)<<24;__asm__ volatile("msr fpcr, %0"::"r"(before_fpcr));}
#elif defined(__SSE__)
    unsigned before_mxcsr=_mm_getcsr();
    if(getenv("H3_TEST_LORA_FLUSH")){before_mxcsr|=0x8040;_mm_setcsr(before_mxcsr);}
#endif
    int before=fegetround();
    if(!h3_lora_selection_create(&options,&s,error,sizeof(error))||!h3_lora_prepare(s,argv[1],mode,progress,NULL,&v,error,sizeof(error))){
        fprintf(stderr,"ERROR: %s\n",error);h3_lora_selection_free(s);return 1;
    }
    if(fegetround()!=before){fprintf(stderr,"ERROR: caller rounding mode changed\n");return 1;}
#if defined(__aarch64__)
    uint64_t after_fpcr;__asm__ volatile("mrs %0, fpcr":"=r"(after_fpcr));
    if(before_fpcr!=after_fpcr){fprintf(stderr,"ERROR: caller FPCR changed\n");return 1;}
#elif defined(__SSE__)
    if((before_mxcsr&0xffc0)!=(_mm_getcsr()&0xffc0)){fprintf(stderr,"ERROR: caller MXCSR changed\n");return 1;}
#endif
    printf("%s\n",v.transformer);fflush(stdout);
    const char *hold=getenv("H3_TEST_LORA_LEASE_SECONDS");if(hold)sleep((unsigned)atoi(hold));
    h3_lora_variant_free(&v);h3_lora_selection_free(s);return 0;
}
