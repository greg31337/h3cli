/* Exercise the exact bootstrap Mach-O parser without asking dyld to map corrupt code. */
#define main h3_test_embedded_main
#include "src/runtime/launcher_macos.m"
#undef main
int main(int argc,char **argv){
    @autoreleasepool {
        if(argc!=2)return 2;
        NSData *data=[NSData dataWithContentsOfFile:@(argv[1])];
        if(!data) return 2;
        uint64_t length=0;section(data.bytes,data.length,&length);
        printf("%llu\n",(unsigned long long)length);return 0;
    }
}
