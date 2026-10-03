#include "src/platform.h"
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
char *h3_executable_path(void) {
#ifdef __APPLE__
    uint32_t size = 0;
    (void)_NSGetExecutablePath(NULL, &size);
    for (;;) {
        if (!size) return NULL;
        char *path = malloc(size);
        if (!path) return NULL;
        if (_NSGetExecutablePath(path, &size) == 0) {
            char *canonical = realpath(path, NULL);
            free(path);
            return canonical;
        }
        free(path);
    }
#else
    size_t size=256;
    for (;;) {
        char *path=malloc(size+1);
        if(!path)return NULL;
        ssize_t count=readlink("/proc/self/exe",path,size);
        if(count<0){free(path);return NULL;}
        if((size_t)count<size){path[count]=0;return path;}
        free(path);
        if(size>SIZE_MAX/2-1)return NULL;
        size*=2;
    }
#endif
}
