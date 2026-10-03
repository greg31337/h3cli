#include "src/runtime/runtime.h"
#if defined(__APPLE__) && defined(H3_PACKAGE_RUNTIME)
#import "src/runtime/macos.h"
#include "src/platform.h"
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

static const char *names[]={"H3_FFMPEG","H3_FFPROBE","H3_SGLANG_INPUT_FFMPEG"};
static char identities[3][72],entries[3][128],runtime_id[72],shader_path[4096];
static void package_start(void) __attribute__((constructor));
static void package_start(void) {
    @autoreleasepool {
        char *exe=h3_executable_path();if(!exe)h3_mac_fail(@"cannot locate core");
        NSString *root=[[@(exe) stringByDeletingLastPathComponent] stringByDeletingLastPathComponent];free(exe);
        int fd=h3_mac_directory(root,NO);
        NSDictionary *ready=h3_mac_verify(fd,nil);
        NSData *data=h3_mac_read(fd,@"share/h3cli/runtime.json",8*1024*1024);
        NSDictionary *manifest=h3_mac_json(data),*files=manifest[@"files"];
        if(![manifest[@"schema"] isEqual:@1]||![manifest[@"platform"] isEqual:@"macos-arm64"]||![files isKindOfClass:[NSDictionary class]]||
           [ready[@"files"] count]!=files.count+1)h3_mac_fail(@"invalid runtime inventory");
        for(NSString *name in files)if(!ready[@"files"][name])h3_mac_fail(@"unverified runtime member");
        NSString *hash=h3_mac_sha(data.bytes,data.length);
        snprintf(runtime_id,sizeof(runtime_id),"sha256:%s",hash.UTF8String);
        setenv("H3_PACKAGED_RUNTIME",runtime_id,1);
        const char *paths[]={"tools/ffmpeg","tools/ffprobe","tools/ffmpeg"};
        for(size_t i=0;i<3;i++){
            NSString *digest=files[@(paths[i])][@"sha256"];
            if(![digest isKindOfClass:[NSString class]]||digest.length!=64)h3_mac_fail(@"missing media identity");
            NSString *path=[root stringByAppendingPathComponent:@(paths[i])];setenv(names[i],path.fileSystemRepresentation,1);
            snprintf(identities[i],sizeof(identities[i]),"sha256:%s",digest.UTF8String);
            snprintf(entries[i],sizeof(entries[i]),"%s=%s",names[i],identities[i]);
        }
        NSString *shader=[root stringByAppendingPathComponent:@"src/metal/shaders.metal"];
        if(!files[@"src/metal/shaders.metal"]||strlen(shader.fileSystemRepresentation)>=sizeof(shader_path))h3_mac_fail(@"missing shader");
        strcpy(shader_path,shader.fileSystemRepresentation);
        unsetenv("H3_SHADER_PATH");setenv("H3CLI_RUNTIME_ROOT",root.fileSystemRepresentation,1);
        h3_mac_clean_environment();close(fd);
    }
}
const char *h3_runtime_shader_path(void){return shader_path;}
const char *h3_runtime_dependency_id(const char *name){
    if(!strcmp(name,"H3_PACKAGED_RUNTIME"))return runtime_id;
    for(size_t i=0;i<3;i++)if(!strcmp(name,names[i]))return identities[i];return NULL;
}
const char *h3_runtime_environment_entry(const char *entry){
    for(size_t i=0;i<3;i++){size_t n=strlen(names[i]);if(!strncmp(entry,names[i],n)&&entry[n]=='=')return entries[i];}return entry;
}
#endif
