#import "src/runtime/macos.h"
#include "src/digest.h"
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

_Noreturn void h3_mac_fail(NSString *message) {
    fprintf(stderr,"h3cli: packaged runtime: %s\n",message.UTF8String);exit(2);
}
NSString *h3_mac_sha(const void *bytes,size_t size) {
    h3_sha256_ctx ctx;h3_sha256_init(&ctx);
    const unsigned char *p=bytes;
    while(size){size_t n=(size<65536?size:65536);h3_sha256_update(&ctx,p,(h3_sha256_size)n);p+=n;size-=n;}
    unsigned char digest[32];h3_sha256_final(digest,&ctx);char hex[65];
    for(size_t i=0;i<32;i++)snprintf(hex+2*i,3,"%02x",digest[i]);
    return @(hex);
}
NSString *h3_mac_file_sha(int fd) {
    h3_sha256_ctx ctx;h3_sha256_init(&ctx);unsigned char buffer[65536],digest[32];ssize_t n;
    if(lseek(fd,0,SEEK_SET)<0)h3_mac_fail(@"cannot seek runtime file");
    while((n=read(fd,buffer,sizeof(buffer)))>0)h3_sha256_update(&ctx,buffer,(h3_sha256_size)n);
    if(n<0)h3_mac_fail(@"cannot read runtime file");
    h3_sha256_final(digest,&ctx);char hex[65];
    for(size_t i=0;i<32;i++)snprintf(hex+2*i,3,"%02x",digest[i]);
    return @(hex);
}
NSDictionary *h3_mac_json(NSData *data) {
    id value=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:NULL]:nil;
    if(![value isKindOfClass:[NSDictionary class]])h3_mac_fail(@"invalid JSON object");
    return value;
}
NSData *h3_mac_encode(id value) {
    NSData *data=[NSJSONSerialization dataWithJSONObject:value options:(NSJSONWritingSortedKeys|NSJSONWritingWithoutEscapingSlashes) error:NULL];
    if(!data)h3_mac_fail(@"cannot encode runtime metadata");return data;
}
uint64_t h3_mac_number(id value,uint64_t maximum) {
    if(![value isKindOfClass:[NSNumber class]]||CFGetTypeID((__bridge CFTypeRef)value)==CFBooleanGetTypeID()||
       strchr("fd",[value objCType][0])||[value longLongValue]<0||[value unsignedLongLongValue]>maximum)
        h3_mac_fail(@"invalid bounded integer");
    return [value unsignedLongLongValue];
}
BOOL h3_mac_path(NSString *path) {
    if(![path isKindOfClass:[NSString class]]||!path.length||path.length>1024||[path hasPrefix:@"/"]||
       [path containsString:@"\\"]||[path rangeOfCharacterFromSet:[NSCharacterSet controlCharacterSet]].location!=NSNotFound)return NO;
    for(NSString *p in [path componentsSeparatedByString:@"/"])
        if(!p.length||[p isEqual:@"."]||[p isEqual:@".."]||[p lengthOfBytesUsingEncoding:NSUTF8StringEncoding]>255)return NO;
    return YES;
}
static void private_dir(int fd) {
    struct stat st;
    if(fstat(fd,&st)||!S_ISDIR(st.st_mode)||st.st_uid!=geteuid()||(st.st_mode&0077))
        h3_mac_fail(@"cache must be a private user-owned directory (mode 0700)");
}
int h3_mac_directory(NSString *path,BOOL create) {
    if(![path hasPrefix:@"/"])h3_mac_fail(@"cache path must be absolute");
    int fd=open("/",O_RDONLY|O_DIRECTORY|O_CLOEXEC);
    NSArray *parts=[path componentsSeparatedByString:@"/"];
    for(NSString *part in parts){
        if(!part.length)continue;
        if([part isEqual:@"."]||[part isEqual:@".."])h3_mac_fail(@"unsafe cache path");
        if(create&&mkdirat(fd,part.fileSystemRepresentation,0700)&&errno!=EEXIST)h3_mac_fail(@"cannot create cache directory");
        int next=openat(fd,part.fileSystemRepresentation,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
        close(fd);if(next<0)h3_mac_fail(@"unsafe or unavailable cache directory (symlinks are not allowed)");
        struct stat ancestor;
        if(fstat(next,&ancestor)||(ancestor.st_uid!=0&&ancestor.st_uid!=geteuid())||
           ((ancestor.st_mode&0022)&&!(ancestor.st_uid==0&&(ancestor.st_mode&S_ISVTX))))
            h3_mac_fail(@"unsafe cache ancestor ownership/permissions");
        fd=next;
    }
    private_dir(fd);return fd;
}
int h3_mac_parent(int root,NSString *path,BOOL create) {
    if(!h3_mac_path(path))h3_mac_fail(@"unsafe runtime member path");
    int fd=dup(root);NSArray *parts=[path componentsSeparatedByString:@"/"];
    for(NSUInteger i=0;i+1<parts.count;i++){
        const char *part=[parts[i] fileSystemRepresentation];
        if(create&&mkdirat(fd,part,0700)&&errno!=EEXIST)h3_mac_fail(@"cannot create member directory");
        int next=openat(fd,part,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);close(fd);
        if(next<0)h3_mac_fail(@"unsafe member directory");private_dir(next);fd=next;
    }
    return fd;
}
NSArray *h3_mac_stamp(struct stat *st) {
    return @[@(st->st_size),@(st->st_mode&07777),@(st->st_ino),@(st->st_dev),
        @(st->st_mtimespec.tv_sec),@(st->st_mtimespec.tv_nsec),@(st->st_ctimespec.tv_sec),@(st->st_ctimespec.tv_nsec)];
}
NSData *h3_mac_read(int root,NSString *path,size_t maximum) {
    int parent=h3_mac_parent(root,path,NO),fd=openat(parent,path.lastPathComponent.fileSystemRepresentation,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
    close(parent);struct stat st;
    if(fd<0||fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_uid!=geteuid()||(st.st_mode&0022)||st.st_size<0||(uint64_t)st.st_size>maximum)
        h3_mac_fail(@"missing, unsafe or oversized runtime metadata");
    NSMutableData *data=[NSMutableData dataWithLength:(NSUInteger)st.st_size];size_t at=0;
    while(at<data.length){ssize_t n=read(fd,(char *)data.mutableBytes+at,data.length-at);if(n<0&&errno==EINTR)continue;if(n<=0)h3_mac_fail(@"short metadata read");at+=(size_t)n;}
    close(fd);return data;
}
void h3_mac_write(int root,NSString *path,NSData *data,mode_t mode) {
    int parent=h3_mac_parent(root,path,YES);
    int fd=openat(parent,path.lastPathComponent.fileSystemRepresentation,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if(fd<0)h3_mac_fail(@"cannot create runtime file");
    size_t at=0;
    while(at<data.length){ssize_t n=write(fd,(const char *)data.bytes+at,data.length-at);if(n<0&&errno==EINTR)continue;if(n<=0)h3_mac_fail(@"runtime write failed (check free space)");at+=(size_t)n;}
    if(fchmod(fd,mode)||fsync(fd)||fsync(parent))h3_mac_fail(@"cannot sync runtime file");close(fd);close(parent);
}
NSDictionary *h3_mac_verify(int root,NSDictionary *expected) {
    private_dir(root);
    NSDictionary *ready=h3_mac_json(h3_mac_read(root,@".ready.json",8*1024*1024));
    NSDictionary *files=ready[@"files"];
    if(![files isKindOfClass:[NSDictionary class]]||files.count<2||files.count>4096)h3_mac_fail(@"invalid readiness inventory");
    if(expected&&![ready[@"manifest"] isEqual:expected])h3_mac_fail(@"runtime manifest changed");
    for(NSString *name in files){
        int parent=h3_mac_parent(root,name,NO);struct stat st;
        if(fstatat(parent,name.lastPathComponent.fileSystemRepresentation,&st,AT_SYMLINK_NOFOLLOW)||!S_ISREG(st.st_mode)||st.st_uid!=geteuid()||(st.st_mode&0022)||![files[name] isEqual:h3_mac_stamp(&st)])
            h3_mac_fail(@"runtime changed; stop its processes and remove this cache entry before retrying");
        close(parent);
    }
    return ready;
}
void h3_mac_clean_environment(void) {
    /* Core and helpers use only absolute/system loads. Preserve model/network policy. */
    NSArray *names=[[[NSProcessInfo processInfo] environment] allKeys];
    for(NSString *name in names)if([name hasPrefix:@"DYLD_"])unsetenv(name.UTF8String);
}
