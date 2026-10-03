/* macOS bootstrap: system Foundation/CommonCrypto/zlib only. No shell or tools. */
#import "src/runtime/macos.h"
#include <mach-o/loader.h>
#include <mach-o/dyld.h>
#include <sys/mman.h>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <zlib.h>
#include <stdlib.h>
#include <string.h>

#define LIMIT (UINT64_C(32)*1024*1024*1024)
#define META_LIMIT (8*1024*1024)
static BOOL bounded(uint64_t offset,uint64_t length,uint64_t size){return offset<=size&&length<=size-offset;}
static uint64_t word(const unsigned char *p){uint64_t v;memcpy(&v,p,8);return v;}
static NSString *digest_hex(const unsigned char *p){char h[65];for(int i=0;i<32;i++)snprintf(h+2*i,3,"%02x",p[i]);return @(h);}
static const unsigned char *section(const unsigned char *file,size_t size,uint64_t *length){
    if(size<sizeof(struct mach_header_64))h3_mac_fail(@"truncated Mach-O");
    struct mach_header_64 h;memcpy(&h,file,sizeof(h));
    if(h.magic!=MH_MAGIC_64||h.cputype!=CPU_TYPE_ARM64||h.filetype!=MH_EXECUTE||h.ncmds>4096||!bounded(sizeof(h),h.sizeofcmds,size))h3_mac_fail(@"unsupported Mach-O");
    size_t at=sizeof(h),end=at+h.sizeofcmds;const unsigned char *found=NULL;
    for(uint32_t i=0;i<h.ncmds;i++){
        struct load_command lc;if(!bounded(at,sizeof(lc),end))h3_mac_fail(@"truncated load command");memcpy(&lc,file+at,sizeof(lc));
        if(lc.cmdsize<sizeof(lc)||lc.cmdsize%8||!bounded(at,lc.cmdsize,end))h3_mac_fail(@"invalid load command");
        if(lc.cmd==LC_SEGMENT_64){
            struct segment_command_64 seg;
            if(lc.cmdsize<sizeof(seg))h3_mac_fail(@"truncated segment");memcpy(&seg,file+at,sizeof(seg));
            if(seg.nsects>(lc.cmdsize-sizeof(seg))/sizeof(struct section_64)||!bounded(seg.fileoff,seg.filesize,size))h3_mac_fail(@"invalid segment");
            for(uint32_t j=0;j<seg.nsects;j++){
                struct section_64 s;memcpy(&s,file+at+sizeof(seg)+(size_t)j*sizeof(s),sizeof(s));
                if(strncmp(s.segname,"__H3CLI",16)||strncmp(s.sectname,"__payload",16))continue;
                if(found||strncmp(seg.segname,"__H3CLI",16)||seg.initprot!=VM_PROT_READ||seg.maxprot!=VM_PROT_READ||
                   (s.flags&SECTION_TYPE)!=S_REGULAR||s.offset<seg.fileoff||!bounded(s.offset,s.size,seg.fileoff+seg.filesize)||s.size>LIMIT||s.size<104)
                    h3_mac_fail(@"invalid payload section");
                found=file+s.offset;*length=s.size;
            }
        }
        at+=lc.cmdsize;
    }
    if(at!=end||!found)h3_mac_fail(@"missing payload section");return found;
}
static NSArray *members(NSDictionary *manifest,uint64_t compressed,uint64_t expanded){
    if(![manifest[@"schema"] isEqual:@1]||![manifest[@"platform"] isEqual:@"macos-arm64"]||![manifest[@"protocol"] isEqual:@1])h3_mac_fail(@"unsupported bundle schema/platform");
    NSArray *files=manifest[@"files"];
    if(![files isKindOfClass:[NSArray class]]||files.count<2||files.count>4096)h3_mac_fail(@"invalid member count");
    NSMutableSet *seen=[NSMutableSet set],*parents=[NSMutableSet set];uint64_t offset=0,total=0;
    for(id f in files){
        if(![f isKindOfClass:[NSDictionary class]])h3_mac_fail(@"invalid member");
        NSString *p=f[@"path"],*hash=f[@"sha256"];
        if(!h3_mac_path(p)||[p hasPrefix:@"."]||[seen containsObject:p]||[parents containsObject:p])h3_mac_fail(@"duplicate/unsafe member");
        for(NSString *parent=p.stringByDeletingLastPathComponent;parent.length;parent=parent.stringByDeletingLastPathComponent){
            if([seen containsObject:parent])h3_mac_fail(@"ancestor member conflict");[parents addObject:parent];
        }
        [seen addObject:p];
        if(![hash isKindOfClass:[NSString class]]||hash.length!=64||[hash rangeOfCharacterFromSet:[[NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdef"] invertedSet]].location!=NSNotFound)h3_mac_fail(@"invalid member digest");
        uint64_t mode=h3_mac_number(f[@"mode"],0755),n=h3_mac_number(f[@"size"],LIMIT),z=h3_mac_number(f[@"compressed_size"],LIMIT);
        if((mode!=0644&&mode!=0755)||h3_mac_number(f[@"offset"],compressed)!=offset||!z||!bounded(offset,z,compressed)||!bounded(total,n,expanded))h3_mac_fail(@"invalid member bounds/mode");
        offset+=z;total+=n;
    }
    if(offset!=compressed||total!=expanded||![seen containsObject:@"bin/h3cli"]||![seen containsObject:@"share/h3cli/runtime.json"])h3_mac_fail(@"incomplete member inventory");
    return files;
}
static void extract(int root,NSArray *files,const unsigned char *payload){
    for(NSDictionary *f in files){
        NSString *path=f[@"path"];int parent=h3_mac_parent(root,path,YES);
        int fd=openat(parent,path.lastPathComponent.fileSystemRepresentation,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
        if(fd<0)h3_mac_fail(@"cannot create extracted file");
        uint64_t remaining=[f[@"compressed_size"] unsignedLongLongValue],at=[f[@"offset"] unsignedLongLongValue],count=0,size=[f[@"size"] unsignedLongLongValue];
        z_stream z={0};if(inflateInit(&z)!=Z_OK)h3_mac_fail(@"cannot initialize decompressor");
        unsigned char buffer[65536];int status=Z_OK;
        while(status==Z_OK){
            if(!z.avail_in&&remaining){uInt n=(uInt)(remaining<65536?remaining:65536);z.next_in=(Bytef *)(payload+at);z.avail_in=n;at+=n;remaining-=n;}
            z.next_out=buffer;z.avail_out=sizeof(buffer);status=inflate(&z,Z_NO_FLUSH);size_t n=sizeof(buffer)-z.avail_out;
            if(!bounded(count,n,size))h3_mac_fail(@"expanded member exceeds its limit");count+=n;
            size_t done=0;while(done<n){ssize_t wrote=write(fd,buffer+done,n-done);if(wrote<0&&errno==EINTR)continue;if(wrote<=0)h3_mac_fail(@"extraction write failed (check free space)");done+=(size_t)wrote;}
            if(status==Z_OK&&!n&&!z.avail_in&&!remaining)h3_mac_fail(@"truncated compressed member");
        }
        if(status!=Z_STREAM_END||remaining||z.avail_in||count!=size)h3_mac_fail(@"invalid compressed member");inflateEnd(&z);
        if(fchmod(fd,(mode_t)[f[@"mode"] unsignedIntValue])||fsync(fd))h3_mac_fail(@"cannot sync extracted file");close(fd);
        fd=openat(parent,path.lastPathComponent.fileSystemRepresentation,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
        if(fd<0||![h3_mac_file_sha(fd) isEqual:f[@"sha256"]])h3_mac_fail(@"extracted hash mismatch");close(fd);
        if(fsync(parent))h3_mac_fail(@"cannot sync member directory");close(parent);
    }
}
int main(int argc,char **argv){
    (void)argc;
    @autoreleasepool {
        umask(0077);uint32_t n=0;_NSGetExecutablePath(NULL,&n);char *path=malloc(n);
        if(!path||_NSGetExecutablePath(path,&n))h3_mac_fail(@"cannot discover launcher");char *canonical=realpath(path,NULL);free(path);
        if(!canonical)h3_mac_fail(@"cannot resolve launcher");int fd=open(canonical,O_RDONLY|O_CLOEXEC);free(canonical);struct stat st;
        if(fd<0||fstat(fd,&st)||st.st_size<=0||(uint64_t)st.st_size>LIMIT)h3_mac_fail(@"cannot open launcher");
        size_t size=(size_t)st.st_size;const unsigned char *mapped=mmap(NULL,size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
        if(mapped==MAP_FAILED)h3_mac_fail(@"cannot map launcher");uint64_t length=0;const unsigned char *p=section(mapped,size,&length);
        if(memcmp(p,"H3CLI_MACOS_V1\0\0\0",16))h3_mac_fail(@"invalid payload magic");
        uint64_t m=word(p+16),z=word(p+24),expanded=word(p+32);
        if(!m||m>META_LIMIT||expanded>LIMIT||!bounded(104,m,length)||!bounded(104+m,z,length)||104+m+z!=length)h3_mac_fail(@"invalid payload bounds");
        const unsigned char *payload=p+104+m;
        if(![h3_mac_sha(p+104,(size_t)m) isEqual:digest_hex(p+40)])h3_mac_fail(@"manifest hash mismatch");
        NSData *manifestData=[NSData dataWithBytes:p+104 length:(NSUInteger)m];
        NSDictionary *manifest=h3_mac_json(manifestData);
        if(![h3_mac_encode(manifest) isEqual:manifestData])h3_mac_fail(@"manifest must use canonical JSON without duplicate keys");
        NSArray *files=members(manifest,z,expanded);
        NSString *identity=h3_mac_sha(p+40,64);
        const char *override=getenv("H3CLI_RUNTIME_CACHE");NSString *cache=nil;
        if(override&&*override)cache=@(override);
        else {
            NSURL *base=[[[NSFileManager defaultManager] URLsForDirectory:NSCachesDirectory inDomains:NSUserDomainMask] firstObject];
            if(!base)h3_mac_fail(@"no user cache directory; set H3CLI_RUNTIME_CACHE");cache=[base.path stringByAppendingPathComponent:@"h3cli/runtime"];
        }
        if(!cache.isAbsolutePath)h3_mac_fail(@"H3CLI_RUNTIME_CACHE must be absolute");
        NSDictionary *info=@{@"schema":@1,@"platform":@"macos-arm64",@"runtime_id":identity,@"compressed_bytes":@(z),@"expanded_bytes":@(expanded),@"cache":cache,@"manifest":manifest};
        if(getenv("H3CLI_BUNDLE_INFO")){NSData *data=h3_mac_encode(info);fwrite(data.bytes,1,data.length,stdout);puts("");return 0;}
        int cachefd=h3_mac_directory(cache,YES);
        /* Separate exclusive creation from reuse. APFS can return ENOENT for
         * O_CREAT|O_NOFOLLOW while another process publishes the same name. */
        int lock=-1;
        for(unsigned attempt=0;attempt<64;attempt++){
            lock=openat(cachefd,".extract.lock",O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
            if(lock>=0)break;
            if(errno==EEXIST)lock=openat(cachefd,".extract.lock",O_RDWR|O_NOFOLLOW|O_CLOEXEC);
            if(lock>=0||errno!=ENOENT)break;
        }
        if(lock<0||fstat(lock,&st)||!S_ISREG(st.st_mode)||st.st_uid!=geteuid()||(st.st_mode&0077))h3_mac_fail(@"unsafe/unavailable extraction lock");
        int locked;do{locked=flock(lock,LOCK_EX);}while(locked&&errno==EINTR);
        if(locked)h3_mac_fail([NSString stringWithFormat:@"cannot lock extraction: %s",strerror(errno)]);
        int root=openat(cachefd,identity.UTF8String,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
        if(root<0){
            if(errno!=ENOENT)h3_mac_fail(@"unsafe existing runtime");
            struct statvfs space;if(fstatvfs(cachefd,&space)||!space.f_frsize||expanded>UINT64_MAX-16*1024*1024||space.f_bavail<((expanded+16*1024*1024)/space.f_frsize)+1)h3_mac_fail(@"insufficient cache free space");
            if(![h3_mac_sha(payload,(size_t)z) isEqual:digest_hex(p+72)])h3_mac_fail(@"payload hash mismatch");
            NSString *stage=[@".stage-" stringByAppendingString:NSUUID.UUID.UUIDString];
            if(mkdirat(cachefd,stage.UTF8String,0700))h3_mac_fail(@"cannot stage runtime");
            root=openat(cachefd,stage.UTF8String,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);if(root<0)h3_mac_fail(@"cannot open staging directory");
            extract(root,files,payload);
            NSMutableDictionary *stamps=[NSMutableDictionary dictionary];
            for(NSDictionary *f in files){NSString *name=f[@"path"];int parent=h3_mac_parent(root,name,NO);if(fstatat(parent,name.lastPathComponent.fileSystemRepresentation,&st,AT_SYMLINK_NOFOLLOW))h3_mac_fail(@"cannot record readiness");stamps[name]=h3_mac_stamp(&st);close(parent);}
            h3_mac_write(root,@".ready.json",h3_mac_encode(@{@"manifest":manifest,@"files":stamps}),0600);
            if(fsync(root)||renameat(cachefd,stage.UTF8String,cachefd,identity.UTF8String)||fsync(cachefd))h3_mac_fail(@"cannot publish runtime");
        }
        h3_mac_verify(root,manifest);close(root);flock(lock,LOCK_UN);close(lock);close(cachefd);munmap((void *)mapped,size);
        h3_mac_clean_environment();
        NSString *runtime=[cache stringByAppendingPathComponent:identity],*core=[runtime stringByAppendingPathComponent:@"bin/h3cli"];
        setenv("H3CLI_RUNTIME_ROOT",runtime.fileSystemRepresentation,1);execv(core.fileSystemRepresentation,argv);
        h3_mac_fail([NSString stringWithFormat:@"cannot execute core: %s",strerror(errno)]);
    }
}
