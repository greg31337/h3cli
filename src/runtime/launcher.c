/* Linux distribution entry point. Static libc/json-c/zlib/libcrypto only.
 * The archive has regular files, each independently compressed; no tar parser,
 * link extraction, shell, mount, network operation or background process.
 */
#define _GNU_SOURCE
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <json-c/json.h>
#include <openssl/sha.h>
#include <zlib.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FOOTER 128
#define MAX_MANIFEST (8U*1024U*1024U)
#define MAX_FILES 4096
#define MAX_UNPACKED (32ULL*1024*1024*1024)
static const unsigned char magic[16]="H3CLI_BUNDLE_V1";
static void die(const char *format,...) {
    va_list ap;va_start(ap,format);fprintf(stderr,"h3cli bundle: ");
    vfprintf(stderr,format,ap);va_end(ap);fputc('\n',stderr);exit(125);
}
static void system_error(const char *what) {die("%s: %s",what,strerror(errno));}
static uint64_t u64(const unsigned char *p) {
    uint64_t v=0;for(int i=7;i>=0;i--)v=(v<<8)|p[i];return v;
}
static void read_at(int fd,void *memory,size_t size,uint64_t offset) {
    unsigned char *p=memory;
    while(size){ssize_t n=pread(fd,p,size,(off_t)offset);if(n<0&&errno==EINTR)continue;
        if (n <= 0)
            die("truncated or unreadable payload");
        p += n;
        size -= (size_t)n;
        offset += (uint64_t)n;
    }
}
static void write_all(int fd,const void *memory,size_t size) {
    const unsigned char *p=memory;
    while(size){ssize_t n=write(fd,p,size);if(n<0&&errno==EINTR)continue;
        if (n <= 0)
            system_error("write extracted file");
        p += n;
        size -= (size_t)n;
    }
}
static void hex(const unsigned char *bytes,char out[65]) {
    for(size_t i=0;i<32;i++)snprintf(out+2*i,3,"%02x",bytes[i]);
}
static void file_digest(int fd,uint64_t offset,uint64_t size,unsigned char out[32]) {
    SHA256_CTX hash;SHA256_Init(&hash);unsigned char buffer[1<<20];
    while(size){size_t n=size>sizeof(buffer)?sizeof(buffer):(size_t)size;
        read_at(fd,buffer,n,offset);SHA256_Update(&hash,buffer,n);offset+=n;size-=n;}
    SHA256_Final(out,&hash);
}
static json_object *field(json_object *o,const char *name,enum json_type type) {
    json_object *v=NULL;if(!o||!json_object_object_get_ex(o,name,&v)||!json_object_is_type(v,type))die("invalid manifest field: %s",name);return v;
}
static uint64_t number(json_object *o,const char *name) {
    int64_t v=json_object_get_int64(field(o,name,json_type_int));if(v<0)die("negative %s",name);return (uint64_t)v;
}
static int safe_path(const char *s) {
    if(!s||!*s||strlen(s)>=PATH_MAX||*s=='/'||strchr(s,'\\'))return 0;
    for(const char *p=s;*p;){const char *end=strchr(p,'/');size_t n=end?(size_t)(end-p):strlen(p);
        if(!n||(n==1&&*p=='.')||(n==2&&!strncmp(p,"..",2)))return 0;
        for(size_t i=0;i<n;i++)if((unsigned char)p[i]<32)return 0;
        if (!end)
            return 1;
        p = end + 1;
        if (!*p)
            return 0;
    }
    return 0;
}
static void check_files(json_object *files,uint64_t packed,uint64_t unpacked) {
    size_t count=json_object_array_length(files);uint64_t offset=0,total=0;
    if(!count||count>MAX_FILES||unpacked>MAX_UNPACKED)die("invalid archive size");
    for(size_t i=0;i<count;i++){
        json_object *f=json_object_array_get_idx(files,i);
        json_object *path_value=field(f,"path",json_type_string);
        const char *path=json_object_get_string(path_value);
        if(strlen(path)!=(size_t)json_object_get_string_len(path_value))die("embedded NUL in archive path");
        const char *hash=json_object_get_string(field(f,"sha256",json_type_string));
        uint64_t size=number(f,"size"),compressed=number(f,"packed"),mode=number(f,"mode");
        if(!safe_path(path)||strlen(hash)!=64||strspn(hash,"0123456789abcdef")!=64||
            (mode!=0644&&mode!=0755)||size>unpacked-total||!compressed||compressed>packed-offset||number(f,"offset")!=offset)die("invalid archive file");
        for(size_t j=0;j<i;j++){
            const char *other=json_object_get_string(field(json_object_array_get_idx(files,j),"path",json_type_string));
            size_t a=strlen(path),b=strlen(other);
            if(!strcmp(path,other)||(a>b&&!strncmp(path,other,b)&&path[b]=='/')||(b>a&&!strncmp(path,other,a)&&other[a]=='/'))die("duplicate or conflicting archive path");
        }
        offset+=compressed;total+=size;
    }
    if(offset!=packed||total!=unpacked)die("archive size totals differ");
}
static void make_parents(const char *path) {
    char *copy=strdup(path);if(!copy)system_error("allocate path");
    for(char *p=copy+1;*p;p++)if(*p=='/'){
        *p=0;if(mkdir(copy,0700)&&errno!=EEXIST)system_error("create cache parent");*p='/';}
    free(copy);
}
static int directory(const char *path) {
    struct stat st;
    if(lstat(path,&st)||!S_ISDIR(st.st_mode)||st.st_uid!=geteuid()||(st.st_mode&0022))die("cache directory must be owned by you and not writable by others: %s",path);
    int fd=open(path,O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)system_error("open cache directory");return fd;
}
static int leaf_parent(int root,const char *name,int create,char leaf[NAME_MAX+1]) {
    char path[PATH_MAX];snprintf(path,sizeof(path),"%s",name);int dir=dup(root);if(dir<0)system_error("duplicate directory");
    char *part=path,*slash;
    while((slash=strchr(part,'/'))){*slash=0;if(create&&mkdirat(dir,part,0700)&&errno!=EEXIST)system_error("create payload directory");
        int next=openat(dir,part,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);close(dir);if(next<0)system_error("unsafe payload directory");dir=next;part=slash+1;}
    if (strlen(part) > NAME_MAX)
        die("archive filename too long");
    strcpy(leaf, part);
    return dir;
}
static json_object *metadata(int root,const char *name) {
    char leaf[NAME_MAX+1];int parent=leaf_parent(root,name,0,leaf);struct stat st;
    if(fstatat(parent,leaf,&st,AT_SYMLINK_NOFOLLOW)||!S_ISREG(st.st_mode)||st.st_uid!=geteuid()){close(parent);return NULL;}close(parent);
    json_object *v=json_object_new_array();
    int64_t values[]={st.st_size,st.st_mode&07777,(int64_t)st.st_ino,(int64_t)st.st_dev,st.st_mtim.tv_sec,st.st_mtim.tv_nsec,st.st_ctim.tv_sec,st.st_ctim.tv_nsec};
    for (size_t i = 0; i < sizeof(values) / sizeof(*values); i++)
        json_object_array_add(v, json_object_new_int64(values[i]));
    return v;
}
static int ready(int root,json_object *files,const char *identity) {
    int fd=openat(root,".ready.json",O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0)return 0;
    struct stat st;if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_uid!=geteuid()||st.st_size<1||st.st_size>MAX_MANIFEST){close(fd);return 0;}
    char *s=calloc((size_t)st.st_size+1,1);if(!s)system_error("allocate ready marker");read_at(fd,s,(size_t)st.st_size,0);close(fd);
    json_object *r=json_tokener_parse(s);free(s);json_object *id=NULL,*recorded=NULL;
    if(!r||!json_object_object_get_ex(r,"id",&id)||!json_object_is_type(id,json_type_string)||strcmp(json_object_get_string(id),identity)||!json_object_object_get_ex(r,"files",&recorded)||!json_object_is_type(recorded,json_type_object)||(size_t)json_object_object_length(recorded)!=json_object_array_length(files)){if(r)json_object_put(r);return 0;}
    int ok=1;
    for(size_t i=0;ok&&i<json_object_array_length(files);i++){
        const char *name=json_object_get_string(field(json_object_array_get_idx(files,i),"path",json_type_string));
        json_object *m=metadata(root,name),*saved=NULL;ok=m&&json_object_object_get_ex(recorded,name,&saved)&&json_object_equal(m,saved);if(m)json_object_put(m);
    }
    json_object_put(r);return ok;
}
static void extract_file(int archive,uint64_t base,int root,json_object *record) {
    const char *name=json_object_get_string(field(record,"path",json_type_string));char leaf[NAME_MAX+1];
    int parent=leaf_parent(root,name,1,leaf),out=openat(parent,leaf,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);close(parent);if(out<0)system_error("create payload file");
    uint64_t pos=base+number(record,"offset"),remaining=number(record,"packed"),written=0,size=number(record,"size");
    unsigned char in[1<<16],buffer[1<<16];z_stream z={0};SHA256_CTX hash;SHA256_Init(&hash);
    if (inflateInit(&z) != Z_OK)
        die("cannot initialize decoder");
    int status = Z_OK;
    while(status!=Z_STREAM_END){
        if(!z.avail_in&&remaining){size_t n=remaining>sizeof(in)?sizeof(in):(size_t)remaining;read_at(archive,in,n,pos);pos+=n;remaining-=n;z.next_in=in;z.avail_in=(uInt)n;}
        z.next_out=buffer;z.avail_out=sizeof(buffer);status=inflate(&z,Z_NO_FLUSH);size_t n=sizeof(buffer)-z.avail_out;
        if (n > size - written)
            die("expanded file exceeds manifest size");
        write_all(out, buffer, n);
        SHA256_Update(&hash, buffer, n);
        written += n;
        if(status!=Z_OK&&status!=Z_STREAM_END)die("corrupt compressed file: %s",name);
        if(status==Z_OK&&!remaining&&!z.avail_in&&!n)die("truncated compressed file: %s",name);
    }
    if (remaining || z.avail_in || written != size)
        die("compressed file length mismatch");
    inflateEnd(&z);
    unsigned char digest[32];char actual[65];SHA256_Final(digest,&hash);hex(digest,actual);
    if(strcmp(actual,json_object_get_string(field(record,"sha256",json_type_string))))die("file checksum mismatch: %s",name);
    if(fchmod(out,(mode_t)number(record,"mode"))||fsync(out)||close(out))system_error("publish extracted file");
}
static void write_ready(int root,json_object *files,const char *identity) {
    json_object *r=json_object_new_object(),*stats=json_object_new_object();json_object_object_add(r,"id",json_object_new_string(identity));
    for(size_t i=0;i<json_object_array_length(files);i++){
        const char *name=json_object_get_string(field(json_object_array_get_idx(files,i),"path",json_type_string));
        json_object *m=metadata(root,name);if(!m)die("extracted file disappeared");json_object_object_add(stats,name,m);}
    json_object_object_add(r,"files",stats);const char *s=json_object_to_json_string_ext(r,JSON_C_TO_STRING_PLAIN);
    int fd=openat(root,".ready.json",O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);if(fd<0)system_error("create ready marker");write_all(fd,s,strlen(s));if(fsync(fd)||close(fd)||fsync(root))system_error("sync ready marker");json_object_put(r);
}
static void launch(const char *root,int argc,char **argv) {
    (void)argc;char core[PATH_MAX],libs[PATH_MAX];
    if(snprintf(core,sizeof(core),"%s/bin/h3cli",root)>=(int)sizeof(core)||snprintf(libs,sizeof(libs),"%s/lib",root)>=(int)sizeof(libs))die("runtime path too long");
    const char *clear[]={"LD_PRELOAD","LD_AUDIT","LD_DEBUG","LD_DEBUG_OUTPUT","LD_PROFILE","LD_ORIGIN_PATH","LD_LIBRARY_PATH","H3CLI_RUNTIME_ROOT"};
    for(size_t i=0;i<sizeof(clear)/sizeof(*clear);i++)unsetenv(clear[i]);
    if(setenv("LD_LIBRARY_PATH",libs,1)||setenv("H3CLI_RUNTIME_ROOT",root,1))system_error("set runtime environment");
    execv(core,argv);system_error("execute h3cli (check cache mount permits execution)");
}
int main(int argc,char **argv) {
    umask(077);char self[PATH_MAX];ssize_t n=readlink("/proc/self/exe",self,sizeof(self)-1);if(n<=0||n>=(ssize_t)sizeof(self)-1)die("cannot locate executable");self[n]=0;
    int fd=open(self,O_RDONLY|O_CLOEXEC);if(fd<0)system_error("open executable");struct stat st;if(fstat(fd,&st)||st.st_size<FOOTER)die("invalid executable");
    unsigned char footer[FOOTER];read_at(fd,footer,sizeof(footer),(uint64_t)st.st_size-FOOTER);
    if(memcmp(footer,magic,16))die("missing bundle trailer");
    uint64_t mo=u64(footer+16),ml=u64(footer+24),po=u64(footer+32),pl=u64(footer+40),total=u64(footer+112),end=(uint64_t)st.st_size-FOOTER;
    if(ml<2||ml>MAX_MANIFEST||po>end||pl>end-po||mo!=po+pl||ml!=end-mo||total>MAX_UNPACKED||u64(footer+120))die("invalid bundle bounds");
    unsigned char digest[32];file_digest(fd,mo,ml,digest);if(memcmp(digest,footer+48,32))die("manifest checksum mismatch");
    char *text=calloc((size_t)ml+1,1);if(!text)system_error("allocate manifest");read_at(fd,text,(size_t)ml,mo);
    json_object *manifest=json_tokener_parse(text);free(text);if(!manifest||number(manifest,"schema")!=1)die("unsupported bundle schema");json_object *files=field(manifest,"files",json_type_array);check_files(files,pl,total);
    /* Include the file table as well as compressed bytes: modes and paths are
       part of a runtime's identity even when its compressed contents match. */
    /* OpenSSL 3's one-shot SHA256 uses EVP and pulls provider loading and
     * network helpers into the static launcher. Keep the same direct SHA
     * implementation used for payload/file hashing, with no runtime loader. */
    SHA256_CTX footer_hash;
    SHA256_Init(&footer_hash);
    SHA256_Update(&footer_hash,footer+48,64);
    SHA256_Final(digest,&footer_hash);
    char identity[65],payload_identity[65];hex(digest,identity);hex(footer+80,payload_identity);char cache[PATH_MAX],root[PATH_MAX];
    const char *override=getenv("H3CLI_RUNTIME_CACHE"),*xdg=getenv("XDG_CACHE_HOME"),*home=getenv("HOME");
    int length;
    if(override&&*override)length=snprintf(cache,sizeof(cache),"%s",override);
    else if(xdg&&*xdg)length=snprintf(cache,sizeof(cache),"%s/h3cli/runtime",xdg);
    else if(home&&*home)length=snprintf(cache,sizeof(cache),"%s/.cache/h3cli/runtime",home);
    else die("set H3CLI_RUNTIME_CACHE to a writable executable directory");
    if(length<0||length>=(int)sizeof(cache)||cache[0]!='/')die("cache root must be an absolute path");
    if(snprintf(root,sizeof(root),"%s/%s",cache,identity)>=(int)sizeof(root))die("cache path too long");
    if(getenv("H3CLI_BUNDLE_INFO")){
        json_object_object_add(manifest,"payload_sha256",json_object_new_string(payload_identity));json_object_object_add(manifest,"runtime_id",json_object_new_string(identity));json_object_object_add(manifest,"cache",json_object_new_string(root));
        puts(json_object_to_json_string_ext(manifest,JSON_C_TO_STRING_PRETTY));json_object_put(manifest);close(fd);return 0;}
    make_parents(cache);if(mkdir(cache,0700)&&errno!=EEXIST)system_error("create runtime cache");int cachefd=directory(cache);
    int lock=openat(cachefd,".extract.lock",O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600);if(lock<0||flock(lock,LOCK_EX))system_error("lock runtime cache");
    if(fstat(lock,&st)||!S_ISREG(st.st_mode)||st.st_uid!=geteuid()||(st.st_mode&0022))die("unsafe extraction lock");
    if(lstat(root,&st)==0){int rootfd=directory(root);if(!ready(rootfd,files,identity))die("cached runtime changed; stop its processes and remove %s before retrying",root);close(rootfd);}
    else {
        if(errno!=ENOENT)system_error("inspect runtime cache");
        struct statvfs space;if(fstatvfs(cachefd,&space))system_error("inspect free cache space");
        if(total>(uint64_t)space.f_bavail*(uint64_t)space.f_frsize)die("not enough disk space to extract runtime (%"PRIu64" bytes needed)",total);
        file_digest(fd,po,pl,digest);if(memcmp(digest,footer+80,32))die("payload checksum mismatch");
        char stage[PATH_MAX];if(snprintf(stage,sizeof(stage),"%s/.extract-XXXXXX",cache)>=(int)sizeof(stage))die("staging path too long");
        if (!mkdtemp(stage))
            system_error("create extraction staging directory");
        int dir = directory(stage);
        for(size_t i=0;i<json_object_array_length(files);i++)extract_file(fd,po,dir,json_object_array_get_idx(files,i));
        write_ready(dir,files,identity);close(dir);if(rename(stage,root)||fsync(cachefd))system_error("publish runtime cache");
    }
    json_object_put(manifest);close(fd);close(lock);close(cachefd);launch(root,argc,argv);return 125;
}
