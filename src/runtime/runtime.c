#include "src/runtime/runtime.h"
#if defined(__linux__) && defined(H3_PACKAGE_RUNTIME)
#include "src/platform.h"
#include <json-c/json.h>
#include <openssl/sha.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <fcntl.h>

static const char *names[]={"H3_SGLANG_CUBLAS_LIBRARY","H3_SGLANG_CUDNN_LIBRARY",
    "H3_SGLANG_JPEG_LIBRARY","H3_SGLANG_INPUT_FFMPEG","H3_FFMPEG","H3_FFPROBE"};
static char identities[6][72], entries[6][128], runtime_id[72];
static void broken(const char *message) {
    fprintf(stderr,"h3cli: invalid packaged runtime: %s\n",message); exit(2);
}
static int relative_path(const char *p) {
    if(!p||!*p||*p=='/'||strchr(p,'\\'))return 0;
    const char *at=p;
    while(*at){const char *end=strchr(at,'/');size_t n=end?(size_t)(end-at):strlen(at);
        if(!n||(n==1&&at[0]=='.')||(n==2&&!strncmp(at,"..",2)))return 0;
        if (!end)
            break;
        at = end + 1;
        if (!*at)
            return 0;
    }
    return 1;
}
static json_object *field(json_object *o,const char *key,enum json_type type) {
    json_object *v=NULL;
    if(!json_object_object_get_ex(o,key,&v)||!json_object_is_type(v,type))broken(key);
    return v;
}
static void file_hash(const char *path,char hash[65]) {
    FILE *file=fopen(path,"rb");if(!file)broken("cannot read identity file");
    SHA256_CTX ctx;unsigned char buffer[65536],digest[32];size_t count;
    SHA256_Init(&ctx);
    while((count=fread(buffer,1,sizeof(buffer),file)))SHA256_Update(&ctx,buffer,count);
    if(ferror(file)){fclose(file);broken("identity read failed");}
    fclose(file);SHA256_Final(digest,&ctx);
    for(size_t i=0;i<32;i++)snprintf(hash+2*i,3,"%02x",digest[i]);
}
/* Only the regression harness opts into historical media. The test profile is
 * outside the shipping payload and is bound to validation provenance by hash.
 * Ordinary H3_FFMPEG/PATH overrides still cannot replace packaged resources. */
static void reference_media(void) {
    const char *profile=getenv("H3_TEST_REFERENCE_MEDIA");
    if(!profile||!*profile)return;
    const char *expected=getenv("H3_TEST_REFERENCE_MEDIA_SHA256");
    if(!expected||strlen(expected)!=64)broken("regression media profile hash missing");
    char *root=realpath(profile,NULL),hash[65],path[PATH_MAX];
    if(!root)broken("regression media profile missing");
    file_hash(root,hash);if(strcmp(hash,expected))broken("regression media profile changed");
    json_object *record=json_object_from_file(root);
    if(!record||json_object_get_int(field(record,"schema",json_type_int))!=1||
       strcmp(json_object_get_string(field(record,"purpose",json_type_string)),"sglang-regression"))broken("regression media profile schema");
    char *slash=strrchr(root,'/');if(!slash)broken("regression profile directory");*slash=0;
    json_object *files=field(record,"files",json_type_object);
    if(json_object_object_length(files)!=3)broken("regression media file count");
    for(size_t i=3;i<6;i++) {
        json_object *item=field(files,names[i],json_type_object);
        const char *relative=json_object_get_string(field(item,"path",json_type_string));
        const char *version=json_object_get_string(field(item,"version",json_type_string));
        const char *digest=json_object_get_string(field(item,"sha256",json_type_string));
        if(!relative_path(relative)||strcmp(version,i==4?"4.2.2":"6.1.1"))broken("regression media selection");
        if(snprintf(path,sizeof(path),"%s/%s",root,relative)>=(int)sizeof(path))broken("regression media path too long");
        struct stat st;
        if(lstat(path,&st)||!S_ISREG(st.st_mode)||st.st_size!=json_object_get_int64(field(item,"size",json_type_int)))broken("regression media missing or changed");
        char *resolved=realpath(path,NULL);
        if(!resolved||strncmp(resolved,root,strlen(root))||resolved[strlen(root)]!='/')broken("regression media escapes its directory");
        free(resolved);file_hash(path,hash);
        if(strcmp(hash,digest))broken("regression media content changed");
        if(setenv(names[i],path,1))broken("regression media environment allocation");
        snprintf(identities[i],sizeof(identities[i]),"sha256:%s",hash);
        snprintf(entries[i],sizeof(entries[i]),"%s=%s",names[i],identities[i]);
    }
    json_object_put(record);free(root);
}
static json_object *verify_ready(const char *root) {
    char path[PATH_MAX];struct stat st;
    if(snprintf(path,sizeof(path),"%s/.ready.json",root)>=(int)sizeof(path))broken("ready path too long");
    if(lstat(path,&st)||!S_ISREG(st.st_mode)||st.st_uid!=geteuid()||st.st_size>8*1024*1024)broken("ready marker missing; launch the downloaded executable or reseal the unpacked runtime");
    json_object *ready=json_object_from_file(path);
    if(!ready)broken("ready marker");
    json_object *files=field(ready,"files",json_type_object);
    if(json_object_object_length(files)<2||json_object_object_length(files)>4096)broken("ready file count");
    int rootfd=open(root,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
    if(rootfd<0)broken("runtime directory");
    json_object_object_foreach(files,name,saved) {
        if(!relative_path(name)||!json_object_is_type(saved,json_type_array)||json_object_array_length(saved)!=8)broken("ready file entry");
        char copy[PATH_MAX];if(strlen(name)>=sizeof(copy))broken("ready file path");strcpy(copy,name);
        char *part=copy,*slash;int dir=dup(rootfd);
        while((slash=strchr(part,'/'))){*slash=0;int next=openat(dir,part,O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);close(dir);if(next<0)broken("unsafe runtime directory");dir=next;part=slash+1;}
        if (fstatat(dir, part, &st, AT_SYMLINK_NOFOLLOW) || !S_ISREG(st.st_mode) || st.st_uid != geteuid())
            broken("runtime file missing or replaced");
        close(dir);
        int64_t values[]={st.st_size,st.st_mode&07777,(int64_t)st.st_ino,(int64_t)st.st_dev,st.st_mtim.tv_sec,st.st_mtim.tv_nsec,st.st_ctim.tv_sec,st.st_ctim.tv_nsec};
        for(size_t i=0;i<8;i++)if(!json_object_is_type(json_object_array_get_idx(saved,i),json_type_int)||json_object_get_int64(json_object_array_get_idx(saved,i))!=values[i])broken("runtime changed; stop its processes and remove its cache entry before retrying");
    }
    close(rootfd);return ready;
}
static void package_start(void) __attribute__((constructor));
static void package_start(void) {
    char *exe=h3_executable_path();if(!exe)broken("cannot resolve executable");
    char *slash=strrchr(exe,'/');if(!slash)broken("executable directory");*slash=0;
    slash=strrchr(exe,'/');if(!slash)broken("runtime directory");*slash=0;
    const char *override=getenv("H3CLI_RUNTIME_ROOT");
    char *root=realpath(override&&*override?override:exe,NULL);free(exe);
    if(!root)broken("runtime root is missing");
    json_object *ready=verify_ready(root);
    char path[PATH_MAX];
    if(snprintf(path,sizeof(path),"%s/share/h3cli/runtime.json",root)>=(int)sizeof(path))broken("path too long");
    json_object *manifest=json_object_from_file(path);
    if(!manifest||json_object_get_int(field(manifest,"schema",json_type_int))!=1)broken("manifest schema");
    FILE *mf=fopen(path,"rb");if(!mf)broken("cannot read manifest");
    SHA256_CTX hash;SHA256_Init(&hash);unsigned char buffer[65536], digest[32];size_t count;
    while((count=fread(buffer,1,sizeof(buffer),mf)))SHA256_Update(&hash,buffer,count);
    if(ferror(mf)){fclose(mf);broken("manifest read failed");}fclose(mf);SHA256_Final(digest,&hash);
    memcpy(runtime_id,"sha256:",7);for(size_t i=0;i<32;i++)snprintf(runtime_id+7+2*i,3,"%02x",digest[i]);
    if(setenv("H3_PACKAGED_RUNTIME",runtime_id,1))broken("runtime identity allocation");
    json_object *files=field(manifest,"files",json_type_object);
    json_object *recorded=field(ready,"files",json_type_object),*entry=NULL;
    if(json_object_object_length(recorded)!=json_object_object_length(files)+1||!json_object_object_get_ex(recorded,"share/h3cli/runtime.json",&entry))broken("incomplete ready marker");
    json_object_object_foreach(files,name,unused) {
        (void)unused;
        if(!json_object_object_get_ex(recorded,name,&entry))broken("unverified runtime member");
    }
    json_object_put(ready);
    json_object *env=field(manifest,"environment",json_type_object);
    for(size_t i=0;i<sizeof(names)/sizeof(*names);i++) {
        const char *relative=json_object_get_string(field(env,names[i],json_type_string));
        if(!relative_path(relative))broken("dependency path");
        json_object *record=field(files,relative,json_type_object);
        const char *hash=json_object_get_string(field(record,"sha256",json_type_string));
        if(strlen(hash)!=64||strspn(hash,"0123456789abcdef")!=64)broken("dependency identity");
        if(snprintf(path,sizeof(path),"%s/%s",root,relative)>=(int)sizeof(path))broken("dependency path too long");
        struct stat st;
        if(lstat(path,&st)||!S_ISREG(st.st_mode)||st.st_size!=json_object_get_int64(field(record,"size",json_type_int)))broken("dependency missing or changed");
        if(setenv(names[i],path,1))broken("environment allocation");
        snprintf(identities[i],sizeof(identities[i]),"sha256:%s",hash);
        snprintf(entries[i],sizeof(entries[i]),"%s=%s",names[i],identities[i]);
    }
    reference_media();
    if(setenv("H3CLI_RUNTIME_ROOT",root,1))broken("runtime environment allocation");
    /* libcurl otherwise retains a build-distribution-specific CA path. */
    if(!getenv("CURL_CA_BUNDLE")&&!getenv("SSL_CERT_FILE")) {
        const char *stores[]={"/etc/ssl/certs/ca-certificates.crt","/etc/pki/tls/certs/ca-bundle.crt","/etc/ssl/ca-bundle.pem"};
        for(size_t i=0;i<sizeof(stores)/sizeof(*stores);i++)if(access(stores[i],R_OK)==0){setenv("SSL_CERT_FILE",stores[i],0);setenv("CURL_CA_BUNDLE",stores[i],0);break;}
    }
    json_object_put(manifest);free(root);
}
const char *h3_runtime_dependency_id(const char *name) {
    if(!strcmp(name,"H3_PACKAGED_RUNTIME"))return runtime_id;
    for(size_t i=0;i<sizeof(names)/sizeof(*names);i++)if(!strcmp(name,names[i]))return identities[i];
    return NULL;
}
const char *h3_runtime_environment_entry(const char *entry) {
    for(size_t i=0;i<sizeof(names)/sizeof(*names);i++){size_t n=strlen(names[i]);if(!strncmp(entry,names[i],n)&&entry[n]=='=')return entries[i];}
    return entry;
}
#endif
