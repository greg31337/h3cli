/* CPU-only metadata-cache regression: no CUDA, model, or rendering. */
#include "src/weights/quant_cache.h"
#include "src/weights/quant.h"
#include "src/platform.h"
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static char error[512];static int checks;
#define CHECK(x) do{checks++;if(!(x)){fprintf(stderr,"FAIL %d %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
static h3_quant_cache *entry(const char *dir,const char *source,uint64_t offset,int mode) {
    return h3_quant_cache_open(dir,source,offset,128,128,mode,9232,8192,9216,error,sizeof(error));
}
static void byte(const char *path,off_t offset,unsigned char value) {
    int fd=open(path,O_WRONLY);CHECK(fd>=0);CHECK(pwrite(fd,&value,1,offset)==1);CHECK(!close(fd));
}
static void flip(const char *path,off_t offset) {
    int fd=open(path,O_RDWR);unsigned char value;CHECK(fd>=0);
    CHECK(pread(fd,&value,1,offset)==1);value^=1;
    CHECK(pwrite(fd,&value,1,offset)==1);CHECK(!close(fd));
}
int main(void) {
    char dir[]="/tmp/h3-quant-cache-XXXXXX";CHECK(mkdtemp(dir));
    char source[512],alias[512];snprintf(source,sizeof(source),"%s/source",dir);snprintf(alias,sizeof(alias),"%s/alias",dir);
    int fd=open(source,O_CREAT|O_RDWR,0600);CHECK(fd>=0);CHECK(!ftruncate(fd,65536));CHECK(!close(fd));
    unsigned char payload[9232],readback[9232];memset(payload,17,sizeof(payload));
    h3_quant_cache *c=entry(dir,source,0,2);CHECK(c);char *artifact=strdup(c->artifact);CHECK(artifact);
    unsigned char original_key[32];memcpy(original_key,c->key,32);
    CHECK(h3_quant_cache_read(c,readback,error,sizeof(error))==0);
    CHECK(h3_quant_cache_write(c,payload,error,sizeof(error)));
    CHECK(h3_quant_cache_read(c,readback,error,sizeof(error))==1);CHECK(!memcmp(payload,readback,sizeof(payload)));
    h3_quant_cache_close(c);
    c=entry(dir,source,0,2);CHECK(c);CHECK(!memcmp(original_key,c->key,32));
    CHECK(h3_quant_cache_read(c,readback,error,sizeof(error))==1);h3_quant_cache_close(c);
    CHECK(!symlink(source,alias));c=entry(dir,alias,0,2);CHECK(c);CHECK(!memcmp(original_key,c->key,32));h3_quant_cache_close(c);
    c=entry(dir,source,2,2);CHECK(c);CHECK(memcmp(original_key,c->key,32));h3_quant_cache_close(c);
    c=entry(dir,source,0,1);CHECK(c);CHECK(memcmp(original_key,c->key,32));h3_quant_cache_close(c);
    CHECK(!entry(dir,source,65536,2));CHECK(strstr(error,"range"));

    /* Version/recipe/shape/source-key/reserved fields and exact file length
     * are validated without calculating a payload checksum. */
    const off_t fields[]={8,12,16,20,24,28,32,40,48,56,88,96,127};
    for(size_t i=0;i<sizeof(fields)/sizeof(*fields);i++) {
        c=entry(dir,source,0,2);CHECK(c);flip(artifact,fields[i]);
        CHECK(h3_quant_cache_read(c,readback,error,sizeof(error))==-1);
        CHECK(h3_quant_cache_write(c,payload,error,sizeof(error)));h3_quant_cache_close(c);
    }
    c=entry(dir,source,0,2);CHECK(c);CHECK(!truncate(artifact,128+sizeof(payload)-1));
    CHECK(h3_quant_cache_read(c,readback,error,sizeof(error))==-1);
    CHECK(h3_quant_cache_write(c,payload,error,sizeof(error)));
    fd=open(artifact,O_WRONLY|O_APPEND);CHECK(fd>=0);CHECK(write(fd,"x",1)==1);CHECK(!close(fd));
    CHECK(h3_quant_cache_read(c,readback,error,sizeof(error))==-1);
    CHECK(h3_quant_cache_write(c,payload,error,sizeof(error)));
    /* Deliberately document the trust boundary: ordinary payload bit changes
     * are not hashed. CUDA's existing scale/nonfinite checks remain separate. */
    byte(artifact,128,18);CHECK(h3_quant_cache_read(c,readback,error,sizeof(error))==1);CHECK(readback[0]==18);
    struct stat before;CHECK(!stat(source,&before));byte(source,0,42);
    struct timespec times[2]={h3_stat_atime(&before),h3_stat_mtime(&before)};
    CHECK(!utimensat(AT_FDCWD,source,times,0));
    CHECK(!h3_quant_cache_unchanged(c,error,sizeof(error)));
    CHECK(!h3_quant_cache_write(c,payload,error,sizeof(error)));h3_quant_cache_close(c);
    c=entry(dir,source,0,2);CHECK(c);CHECK(memcmp(original_key,c->key,32));
    CHECK(h3_quant_cache_read(c,readback,error,sizeof(error))==0);h3_quant_cache_close(c);

    /* A metadata lookup and warm hit need no read permission on the source. */
    CHECK(!chmod(source,0000));c=entry(dir,source,0,2);CHECK(c);
    CHECK(h3_quant_cache_write(c,payload,error,sizeof(error)));h3_quant_cache_close(c);
    c=entry(dir,source,0,2);CHECK(c);CHECK(h3_quant_cache_read(c,readback,error,sizeof(error))==1);h3_quant_cache_close(c);
    CHECK(!chmod(source,0600));
    /* Concurrent publishers serialize and leave a complete reusable artifact. */
    pid_t jobs[2];
    for(int i=0;i<2;i++) {
        jobs[i]=fork();CHECK(jobs[i]>=0);
        if(!jobs[i]) {
            h3_quant_cache *child=entry(dir,source,0,2);if(!child)_exit(2);
            int hit=h3_quant_cache_read(child,readback,error,sizeof(error));
            int ok=hit==1||(hit==0&&h3_quant_cache_write(child,payload,error,sizeof(error)));
            h3_quant_cache_close(child);_exit(ok?0:3);
        }
    }
    for(int i=0;i<2;i++){int status;CHECK(waitpid(jobs[i],&status,0)==jobs[i]);CHECK(WIFEXITED(status)&&WEXITSTATUS(status)==0);}
    c=entry(dir,source,0,2);CHECK(c);CHECK(h3_quant_cache_read(c,readback,error,sizeof(error))==1);h3_quant_cache_close(c);
    unsetenv("H3_QUANT_VERIFY");CHECK(!h3_quant_verify());
    setenv("H3_QUANT_VERIFY","1",1);CHECK(h3_quant_verify());CHECK(h3_quant_options(2,NULL,error,sizeof(error)));
    setenv("H3_QUANT_VERIFY","bad",1);CHECK(!h3_quant_options(2,NULL,error,sizeof(error)));unsetenv("H3_QUANT_VERIFY");
    DIR *d=opendir(dir);CHECK(d);struct dirent *e;
    while((e=readdir(d)))if(strcmp(e->d_name,".")&&strcmp(e->d_name,"..")) {
        CHECK(!strstr(e->d_name,".tmp."));char path[1024];snprintf(path,sizeof(path),"%s/%s",dir,e->d_name);CHECK(!unlink(path));
    }
    closedir(d);CHECK(!rmdir(dir));free(artifact);
    printf("PASS: %d metadata-cache checks (no GPU, no weight hashes)\n",checks);return 0;
}
