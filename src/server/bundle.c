/* A bounded, non-archive state transfer format. Names are fixed, never paths. */
#include "src/server/internal.h"
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static const unsigned char magic[8]={'H','3','B','N','D','L','1','\n'};
static int copy_bytes(int in,int out,uint64_t count){char buf[65536];while(count){size_t want=count<sizeof(buf)?(size_t)count:sizeof(buf);ssize_t n=read(in,buf,want);if(n<0&&errno==EINTR)continue;if(n<=0)return 0;size_t at=0;while(at<(size_t)n){ssize_t k=write(out,buf+at,(size_t)n-at);if(k<0&&errno==EINTR)continue;if(k<=0)return 0;at+=(size_t)k;}count-=(uint64_t)n;}return 1;}
static int all_write(int fd,const void *data,size_t size){const char *p=data;while(size){ssize_t n=write(fd,p,size);if(n<0&&errno==EINTR)continue;if(n<=0)return 0;p+=n;size-=(size_t)n;}return 1;}
int srv_bundle_write(const char *state,const char *output,char *error,size_t size){
    const char *suffixes[]={"",".presentation",".lora.json"};char *paths[3]={0};uint64_t bytes[3]={0};int count=0,ok=1;
    sj_value *manifest=sj_object(),*files=sj_array();sj_add(manifest,"schema",sj_int(1));sj_add(manifest,"files",files);
    for(int i=0;i<3;i++){char *path=NULL;if(asprintf(&path,"%s%s",state,suffixes[i])<0){ok=0;break;}struct stat st;if(lstat(path,&st)){free(path);if(i==0||errno!=ENOENT)ok=0;continue;}
        char hash[65];if(!S_ISREG(st.st_mode)||st.st_size<0||!srv_file_hash(path,hash)){free(path);ok=0;break;}paths[count]=path;bytes[count++]=(uint64_t)st.st_size;
        sj_value *file=sj_object();sj_add(file,"suffix",sj_string(suffixes[i]));sj_add(file,"bytes",sj_uint((uint64_t)st.st_size));sj_add(file,"sha256",sj_string(hash));sj_add(files,NULL,file);
    }
    char *text=sj_dump(manifest);sj_free(manifest);size_t len=text?strlen(text):0;unsigned char header[16];memcpy(header,magic,8);for(int i=0;i<8;i++)header[8+i]=(unsigned char)((uint64_t)len>>(8*i));
    int fd=ok&&text?open(output,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600):-1;ok=fd>=0&&all_write(fd,header,16)&&all_write(fd,text,len);free(text);
    for(int i=0;ok&&i<count;i++){int in=open(paths[i],O_RDONLY|O_NOFOLLOW|O_CLOEXEC);ok=in>=0&&copy_bytes(in,fd,bytes[i]);if(in>=0)close(in);}
    if(fd>=0){if(fsync(fd)||close(fd))ok=0;}for(int i=0;i<3;i++)free(paths[i]);if(!ok){unlink(output);snprintf(error,size,"cannot create state bundle");}return ok;
}
/* Returns 2 for a non-bundle, 1 for successful extraction, 0 for a corrupt bundle. */
int srv_bundle_unpack(const char *input,const char *output,uint64_t limit,char *error,size_t size){
    int fd=open(input,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);if(fd<0)return 0;unsigned char header[16];ssize_t n=read(fd,header,16);
    if(n<8||memcmp(header,magic,8)){close(fd);return 2;}int ok=n==16;uint64_t len=0;for(int i=0;i<8;i++)len|=(uint64_t)header[8+i]<<(8*i);
    if(!ok||!len||len>16384){close(fd);snprintf(error,size,"invalid bundle header");return 0;}
    char *text=malloc((size_t)len);size_t got=0;while(text&&got<len){ssize_t r=read(fd,text+got,(size_t)len-got);if(r<0&&errno==EINTR)continue;if(r<=0)break;got+=(size_t)r;}
    sj_value *v=text&&got==len?sj_parse(text,(size_t)len,error,size):NULL;free(text);int64_t version=0;const sj_value *files=sj_get(v,"files");
    ok=v&&sj_i64(sj_get(v,"schema"),&version)&&version==1&&files&&files->type==LJ_ARRAY&&files->count>=1&&files->count<=3;unsigned seen=0;uint64_t total=0;char *paths[3]={0};
    for(size_t i=0;ok&&i<files->count;i++){const sj_value *file=files->items[i];const char *suffix=sj_field(file,"suffix"),*expected=sj_field(file,"sha256");uint64_t bytes=0;unsigned bit=0;
        if(suffix){if(!*suffix)bit=1;else if(!strcmp(suffix,".presentation"))bit=2;else if(!strcmp(suffix,".lora.json"))bit=4;}
        if(!bit||(seen&bit)||(i==0&&bit!=1)||!expected||strlen(expected)!=64||!sj_u64(sj_get(file,"bytes"),&bytes)||bytes>limit-total){ok=0;break;}seen|=bit;total+=bytes;
        if(asprintf(&paths[i],"%s%s",output,suffix)<0){ok=0;break;}int out=open(paths[i],O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
        ok=out>=0&&copy_bytes(fd,out,bytes);if(out>=0){if(fsync(out))ok=0;if(close(out))ok=0;}char hash[65];if(ok)ok=srv_file_hash(paths[i],hash)&&!strcmp(hash,expected);
    }
    char extra;if(ok&&read(fd,&extra,1)!=0)ok=0;close(fd);sj_free(v);
    for(int i=0;i<3;i++){if(paths[i]&&!ok)unlink(paths[i]);free(paths[i]);}if(!ok)snprintf(error,size,"invalid, excessive or damaged state bundle");return ok;
}
