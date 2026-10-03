#include "src/server/internal.h"
#include "src/digest.h"
#include "src/platform.h"
#include <curl/curl.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#ifdef __APPLE__
#include <sys/mount.h>
#else
#include <sys/vfs.h>
#endif
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
extern char **environ;
static volatile sig_atomic_t stop_signal;
static void stopping(int sig){(void)sig;stop_signal=1;}
int srv_fail(srv_error *e,int status,const char *code,const char *field,const char *format,...){
    if(e){e->status=status;snprintf(e->code,sizeof(e->code),"%s",code);snprintf(e->field,sizeof(e->field),"%s",field?field:"");
        va_list ap;va_start(ap,format);vsnprintf(e->message,sizeof(e->message),format,ap);va_end(ap);}return 0;
}
char *srv_redact(srv *s,const char *text){
    char *result=strdup(text);if(!result)return NULL;
    size_t count=3+s->config.read_root_count;
    for(size_t i=0;i<count;i++){const char *from=i==0?s->config.state:i==1?s->config.model:i==2?s->config.source_root:s->config.read_roots[i-3];
        if (!from || strlen(from) < 2)
            continue;
        const char *to = i == 0 ? "[managed]" : i == 1 ? "[model]" : i == 2 ? "[source]" : "[read-root]";
        const char *at=result;size_t len=0;char *next=NULL;FILE *f=open_memstream(&next,&len);if(!f){free(result);return NULL;}
        const char *match;while((match=strstr(at,from))){fwrite(at,1,(size_t)(match-at),f);fputs(to,f);at=match+strlen(from);}fputs(at,f);int ok=fclose(f)==0;free(result);if(!ok){free(next);return NULL;}result=next;
    }return result;
}
double srv_now(void){struct timespec t;clock_gettime(CLOCK_REALTIME,&t);return (double)t.tv_sec+(double)t.tv_nsec/1e9;}
int srv_id(char out[SRV_ID],const char *prefix){
    unsigned char bytes[16];int fd=open("/dev/urandom",O_RDONLY|O_CLOEXEC);if(fd<0)return 0;
    size_t got=0;while(got<sizeof(bytes)){ssize_t n=read(fd,bytes+got,sizeof(bytes)-got);if(n<=0){if(errno==EINTR)continue;close(fd);return 0;}got+=(size_t)n;}close(fd);
    size_t start=strlen(prefix);if(start+33>SRV_ID)return 0;memcpy(out,prefix,start);
    for (size_t i = 0; i < 16; i++)
        snprintf(out + start + i * 2, 3, "%02x", bytes[i]);
    return 1;
}
char *srv_path(const char *a,const char *b){char *p=NULL;if(asprintf(&p,"%s/%s",a,b)<0)return NULL;return p;}
int srv_mkdirs(const char *path,char *error,size_t size){
    char *p=strdup(path);if(!p)return 0;
    for(char *c=p+1;;c++)if(*c=='/'||!*c){char save=*c;*c=0;
        struct stat st;int created=mkdir(p,0700)==0;if(!created&&errno!=EEXIST){snprintf(error,size,"mkdir: %s",strerror(errno));free(p);return 0;}
        if(lstat(p,&st)||!S_ISDIR(st.st_mode)){snprintf(error,size,"path component is not a real directory");free(p);return 0;}
        if(created){char *parent=strdup(p);char *slash=parent?strrchr(parent,'/'):NULL;if(slash){if(slash==parent)slash[1]=0;else *slash=0;}
            int fd=open(slash?parent:".",O_RDONLY|O_CLOEXEC);int synced=fd>=0&&fsync(fd)==0;if(fd>=0)close(fd);free(parent);
            if(!synced){snprintf(error,size,"sync directory: %s",strerror(errno));free(p);return 0;}}
        *c=save;if(!save)break;}
    free(p);return 1;
}
int srv_write_json(const char *path,const sj_value *v,char *error,size_t size){
    char *temp=NULL;if(asprintf(&temp,"%s.tmp",path)<0)return 0;int fd=open(temp,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(fd<0){snprintf(error,size,"create manifest: %s",strerror(errno));free(temp);return 0;}
    FILE *f=fdopen(fd,"wb");int ok=f&&sj_write(f,v)&&fputc('\n',f)!=EOF&&fflush(f)==0&&fsync(fd)==0;
    if(f){if(fclose(f))ok=0;}else close(fd);
    if(ok&&rename(temp,path))ok=0;
    if(ok){char *parent=strdup(path);char *slash=parent?strrchr(parent,'/'):NULL;if(slash){*slash=0;int dir=open(parent,O_RDONLY|O_CLOEXEC);if(dir>=0){if(fsync(dir))ok=0;close(dir);}}free(parent);}
    if(!ok){snprintf(error,size,"publish manifest: %s",strerror(errno));unlink(temp);}free(temp);return ok;
}
sj_value *srv_read_json(const char *path,char *error,size_t size){
    int fd=open(path,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);struct stat st;
    if(fd<0||fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<0||st.st_size>4*SRV_JSON_LIMIT){if(fd>=0)close(fd);snprintf(error,size,"invalid manifest file");return NULL;}
    char *text=malloc((size_t)st.st_size+1);if(!text){close(fd);return NULL;}size_t got=0;
    while(got<(size_t)st.st_size){ssize_t n=read(fd,text+got,(size_t)st.st_size-got);if(n<=0){if(errno==EINTR)continue;break;}got+=(size_t)n;}
    close(fd);sj_value *v=got==(size_t)st.st_size?sj_parse(text,got,error,size):NULL;free(text);return v;
}
static void hex_digest(h3_sha256_ctx *ctx,char out[65]){uint8_t d[32];h3_sha256_final(d,ctx);for(size_t i=0;i<32;i++)snprintf(out+i*2,3,"%02x",d[i]);}
int srv_text_hash(const char *text,char out[65]){h3_sha256_ctx ctx;h3_sha256_init_fast(&ctx);size_t n=strlen(text);if(n>UINT32_MAX)return 0;h3_sha256_update(&ctx,text,(h3_sha256_size)n);hex_digest(&ctx,out);return 1;}
int srv_file_hash(const char *path,char out[65]){
    int fd=open(path,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);if(fd<0)return 0;h3_sha256_ctx ctx;h3_sha256_init_fast(&ctx);char buf[65536];int ok=1;
    for(;;){ssize_t n=read(fd,buf,sizeof(buf));if(n<0){if(errno==EINTR)continue;ok=0;break;}if(!n)break;h3_sha256_update(&ctx,buf,(h3_sha256_size)n);}close(fd);if(ok)hex_digest(&ctx,out);return ok;
}
int srv_remove_tree(const char *path){
    struct stat st;if(lstat(path,&st))return errno==ENOENT;
    if(!S_ISDIR(st.st_mode))return unlink(path)==0;
    DIR *d=opendir(path);if(!d)return 0;struct dirent *entry;int ok=1;
    while((entry=readdir(d)))if(strcmp(entry->d_name,".")&&strcmp(entry->d_name,"..")){char *child=srv_path(path,entry->d_name);if(!child||!srv_remove_tree(child))ok=0;free(child);}
    closedir(d);if(rmdir(path))ok=0;return ok;
}
uint64_t srv_tree_bytes(const char *path){
    struct stat st;if(lstat(path,&st))return 0;if(!S_ISDIR(st.st_mode))return S_ISREG(st.st_mode)?(uint64_t)st.st_size:0;
    DIR *d=opendir(path);if(!d)return 0;uint64_t total=0;struct dirent *entry;
    while((entry=readdir(d)))if(strcmp(entry->d_name,".")&&strcmp(entry->d_name,"..")){char *p=srv_path(path,entry->d_name);if(p){uint64_t n=srv_tree_bytes(p);total=n>UINT64_MAX-total?UINT64_MAX:total+n;free(p);}}
    closedir(d);return total;
}
int srv_capture(char *const argv[],char **text,size_t limit,int timeout,char *error,size_t size){
    *text=NULL;int pipefd[2];if(pipe(pipefd))return 0;fcntl(pipefd[0],F_SETFD,FD_CLOEXEC);fcntl(pipefd[1],F_SETFD,FD_CLOEXEC);
    posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions,pipefd[1],STDOUT_FILENO);
    posix_spawn_file_actions_addopen(&actions,STDERR_FILENO,"/dev/null",O_WRONLY,0600);
    posix_spawn_file_actions_addopen(&actions,STDIN_FILENO,"/dev/null",O_RDONLY,0600);
    posix_spawnattr_t attrs;posix_spawnattr_init(&attrs);posix_spawnattr_setflags(&attrs,POSIX_SPAWN_SETPGROUP);posix_spawnattr_setpgroup(&attrs,0);
    pid_t pid;int rc=posix_spawnp(&pid,argv[0],&actions,&attrs,argv,environ);posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attrs);close(pipefd[1]);
    if(rc){close(pipefd[0]);snprintf(error,size,"spawn %s: %s",argv[0],strerror(rc));return 0;}
    fcntl(pipefd[0],F_SETFL,O_NONBLOCK);char *buf=malloc(limit+1);size_t used=0;double end=srv_now()+timeout;int status=0,done=0,ok=buf!=NULL;
    while(ok&&!done){
        for(;;){char chunk[4096];ssize_t n=read(pipefd[0],chunk,sizeof(chunk));if(n>0){if((size_t)n>limit-used){ok=0;break;}memcpy(buf+used,chunk,(size_t)n);used+=(size_t)n;}else {if(n<0&&errno==EINTR)continue;break;}}
        pid_t result=waitpid(pid,&status,WNOHANG);if(result==pid){done=1;break;}if(result<0&&errno!=EINTR){ok=0;break;}
        if(srv_now()>end){ok=0;break;}struct pollfd pfd={.fd=pipefd[0],.events=POLLIN};poll(&pfd,1,25);
    }
    if(!done){kill(-pid,SIGKILL);while(waitpid(pid,&status,0)<0&&errno==EINTR){}}
    if(ok){for(;;){ssize_t n=read(pipefd[0],buf+used,limit-used);if(n<=0)break;used+=(size_t)n;if(used==limit){ok=0;break;}}}
    close(pipefd[0]);ok=ok&&WIFEXITED(status)&&WEXITSTATUS(status)==0;
    if(ok){buf[used]=0;*text=buf;}else {free(buf);snprintf(error,size,"%s failed, exceeded output limit or timed out",argv[0]);}return ok;
}
int srv_probe(const char *path,sj_value **out,char *error,size_t size){
    const char *program=getenv("H3_FFPROBE");if(!program||!*program)program="ffprobe";
    char *argv[]={(char *)program,"-v","error","-show_streams","-show_format","-of","json",(char *)path,NULL};char *text;
    if (!srv_capture(argv, &text, 256 * 1024, 30, error, size))
        return 0;
    *out = sj_parse(text, strlen(text), error, size);
    free(text);
    return *out != NULL;
}
static void config_free(srv_config *c){free(c->host);free(c->state);free(c->model);free(c->model_id);free(c->executable);free(c->source_root);free(c->api_key);free(c->models_root);
    for (size_t i = 0; i < c->read_root_count; i++)
        free(c->read_roots[i]);
    free(c->read_roots);
}
static int number(const char *s,uint64_t min,uint64_t max,uint64_t *out){
    if (!*s)
        return 0;
    for (const char *p = s; *p; p++)
        if (*p < '0' || *p > '9')
            return 0;
    char *end;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || *end || v < min || v > max)
        return 0;
    *out = v;
    return 1;
}
static int assign(char **where,const char *value){char *p=strdup(value);if(!p)return 0;free(*where);*where=p;return 1;}
static int configure(srv_config *c,int argc,char **argv,char *error,size_t size){
    *c=(srv_config){.port=30000,.queue_limit=32,.timeout=3600,.model_timeout=21600,.offline=h3_models_offline(),.upload_limit=UINT64_C(2048)*1024*1024,.storage_limit=UINT64_C(65536)*1024*1024};
    c->host=strdup("127.0.0.1");c->state=strdup("outputs/server");c->models_root=strdup("models");c->model_id=strdup("h3cli");c->executable=h3_executable_path();c->source_root=getcwd(NULL,0);
    if(!c->host||!c->state||!c->models_root||!c->model_id||!c->executable||!c->source_root)return 0;
    for(int i=1;i<argc;i++){
        const char *name=argv[i],*value=NULL;char key[128];const char *eq=strchr(name,'=');
        if(eq){size_t n=(size_t)(eq-name);if(n>=sizeof(key))goto bad;memcpy(key,name,n);key[n]=0;name=key;value=eq+1;}
        if(!strcmp(name,"--offline")){if(value)goto bad;c->offline=1;continue;}
        if(!strcmp(name,"--server")||!strcmp(name,"--server-allow-url-inputs")){if(value)goto bad;if(strcmp(name,"--server"))c->allow_urls=1;continue;}
        if(!value){if(++i>=argc)goto bad;value=argv[i];}
        if(!strcmp(name,"-d")||!strcmp(name,"--model-dir")){if(!assign(&c->model,value))return 0;}
        else if(!strcmp(name,"--models-path")){if(!assign(&c->models_root,value))return 0;}
        else if(!strcmp(name,"--server-host")){if(!assign(&c->host,value))return 0;}
        else if(!strcmp(name,"--server-model-id")){if(!*value||strchr(value,'\n')||!assign(&c->model_id,value))goto bad;}
        else if(!strcmp(name,"--server-state-dir")){if(!assign(&c->state,value))return 0;}
        else if(!strcmp(name,"--server-api-key-file")){
            FILE *f=fopen(value,"rb");if(!f)goto bad;char b[4097];size_t n=fread(b,1,4096,f);int excess=fgetc(f)!=EOF;fclose(f);
            while (n && (b[n - 1] == '\n' || b[n - 1] == '\r'))
                n--;
            b[n] = 0;
            if (excess || n < 16 || memchr(b, 0, n))
                goto bad;
            for (size_t k = 0; k < n; k++)
                if ((unsigned char)b[k] <= 32 || (unsigned char)b[k] >= 127)
                    goto bad;
            if (!assign(&c->api_key, b))
                return 0;
        }else if(!strcmp(name,"--server-read-root")){
            char *p=realpath(value,NULL);struct stat st;if(!p||stat(p,&st)||!S_ISDIR(st.st_mode)){free(p);goto bad;}
            char **roots=realloc(c->read_roots,(c->read_root_count+1)*sizeof(*roots));if(!roots){free(p);return 0;}c->read_roots=roots;c->read_roots[c->read_root_count++]=p;
        }else{
            uint64_t v;if(!number(value,1,UINT64_C(1048576),&v))goto bad;
            if(!strcmp(name,"--server-port")&&v<=65535)c->port=(int)v;
            else if(!strcmp(name,"--server-queue-limit")&&v<=1024)c->queue_limit=(int)v;
            else if(!strcmp(name,"--server-job-timeout")&&v<=86400)c->timeout=(int)v;
            else if(!strcmp(name,"--server-model-download-timeout")&&v<=604800)c->model_timeout=(int)v;
            else if(!strcmp(name,"--server-max-upload-mib"))c->upload_limit=v*1024*1024;
            else if(!strcmp(name,"--server-max-storage-mib"))c->storage_limit=v*1024*1024;
            else goto bad;
        }continue;
bad:snprintf(error,size,"invalid or job-specific server startup option: %.120s",name);return 0;
    }
    if(strcmp(c->host,"127.0.0.1")&&strcmp(c->host,"::1")&&!c->api_key){snprintf(error,size,"non-loopback binding requires --server-api-key-file");return 0;}
    h3_request request={0};h3_model_paths paths={0};struct stat st;
    int resolved=h3_request_add(&request,"models-path",c->models_root,NULL,H3_SOURCE_MANAGED,error,size)&&
        (!c->model||h3_request_add(&request,"model-dir",c->model,NULL,H3_SOURCE_MANAGED,error,size))&&
        h3_models_paths(&request,&paths,error,size);
    h3_request_free(&request);
    if(!resolved)return 0;
    free(c->model);c->model=paths.main;paths.main=NULL;
    free(c->models_root);c->models_root=paths.root;paths.root=NULL;h3_models_paths_free(&paths);
    if (!srv_mkdirs(c->state, error, size))
        return 0;
    char *root = realpath(c->state, NULL);
    if (!root)
        return 0;
    free(c->state);
    c->state = root;
    struct statfs fs;if(statfs(root,&fs)){snprintf(error,size,"cannot inspect state filesystem");return 0;}
#ifdef __APPLE__
    if(!(fs.f_flags&MNT_LOCAL)){snprintf(error,size,"server state requires a local filesystem");return 0;}
#else
    if(fs.f_type==0x6969||fs.f_type==0xff534d42){snprintf(error,size,"server state requires a local filesystem");return 0;}
#endif
    if(stat(root,&st)||st.st_uid!=geteuid()||chmod(root,0700)){snprintf(error,size,"state directory must be owned by the server user");return 0;}
    return 1;
}
int h3_server_main(int argc,char **argv){
    if(argc==3&&(!strcmp(argv[1],"--server")&&!strcmp(argv[2],"--help"))){
        puts("Usage: h3cli --server [-d MODEL] [--server-host 127.0.0.1] [--server-port 30000]\n"
             "  --models-path PATH           Shared models root (models)\n"
             "  --offline                    Disable model downloads and URL imports\n"
             "  --server-model-download-timeout SEC  Preparation deadline (21600)\n"
             "  --server-state-dir PATH       Durable local state (outputs/server)\n"
             "  --server-model-id NAME        Served model (h3cli)\n"
             "  --server-queue-limit N        Waiting variants (32)\n"
             "  --server-job-timeout SEC      Per-variant deadline (3600)\n"
             "  --server-max-upload-mib N     Per-upload limit (2048)\n"
             "  --server-max-storage-mib N    Managed storage limit (65536)\n"
             "  --server-read-root PATH       Allow local input imports; repeatable\n"
             "  --server-api-key-file PATH    Bearer key; required off loopback\n"
             "  --server-allow-url-inputs     Allow bounded public HTTP(S) imports\n"
             "See docs/features/server.md for requests and saved-state workflows.");return 0;
    }
    srv s={.lock_fd=-1};char error[SRV_ERROR]={0};int ok=0,scheduler_started=0;
    pthread_mutex_init(&s.mutex,NULL);pthread_mutex_init(&s.admission,NULL);pthread_cond_init(&s.changed,NULL);
    if(!configure(&s.config,argc,argv,error,sizeof(error)))goto done;
    char *lock=srv_path(s.config.state,"owner.lock");s.lock_fd=lock?open(lock,O_RDWR|O_CREAT|O_CLOEXEC|O_NOFOLLOW,0600):-1;free(lock);
    if(s.lock_fd<0||flock(s.lock_fd,LOCK_EX|LOCK_NB)){snprintf(error,sizeof(error),"state directory already owned or unavailable");goto done;}
    const char *dirs[]={"jobs","assets","incoming","cache"};for(size_t i=0;i<4;i++){char *p=srv_path(s.config.state,dirs[i]);int made=p&&srv_mkdirs(p,error,sizeof(error));free(p);if(!made)goto done;}
    if(!srv_file_hash(s.config.executable,s.executable_sha)||!h3_models_global_init(error,sizeof(error))){snprintf(error,sizeof(error),"cannot initialize service dependencies");goto done;}
    if(!srv_db_open(&s,error,sizeof(error)))goto done;
    srv_gc(&s,1);
    if(!srv_http_start(&s,error,sizeof(error)))goto done;
    signal(SIGPIPE,SIG_IGN);signal(SIGINT,stopping);signal(SIGTERM,stopping);
    if(pthread_create(&s.scheduler,NULL,srv_scheduler,&s)){snprintf(error,sizeof(error),"cannot start queue scheduler");goto done;}scheduler_started=1;
    fprintf(stderr,"h3cli: server listening on %s:%d; one inference worker; state=%s\n",s.config.host,s.config.port,s.config.state);
    double last_gc=0;while(!stop_signal){struct timespec delay={0,100000000};nanosleep(&delay,NULL);if(srv_now()-last_gc>2){srv_gc(&s,0);last_gc=srv_now();}}ok=1;
done:
    atomic_store(&s.stopping,1);pthread_mutex_lock(&s.mutex);pthread_cond_broadcast(&s.changed);pthread_mutex_unlock(&s.mutex);
    srv_http_stop(&s);if(scheduler_started)pthread_join(s.scheduler,NULL);
    if (s.db)
        sqlite3_close(s.db);
    if (s.lock_fd >= 0)
        close(s.lock_fd);
    config_free(&s.config);pthread_cond_destroy(&s.changed);pthread_mutex_destroy(&s.mutex);pthread_mutex_destroy(&s.admission);
    if (!ok)
        fprintf(stderr, "h3cli: server: %s\n", *error ? error : "out of memory");
    return ok ? 0 : 2;
}
