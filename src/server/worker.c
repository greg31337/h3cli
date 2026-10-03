#include "src/server/internal.h"
#include "src/cli/dispatch.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
enum { REQUEST_FD=100, EVENTS_FD=101, CONTROL_FD=102 };
static volatile sig_atomic_t worker_cancelled;
typedef struct {const char *work;sj_value *result;double last_frame;} worker_state;
static void worker_signal(int signal_number){(void)signal_number;worker_cancelled=1;}
static void *control_watch(void *opaque){
    (void)opaque;char c;ssize_t n;do n=read(CONTROL_FD,&c,1);while(n<0&&errno==EINTR);
    worker_cancelled=1;
    // Parent EOF and explicit cancellation both have a bounded grace period.
    struct timespec delay={3,0};while(nanosleep(&delay,&delay)&&errno==EINTR){}
    kill(0,SIGKILL);return NULL;
}
static void worker_event(sj_value *event){
    char *text=sj_dump(event);sj_free(event);if(!text)return;size_t n=strlen(text);
    if(n<3000){text[n]='\n';ssize_t sent=write(EVENTS_FD,text,n+1);(void)sent;}free(text);
}
static int worker_progress(const char *phase,int completed,int total,void *opaque){
    (void)opaque;if(worker_cancelled)return 1;sj_value *event=sj_object();
    sj_add(event,"type",sj_string("progress"));sj_add(event,"phase",sj_string(phase));
    sj_add(event,"completed",sj_int(completed));sj_add(event,"total",sj_int(total));worker_event(event);return worker_cancelled;
}
static int worker_model_progress(const h3_model_progress *p,void *opaque){
    (void)opaque;if(worker_cancelled)return 1;
    sj_value *event=sj_object();sj_add(event,"type",sj_string("model_preparation"));
    sj_add(event,"phase",sj_string(p->phase));sj_add(event,"component",sj_string(p->component));
    sj_add(event,"downloaded_bytes",sj_uint(p->completed));sj_add(event,"required_bytes",sj_uint(p->total));
    sj_add(event,"reused_bytes",sj_uint(p->reused));sj_add(event,"seconds",sj_real(p->elapsed));
    worker_event(event);return worker_cancelled;
}
static int worker_frame(const h3_frame *frame,void *opaque){
    worker_state *s=opaque;if(worker_cancelled)return 1;double now=srv_now();
    if (now - s->last_frame < 1)
        return 0;
    s->last_frame = now;
    char *path=srv_path(s->work,"preview.ppm"),*temp=srv_path(s->work,"preview.tmp");
    if(!path||!temp){free(path);free(temp);return 1;}
    FILE *f=fopen(temp,"wb");int ok=f&&fprintf(f,"P6\n%d %d\n255\n",frame->width,frame->height)>0;
    for(int y=0;ok&&y<frame->height;y++)ok=fwrite(frame->rgb+(size_t)y*(size_t)frame->stride,3,(size_t)frame->width,f)==(size_t)frame->width;
    if (f && fclose(f))
        ok = 0;
    if (ok)
        ok = rename(temp, path) == 0;
    if (!ok)
        unlink(temp);
    free(path);
    free(temp);
    if(ok){sj_value *event=sj_object();sj_add(event,"type",sj_string("preview"));sj_add(event,"name",sj_string("preview.ppm"));
        sj_add(event,"frame",sj_int(frame->frame_index));sj_add(event,"step",sj_int(frame->denoise_step));worker_event(event);}return !ok;
}
static void worker_result(const h3_result *r,void *opaque){
    worker_state *s=opaque;sj_value *v=sj_object();sj_add(v,"schema",sj_int(1));
    sj_add(v,"kind",sj_string(r->status==H3_RESULT_PAUSED?"paused":r->kind==H3_RESULT_STILL?"still":"video"));
    sj_add(v,"width",sj_int(r->width));sj_add(v,"height",sj_int(r->height));sj_add(v,"frames",sj_int(r->frames));
    sj_add(v,"fps",sj_int(r->fps));sj_add(v,"seed",sj_uint(r->seed));sj_add(v,"audio_samples",sj_int(r->audio_samples));
    sj_add(v,"sample_rate",sj_int(r->sample_rate));sj_add(v,"completed_steps",sj_int(r->completed_steps));sj_add(v,"total_steps",sj_int(r->total_steps));
    sj_free(s->result);s->result=v;
}
#ifdef H3_SERVER_TESTING
#include "tests/server_fake_worker.inc"
#endif
int h3_server_worker_main(int argc,char **argv){
    if(argc!=2||strcmp(argv[1],"--server-worker")){fprintf(stderr,"h3cli: invalid internal worker invocation\n");return 2;}
    struct stat st;if(fstat(REQUEST_FD,&st)||!S_ISREG(st.st_mode)||st.st_size<=0||st.st_size>4*SRV_JSON_LIMIT||fcntl(CONTROL_FD,F_GETFD)<0||fcntl(EVENTS_FD,F_GETFD)<0){
        fprintf(stderr,"h3cli: worker requires private request/control descriptors\n");return 2;}
    fcntl(REQUEST_FD,F_SETFD,FD_CLOEXEC);fcntl(CONTROL_FD,F_SETFD,FD_CLOEXEC);fcntl(EVENTS_FD,F_SETFD,FD_CLOEXEC);fcntl(EVENTS_FD,F_SETFL,O_NONBLOCK);
    char error[SRV_ERROR]={0};char *text=malloc((size_t)st.st_size+1);if(!text)return 1;size_t got=0;
    while(got<(size_t)st.st_size){ssize_t n=read(REQUEST_FD,text+got,(size_t)st.st_size-got);if(n<=0){if(errno==EINTR)continue;break;}got+=(size_t)n;}close(REQUEST_FD);
    sj_value *packet=got==(size_t)st.st_size?sj_parse(text,got,error,sizeof(error)):NULL;free(text);
    h3_request request={0};uint64_t maximum=0;int64_t parent=0;
    const char *mode=sj_field(packet,"mode"),*work=sj_field(packet,"work");
    if(!packet||!mode||!work||!sj_i64(sj_get(packet,"parent"),&parent)||parent!=getppid()||
       !sj_u64(sj_get(packet,"max_file_bytes"),&maximum)||
       !h3_request_from_json(sj_get(packet,"request"),&request,error,sizeof(error))){fprintf(stderr,"h3cli: invalid worker IPC: %s\n",error);sj_free(packet);h3_request_free(&request);return 2;}
    int validate=!strcmp(mode,"validate"),prepare=!strcmp(mode,"prepare");if(!validate&&!prepare&&strcmp(mode,"run")){sj_free(packet);h3_request_free(&request);return 2;}
    if(!validate&&!srv_resources_check(work,error,sizeof(error))){fprintf(stderr,"h3cli: %s\n",error);sj_free(packet);h3_request_free(&request);return 1;}
    struct rlimit cap={(rlim_t)maximum,(rlim_t)maximum};if(!prepare&&setrlimit(RLIMIT_FSIZE,&cap)){sj_free(packet);h3_request_free(&request);return 1;}
    signal(SIGPIPE,SIG_IGN);signal(SIGTERM,worker_signal);signal(SIGINT,worker_signal);worker_cancelled=0;
    pthread_t watcher;if(pthread_create(&watcher,NULL,control_watch,NULL)){sj_free(packet);h3_request_free(&request);return 1;}
    if(!validate&&!prepare)setenv("H3_OFFLINE","1",1);
    int cli_argc=0;char **cli_argv=h3_request_argv(&request,&cli_argc);worker_state state={.work=work};
    h3_cli_hooks hooks={.validate_only=validate,.prepare_only=prepare,.model_progress=worker_model_progress,.progress=worker_progress,.frame=worker_frame,.result=worker_result,.opaque=&state};
    double begin=srv_now();int status;
#ifdef H3_SERVER_TESTING
    if(!validate&&getenv("H3_SERVER_TEST_WORKER"))status=prepare?fake_prepare(&request):fake_execute(&request,&state);
    else
#endif
    status=cli_argv?h3_cli_run(cli_argc,cli_argv,&hooks):1;
    h3_argv_free(cli_argv);
    if(worker_cancelled)status=130;
    if(!validate&&!status&&!srv_resources_check(work,error,sizeof(error))){fprintf(stderr,"h3cli: %s\n",error);status=1;}
    if(prepare&&!status&&!srv_resources_refresh(&request,work,error,sizeof(error))){fprintf(stderr,"h3cli: %s\n",error);status=1;}
    if(!validate&&!prepare&&!status){
        if(!state.result){state.result=sj_object();sj_add(state.result,"schema",sj_int(1));sj_add(state.result,"kind",sj_string("inspection"));}
        sj_add(state.result,"operation",sj_string(h3_request_operation(&request)));sj_add(state.result,"wall_seconds",sj_real(srv_now()-begin));
        const h3_option_value *q=h3_request_get(&request,"quality");sj_add(state.result,"quality",sj_string(q?q->value:NULL));
        sj_value *outputs=sj_array();
        for(size_t i=0;i<request.count;i++) {
            const h3_option_value *v=&request.values[i];
            if(v->option->path!=H3_PATH_OUTPUT)continue;
            if(!strcmp(v->option->name,"output")&&(h3_request_get(&request,"state-only")||
              (sj_field(state.result,"kind")&&!strcmp(sj_field(state.result,"kind"),"paused")&&!h3_request_get(&request,"preview-on-stop"))))continue;
            if(strncmp(v->value,work,strlen(work))||v->value[strlen(work)]!='/'){status=1;break;}
            sj_value *output=sj_object();sj_add(output,"option",sj_string(v->option->name));sj_add(output,"name",sj_string(v->value+strlen(work)+1));sj_add(outputs,NULL,output);
        }
        sj_add(state.result,"outputs",outputs);
        char *path=srv_path(work,"completion.json");if(!path||!srv_write_json(path,state.result,error,sizeof(error))){fprintf(stderr,"h3cli: %s\n",error);status=1;}free(path);
    }
    sj_value *finished=sj_object();sj_add(finished,"type",sj_string("exit"));sj_add(finished,"status",sj_int(status));worker_event(finished);
    pthread_cancel(watcher);pthread_join(watcher,NULL);close(EVENTS_FD);close(CONTROL_FD);
    sj_free(state.result);h3_request_free(&request);sj_free(packet);return status;
}
static int make_pipe(int fds[2]){
    int temp[2];if(pipe(temp))return 0;
    fds[0]=fcntl(temp[0],F_DUPFD_CLOEXEC,103);fds[1]=fcntl(temp[1],F_DUPFD_CLOEXEC,103);close(temp[0]);close(temp[1]);
    if(fds[0]<0||fds[1]<0){if(fds[0]>=0)close(fds[0]);if(fds[1]>=0)close(fds[1]);return 0;}return 1;
}
int srv_worker_launch(srv *s,const h3_request *request,int validate,const char *work,pid_t *pid,int *events,int *control,int *logs,char *error,size_t size){
    char hash[65];if(!srv_file_hash(s->config.executable,hash)||strcmp(hash,s->executable_sha)){snprintf(error,size,"worker executable changed; restart the server");return 0;}
    int input=-1,ep[2]={-1,-1},cp[2]={-1,-1},lp[2]={-1,-1},ok=0;
    char *temp=srv_path(work,"request-XXXXXX");if(!temp)return 0;int fd=mkstemp(temp);if(fd>=0){unlink(temp);input=fcntl(fd,F_DUPFD_CLOEXEC,103);close(fd);}free(temp);
    if(input<0||!make_pipe(ep)||!make_pipe(cp)||!make_pipe(lp))goto done;
    sj_value *packet=sj_object();sj_add(packet,"mode",sj_string(validate==2?"prepare":validate?"validate":"run"));sj_add(packet,"work",sj_string(work));
    sj_add(packet,"parent",sj_int(getpid()));sj_add(packet,"max_file_bytes",sj_uint(s->config.storage_limit));sj_add(packet,"request",h3_request_json(request));
    char *body=sj_dump(packet);sj_free(packet);if(!body)goto done;size_t n=strlen(body),at=0;
    while(at<n){ssize_t wrote=write(input,body+at,n-at);if(wrote<0&&errno==EINTR)continue;if(wrote<=0)break;at+=(size_t)wrote;}free(body);if(at!=n||lseek(input,0,SEEK_SET)<0)goto done;
    posix_spawn_file_actions_t actions;posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions,input,REQUEST_FD);posix_spawn_file_actions_adddup2(&actions,ep[1],EVENTS_FD);
    posix_spawn_file_actions_adddup2(&actions,cp[0],CONTROL_FD);posix_spawn_file_actions_adddup2(&actions,lp[1],STDOUT_FILENO);posix_spawn_file_actions_adddup2(&actions,lp[1],STDERR_FILENO);
    posix_spawn_file_actions_addopen(&actions,STDIN_FILENO,"/dev/null",O_RDONLY,0600);
    posix_spawnattr_t attrs;posix_spawnattr_init(&attrs);sigset_t empty,defaults;sigemptyset(&empty);sigemptyset(&defaults);sigaddset(&defaults,SIGTERM);sigaddset(&defaults,SIGINT);
    posix_spawnattr_setsigmask(&attrs,&empty);posix_spawnattr_setsigdefault(&attrs,&defaults);
    posix_spawnattr_setflags(&attrs,POSIX_SPAWN_SETPGROUP|POSIX_SPAWN_SETSIGMASK|POSIX_SPAWN_SETSIGDEF);posix_spawnattr_setpgroup(&attrs,0);
    char *argv[]={s->config.executable,"--server-worker",NULL};
    int rc=posix_spawn(pid,s->config.executable,&actions,&attrs,argv,environ);
    posix_spawnattr_destroy(&attrs);posix_spawn_file_actions_destroy(&actions);
    if(rc){snprintf(error,size,"spawn worker: %s",strerror(rc));goto done;}
    *events=ep[0];ep[0]=-1;*control=cp[1];cp[1]=-1;*logs=lp[0];lp[0]=-1;
    fcntl(*events,F_SETFL,O_NONBLOCK);fcntl(*logs,F_SETFL,O_NONBLOCK);ok=1;
done:
    if (input >= 0)
        close(input);
    for (int i = 0; i < 2; i++) {
        if (ep[i] >= 0)
            close(ep[i]);
        if (cp[i] >= 0)
            close(cp[i]);
        if (lp[i] >= 0)
            close(lp[i]);
    }
    if (!ok && !*error)
        snprintf(error, size, "worker IPC setup: %s", strerror(errno));
    return ok;
}
int srv_worker_validate(srv *s,const h3_request *request,const char *work,char *error,size_t size){
    pid_t pid;int events,control,logs;
    if(!srv_worker_launch(s,request,1,work,&pid,&events,&control,&logs,error,size))return 0;
    double deadline=srv_now()+60;int status=0,done=0;size_t got=0;char message[SRV_ERROR]={0};
    while(!done){char buf[4096];ssize_t n;
        while((n=read(logs,buf,sizeof(buf)))>0){size_t take=(size_t)n;if(take>=sizeof(message)){memcpy(message,buf+take-sizeof(message)+1,sizeof(message)-1);got=sizeof(message)-1;}
            else {if(got+take>=sizeof(message)){size_t drop=got+take-sizeof(message)+1;memmove(message,message+drop,got-drop);got-=drop;}memcpy(message+got,buf,take);got+=take;}message[got]=0;}
        while(read(events,buf,sizeof(buf))>0){}
        pid_t result=waitpid(pid,&status,WNOHANG);if(result==pid){done=1;break;}
        if((result<0&&errno!=EINTR)||srv_now()>deadline||atomic_load(&s->stopping))break;
        struct pollfd fds[2]={{logs,POLLIN,0},{events,POLLIN,0}};poll(fds,2,20);
    }
    if(!done){kill(-pid,SIGKILL);while(waitpid(pid,&status,0)<0&&errno==EINTR){}}
    // The control descriptor remains open until exit so a successful validator is not cancelled.
    close(control);close(events);close(logs);
    if(done&&WIFEXITED(status)&&WEXITSTATUS(status)==0)return 1;
    snprintf(error,size,"%s",*message?message:"native preflight failed or timed out");return 0;
}
