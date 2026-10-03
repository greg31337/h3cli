/* Linked only into bin/h3cli-server-test. Never part of the production server. */
static int fake_execute(const h3_request *request,worker_state *state){
    const h3_option_value *p=h3_request_get(request,"prompt");const char *prompt=p?p->value:"";
    double end=srv_now()+(strstr(prompt,"slow")?4.0:0.20);
    if(strstr(prompt,"crash"))_exit(73);
    if(strstr(prompt,"hang"))end=srv_now()+120;
    if(strstr(prompt,"logflood"))for(int i=0;i<10000;i++)fprintf(stderr,"FAKE test worker log line %d: abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz\n",i);
    while(srv_now()<end){if(worker_progress("denoise",1,2,state))return 130;struct timespec delay={0,50000000};nanosleep(&delay,NULL);}
    for(size_t i=0;i<request->count;i++){const h3_option_value *v=&request->values[i];if(v->option->path!=H3_PATH_OUTPUT||!strcmp(v->option->name,"frames-dir"))continue;
        FILE *f=fopen(v->value,"wb");if(!f)return 1;fprintf(f,"FAKE SERVER LIFECYCLE FIXTURE, NOT H3 MEDIA\n%s\n",prompt);fclose(f);}
    state->result=sj_object();sj_add(state->result,"schema",sj_int(1));sj_add(state->result,"kind",sj_string("fake_test_only"));sj_add(state->result,"test_only",sj_bool(1));
    sj_add(state->result,"width",sj_int(256));sj_add(state->result,"height",sj_int(256));sj_add(state->result,"frames",sj_int(22));return 0;
}

/* The fake backend has no model dependency. Its preparation delay only tests
 * process/queue supervision; transfer tests exercise the real downloader. */
static int fake_prepare(const h3_request *request){
    const h3_option_value *p=h3_request_get(request,"prompt");
    double duration=p&&strstr(p->value,"prepare-slow")?5.0:.01,begin=srv_now();
    while(srv_now()-begin<duration){
        h3_model_progress event={"downloading_models","fixture",1,2,0,srv_now()-begin};
        if(worker_model_progress(&event,NULL))return 130;
        struct timespec delay={0,50000000};nanosleep(&delay,NULL);
    }
    return 0;
}
