#include "src/models/internal.h"
#include <curl/curl.h>
#include <errno.h>
#include <fcntl.h>
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    model_job *job;
    const h3_model_artifact *artifact;
    int fd, metadata_dir, storage_error, bad_range;
    char metadata_name[180],location[4096];
    h3_sha256_ctx hash;
    uint64_t checkpoint;
    long status, retry_after;
    uint64_t offset, written, range_start, range_total;
    int have_range;
} transfer;

typedef struct {
    uint64_t version,total,prefix;
    char expected[65],prefix_sha256[65],origin_sha256[65];
} partial_record;
static void prefix_hash(h3_sha256_ctx *ctx,char out[65]) {
    h3_sha256_ctx copy=*ctx;uint8_t raw[32];h3_sha256_final(raw,&copy);
    for(int i=0;i<32;i++)snprintf(out+2*i,3,"%02x",raw[i]);
}
static int checkpoint(transfer *t) {
    partial_record r={.version=1,.total=t->artifact->bytes,.prefix=t->offset+t->written};
    snprintf(r.expected,sizeof(r.expected),"%s",t->artifact->sha256);
    model_digest(t->artifact->url,strlen(t->artifact->url),r.origin_sha256);
    prefix_hash(&t->hash,r.prefix_sha256);
    if(fsync(t->fd)||!model_write_atomic(t->metadata_dir,t->metadata_name,&r,sizeof(r))){
        t->storage_error=errno?errno:EIO;return 0;
    }
    t->checkpoint=r.prefix;return 1;
}
static int load_prefix(transfer *t) {
    partial_record r={0};int have=0;
    int fd=openat(t->metadata_dir,t->metadata_name,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
    if(fd>=0){char extra;have=read(fd,&r,sizeof(r))==sizeof(r)&&read(fd,&extra,1)==0;close(fd);if(!have)return 0;}
    else if(errno!=ENOENT)return 0;
    char origin[65];model_digest(t->artifact->url,strlen(t->artifact->url),origin);
    if(have&&(r.version!=1||r.total!=t->artifact->bytes||r.prefix>t->offset||
        r.expected[64]||r.origin_sha256[64]||r.prefix_sha256[64]||
        strcmp(r.expected,t->artifact->sha256)||strcmp(r.origin_sha256,origin)))return 0;
    h3_sha256_init_fast(&t->hash);uint64_t at=0;
    unsigned char *buffer=malloc(4u*1024u*1024u);if(!buffer)return 0;
    int ok=1;
    while(at<t->offset){
        uint64_t left=t->offset-at;
        if(have&&at<r.prefix&&left>r.prefix-at)left=r.prefix-at;
        size_t n=left>4u*1024u*1024u?4u*1024u*1024u:(size_t)left;
        ssize_t got=pread(t->fd,buffer,n,(off_t)at);
        if(got<0&&errno==EINTR)continue;
        if(got<=0){ok=0;break;}
        h3_sha256_update(&t->hash,buffer,(h3_sha256_size)got);at+=(uint64_t)got;
        if(have&&at==r.prefix){char hash[65];prefix_hash(&t->hash,hash);if(strcmp(hash,r.prefix_sha256)){ok=0;break;}}
        if(!model_progress(t->job,"verifying_partial",t->artifact->path,0,0,0)){ok=0;break;}
    }
    free(buffer);return ok&&checkpoint(t);
}
static size_t headers(char *data, size_t unit, size_t count, void *opaque) {
    transfer *t = opaque;
    if (unit && count > SIZE_MAX/unit) return 0;
    size_t n = unit*count;
    if (n >= 4096) return 0;
    char line[4096]; memcpy(line, data, n); line[n] = 0;
    if (!strncmp(line, "HTTP/", 5)) {
        const char *space = strchr(line, ' ');
        t->status = space ? strtol(space+1, NULL, 10) : 0;
        t->have_range = 0; t->retry_after = 0; t->location[0]=0;
    } else if (!strncasecmp(line,"Location:",9)) {
        char *value=line+9;while(*value==' '||*value=='\t')value++;
        size_t length=strcspn(value,"\r\n");
        if(length>=sizeof(t->location))return 0;
        memcpy(t->location,value,length);t->location[length]=0;
    } else if (!strncasecmp(line, "Content-Range:", 14)) {
        uint64_t end;
        if (sscanf(line+14, " bytes %" SCNu64 "-%" SCNu64 "/%" SCNu64,
                   &t->range_start, &end, &t->range_total) == 3 &&
            end >= t->range_start && end < t->range_total) t->have_range = 1;
    } else if (!strncasecmp(line, "Retry-After:", 12)) {
        char *end;
        long delay = strtol(line+12, &end, 10);
        if (end == line+12) {
            time_t date = curl_getdate(line+12, NULL), now = time(NULL);
            delay = date > now ? (long)(date-now) : 0;
        }
        if (delay > 0) t->retry_after = delay > 60 ? 60 : delay;
    }
    return n;
}
static size_t write_data(char *data, size_t unit, size_t count, void *opaque) {
    transfer *t = opaque;
    if (unit && count > SIZE_MAX/unit) return 0;
    size_t n = unit*count;
    if (t->status >= 300 && t->status < 400) return 0;
    if ((t->offset && (t->status != 206 || !t->have_range ||
            t->range_start != t->offset || t->range_total != t->artifact->bytes)) ||
        (!t->offset && t->status != 200 && !(t->status == 206 && t->have_range &&
            !t->range_start && t->range_total == t->artifact->bytes))) {
        t->bad_range = 1; return 0;
    }
    if (t->written > t->artifact->bytes-t->offset ||
        n > t->artifact->bytes-t->offset-t->written) {
        t->bad_range = 1; return 0;
    }
    size_t at = 0;
    while (at < n) {
        ssize_t wrote = write(t->fd, data+at, n-at);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) { t->storage_error = errno ? errno : EIO; return 0; }
        at += (size_t)wrote;
    }
    h3_sha256_update(&t->hash,data,(h3_sha256_size)n);
    t->written += n;
    if(t->offset+t->written-t->checkpoint>=64u*1024u*1024u&&!checkpoint(t))return 0;
    return model_progress(t->job, "downloading_models", t->artifact->path, n, 0, 0) ? n : 0;
}
static int progress(void *opaque, curl_off_t total, curl_off_t now,
                     curl_off_t upload_total, curl_off_t uploaded) {
    (void)total; (void)now; (void)upload_total; (void)uploaded;
    transfer *t = opaque;
    return !model_progress(t->job, "downloading_models", t->artifact->path, 0, 0, 0);
}
static int backoff(model_job *job, const char *component, long seconds) {
    double until = model_now()+(double)seconds;
    do {
        if (!model_progress(job, "retrying_model", component, 0, 0, 0)) return 0;
        struct timespec delay = {0, 100000000};
        while (nanosleep(&delay, &delay) && errno == EINTR) {
            if (atomic_load(&job->stopped)) return 0;
        }
    } while (model_now() < until);
    return 1;
}
int model_transfer(model_job *job, const h3_model_artifact *artifact, int fd,int metadata_dir,const char *part) {
    char why[256];
    if (!h3_models_global_init(why, sizeof(why))) return model_fail(job, "%s", why);
    if (job->plan->offline) return model_fail(job, "offline: missing model %s", artifact->path);
#ifndef H3_MODEL_TESTING
    if (strncmp(artifact->url, "https://", 8)) return model_fail(job, "model source must use HTTPS");
#endif
    for (int attempt = 0; attempt < 5; attempt++) {
        off_t offset = lseek(fd, 0, SEEK_END);
        if (offset < 0 || (uint64_t)offset > artifact->bytes) return model_fail(job, "invalid partial model size: %s", artifact->path);
        transfer t = {.job=job, .artifact=artifact, .fd=fd, .metadata_dir=metadata_dir,.offset=(uint64_t)offset};
        snprintf(t.metadata_name,sizeof(t.metadata_name),"%s.meta",part);
        if(!load_prefix(&t))return model_fail(job,"partial metadata/checksum conflict: %s",artifact->path);
        if((uint64_t)offset==artifact->bytes)return 1;
        char *url = strdup(artifact->url);
        if (!url) return model_fail(job, "cannot allocate model URL");
        CURLcode result = CURLE_OK; long status = 0;
        for (int redirect = 0; redirect <= 8; redirect++) {
        CURL *curl = curl_easy_init();
        if (!curl) { free(url); return model_fail(job, "cannot create model HTTPS transfer"); }
        struct curl_slist *authorization = NULL;
        const char *token = getenv("HF_TOKEN");
        if (token && *token && !strncmp(url, "https://huggingface.co/", 23)) {
            if (strchr(token, '\r') || strchr(token, '\n') || strlen(token) > 4096) {
                curl_easy_cleanup(curl); free(url); return model_fail(job, "invalid HF_TOKEN format");
            }
            char *header = NULL;
            if (asprintf(&header, "Authorization: Bearer %s", token) < 0) {
                curl_easy_cleanup(curl); free(url); return model_fail(job, "cannot allocate HF authentication");
            }
            authorization = curl_slist_append(NULL, header);
            memset(header, 0, strlen(header)); free(header);
            if (!authorization) { curl_easy_cleanup(curl); free(url); return model_fail(job, "cannot allocate HF authentication"); }
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, authorization);
        }
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "h3cli-models/1");
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 8L);
        curl_easy_setopt(curl, CURLOPT_UNRESTRICTED_AUTH, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
#ifdef H3_MODEL_TESTING
        long protocols = CURLPROTO_HTTPS|CURLPROTO_HTTP;
#else
        long protocols = CURLPROTO_HTTPS;
#endif
#if LIBCURL_VERSION_NUM >= 0x075500
        const char *schemes = protocols == CURLPROTO_HTTPS ? "https" : "https,http";
        curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, schemes);
        curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, schemes);
#else
        curl_easy_setopt(curl, CURLOPT_PROTOCOLS, protocols);
        curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS, protocols);
#endif
#ifdef H3_MODEL_TESTING
        struct curl_slist *connect=NULL;
        const char *test_connect=getenv("H3_MODEL_TEST_CONNECT");
        if(test_connect){connect=curl_slist_append(NULL,test_connect);curl_easy_setopt(curl,CURLOPT_CONNECT_TO,connect);}
#endif
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 120L);
#ifdef H3_MODEL_TESTING
        if (getenv("H3_MODEL_TEST_TIMEOUT"))
            curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 250L);
#endif
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, (curl_off_t)offset);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, headers);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &t);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &t);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &t);
        const char *ca = getenv("CURL_CA_BUNDLE");
        if (!ca || !*ca) ca = getenv("SSL_CERT_FILE");
        if (ca && *ca) curl_easy_setopt(curl, CURLOPT_CAINFO, ca);
        model_progress(job, "downloading_models", artifact->path, 0, 0, 1);
        result = curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
        char *next = NULL;
        /* Older supported curl releases reject a Range response at the origin's
         * 302 before its body callback. The range belongs to the final CDN
         * response; follow this validated redirect with the same offset. */
        if ((result == CURLE_OK || result == CURLE_WRITE_ERROR || result == CURLE_RANGE_ERROR) &&
            status >= 300 && status < 400 && *t.location) {
            CURLU *target=curl_url();char *resolved=NULL;
            if(target&&!curl_url_set(target,CURLUPART_URL,url,0)&&
                !curl_url_set(target,CURLUPART_URL,t.location,0)&&!curl_url_get(target,CURLUPART_URL,&resolved,0))
                next=strdup(resolved);
            curl_free(resolved);curl_url_cleanup(target);
        }
        curl_easy_cleanup(curl); curl_slist_free_all(authorization); free(url); url = NULL;
#ifdef H3_MODEL_TESTING
        curl_slist_free_all(connect);
#endif
        if (!next) break;
        int valid = !strncmp(next, "https://", 8);
#ifdef H3_MODEL_TESTING
        valid = valid || !strncmp(next, "http://", 7);
#endif
        if (!valid || redirect == 8) { free(next); return model_fail(job, "model %s: unsafe or excessive redirects", artifact->path); }
        /* Recreate headers for each origin, including with pinned curl 7.81.
         * Credentials never reach a CDN, changed port or third-party host. */
        url = next;
        }
        free(url);
        if(!t.storage_error&&!checkpoint(&t))return model_fail(job,"cannot checkpoint partial %s",artifact->path);
        if (atomic_load(&job->stopped)) return 0;
        if (t.storage_error) return model_fail(job, "writing model %s: %s", artifact->path, strerror(t.storage_error));
        if (result == CURLE_OK && t.offset+t.written == artifact->bytes && !t.bad_range) return 1;
        if (offset && status == 200 && (result == CURLE_RANGE_ERROR || t.bad_range)) {
            if (ftruncate(fd, 0) || lseek(fd, 0, SEEK_SET) < 0)
                return model_fail(job, "cannot restart partial model %s", artifact->path);
            if(unlinkat(metadata_dir,t.metadata_name,0)&&errno!=ENOENT)return model_fail(job,"cannot reset partial metadata");
            continue;
        }
        if (status >= 400 && status < 500 && status != 408 && status != 416 && status != 429) {
            /* A previously issued signed CDN URL may expire. A fresh request
             * to the immutable origin is allowed once; persistent denial fails. */
            if (status == 403 && !attempt) continue;
            return model_fail(job, "model %s: HTTP %ld; check pinned source availability and HF_TOKEN permissions", artifact->path, status);
        }
        if (t.bad_range || result == CURLE_PEER_FAILED_VERIFICATION ||
            result == CURLE_SSL_CACERT_BADFILE || result == CURLE_FILESIZE_EXCEEDED)
            return model_fail(job, "model %s: invalid range/size or TLS trust (%s); check source and CA store", artifact->path, curl_easy_strerror(result));
        if (attempt == 4) return model_fail(job, "model %s: transfer failed after five attempts (HTTP %ld, %s)", artifact->path, status, curl_easy_strerror(result));
        if (!backoff(job, artifact->path, t.retry_after ? t.retry_after : 1L<<attempt)) return 0;
    }
    return model_fail(job, "model %s: server repeatedly ignored resume requests", artifact->path);
}
