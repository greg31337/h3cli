#include "src/models/internal.h"
#include "src/platform.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/clonefile.h>
#else
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif

typedef struct {
    uint64_t version, device, inode, bytes;
    int64_t mtime, mtime_nsec, ctime, ctime_nsec;
    char sha256[65];
} receipt;
typedef struct { char *key; int dir, lock; } lease;

static receipt file_receipt(const struct stat *st, const char *hash) {
    struct timespec mt = h3_stat_mtime(st), ct = h3_stat_ctime(st);
    receipt r = {.version=1, .device=(uint64_t)st->st_dev, .inode=(uint64_t)st->st_ino,
        .bytes=(uint64_t)st->st_size, .mtime=mt.tv_sec, .mtime_nsec=mt.tv_nsec,
        .ctime=ct.tv_sec, .ctime_nsec=ct.tv_nsec};
    snprintf(r.sha256, sizeof(r.sha256), "%s", hash);
    return r;
}
static int same_receipt(const receipt *a, const receipt *b) {
    return a->version == b->version && a->device == b->device && a->inode == b->inode &&
        a->bytes == b->bytes && a->mtime == b->mtime && a->mtime_nsec == b->mtime_nsec &&
        a->ctime == b->ctime && a->ctime_nsec == b->ctime_nsec && !strcmp(a->sha256, b->sha256);
}
static char *metadata_file(const h3_model_plan *p, const h3_model_item *item, const char *leaf) {
    const char *root = (item->artifact->groups & 3u) ? p->paths.main : item->destination;
    char hash[65]; model_digest(root, strlen(root), hash);
    char *parent = strdup(root);
    if (!parent) return NULL;
    char *slash = strrchr(parent, '/');
    if (!slash) { free(parent); return NULL; }
    if (slash == parent) parent[1] = 0; else *slash = 0;
    char *path = NULL;
    if (asprintf(&path, "%s/.h3cli-downloads/%s/%s", parent, hash, leaf) < 0) path = NULL;
    free(parent); return path;
}
static void item_key(const h3_model_item *item, char out[65]) {
    model_digest(item->destination, strlen(item->destination), out);
}
static int metadata_dir(const h3_model_plan *p, const h3_model_item *item, int create,
                         char *error, size_t size) {
    char *path = metadata_file(p, item, "lock"), *leaf = NULL;
    int dir = path ? model_parent(path, create, &leaf, error, size) : -1;
    free(path); free(leaf); return dir;
}
static int get_receipt(int dir, const char *key, receipt *r) {
    char name[90]; snprintf(name, sizeof(name), "%s.receipt", key);
    int fd = openat(dir, name, O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
    if (fd < 0) return 0;
    ssize_t got = read(fd, r, sizeof(*r)); char extra;
    int ok = got == sizeof(*r) && read(fd, &extra, 1) == 0 && r->version == 1 && r->sha256[64] == 0;
    close(fd); return ok;
}
static int save_receipt(int dir, const char *key, int fd, const char *hash) {
    struct stat st;
    if (fstat(fd, &st)) return 0;
    receipt r = file_receipt(&st, hash);
    char name[90]; snprintf(name, sizeof(name), "%s.receipt", key);
    return model_write_atomic(dir, name, &r, sizeof(r));
}
/* Existing unmanaged files remain custom inputs unless extending/verifying an
 * installation. A managed receipt must still detect a changed on-disk file. */
static int check_existing(model_job *job, size_t index, int strict, int save) {
    const h3_model_item *item = &job->plan->items[index];
    int fd = open(item->destination, O_RDONLY|O_CLOEXEC);
    if (fd < 0) return model_fail(job,"cannot read existing model %s: %s",item->destination,strerror(errno));
    struct stat st;
    if (fstat(fd, &st) || !S_ISREG(st.st_mode)) { close(fd); return model_fail(job, "model is not a regular file: %s", item->destination); }
    char ignored[512] = {0}, key[65], hash[65]; item_key(item, key);
    int dir = metadata_dir(job->plan, item, 0, ignored, sizeof(ignored));
    receipt saved = {0}; int managed = dir >= 0 && get_receipt(dir, key, &saved);
    struct stat pin;int recorded=dir>=0&&!fstatat(dir,"catalog",&pin,AT_SYMLINK_NOFOLLOW)&&S_ISREG(pin.st_mode)&&pin.st_size==64;
    receipt current = file_receipt(&st, managed ? saved.sha256 : item->artifact->sha256);
    int valid = managed && same_receipt(&saved, &current) && (!strict || !strcmp(saved.sha256, item->artifact->sha256));
    /* Lost/corrupt receipts in a managed installation must be recovered by
     * verification. Complete unmanaged custom installations remain untouched. */
    if (!strict && !managed && !recorded) valid = 1;
    if (!valid) {
        h3_model_artifact expected = *item->artifact;
        if (managed && !strict) { expected.bytes = saved.bytes; expected.sha256 = saved.sha256; }
        valid = model_hash_fd(fd, &expected, job, hash) && !strcmp(hash, expected.sha256);
        if (valid && dir >= 0 && (save || recorded)) {
            int written=save_receipt(dir,key,fd,expected.sha256);
            if(save&&!written)valid=0; /* Warm read-only reuse may simply reverify. */
        }
    } else if (strict && save && dir >= 0 && !managed) {
        valid = save_receipt(dir, key, fd, item->artifact->sha256);
    }
    if (dir >= 0) close(dir);
    close(fd);
    if (!valid) return model_fail(job, "existing model conflicts with pinned content or changed: %s; preserve it and select a separate empty destination", item->destination);
    pthread_mutex_lock(&job->mutex); job->verified[index] = strict ? 1 : 2; pthread_mutex_unlock(&job->mutex);
    return 1;
}
static int lease_compare(const void *a, const void *b) {
    return strcmp(((const lease *)a)->key, ((const lease *)b)->key);
}
static int installation_missing(const h3_model_plan *p,size_t index) {
    int main=(p->items[index].artifact->groups&3u)!=0;
    for(size_t i=0;i<p->count;i++) {
        int other=(p->items[i].artifact->groups&3u)!=0;
        if((main&&other)||(!main&&!strcmp(p->items[index].destination,p->items[i].destination))) {
            struct stat st;if(stat(p->items[i].destination,&st)||!S_ISREG(st.st_mode))return 1;
        }
    }
    return 0;
}
static int acquire_leases(model_job *job, lease **result, size_t *count) {
    lease *leases = calloc(job->plan->count, sizeof(*leases));
    if (!leases) return model_fail(job, "cannot allocate model locks");
    *result = leases; *count = 0;
    for (size_t i = 0; i < job->plan->count; i++) {
        if(!installation_missing(job->plan,i))continue;
        char *key = metadata_file(job->plan, &job->plan->items[i], "lock");
        if (!key) return model_fail(job, "cannot allocate model lock name");
        int duplicate = 0;
        for (size_t j = 0; j < *count; j++) if (!strcmp(key, leases[j].key)) duplicate = 1;
        if (duplicate) { free(key); continue; }
        leases[(*count)++] = (lease){key, -1, -1};
    }
    qsort(leases, *count, sizeof(*leases), lease_compare);
    for (size_t i = 0; i < *count; i++) {
        char *leaf = NULL, error[512];
        leases[i].dir = model_parent(leases[i].key, 1, &leaf, error, sizeof(error));
        if (leases[i].dir < 0) return model_fail(job, "%s", error);
        leases[i].lock = openat(leases[i].dir, leaf, O_RDWR|O_CREAT|O_NOFOLLOW|O_CLOEXEC, 0600);
        free(leaf);
        if (leases[i].lock < 0) return model_fail(job, "cannot open model installation lock");
        while (flock(leases[i].lock, LOCK_EX|LOCK_NB)) {
            if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
                return model_fail(job, "cannot lock model installation: %s", strerror(errno));
            if (!model_progress(job, "waiting_for_models", "installation lock", 0, 0, 0)) return 0;
            struct timespec wait = {0, 100000000}; nanosleep(&wait, NULL);
        }
    }
    return 1;
}
static int pin_installations(model_job *job, lease *leases, size_t count) {
    for (size_t i = 0; i < count; i++) {
        char stored[66] = {0};
        int fd = openat(leases[i].dir, "catalog", O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
        if (fd >= 0) {
            ssize_t n = read(fd, stored, sizeof(stored)-1); close(fd);
            if (n != 64 || strcmp(stored, h3_model_catalog_sha256))
                return model_fail(job, "partial installation belongs to another catalog; select a separate models path");
        } else if (errno != ENOENT) return model_fail(job, "cannot read model catalog receipt");
        else if (!model_write_atomic(leases[i].dir, "catalog", h3_model_catalog_sha256, 64))
            return model_fail(job, "cannot record model catalog receipt");
    }
    return 1;
}
/* The first main-root receipt covers both modes. Before adopting that root,
 * validate existing catalog files outside this request as well: a complete
 * custom FL2VA tree must not become managed as a side effect of adding Ref2VA.
 * Complete unmanaged trees without a missing main component stay local inputs.
 */
static int adoption_neighbors(model_job *job) {
    const h3_model_plan *p=job->plan;
    size_t first=p->count;
    for(size_t i=0;i<p->count;i++)if((p->items[i].artifact->groups&3u)&&installation_missing(p,i)){first=i;break;}
    if(first==p->count)return 1;
    char error[512]={0};int dir=metadata_dir(p,&p->items[first],0,error,sizeof(error));
    if(dir<0)return model_fail(job,"%s",error);
    struct stat st;int pinned=!fstatat(dir,"catalog",&st,AT_SYMLINK_NOFOLLOW);close(dir);
    if(pinned)return 1; /* pin_installations checks the existing catalog identity. */
    for(size_t i=0;i<h3_model_catalog_count;i++) {
        const h3_model_artifact *a=&h3_model_catalog[i];
        if(!(a->groups&3u))continue;
        char *path=NULL;if(asprintf(&path,"%s/%s",p->paths.main,a->path)<0)return model_fail(job,"cannot inspect neighboring model mode");
        int selected=0;for(size_t j=0;j<p->count;j++)if(!strcmp(path,p->items[j].destination)){selected=1;break;}
        if(selected){free(path);continue;}
        int fd=open(path,O_RDONLY|O_CLOEXEC);
        if(fd<0&&errno==ENOENT){free(path);continue;}
        char hash[65];int ok=fd>=0&&model_hash_fd(fd,a,job,hash)&&!strcmp(hash,a->sha256);
        if(fd>=0)close(fd);
        if(!ok){model_fail(job,"cannot extend a custom or conflicting main-model tree: %s; choose a separate destination",path);free(path);return 0;}
        free(path);
    }
    return 1;
}
static int copy_verified(model_job *job, const h3_model_artifact *a, const char *from, int *target, int dir, const char *part) {
    int source = open(from, O_RDONLY|O_NOFOLLOW|O_CLOEXEC);
    if (source < 0) return 0;
    int cloned = 0;
#ifdef H3_MODEL_TESTING
    int force_copy=getenv("H3_MODEL_TEST_COPY")!=NULL;
#else
    int force_copy=0;
#endif
#ifdef __APPLE__
    /* clonefile creates a new inode; the private staging name is protected by
     * the installation lock and has never been exposed as a model. */
    close(*target); *target = -1;
    if (unlinkat(dir, part, 0) || (!force_copy && fclonefileat(source, dir, part, 0) && errno != ENOTSUP && errno != EXDEV && errno != EINVAL)) {
        close(source); return 0;
    }
    *target = openat(dir, part, O_RDWR|O_CREAT|O_NOFOLLOW|O_CLOEXEC, 0600);
    struct stat clone;
    cloned = *target >= 0 && !fstat(*target, &clone) && (uint64_t)clone.st_size == a->bytes;
#else
    (void)dir; (void)part;
    cloned = !force_copy && ioctl(*target, FICLONE, source) == 0;
#endif
    int fd = *target;
    if (fd < 0) { close(source); return 0; }
    if (!cloned) {
        unsigned char *buffer = malloc(4u*1024u*1024u);
        if (!buffer) { close(source); return 0; }
        int ok = !ftruncate(fd, 0) && lseek(fd, 0, SEEK_SET) >= 0;
        uint64_t copied = 0;
        while (ok && copied < a->bytes) {
            size_t chunk = a->bytes-copied > 4u*1024u*1024u ? 4u*1024u*1024u : (size_t)(a->bytes-copied);
            ssize_t n = read(source, buffer, chunk);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { ok = 0; break; }
            size_t at = 0;
            while (at < (size_t)n) {
                ssize_t sent = write(fd, buffer+at, (size_t)n-at);
                if (sent < 0 && errno == EINTR) continue;
                if (sent <= 0) { ok = 0; break; }
                at += (size_t)sent;
            }
            copied += (uint64_t)n;
            if (!model_progress(job, "copying_model", a->path, 0, 0, 0)) ok = 0;
        }
        free(buffer);
        if (!ok) { close(source); return 0; }
    }
    close(source);
    return 1;
}
static int discovery(const h3_model_item *item);
static char *shared_local(model_job *job,const h3_model_item *item) {
    if(!(item->artifact->groups&3u))return NULL;
    for(size_t i=0;i<h3_model_catalog_count;i++) {
        const h3_model_artifact *a=&h3_model_catalog[i];
        if(!(a->groups&3u)||strcmp(a->sha256,item->artifact->sha256)||!strcmp(a->path,item->artifact->path))continue;
        char *path=NULL;if(asprintf(&path,"%s/%s",job->plan->paths.main,a->path)<0)return NULL;
        int in_plan=0;for(size_t j=0;j<job->plan->count;j++)if(!strcmp(path,job->plan->items[j].destination))in_plan=1;
        if(in_plan){free(path);continue;}
        int fd=open(path,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);char hash[65];
        int ok=fd>=0&&model_hash_fd(fd,a,job,hash)&&!strcmp(hash,a->sha256);
        if(fd>=0)close(fd);
        if(ok)return path;
        free(path);if(atomic_load(&job->stopped))break;
    }
    return NULL;
}
static int install_item(model_job *job, size_t index) {
    const h3_model_item *item = &job->plan->items[index];
    const h3_model_artifact *a = item->artifact;
    if (atomic_load(&job->stopped)) return 0;
    struct stat existing;
    if (!lstat(item->destination, &existing)) return check_existing(job, index, 1, 1);
    char error[512] = {0}, *name = NULL, key[65]; item_key(item, key);
    int dir = metadata_dir(job->plan, item, 1, error, sizeof(error));
    int parent = model_parent(item->destination, 1, &name, error, sizeof(error));
    if (dir < 0 || parent < 0) {
        if (dir >= 0) close(dir);
        if (parent >= 0) close(parent);
        free(name); return model_fail(job, "%s", error);
    }
    char part[150]; snprintf(part, sizeof(part), "%s-%s.part", key, a->sha256);
    int fd = openat(dir, part, O_RDWR|O_CREAT|O_NOFOLLOW|O_CLOEXEC, 0600), ok = fd >= 0;
    if (!ok) model_fail(job, "cannot open partial model %s: %s", item->destination, strerror(errno));
    int reuse = -1;
    for (size_t j = 0; ok && j < job->plan->count; j++) {
        if (j == index || strcmp(a->sha256, job->plan->items[j].artifact->sha256)) continue;
        pthread_mutex_lock(&job->mutex); int ready = job->verified[j] == 1; pthread_mutex_unlock(&job->mutex);
        if (ready) { reuse = (int)j; break; }
        if (j < index && (!discovery(&job->plan->items[j]) || discovery(item))) {
            while (!ready && !atomic_load(&job->stopped)) {
                if (!model_progress(job, "waiting_for_models", a->path, 0, 0, 0)) break;
                struct timespec delay = {0, 100000000}; nanosleep(&delay, NULL);
                pthread_mutex_lock(&job->mutex); ready = job->verified[j] == 1; pthread_mutex_unlock(&job->mutex);
            }
            if (ready) reuse = (int)j;
            break;
        }
    }
    if (ok && atomic_load(&job->stopped)) ok = 0;
    char *shared=ok&&reuse<0?shared_local(job,item):NULL;
    if (ok && (reuse >= 0 || shared)) {
        ok = copy_verified(job, a, shared?shared:job->plan->items[(size_t)reuse].destination, &fd, dir, part);
        if (!ok && !atomic_load(&job->stopped)) model_fail(job, "cannot materialize shared model %s: %s", a->path, strerror(errno));
        if (ok) model_progress(job, "reusing_model", a->path, 0, a->bytes, 1);
    } else if (ok) ok = model_transfer(job, a, fd, dir, part);
    free(shared);
    char hash[65];
    if (ok) ok = model_hash_fd(fd, a, job, hash) && !strcmp(hash, a->sha256);
    if (!ok && !atomic_load(&job->stopped)) {
        /* Bad content must not become a complete, repeatedly reused partial. */
        if (fd >= 0 && ftruncate(fd, 0)) model_fail(job, "cannot discard corrupt partial model");
        char meta[180];snprintf(meta,sizeof(meta),"%s.meta",part);unlinkat(dir,meta,0);
        model_fail(job, "model SHA-256 or byte size mismatch: %s", a->path);
    }
    if (ok) ok = !fsync(fd);
    /* A path ancestor must not have been substituted while a transfer ran. */
    if(ok){
        char *check_name=NULL;int check=model_parent(item->destination,0,&check_name,error,sizeof(error));
        struct stat before,after;
        ok=check>=0&&!fstat(parent,&before)&&!fstat(check,&after)&&before.st_dev==after.st_dev&&before.st_ino==after.st_ino;
        if(check>=0)close(check);
        free(check_name);
    }
    /* Atomically rename without replacement. No writable staging hard link
     * remains after a process crash at the publication boundary. */
#ifdef H3_MODEL_TESTING
    if(ok&&getenv("H3_MODEL_TEST_CRASH_BEFORE_PUBLISH"))_exit(71);
#endif
    int raced=0,published=-1;
    if(ok){
#ifdef __APPLE__
        published=renameatx_np(dir,part,parent,name,RENAME_EXCL);
#else
        published=(int)syscall(SYS_renameat2,dir,part,parent,name,RENAME_NOREPLACE);
#endif
        if(published){
            if(errno==EEXIST){raced=1;ok=check_existing(job,index,1,1);}
            else ok=0;
        }
    }
#ifdef H3_MODEL_TESTING
    if(ok&&getenv("H3_MODEL_TEST_CRASH_AFTER_PUBLISH"))_exit(72);
#endif
    if(ok&&raced)ok=!unlinkat(dir,part,0);
    if(ok)ok=!fsync(parent)&&!fsync(dir);
    if(ok&&!raced)ok=save_receipt(dir,key,fd,a->sha256);
    if(ok){char meta[180];snprintf(meta,sizeof(meta),"%s.meta",part);if(unlinkat(dir,meta,0)&&errno!=ENOENT)ok=0;}
    if (fd >= 0) close(fd);
    close(dir); close(parent); free(name);
    if (!ok && !atomic_load(&job->stopped)) return model_fail(job, "cannot publish model %s: %s", item->destination, strerror(errno));
    if (ok) { pthread_mutex_lock(&job->mutex); job->verified[index] = 1; pthread_mutex_unlock(&job->mutex); }
    return ok;
}
static int discovery(const h3_model_item *item) {
    return strstr(item->artifact->path, "/transformer/config.json") != NULL;
}
static void *install_worker(void *opaque) {
    model_job *job = opaque;
    for (;;) {
        pthread_mutex_lock(&job->mutex);
        size_t i = job->next++;
        while (i < job->plan->count && (job->verified[i] || discovery(&job->plan->items[i]))) i = job->next++;
        pthread_mutex_unlock(&job->mutex);
        if (i >= job->plan->count || atomic_load(&job->stopped)) break;
        if (!install_item(job, i)) break;
    }
    return NULL;
}
int h3_models_prepare(const h3_model_plan *p, h3_model_callback callback,
                      void *opaque, char *error, size_t size) {
    if (!p->count || p->inspect) return 1;
    model_job job = {.plan=p, .callback=callback, .opaque=opaque, .mutex=PTHREAD_MUTEX_INITIALIZER,
        .began=model_now()};
    job.verified = calloc(p->count, 1);
    if (!job.verified) { snprintf(error, size, "cannot allocate model preparation"); return 0; }
    int missing = h3_models_missing(p), ok = 1;
    lease *leases = NULL; size_t lease_count = 0;
    if (missing && p->offline) {
        for (size_t i = 0; i < p->count; i++) {
            struct stat st;
            if (stat(p->items[i].destination, &st)) { model_fail(&job, "offline: missing required model file %s at %s", p->items[i].artifact->path, p->items[i].destination); break; }
        }
        ok = 0;
    }
    if (ok && missing) ok = acquire_leases(&job, &leases, &lease_count);
    for (size_t i = 0; ok && i < p->count; i++) {
        struct stat st;
        if (!lstat(p->items[i].destination, &st)) ok = check_existing(&job, i, installation_missing(p,i) || p->verify, installation_missing(p,i));
        else if (errno != ENOENT) ok = model_fail(&job, "cannot read model %s: %s", p->items[i].destination, strerror(errno));
        else job.total += p->items[i].artifact->bytes;
    }
    if (ok && missing) ok = adoption_neighbors(&job);
    if (ok && missing) ok = pin_installations(&job, leases, lease_count);
    /* Copy fallback requires logical space on its own destination filesystem.
     * Existing staged blocks already consume free space and are not counted twice. */
    for(size_t i=0;ok&&i<lease_count;i++) {
        struct stat mount;struct statvfs fs;
        if(fstat(leases[i].dir,&mount)||fstatvfs(leases[i].dir,&fs)){ok=model_fail(&job,"cannot inspect model free space");break;}
        uint64_t required=0;
        for(size_t j=0;j<p->count;j++)if(!job.verified[j]) {
            char ignored[512]={0};int dir=metadata_dir(p,&p->items[j],0,ignored,sizeof(ignored));struct stat other;
            if(dir<0||fstat(dir,&other)){if(dir>=0)close(dir);ok=model_fail(&job,"cannot inspect staging filesystem");break;}
            if(other.st_dev==mount.st_dev){
                char key[65],part[150];item_key(&p->items[j],key);
                snprintf(part,sizeof(part),"%s-%s.part",key,p->items[j].artifact->sha256);
                struct stat partial;uint64_t allocated=0;
                if(!fstatat(dir,part,&partial,AT_SYMLINK_NOFOLLOW)&&S_ISREG(partial.st_mode)&&partial.st_blocks>0)allocated=(uint64_t)partial.st_blocks*512;
                uint64_t bytes=p->items[j].artifact->bytes;
                required+=bytes>allocated?bytes-allocated:0;
            }
            close(dir);
        }
        if(ok&&(long double)fs.f_bavail*(long double)fs.f_frsize<(long double)required)
            ok=model_fail(&job,"insufficient model disk space; need up to %" PRIu64 " additional bytes on destination filesystem",required);
    }
    if (ok && missing) {
        pthread_t threads[4]; int started = 0;
        for (int i = 0; i < 4; i++) {
            if (pthread_create(&threads[i], NULL, install_worker, &job)) { ok = model_fail(&job, "cannot start model transfer thread"); break; }
            started++;
        }
        for (int i = 0; i < started; i++) pthread_join(threads[i], NULL);
        ok = ok && !atomic_load(&job.stopped);
        for (size_t i = 0; ok && i < p->count; i++)
            if (!job.verified[i]) ok = install_item(&job, i);
        for (size_t i = 0; ok && i < lease_count; i++)
            if (!model_write_atomic(leases[i].dir, "ready", h3_model_catalog_sha256, 64))
                ok = model_fail(&job, "cannot record complete model installation");
        if (ok) model_progress(&job, "models_ready", "required components", 0, 0, 1);
    }
    for (size_t i = 0; i < lease_count; i++) {
        if (leases[i].lock >= 0) close(leases[i].lock);
        if (leases[i].dir >= 0) close(leases[i].dir);
        free(leases[i].key);
    }
    free(leases); free(job.verified);
    if (!ok || atomic_load(&job.stopped)) {
        snprintf(error, size, "%s", job.error[0] ? job.error : "model download cancelled"); ok = 0;
    }
    pthread_mutex_destroy(&job.mutex);
    return ok;
}
