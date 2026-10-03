#include "src/sampling/av_state.h"

/* CommonCrypto remains the platform C SHA-256 API on macOS. */
#include "src/digest.h"
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "h3av requires IEEE-754 binary32");
enum { HEADER_SIZE = 160, HASH_OFFSET = 128 };
static const uint8_t magic[8] = {'H','3','A','V','\r','\n',0x1a,'\n'};

double h3_av_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int fail(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return 0;
}

static int little_endian(void) {
    const uint32_t one = 1;
    return *(const uint8_t *)&one == 1;
}

static int multiply(size_t *n, size_t v) {
    if (v && *n > SIZE_MAX / v) return 0;
    *n *= v;
    return 1;
}

int h3_geometry_valid(int width,int height,int profile) {
    return (profile==0||profile==1)&&width>=32&&height>=32&&!(width%32)&&!(height%32)&&
        (int64_t)width*height<=(profile?INT64_C(2088960):H3_MAX_PIXELS)&&
        (!profile||(width<=1920&&height<=1920));
}
int h3_av_state_shape_profile(int width,int height,int frames,int profile,h3_av_state_info *info) {
    if (!info || !h3_geometry_valid(width,height,profile) || frames < 5 || frames > 362 ||
        (frames - 5) % 17) return 0;
    h3_av_state_info result = {0};
    result.version = 3; result.geometry_profile = profile;
    result.render_width = width; result.render_height = height;
    result.frames = frames;
    result.video_t = h3_video_latent_t(frames);
    result.latent_h = height / 16; result.latent_w = width / 16;
    result.audio_t = h3_temporal(frames).audio_t;
    size_t video = 24, audio = 64;
    if (!multiply(&video, (size_t)result.video_t) ||
        !multiply(&video, (size_t)result.latent_h) ||
        !multiply(&video, (size_t)result.latent_w) ||
        !multiply(&audio, (size_t)result.audio_t) ||
        video > SIZE_MAX / sizeof(float) || audio > SIZE_MAX / sizeof(float) ||
        video > SIZE_MAX / sizeof(float) - audio) return 0;
    result.video_elements = video; result.audio_elements = audio;
    *info = result;
    return 1;
}
int h3_av_state_shape(int width,int height,int frames,h3_av_state_info *info) {
    return h3_av_state_shape_profile(width,height,frames,0,info);
}

static int valid(const h3_av_state *s) {
    h3_av_state_info expected;
    return s && s->video && s->audio && s->info.version == 3 &&
        h3_av_state_shape_profile(s->info.render_width, s->info.render_height,
                          s->info.frames, s->info.geometry_profile, &expected) &&
        expected.video_t == s->info.video_t &&
        expected.audio_t == s->info.audio_t &&
        expected.latent_h == s->info.latent_h &&
        expected.latent_w == s->info.latent_w &&
        expected.video_elements == s->info.video_elements &&
        expected.audio_elements == s->info.audio_elements;
}

h3_av_state *h3_av_state_new_profile(int width, int height, int frames,int profile,uint64_t seed,
                             const uint8_t signature[32]) {
    h3_av_state_info info;
    if (!signature || !h3_av_state_shape_profile(width, height, frames,profile,&info)) return NULL;
    h3_av_state *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->info = info; s->info.seed = seed;
    memcpy(s->info.compatibility, signature, 32);
    s->video = malloc(info.video_elements * sizeof(float));
    s->audio = malloc(info.audio_elements * sizeof(float));
    if (!s->video || !s->audio) { h3_av_state_free(s); return NULL; }
    return s;
}
h3_av_state *h3_av_state_new(int width,int height,int frames,uint64_t seed,const uint8_t signature[32]) {
    return h3_av_state_new_profile(width,height,frames,0,seed,signature);
}

const h3_av_state *h3_result_av_state(const h3_result *r) { return r ? r->av_state : NULL; }
const h3_av_state_info *h3_av_state_get_info(const h3_av_state *s) { return s ? &s->info : NULL; }
const float *h3_av_state_video(const h3_av_state *s) { return s ? s->video : NULL; }
const float *h3_av_state_audio(const h3_av_state *s) { return s ? s->audio : NULL; }
void h3_av_state_free(h3_av_state *s) {
    if (s) { free(s->video); free(s->audio); free(s); }
}
h3_av_state *h3_av_state_clone(const h3_av_state *s) {
    if (!valid(s)) return NULL;
    h3_av_state *copy = h3_av_state_new_profile(s->info.render_width, s->info.render_height,
        s->info.frames, s->info.geometry_profile, s->info.seed, s->info.compatibility);
    if (copy) {
        memcpy(copy->video, s->video, s->info.video_elements * sizeof(float));
        memcpy(copy->audio, s->audio, s->info.audio_elements * sizeof(float));
    }
    return copy;
}

static void put(uint8_t *p, uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; i++) p[i] = (uint8_t)(value >> (8 * i));
}
static uint64_t get(const uint8_t *p, unsigned bytes) {
    uint64_t value = 0;
    for (unsigned i = 0; i < bytes; i++) value |= (uint64_t)p[i] << (8 * i);
    return value;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
static void hash_bytes(h3_sha256_ctx *hash, const void *data, size_t size) {
    const uint8_t *p = data;
    while (size) {
        h3_sha256_size chunk = (h3_sha256_size)(size > 1048576 ? 1048576 : size);
        h3_sha256_update(hash, p, chunk);
        p += chunk; size -= chunk;
    }
}
static void checksum(uint8_t header[HEADER_SIZE], const h3_av_state *s,
                     uint8_t digest[32]) {
    h3_sha256_ctx hash;
    h3_sha256_init(&hash);
    hash_bytes(&hash, header, HASH_OFFSET);
    hash_bytes(&hash, s->video, s->info.video_elements * sizeof(float));
    hash_bytes(&hash, s->audio, s->info.audio_elements * sizeof(float));
    h3_sha256_final(digest, &hash);
}

int h3_av_state_metadata_signature(const char *model_dir,int ref2va,
                           uint8_t signature[32],char *error,size_t size) {
    if(!model_dir||!signature||(ref2va!=0&&ref2va!=1))return fail(error,size,"invalid reference AV identity");
    const char *files[]={"transformer/config.json","video_vae/config.json",
        "video_vae/source/config.json","audio_vae/config.json","audio_vae/config.yaml",
        "audio_vae/metadata.json","video_vae/source/model.safetensors","audio_vae/model.safetensors"};
    const char contract[]="h3-sglang-local-av-metadata-v1;v=24,16,17,5;a=32,2,40,800;f32";
    h3_sha256_ctx hash;h3_sha256_init(&hash);hash_bytes(&hash,contract,sizeof(contract));
    for(size_t i=0;i<sizeof(files)/sizeof(*files);i++) {
        char path[4096];struct stat st;
        int n=snprintf(path,sizeof(path),"%s/%s/%s",model_dir,ref2va?"Ref2VA":"FL2VA",files[i]);
        if(n<0||(size_t)n>=sizeof(path)||stat(path,&st)||!S_ISREG(st.st_mode))return fail(error,size,"cannot stat reference AV compatibility file");
        uint8_t stamp[56];put(stamp,st.st_dev,8);put(stamp+8,st.st_ino,8);put(stamp+16,st.st_size,8);
#ifdef __APPLE__
        put(stamp+24,st.st_mtimespec.tv_sec,8);put(stamp+32,st.st_mtimespec.tv_nsec,8);
        put(stamp+40,st.st_ctimespec.tv_sec,8);put(stamp+48,st.st_ctimespec.tv_nsec,8);
#else
        put(stamp+24,st.st_mtim.tv_sec,8);put(stamp+32,st.st_mtim.tv_nsec,8);
        put(stamp+40,st.st_ctim.tv_sec,8);put(stamp+48,st.st_ctim.tv_nsec,8);
#endif
        hash_bytes(&hash,files[i],strlen(files[i])+1);hash_bytes(&hash,stamp,sizeof(stamp));
    }
    h3_sha256_final(signature,&hash);return 1;
}
#pragma GCC diagnostic pop

static void state_header(const h3_av_state *s, uint8_t h[HEADER_SIZE]) {
    memset(h,0,HEADER_SIZE);
    memcpy(h, magic, 8);
    put(h+8, s->info.version, 4); put(h+12, HEADER_SIZE, 4);
    put(h+16, 0x01020304, 4); put(h+20, 1, 4); /* LE, IEEE F32 */
    const int fields[] = {s->info.render_width, s->info.render_height, s->info.frames,
        s->info.video_t, s->info.latent_h, s->info.latent_w, s->info.audio_t, 24, 32, 2};
    for (unsigned i = 0; i < 10; i++) put(h+24+4*i, (uint32_t)fields[i], 4);
    put(h+64, s->info.seed, 8);
    put(h+72, s->info.video_elements * sizeof(float), 8);
    put(h+80, s->info.audio_elements * sizeof(float), 8);
    memcpy(h+88, s->info.compatibility, 32);
    put(h+120,(unsigned)s->info.geometry_profile,4);
    checksum(h, s, h+HASH_OFFSET);
}

void h3_av_state_fingerprint(const h3_av_state *s, uint8_t digest[32]) {
    uint8_t h[HEADER_SIZE]; state_header(s,h);
    h3_sha256_ctx hash; h3_sha256_init(&hash);
    hash_bytes(&hash,h,HEADER_SIZE);
    hash_bytes(&hash,s->video,s->info.video_elements*4);
    hash_bytes(&hash,s->audio,s->info.audio_elements*4);
    h3_sha256_final(digest,&hash);
}

int h3_av_state_save(const h3_av_state *s, const char *path, char *error, size_t size) {
    double start = h3_av_now();
    if (error && size) *error = 0;
    if (!valid(s) || !path || !*path || !little_endian())
        return fail(error, size, "invalid h3av state/path or unsupported host endianness");
    uint8_t h[HEADER_SIZE]; state_header(s,h);
    size_t length = strlen(path);
    if (length > SIZE_MAX - 16) return fail(error, size, "h3av path too long");
    char *temporary = malloc(length + 16);
    if (!temporary) return fail(error, size, "out of memory saving h3av");
    snprintf(temporary, length + 16, "%s.tmp.XXXXXX", path);
    int fd = mkstemp(temporary);
    FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
    int ok = f && fwrite(h, 1, sizeof(h), f) == sizeof(h) &&
        fwrite(s->video, sizeof(float), s->info.video_elements, f) == s->info.video_elements &&
        fwrite(s->audio, sizeof(float), s->info.audio_elements, f) == s->info.audio_elements &&
        fflush(f) == 0 && fsync(fd) == 0;
    if (f) { if (fclose(f)) ok = 0; } else if (fd >= 0) close(fd);
    if (ok && rename(temporary, path)) ok = 0;
    if (!ok) {
        if (error && size) snprintf(error, size, "cannot save h3av %s: %s", path, strerror(errno));
        unlink(temporary);
    }
    free(temporary);
    if (ok && getenv("H3_PROFILE")) fprintf(stderr, "h3cli: AV state save %.3f s, %zu bytes\n",
        h3_av_now()-start, HEADER_SIZE + (s->info.video_elements+s->info.audio_elements)*4);
    return ok;
}

h3_av_state *h3_av_state_load(const char *path, char *error, size_t size) {
    double start = h3_av_now();
    if (error && size) *error = 0;
    FILE *f = path && *path ? fopen(path, "rb") : NULL;
    if (!f) { fail(error, size, "cannot open h3av state"); return NULL; }
    uint8_t h[HEADER_SIZE], digest[32];
    h3_av_state *s = NULL;
    struct stat st;
    if (!little_endian() || fread(h, 1, sizeof(h), f) != sizeof(h) ||
        memcmp(h, magic, 8) || get(h+8,4)!=3 || get(h+12,4) != HEADER_SIZE ||
        get(h+16,4) != 0x01020304 || get(h+20,4) != 1 || get(h+124,4) != 0 ||
        get(h+120,4)>1) {
        fail(error, size, "invalid h3av magic, version, geometry profile, endianness or header"); goto done;
    }
    int v[10];
    for (unsigned i=0; i<10; i++) {
        uint64_t n = get(h+24+4*i, 4);
        if (n > INT_MAX) { fail(error,size,"h3av dimension exceeds INT_MAX"); goto done; }
        v[i] = (int)n;
    }
    h3_av_state_info expected;
    if (!h3_av_state_shape_profile(v[0],v[1],v[2],(int)get(h+120,4),&expected) ||
        v[3]!=expected.video_t || v[4]!=expected.latent_h || v[5]!=expected.latent_w ||
        v[6]!=expected.audio_t || v[7]!=24 || v[8]!=32 || v[9]!=2 ||
        get(h+72,8)!=expected.video_elements*4 || get(h+80,8)!=expected.audio_elements*4 ||
        fstat(fileno(f),&st) || st.st_size < 0 ||
        (uint64_t)st.st_size != HEADER_SIZE+(expected.video_elements+expected.audio_elements)*4) {
        fail(error,size,"h3av geometry or payload length mismatch (expected [24,T,H,W] and [32,2,T])"); goto done;
    }
    s=h3_av_state_new_profile(v[0],v[1],v[2],(int)get(h+120,4),get(h+64,8),h+88);
    if (!s) { fail(error,size,"out of memory loading h3av"); goto done; }
    int ok = fread(s->video,4,expected.video_elements,f)==expected.video_elements &&
        fread(s->audio,4,expected.audio_elements,f)==expected.audio_elements && fgetc(f)==EOF && !ferror(f);
    if (ok) checksum(h,s,digest);
    if (!ok || memcmp(digest,h+HASH_OFFSET,32)) {
        fail(error,size,"h3av corruption: payload checksum mismatch"); h3_av_state_free(s); s=NULL;
    }
done:
    fclose(f);
    if (s && getenv("H3_PROFILE")) fprintf(stderr,"h3cli: AV state load %.3f s, %zu bytes\n",
        h3_av_now()-start, HEADER_SIZE+(s->info.video_elements+s->info.audio_elements)*4);
    return s;
}

int h3_av_state_validate_continuation(const h3_av_state *s,
    int width, int height, int frames, int context,
    const uint8_t signature[32], h3_denoise_prefix *prefix, char *error, size_t size) {
    h3_av_state_info target;
    if (!valid(s) || s->info.geometry_profile!=0 || !h3_av_state_shape(width,height,frames,&target))
        return fail(error,size,"invalid continuation source or target geometry");
    if (!h3_continuation_context(context,prefix)) {
        if (error && size) snprintf(error,size,"continuation context %d must be 39 + 51*k frames",context);
        return 0;
    }
    if (s->info.render_width!=width || s->info.render_height!=height) {
        if (error && size) snprintf(error,size,"continuation geometry expected %dx%d, got %dx%d",
            s->info.render_width,s->info.render_height,width,height);
        return 0;
    }
    if (signature && memcmp(signature,s->info.compatibility,32)) {
        if (error && size) snprintf(error,size,
            "incompatible continuation model/VAE signature: expected %02x%02x%02x%02x..., state has %02x%02x%02x%02x...; use matching normalization and VAE weights",
            signature[0],signature[1],signature[2],signature[3],
            s->info.compatibility[0],s->info.compatibility[1],s->info.compatibility[2],s->info.compatibility[3]);
        return 0;
    }
    if (s->info.frames<context || s->info.video_t<prefix->video_prefix_t ||
        s->info.audio_t<prefix->audio_prefix_t) {
        if (error && size) snprintf(error,size,"continuation needs %d history frames; source has %d",context,s->info.frames);
        return 0;
    }
    if (frames<=context || target.video_t<=prefix->video_prefix_t || target.audio_t<=prefix->audio_prefix_t)
        return fail(error,size,"continuation context consumes target; a nonempty generated suffix is required");
    return 1;
}

int h3_av_state_insert_prefix(const h3_av_state *s,
    const h3_av_state_info *target, h3_denoise_prefix p,
    float *video, float *audio, int augment_video) {
    h3_av_state_info checked;
    if (!target || !h3_av_state_shape(target->render_width,target->render_height,target->frames,&checked) ||
        target->version!=3 || target->geometry_profile!=0 || target->video_t!=checked.video_t || target->audio_t!=checked.audio_t ||
        target->latent_h!=checked.latent_h || target->latent_w!=checked.latent_w ||
        target->video_elements!=checked.video_elements || target->audio_elements!=checked.audio_elements) return 0;
    if (!valid(s) || !target || !video || !audio || p.video_prefix_t<0 || p.audio_prefix_t<0 ||
        target->latent_h!=s->info.latent_h || target->latent_w!=s->info.latent_w ||
        p.video_prefix_t>s->info.video_t || p.video_prefix_t>target->video_t ||
        p.audio_prefix_t>s->info.audio_t || p.audio_prefix_t>target->audio_t) return 0;
    size_t hw=(size_t)target->latent_h*(size_t)target->latent_w;
    size_t count=(size_t)p.video_prefix_t*hw;
    for (size_t c=0;c<24;c++) {
        const float *src=s->video+(c*(size_t)s->info.video_t+(size_t)(s->info.video_t-p.video_prefix_t))*hw;
        float *dst=video+c*(size_t)target->video_t*hw;
        if (!augment_video) memcpy(dst,src,count*sizeof(float));
        else for (size_t i=0;i<count;i++) {
            /* Separate products match the F32 native equations (no FMA). */
            volatile float clean=0.999f*src[i];
            volatile float noise=0.001f*dst[i];
            dst[i]=clean+noise;
        }
    }
    for (size_t c=0;c<64;c++) memcpy(audio+c*(size_t)target->audio_t,
        s->audio+c*(size_t)s->info.audio_t+(size_t)(s->info.audio_t-p.audio_prefix_t),
        (size_t)p.audio_prefix_t*sizeof(float));
    return 1;
}

int h3_av_state_insert_bridge_video(const h3_av_state *s,
    const h3_av_state_info *target, const h3_bridge_profile *p,
    float initial_sigma, float *video) {
    h3_av_state_info checked;
    if (!valid(s) || !target || !video || !h3_bridge_profile_valid(p) ||
        !isfinite(initial_sigma) || initial_sigma < 0 || initial_sigma > 1 ||
        !h3_av_state_shape(target->render_width, target->render_height, target->frames, &checked) ||
        target->version != checked.version || target->video_t != checked.video_t ||
        target->audio_t != checked.audio_t || target->latent_h != checked.latent_h ||
        target->latent_w != checked.latent_w || target->video_elements != checked.video_elements ||
        target->audio_elements != checked.audio_elements ||
        target->render_width != s->info.render_width || target->render_height != s->info.render_height ||
        s->info.frames < p->context_frames || target->frames <= p->context_frames) return 0;
    size_t hw = (size_t)target->latent_h * (size_t)target->latent_w;
    int context = p->prefix.video_prefix_t;
    for (size_t c = 0; c < 24; c++) for (int t = 0; t < context; t++) {
        const float *src = s->video + (c * (size_t)s->info.video_t +
            (size_t)(s->info.video_t - context + t)) * hw;
        float *dst = video + (c * (size_t)target->video_t + (size_t)t) * hw;
        float mask = p->class_mask[p->video_classes[t]];
        if (mask == 0) {
            /* Keep the literal hard-mode arithmetic: reconstructing .001f
             * from 1 - .999f changes its F32 bits. */
            for (size_t i = 0; i < hw; i++) {
                volatile float clean = 0.999f * src[i];
                volatile float noise = 0.001f * dst[i];
                dst[i] = clean + noise;
            }
        } else {
            float sigma = mask * initial_sigma;
            for (size_t i = 0; i < hw; i++) dst[i] = h3_flow_mix(src[i], dst[i], sigma);
        }
    }
    return 1;
}

int h3_av_state_insert_bridge(const h3_av_state *s,
    const h3_av_state_info *target, const h3_bridge_profile *p,
    float initial_sigma_v, float initial_sigma_a, float *video, float *audio) {
    /* Validate the additional audio arguments before the video helper can
     * mutate anything. Its geometry validation covers both AV stream shapes. */
    if (!audio || !isfinite(initial_sigma_a) || initial_sigma_a < 0 || initial_sigma_a > 1 ||
        !h3_av_state_insert_bridge_video(s, target, p, initial_sigma_v, video)) return 0;
    int context = p->prefix.audio_prefix_t;
    for (size_t c = 0; c < 64; c++) for (int t = 0; t < context; t++) {
        const float *src = s->audio + c * (size_t)s->info.audio_t +
            (size_t)(s->info.audio_t - context + t);
        float *dst = audio + c * (size_t)target->audio_t + (size_t)t;
        float mask = p->class_mask[p->audio_classes[t]];
        if (mask == 0) memcpy(dst, src, sizeof(*dst));
        else *dst = h3_flow_mix(*src, *dst, mask * initial_sigma_a);
    }
    return 1;
}
