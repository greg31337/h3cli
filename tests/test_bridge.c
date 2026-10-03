#include "src/sampling/av_state.h"
#include "src/denoise/dit.h"
#include "src/denoise/dit_schedule.h"
#include "src/internal.h"
#include "src/media/ffmpeg.h"
#include "src/vae/video_encoder.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t checks;
static char error[512];
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #x, error); exit(1); } } while (0)

static int equal_float(float a, float b) { return !memcmp(&a, &b, sizeof(a)); }

static void profiles(void) {
    h3_params defaults = H3_PARAMS_DEFAULT;
    CHECK(H3_CONTINUE_HARD == 0 && defaults.continuation_mode == H3_CONTINUE_HARD);
    CHECK(defaults.bridge_video_steps == 8 && defaults.bridge_max_strength == .5f && defaults.bridge_profile == H3_BRIDGE_STEPPED);
    CHECK(h3_video_time_boundary_frames(-1) == -1);
    CHECK(h3_video_time_boundary_frames(INT_MAX) == -1);
    int frames = 0;
    const int spans[] = {1,4,4,4,4};
    for (int t = 0; t <= 107; t++) {
        CHECK(h3_video_time_boundary_frames(t) == frames);
        frames += spans[t % 5];
    }
    const float maxima[] = {0, .25f, .40f, .50f, .65f, 1, FLT_MIN, 0x1p-149f};
    for (int context = 39; context <= 345; context += 51) {
        for (int type = 0; type < 3; type++) for (size_t m = 0; m < sizeof(maxima)/sizeof(*maxima); m++) {
            for (int steps = 1; steps < h3_video_latent_t(context); steps++) {
                h3_bridge_profile p, again;
                char warning[256];
                CHECK(h3_bridge_profile_build(context, steps, maxima[m], (h3_bridge_profile_type)type,
                    &p, warning, sizeof(warning), error, sizeof(error)));
                CHECK(h3_bridge_profile_valid(&p));
                CHECK(h3_bridge_profile_build(context, steps, maxima[m], (h3_bridge_profile_type)type,
                    &again, NULL, 0, error, sizeof(error)));
                CHECK(!memcmp(p.video_classes, again.video_classes, sizeof(p.video_classes)));
                CHECK(!memcmp(p.audio_classes, again.audio_classes, sizeof(p.audio_classes)));
                CHECK(!!*warning == (p.video_exact_t == 1));
                CHECK(p.video_exact_t >= 1 && p.audio_exact_t >= 1);
                CHECK(p.bridge_frames < context);
                CHECK(abs(p.audio_bridge_t * 24 - p.bridge_frames * 40) <= 12);
                CHECK(p.prefix.audio_prefix_t * 24 == context * 40);
                CHECK(p.class_count <= H3_TARGET_ROW_CLASSES);
                for (int t = 0; t < p.prefix.video_prefix_t; t++) {
                    int frame = h3_video_time_boundary_frames(t);
                    if (frame * 40 % 24 == 0) {
                        int tick = frame * 40 / 24;
                        CHECK(p.class_mask[p.video_classes[t]] == p.class_mask[p.audio_classes[tick]]);
                    }
                }
                for (int audio = 0; audio < 2; audio++) {
                    int count = audio ? p.prefix.audio_prefix_t : p.prefix.video_prefix_t;
                    const uint8_t *classes = audio ? p.audio_classes : p.video_classes;
                    int boundary = audio ? p.audio_bridge_t : p.video_bridge_t;
                    float previous = 1;
                    for (int t = 0; t < count; t++) {
                        int c = classes[t];
                        CHECK(c >= 0 && c < H3_TARGET_ROW_CLASSES && p.active[c]);
                        CHECK(p.class_mask[c] <= previous && p.class_mask[c] <= maxima[m] && p.class_mask[c] >= 0);
                        if (t >= boundary) CHECK(c == (audio ? H3_ROW_PRESERVED_AUDIO : H3_ROW_PRESERVED_VIDEO));
                        previous = p.class_mask[c];
                    }
                }
            }
        }
    }
    h3_bridge_profile p;
    CHECK(h3_bridge_profile_build(39, 8, .5f, H3_BRIDGE_STEPPED, &p, NULL, 0, error, sizeof(error)));
    CHECK(p.bridge_frames == 26 && p.audio_bridge_t == 43 && p.audio_exact_t == 22 && p.video_exact_t == 4);
    const float mask[] = {.5f,.5f,.5f,.4f,.3f,.2f,.2f,.1f,0,0,0,0};
    for (int t = 0; t < 12; t++) CHECK(p.class_mask[p.video_classes[t]] == mask[t]);
    for (int t = 0; t < 100; t++) {
        double time = (double)t / 100;
        CHECK(fabsf(h3_bridge_strength(H3_BRIDGE_LINEAR, .5f, time) - (float)(.5 * (1-time))) < 1e-7f);
        CHECK(fabsf(h3_bridge_strength(H3_BRIDGE_EASE_OUT, .5f, time) - (float)(.5 * (1-time*time))) < 1e-7f);
    }
    CHECK(h3_bridge_strength(H3_BRIDGE_LINEAR, .5f, 1) == 0);
    CHECK(isnan(h3_bridge_strength(H3_BRIDGE_LINEAR, .5f, NAN)));
    CHECK(isnan(h3_bridge_strength((h3_bridge_profile_type)99, .5f, 0)));
    h3_bridge_profile before = p;
    const int bad_steps[] = {INT_MIN, -1, 0, 12, 13, INT_MAX};
    for (size_t i = 0; i < sizeof(bad_steps)/sizeof(*bad_steps); i++)
        CHECK(!h3_bridge_profile_build(39, bad_steps[i], .5f, H3_BRIDGE_STEPPED, &p, NULL, 0, error, sizeof(error)));
    const float bad_max[] = {-1, 1.01f, NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < sizeof(bad_max)/sizeof(*bad_max); i++)
        CHECK(!h3_bridge_profile_build(39, 8, bad_max[i], H3_BRIDGE_STEPPED, &p, NULL, 0, error, sizeof(error)));
    CHECK(!h3_bridge_profile_build(40, 8, .5f, H3_BRIDGE_STEPPED, &p, NULL, 0, error, sizeof(error)));
    CHECK(!h3_bridge_profile_build(INT_MAX, 8, .5f, H3_BRIDGE_STEPPED, &p, NULL, 0, error, sizeof(error)));
    CHECK(!h3_bridge_profile_build(39, 8, .5f, (h3_bridge_profile_type)99, &p, NULL, 0, error, sizeof(error)));
    CHECK(!memcmp(p.video_classes, before.video_classes, sizeof(p.video_classes)));
    p.video_classes[0] = 255;
    CHECK(!h3_bridge_profile_valid(&p));
}

static void row_maps(void) {
    h3_sigma_schedule sigmas = {0}; sigmas.steps = 6;
    float sv[] = {1,.7f,.2f,.001f,.0001f,0}, sa[] = {1,.4f,.1f,.0001f,0,0};
    memcpy(sigmas.video, sv, sizeof(sv)); memcpy(sigmas.audio, sa, sizeof(sa));
    h3_dit_schedule *hard = h3_dit_schedule_plan(&sigmas, 1, 1, error, sizeof(error)); CHECK(hard);
    h3_layout_ref refs[] = {{H3_LAYOUT_REF_IMAGE,1,4,6,0}, {H3_LAYOUT_REF_VIDEO,7,4,6,80}, {H3_LAYOUT_REF_AUDIO,0,0,0,80}};
    h3_layout_spec spec = {3,17,4,6,93,56,NULL,0,refs,3};
    h3_layout layout;
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(h3_continuation_context(39, &layout.prefix));
    uint32_t *original = malloc(layout.seq_len * sizeof(*original)), *map = malloc(layout.seq_len * sizeof(*map));
    CHECK(original && map);
    uint8_t tags[] = {1,0,1};
    for (int type = 0; type < 3; type++) for (int maximum = 0; maximum <= 2; maximum++) {
        h3_bridge_profile p;
        CHECK(h3_bridge_profile_build(39, 8, (float)maximum * .5f, (h3_bridge_profile_type)type,
            &p, NULL, 0, error, sizeof(error)));
        h3_dit_schedule *schedule = h3_dit_schedule_plan_bridge(&sigmas, &p, error, sizeof(error)); CHECK(schedule);
        CHECK(!h3_dit_schedule_final(schedule) && !h3_dit_schedule_block(schedule, 0));
        CHECK(h3_dit_schedule_time_rows(schedule) <= h3_dit_schedule_time_rows(hard) + 6 * 20);
        float vv[24*17*4*6], av[64*93], vp[24*17*4*6], ap[64*93];
        for (size_t i = 0; i < sizeof(vv)/sizeof(*vv); i++) vv[i] = 2;
        for (size_t i = 0; i < sizeof(av)/sizeof(*av); i++) av[i] = 3;
        CHECK(h3_bridge_mask_velocity(&p,17,4,6,93,vv,av));
        CHECK(h3_dit_patchify_video(vv,24,17,4,6,vp,sizeof(vp)/sizeof(*vp)));
        CHECK(h3_dit_pack_audio(av,32,93,ap,sizeof(ap)/sizeof(*ap)));
        for (int t = 0; t < 93; t++) {
            CHECK(h3_bridge_row_class(&p,1,(size_t)t,4,6,93) == h3_bridge_row_class(&p,1,(size_t)(t+93),4,6,93));
            for (int c = 0; c < 32; c++) CHECK(ap[t*32+c] == ap[(t+93)*32+c]);
        }
        for (int step = 0; step < sigmas.steps; step++) {
            CHECK(h3_dit_schedule_row_map(hard, step, &layout, tags, 3, original, layout.seq_len));
            CHECK(h3_dit_schedule_row_map(schedule, step, &layout, tags, 3, map, layout.seq_len));
            for (size_t seg = 0; seg < layout.segment_count; seg++) {
                const h3_segment *s = &layout.segments[seg];
                for (size_t row = s->start; row < s->stop; row++) {
                    if (s->kind != H3_SEG_VIDEO && s->kind != H3_SEG_AUDIO) {
                        CHECK(map[row] == original[row]);
                        CHECK(h3_dit_schedule_timestep(schedule,map[row]/3) == h3_dit_schedule_timestep(hard,original[row]/3));
                        continue;
                    }
                    int audio = s->kind == H3_SEG_AUDIO;
                    size_t local = row - s->start;
                    size_t t = audio ? local % 93 : local / 6;
                    float strength = t < (size_t)(audio ? 65 : 12) ? p.class_mask[(audio ? p.audio_classes : p.video_classes)[t]] : 1;
                    h3_target_row_class kind = h3_bridge_row_class(&p,audio,local,4,6,93);
                    CHECK(p.class_mask[kind] == strength);
                    volatile float sigma = strength * (audio ? sa[step] : sv[step]);
                    float expected = strength == 0 ? (audio ? 1 : fmaxf(1-sv[step],.999f)) : 1 - sigma;
                    if (h3_dit_schedule_timestep(schedule,map[row]/3) != expected)
                        fprintf(stderr,"row mismatch: profile=%d max=%d step=%d audio=%d t=%zu class=%d strength=%.9g got=%.9g expected=%.9g\n",
                            type,maximum,step,audio,t,kind,(double)strength,
                            (double)h3_dit_schedule_timestep(schedule,map[row]/3),(double)expected);
                    CHECK(h3_dit_schedule_timestep(schedule,map[row]/3) == expected);
                    CHECK(map[row]%3 == (audio ? 2u : 0u));
                    if (strength == 0 || strength == 1) CHECK(map[row] == original[row] || strength == 1);
                    for (size_t column = 0; column < (audio ? 32u : 96u); column++)
                        CHECK((audio ? ap[local*32+column] : vp[local*96+column]) == (audio ? 3 : 2) * strength);
                    if (!audio) {
                        double relative = layout.positions[row].t - layout.positions[s->start].t;
                        CHECK(fabs(relative - (double)h3_video_time_boundary_frames((int)t)*5/3) < 1e-10);
                    }
                }
            }
            if (!maximum) CHECK(!memcmp(map, original, layout.seq_len * sizeof(*map)));
            /* At terminal zero, every fractional class reuses the generated
             * time row; no per-token modulation allocation is needed. */
            if (step == 5) for (int c = 4; c < H3_TARGET_ROW_CLASSES; c++) if (p.active[c])
                CHECK(h3_dit_schedule_class_row(schedule,step,(h3_target_row_class)c) == h3_dit_schedule_video_row(schedule,step));
        }
        CHECK(h3_dit_schedule_class_row(schedule,0,H3_TARGET_ROW_CLASSES) == UINT32_MAX);
        CHECK(!h3_bridge_mask_velocity(&p,1,4,6,93,vv,av));
        vv[16*24] = -0.0f; /* Generated video t=16, first channel. */
        av[92] = -0.0f;
        vv[11*24] = NAN; /* Exact endpoint must clear even nonfinite predictions. */
        av[64] = NAN;
        CHECK(h3_bridge_mask_velocity(&p,17,4,6,93,vv,av));
        CHECK(equal_float(vv[16*24],-0.0f) && equal_float(av[92],-0.0f));
        CHECK(equal_float(vv[11*24],0.0f) && equal_float(av[64],0.0f));
        CHECK(h3_bridge_row_class(&p,1,186,4,6,93) == H3_TARGET_ROW_CLASSES);
        h3_dit_schedule_free(schedule);
    }
    free(original); free(map); h3_layout_free(&layout); h3_dit_schedule_free(hard);
}

static void initialize(const h3_av_state *source, int context, int frames) {
    const h3_av_state_info *s = &source->info;
    h3_av_state *noise = h3_av_state_new(s->render_width,s->render_height,frames,43,s->compatibility); CHECK(noise);
    h3_rng rng;
    h3_rng_seed(&rng,43); h3_rng_fill_normal(&rng,noise->video,noise->info.video_elements);
    h3_rng_fill_normal(&rng,noise->audio,noise->info.audio_elements);
    h3_rng pristine_rng = rng;
    h3_av_state *pristine_source = h3_av_state_clone(source); CHECK(pristine_source);
    size_t hw = (size_t)s->latent_h * (size_t)s->latent_w;
    float maxima[] = {0,.5f,1}, sigmas[] = {0,.3f,1};
    for (int type = 0; type < 3; type++) for (int m = 0; m < 3; m++) for (int si = 0; si < 3; si++) {
        h3_bridge_profile p;
        CHECK(h3_bridge_profile_build(context,8,maxima[m],(h3_bridge_profile_type)type,&p,NULL,0,error,sizeof(error)));
        h3_av_state *target = h3_av_state_clone(noise), *hard = h3_av_state_clone(noise); CHECK(target && hard);
        CHECK(h3_av_state_insert_prefix(source,&hard->info,p.prefix,hard->video,hard->audio,1));
        float sigma_a = sigmas[2-si];
        CHECK(h3_av_state_insert_bridge(source,&target->info,&p,sigmas[si],sigma_a,target->video,target->audio));
        for (size_t c=0;c<64;c++) for (int t=0;t<target->info.audio_t;t++) {
            size_t dst=c*(size_t)target->info.audio_t+(size_t)t;
            float expected=noise->audio[dst];
            if (t<p.prefix.audio_prefix_t) {
                size_t src=c*(size_t)s->audio_t+(size_t)(s->audio_t-p.prefix.audio_prefix_t+t);
                float strength=p.class_mask[p.audio_classes[t]], effective=strength*sigma_a;
                if (strength==0 || effective==0) expected=source->audio[src];
                else if (effective==1) expected=noise->audio[dst];
                else {
                    volatile float clean=(1-effective)*source->audio[src], perturbation=effective*noise->audio[dst];
                    expected=clean+perturbation;
                }
            }
            CHECK(equal_float(target->audio[dst],expected));
        }
        for (size_t c = 0; c < 24; c++) for (int t = 0; t < target->info.video_t; t++) {
            size_t dst = (c*(size_t)target->info.video_t + (size_t)t)*hw;
            if (t >= p.prefix.video_prefix_t) {
                CHECK(!memcmp(target->video+dst,noise->video+dst,hw*4)); continue;
            }
            size_t src = (c*(size_t)s->video_t + (size_t)(s->video_t-p.prefix.video_prefix_t+t))*hw;
            float mask = p.class_mask[p.video_classes[t]];
            if (mask == 0) CHECK(!memcmp(target->video+dst,hard->video+dst,hw*4));
            for (size_t i = 0; i < hw; i++) {
                float expected;
                float sigma = mask * sigmas[si];
                if (mask == 0) {
                    volatile float a = .999f*source->video[src+i], b = .001f*noise->video[dst+i];
                    expected = a+b;
                } else if (sigma == 0) expected = source->video[src+i];
                else if (sigma == 1) expected = noise->video[dst+i];
                else {
                    volatile float a = (1-sigma)*source->video[src+i], b = sigma*noise->video[dst+i];
                    expected = a+b;
                }
                CHECK(equal_float(target->video[dst+i],expected));
            }
        }
        if (!m) {
            CHECK(!memcmp(target->video,hard->video,target->info.video_elements*4));
            CHECK(!memcmp(target->audio,hard->audio,target->info.audio_elements*4));
        }
        h3_av_state_info malformed = target->info; malformed.video_t = INT_MAX;
        CHECK(!h3_av_state_insert_bridge_video(source,&malformed,&p,1,target->video));
        CHECK(!h3_av_state_insert_bridge_video(source,&target->info,&p,NAN,target->video));
        CHECK(!h3_av_state_insert_bridge(source,&target->info,&p,1,NAN,target->video,target->audio));
        h3_av_state_free(target); h3_av_state_free(hard);
    }
    CHECK(!memcmp(source->video,pristine_source->video,s->video_elements*4));
    CHECK(!memcmp(source->audio,pristine_source->audio,s->audio_elements*4));
    CHECK(h3_rng_u32(&rng) == h3_rng_u32(&pristine_rng));
    h3_av_state_free(noise); h3_av_state_free(pristine_source);
}

static void initialization(void) {
    uint8_t signature[32] = {0};
    h3_av_state *source = h3_av_state_new(64,32,243,42,signature); CHECK(source);
    for (size_t i = 0; i < source->info.video_elements; i++) source->video[i] = (float)i/8192;
    for (size_t i = 0; i < source->info.audio_elements; i++) source->audio[i] = (float)i/32768;
    source->video[0] = -0.0f;
    initialize(source,39,56); initialize(source,90,141); initialize(source,192,243);
    h3_av_state_free(source);
    CHECK(equal_float(h3_flow_mix(-0.0f, NAN, 0), -0.0f));
    CHECK(equal_float(h3_flow_mix(NAN, -0.0f, 1), -0.0f));
    CHECK(h3_flow_mix(2,10,.25f) == 4);
    CHECK(isnan(h3_flow_mix(1,2,NAN)));
    CHECK(isnan(h3_flow_mix(1,2,-.1f)));
    CHECK(isnan(h3_flow_mix(1,2,1.1f)));
}

static void validation(void) {
    h3_ctx ctx = {0}; h3_params p = H3_PARAMS_DEFAULT;
    p.steps = 6;
    p.width = p.height = 256; p.continuation_mode = H3_CONTINUE_BRIDGE;
    CHECK(!h3_generate(&ctx,"test",&p) && strstr(ctx.error,"requires a continuation state"));
    uint8_t signature[32] = {0};
    h3_av_state *source = h3_av_state_new(256,256,56,42,signature); CHECK(source);
    p.continuation = source;
    p.denoise_reuse=4;
    CHECK(!h3_generate(&ctx,"test",&p) && strstr(ctx.error,"denoise reuse must be in [1, 3]"));
    p.denoise_reuse=1;
    for (int core=2;core<=5;core++) if (core!=4) {
        p.core_reuse=core;
        CHECK(!h3_generate(&ctx,"test",&p) && strstr(ctx.error,"supports --core-reuse 1, 4, or 6"));
    }
    p.core_reuse=1;
    h3_reference refs[13] = {{H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0}};
    p.references=refs; p.reference_count=13;
    // Validated combinations reach the next unrelated validation guard.
    for (int reuse=1;reuse<=3;reuse++) {
        p.denoise_reuse=reuse;
        CHECK(!h3_generate(&ctx,"test",&p) && strstr(ctx.error,"12 references"));
    }
    p.denoise_reuse=1;
    const int cores[]={1,4,6};
    for (size_t i=0;i<sizeof(cores)/sizeof(*cores);i++) {
        p.core_reuse=cores[i];
        CHECK(!h3_generate(&ctx,"test",&p) && strstr(ctx.error,"12 references"));
    }
    p.core_reuse=1; p.references=NULL; p.reference_count=0;
    p.bridge_video_steps = 12;
    CHECK(!h3_generate(&ctx,"test",&p) && strstr(ctx.error,"at least one exact"));
    p.bridge_video_steps = 8; p.bridge_max_strength = NAN;
    CHECK(!h3_generate(&ctx,"test",&p) && strstr(ctx.error,"finite"));
    p.continuation_mode = H3_CONTINUE_HARD;
    p.bridge_profile = (h3_bridge_profile_type)999; p.bridge_video_steps = -1;
    p.references = refs; p.reference_count = 13;
    CHECK(!h3_generate(&ctx,"test",&p) && strstr(ctx.error,"12 references"));
    p.continuation_mode = (h3_continuation_mode)2;
    CHECK(!h3_generate(&ctx,"test",&p) && strstr(ctx.error,"unknown continuation mode"));
    h3_av_state_free(source);
}

static void bridge_euler(void) {
    uint8_t signature[32]={0};
    h3_av_state *initial=h3_av_state_new(64,32,90,1,signature); CHECK(initial);
    for (size_t i=0;i<initial->info.video_elements;i++) initial->video[i]=(float)(i%47)/32;
    for (size_t i=0;i<initial->info.audio_elements;i++) initial->audio[i]=(float)(i%31)/32;
    h3_sigma_schedule sigmas; CHECK(h3_serving_schedule_build(20,&sigmas));
    for (int type=0;type<3;type++) for (int maximum=0;maximum<3;maximum++) {
        h3_bridge_profile p; CHECK(h3_bridge_profile_build(39,8,(float)maximum*.5f,
            (h3_bridge_profile_type)type,&p,NULL,0,error,sizeof(error)));
        h3_av_state *target=h3_av_state_clone(initial), *expected=h3_av_state_clone(initial);
        h3_av_state *velocity=h3_av_state_clone(initial), *scaled=h3_av_state_clone(initial);
        CHECK(target && expected && velocity && scaled);
        target->video[11*8]=expected->video[11*8]=initial->video[11*8]=-0.0f;
        target->audio[64]=expected->audio[64]=initial->audio[64]=-0.0f;
        for (int step=0;step<20;step++) {
            for (size_t i=0;i<velocity->info.video_elements;i++) velocity->video[i]=(float)(i%17+1)/16;
            for (size_t i=0;i<velocity->info.audio_elements;i++) velocity->audio[i]=-(float)(i%23+1)/16;
            for (int stream=0;stream<2;stream++) {
                int length=stream?150:27, prefix=stream?65:12;
                size_t spatial=stream?1:8;
                for (size_t c=0;c<(stream?64u:24u);c++) for (int t=0;t<length;t++) {
                    float strength=t<prefix?p.class_mask[(stream?p.audio_classes:p.video_classes)[t]]:1;
                    size_t offset=(c*(size_t)length+(size_t)t)*spatial;
                    float *want=(stream?expected->audio:expected->video)+offset;
                    float *v=(stream?scaled->audio:scaled->video)+offset;
                    const float *raw=(stream?velocity->audio:velocity->video)+offset;
                    for (size_t i=0;i<spatial;i++) v[i]=raw[i]*strength;
                    if (strength==0) continue;
                    CHECK(h3_euler_velocity_step(want,v,spatial,
                        stream?sigmas.audio[step]:sigmas.video[step],stream?sigmas.audio[step+1]:sigmas.video[step+1]));
                }
            }
            h3_bridge_step_stats stats;
            CHECK(h3_bridge_euler_step(&p,27,2,4,150,target->video,target->audio,
                velocity->video,velocity->audio,sigmas.video[step],sigmas.video[step+1],
                sigmas.audio[step],sigmas.audio[step+1],&stats));
            CHECK(!memcmp(target->video,expected->video,target->info.video_elements*4));
            CHECK(!memcmp(target->audio,expected->audio,target->info.audio_elements*4));
            CHECK(h3_bridge_check_exact(&p,27,2,4,150,target->video,target->audio,initial->video,initial->audio));
            for (int c=0;c<H3_TARGET_ROW_CLASSES;c++) if (stats.elements[c]) {
                double strength=p.class_mask[c];
                if (strength==0) CHECK(stats.changed[c]==0 && stats.update_square[c]==0 && stats.scaled_square[c]==0);
                else {
                    CHECK(stats.changed[c]>0);
                    CHECK(fabs(stats.scaled_square[c]/stats.raw_square[c]-strength*strength)<1e-7);
                }
            }
        }
        target->audio[64]=1;
        CHECK(!h3_bridge_check_exact(&p,27,2,4,150,target->video,target->audio,initial->video,initial->audio));
        target->audio[64]=initial->audio[64]; target->video[11*8]=1;
        CHECK(!h3_bridge_check_exact(&p,27,2,4,150,target->video,target->audio,initial->video,initial->audio));
        CHECK(!h3_bridge_euler_step(&p,27,2,4,150,target->video,target->audio,
            velocity->video,velocity->audio,1,.9f,.8f,.9f,NULL));
        h3_av_state_free(target); h3_av_state_free(expected); h3_av_state_free(velocity); h3_av_state_free(scaled);
    }
    h3_av_state_free(initial);
}

static void bridge_cache(void) {
    h3_params p=H3_PARAMS_DEFAULT;
    uint8_t signature[32]={0};
    h3_av_state *source=h3_av_state_new(64,32,90,42,signature); CHECK(source);
    h3_reference refs[]={{H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0},{H3_REFERENCE_IMAGE,"inputs/body1.jpg",NULL,0}};
    p.continuation=source; p.references=refs; p.reference_count=2;
    char *condition=h3_conditioning_key("prompt",&p,64,32,1); CHECK(condition);
    char *hard=h3_prepared_key(condition,&p,64,32); CHECK(hard);
    p.bridge_video_steps=11; p.bridge_max_strength=.65f; p.bridge_profile=H3_BRIDGE_LINEAR;
    char *key=h3_prepared_key(condition,&p,64,32); CHECK(key && !strcmp(hard,key)); free(key);
    p.continuation_mode=H3_CONTINUE_BRIDGE;
    char *bridge=h3_prepared_key(condition,&p,64,32); CHECK(bridge && strcmp(hard,bridge));
    key=h3_conditioning_key("prompt",&p,64,32,1); CHECK(key && !strcmp(condition,key)); free(key);
    p.bridge_video_steps=8; key=h3_prepared_key(condition,&p,64,32); CHECK(key && strcmp(bridge,key)); free(key);
    p.bridge_video_steps=11; p.bridge_max_strength=.5f; key=h3_prepared_key(condition,&p,64,32); CHECK(key && strcmp(bridge,key)); free(key);
    p.bridge_max_strength=.65f; p.bridge_profile=H3_BRIDGE_EASE_OUT; key=h3_prepared_key(condition,&p,64,32); CHECK(key && strcmp(bridge,key)); free(key);
    free(bridge); free(hard); free(condition); h3_av_state_free(source);
}

/* Encode the requested real image fixtures with the released VAE, then tile
 * each clean image latent through a temporal test state. This exercises actual
 * normalized latent values without pretending an image is a generated clip.
 * No decoded image/video is produced by this numerical fixture test. */
static void inputs(const char *model, const char *directory) {
    const char *names[] = {"face1", "body1", "face2", "body2", "12"};
    char weights[4096];
    CHECK(snprintf(weights,sizeof(weights),"%s/Ref2VA/video_vae/source",model) < (int)sizeof(weights));
    uint8_t signature[32];
    CHECK(h3_av_state_metadata_signature(model,1,signature,error,sizeof(error)));
    for (size_t file = 0; file < sizeof(names)/sizeof(*names); file++) {
        char path[4096], output[4096];
        CHECK(snprintf(path,sizeof(path),"inputs/%s.jpg",names[file]) < (int)sizeof(path));
        CHECK(snprintf(output,sizeof(output),"%s/%s.h3av",directory,names[file]) < (int)sizeof(output));
        float *pixels = NULL; h3_video_latent latent = {0};
        CHECK(h3_ffmpeg_read_image_f32(path,256,256,H3_IMAGE_FIT_COVER,&pixels,error,sizeof(error)));
        CHECK(h3_video_vae_encode(weights,"src/metal/shaders.metal",pixels,1,256,256,NULL,NULL,&latent,error,sizeof(error)));
        CHECK(latent.time == 1 && latent.height == 16 && latent.width == 16);
        for (size_t i = 0; i < 24*16*16; i++) CHECK(isfinite(latent.values[i]));
        h3_av_state *source = h3_av_state_new(256,256,90,42,signature); CHECK(source);
        for (size_t c = 0; c < 24; c++) for (int t = 0; t < source->info.video_t; t++)
            memcpy(source->video+(c*(size_t)source->info.video_t+(size_t)t)*256,latent.values+c*256,256*4);
        memset(source->audio,0,source->info.audio_elements*4);
        CHECK(h3_av_state_save(source,output,error,sizeof(error)));
        char metadata[4096];
        CHECK(snprintf(metadata,sizeof(metadata),"%s/%s.json",directory,names[file]) < (int)sizeof(metadata));
        FILE *meta = fopen(metadata,"w"); CHECK(meta);
        fprintf(meta,"{\"input\":\"%s\",\"fixture_kind\":\"encoded image latent repeated through time; zero audio latent; not a generated clip\",\"encoder_canvas\":[256,256],\"context_frames\":39,\"target_frames\":90,\"initialization_combinations\":27}\n",path);
        CHECK(!fclose(meta));
        h3_av_state *loaded = h3_av_state_load(output,error,sizeof(error)); CHECK(loaded);
        initialize(loaded,39,90);
        printf("ok: encoded %s; 27 profile/strength/sigma initialization checks, exact rows and suffix preserved\n",path);
        fflush(stdout);
        h3_av_state_free(loaded); h3_av_state_free(source); free(pixels); h3_video_latent_free(&latent);
    }
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1],"--inputs")) inputs(argv[2],argv[3]);
    else if (argc == 3 && !strcmp(argv[1],"--state")) {
        h3_av_state *source = h3_av_state_load(argv[2],error,sizeof(error)); CHECK(source);
        initialize(source,39,90); h3_av_state_free(source);
    } else if (argc == 1) { profiles(); row_maps(); initialization(); validation(); bridge_euler(); bridge_cache(); }
    else { fprintf(stderr,"usage: %s [--inputs MODEL EXISTING_OUTPUT_DIR | --state SOURCE.h3av]\n",argv[0]); return 2; }
    printf("ok: %zu bridge checks\n",checks);
    return 0;
}
