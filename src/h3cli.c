#include "src/log.h"
#include "src/denoise/attention.h"
#include "src/denoise/adaptive_cache.h"
#include "src/denoise/approximate.h"
#include "src/vae/image_vae.h"
#include "src/h3.h"
#include "src/weights/lora.h"
#include "src/weights/quant.h"
#include "src/media/delivery.h"
#include "src/device.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include "src/cli/cli_progress.h"
#include "src/sglang/sglang.h"
#include "src/host.h"
#include "src/cli/terminal.h"

#include <errno.h>
#include <getopt.h>
#include "src/cli/options.h"
#include "src/cli/dispatch.h"
#include "src/request.h"
#include "src/internal.h"
#include "src/server/server.h"
#include <setjmp.h>
#include "src/cli/option_ids.h"
#include "src/upscale/upscale.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>

#define DEFAULT_MODEL_DIR "models/MiniMaxH3"
#define DEFAULT_IMAGE_VAE "models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors"
#define DEFAULT_UPSCALE_MODEL "models/latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"

static void usage(const char *program) {
    fprintf(stderr,
        "Usage: %s [-d MODEL_DIR] -p PROMPT [-o OUTPUT] [options]\n"
        "       %s [-d MODEL_DIR] --info\n"
        "       %s [-d MODEL_DIR] --resume-sampler-state PATH [options]\n"
        "       %s [-d MODEL_DIR] --decode-av-state PATH [-o OUTPUT] [options]\n"
        "       %s --decode-still-latent PATH [--image-vae PATH] [-o OUTPUT] [options]\n\n",
        program, program, program, program, program);
    fputs(
        "Options:\n"
        "      --server           Start queued HTTP service (--server --help)\n"
        "      --models-path PATH Shared model root (default: models)\n"
        "      --download-models GROUPS Prefetch base,references,preview,image-vae,upscale,all\n"
        "      --list-models      Show pinned models and local availability\n"
        "      --offline          Disable model downloads (also H3_OFFLINE=1)\n"
        "                        Missing required models download automatically\n"
        "  -d, --model-dir PATH   Model directory (default: " DEFAULT_MODEL_DIR ")\n"
        "  -p, --prompt TEXT      Raw H3 prompt\n"
        "  -o, --output PATH      Output MP4 or still PNG (default: outputs/h3cli.mp4/.png)\n"
        "      --continue-from PATH  Resume a complete .h3av latent state\n"
        "      --continue-context N  Protected frames: 39 + 51*k (default: 39)\n"
        "      --continue-mode hard|bridge  Default: hard; CUDA uses CPU-state F32 Euler\n"
        "      --continue-bridge-steps N  Adaptable video latent rows (default: 8)\n"
        "      --continue-bridge-max-strength X  Fraction in [0,1] (default: 0.50)\n"
        "      --continue-bridge-profile NAME  stepped (default), linear, ease-out\n"
        "      --save-upscale-state PATH  Save portable clean source before decoding\n"
        "      --upscale-state PATH  Start exact 2x spatial upscaling from .h3up\n"
        "      --upscale-model PATH  Pinned BF16 latent upscaler safetensors\n"
        "                            Default: " DEFAULT_UPSCALE_MODEL "\n"
        "      --upscale-refine-steps N  0, 2, 3 or 4 (default: 4)\n"
        "      --upscale-noise X   Starting video sigma, 0 < X <= 0.5 (default: 0.25)\n"
        "      --upscale-seed N    Fresh video noise seed (default: source seed)\n"
        "      --upscale-import-sampler  Explicit completed text-only sampler import\n"
        "      --inspect-upscale-state PATH  Inspect source and 2x plan without models\n"
        "      --state-only        Save source/AV/checkpoint without VAEs or MP4\n"
        "      --save-av-state PATH  Save complete latents and presentation metadata\n"
        "      --stop-after-step N  Absolute completed transitions of the full schedule\n"
        "      --save-sampler-state PATH  Atomically save .h3sample state\n"
        "      --resume-sampler-state PATH  Resume serialized sampler state\n"
        "      --preview-on-stop  Decode a silent preview after saving checkpoint\n"
        "      --keep-continuation-prefix  Deliver protected history for debugging\n"
        , stderr);
    fputs(
        "      --width N          Output width (default: 864)\n"
        "      --height N         Output height (default: 480)\n"
        "      --render-width N   Lower internal model width (optional)\n"
        "      --render-height N  Lower internal model height (optional)\n"
        "      --frames N         Raw target frames, including continuation context (default: 56)\n"
        "      --seconds N        Requested duration at 24 fps (instead of --frames)\n"
        "      --steps N          Denoising passes (default: 20)\n"
        "      --quality LEVEL    lossless, extra-high, high, preview or fast-preview\n"
        "                        Preset defaults; explicit flags win in any order\n"
        "                        50/50/50/12/6 steps; high uses CUDA adaptive cache\n"
        "                        (Metal reuse 2); previews use reuse 2/3 and tiny VAE\n"
        "      --output-quality Q MP4 compression: default, maximum, high, medium, low\n"
        "                         Omitted: CRF 18; explicit default: SGLang CRF 25\n"
        "      --ffmpeg-crf N     H.264 CRF 0..51; overrides output-quality\n"
        "      --lossless-video   Preserve RGB24 frames exactly (H.264 RGB); audio AAC\n"
        "      --reuse N          Denoiser reuse: 1 close, 2 fast, 3 aggressive\n"
        "      --layers N         DiT blocks: 50 exact, 45 fast, 40 aggressive\n"
        "      --core-reuse N     Core refresh: 1 exact, 4 fast, 6 aggressive\n"
        "      --token-reduction  Pair video tokens in middle DiT blocks\n"
        "      --ssd-streaming    Stream original BF16 DiT layers from SSD\n"
        "      --use-int8-row-fc2 Faster one-scale int8 FC2 (M5)\n"
        "      --use-reference-rope  Disable native 256 RoPE adaptation\n"
        "      --use-slower-bf16-mlp  Force close-reference BF16/MPS MLP\n"
        "      --use-slower-bf16-qkv  Force close-reference BF16 QKV\n"
        "      --use-slower-bf16-attention-output  Force BF16 attention output\n"
        "      --use-slower-row-major-attention-output  Restore SDPA transpose\n"
        "      --use-slower-unfused-int8-inputs  Keep standalone quantizers\n"
        "      --use-slower-unfused-qkv-rope  Keep separate Q/K norm/RoPE\n"
        "      --use-slower-scalar-qkv-rms  Force scalar Q/K RMS loads\n"
        "      --use-slower-uncached-int8-scales  Reread projection scales\n"
        "      --use-slower-dynamic-fc1-k  Use runtime-bound FC1 K loop\n"
        "      --use-slower-grouped-quantizer  Force 256-thread FC2 quantizer\n"
        "      --seed N           Random seed (default: 42)\n"
        "      --first-frame PATH First-frame image (aspect-cover scale, center-crop)\n"
        "      --last-frame PATH  Last-frame image (aspect-cover scale, center-crop)\n"
        "      --ref-image PATH    Append an ordered Ref2VA image\n"
        "      --ref-image-size S  match (default), high (long edge 2048), max (short edge 2048)\n"
        "      --ref-video PATH    Append video, including embedded audio\n"
        "      --ref-silent-video PATH  Append video without its audio\n"
        "      --ref-video-audio VIDEO AUDIO  Append video + soundtrack\n"
        "      Ref2VA videos use released-v1 preprocessing and VAE encoding by default\n"
        "      --ref-audio PATH    Append an ordered standalone audio clip\n"
        "      --frames-dir PATH  Write generated frames as PPM files\n"
        "      --show             Display a frame after every denoising step (M5)\n"
        "      --zoom N           Terminal image zoom (default: 2 for Retina)\n"
        "      --verbose          Include backend, allocation and execution diagnostics\n"
        "      --profile          Include diagnostics and instrument GPU timings\n"
        "      --info             Inspect model/device without mapping weights\n"
        "      --cuda-device N    Select Linux CUDA device (default: 0)\n"
        "      --cuda-weight-mode auto|resident|stream\n"
        "                            auto: full or partial BF16 residency as capacity permits\n"
        "                            resident: require the full core; stream: retain no core blocks\n"
        "  -h, --help             Show this help\n",
        stderr);
    fputs("      --backend MODE     mpsgraph (default) or metal hybrid\n"
        "      --metal-attention MODE  dense (default) or sol\n"
        "      --metal-attention-kernel NAME  steel (default) or steel-routed\n"
        "      --metal-attention-dtype NAME   bf16 (default) or fp16 (steel-routed, dense)\n"
        "      --metal-tier NAME              diagnostic, reference, preview\n"
        "      --metal-attention-layout MODE  adapter (default) or fused (FP16 only)\n"
        "      --metal-ane MODE   off (default), serial, static or dynamic; QKV split\n"
        "      --metal-ane-rows N  Maximum ANE rows per QKV (default 4096; <=16384)\n"
        "      --metal-ane-chunk N  Compiled row chunk: 256, 512 (default), 1024\n"
        "      --metal-weight-format F  bf16 (default) or weight-only q8\n"
        "      --metal-q8-kernel K  mpsgraph (bounded dequant, default) or simdgroup\n"
        "      --sol-q-block N    32 (default) or 64 query rows\n"
        "      --sol-kv-block N   32, 64 (default), or 128 key/value rows\n"
        "      --sol-tau F        Threshold standard deviations (default 1)\n"
        "      --sol-dense-layers N  Initial exact layers (default 1)\n"
        "      --sol-dense-steps N  Dense evaluations [0,N) (Metal 0, CUDA 1)\n"
        "      --sol-dense-sigma F  Dense when max(video,audio sigma) >= F (-1 disables)\n"
        "      --sol-local-radius N  Exact temporal radius in latent video frames (default 1)\n"
        "      --sol-min-exact F  Minimum exact block fraction (default 0 on Metal/CUDA)\n"
        "      --save-conditioning PATH  Persist reusable conditioning (.h3cond)\n"
        "      --load-conditioning PATH  Restore strictly matching conditioning\n"
        "      --conditioning-schedule   Also cache exact-schedule AdaLN tensors\n"
        "      --lora PATH[:SCALE]    Fold ordered adapters at runtime (repeatable; scale 1)\n"
        "      --lora-cache DIR       Cache (XDG_CACHE_HOME/h3/lora or ~/.cache/h3/lora)\n"
        "      --lora-memory-mib N    Folding scratch budget (default 512 MiB)\n"
        "      --preview-vae      Approximate TAEH3 video decoding (Metal/CUDA)\n"
        "      --no-preview-vae   Use the full VAE, including with preview quality\n"
        "      --preview-vae-model PATH  Optional tiny model (default: models/preview-vae/taeh3.safetensors)\n"
        "      --decode-still-latent PATH  Decode one normalized .safetensors latent to PNG\n"
        "      --image-vae PATH    full_decoder_v1 image VAE checkpoint\n"
        "                            Default: " DEFAULT_IMAGE_VAE "\n"
        "      --still             Single-image generation, PNG output\n"
        "      --save-still-latent PATH  Save normalized still .safetensors latent\n"
        "      --decode-av-state PATH  Decode saved final latents without sampling\n"
          "                              Ordinary CUDA BF16 video defaults to SGLang\n"
          "      --adaptive-cache MODE      off (default), conservative or aggressive\n"
          "                                BF16 CUDA supports hard/bridge continuation and references\n"
          "      --adaptive-cache-threshold T  Strict score threshold 0..1 (defaults 0.04 / 0.08)\n"
          "      --adaptive-cache-max-hits N   Consecutive hit ceiling 1..16 (defaults 1 / 3)\n"
          "      --adaptive-cache-max-mib N  Cache ceiling in MiB (default 4096; resume restores saved)\n"
          "      --adaptive-cache-warmup N  Exact initial steps: 2..16, <= steps-2 (default 4)\n"
          "      --subblock-warmup N        Dense initial steps: 2..16, <= steps-2 (default 10)\n"
          "      --subblock-sparsity F       Drop fraction [0,1), default 0.75; requires subblock\n"
          "      --cuda-attention MODE      default, sage2++, sage3, sol or subblock; main CUDA DiT only\n"
          "      --cuda-denoise-quant MODE  off (default), fp8 or nvfp4; CUDA DiT only\n"
          "      --cuda-denoise-quant-cache DIR  Packed denoiser artifact cache\n"
          "      --cuda-denoise-quant-verify  Strict weight hashes (slow; default metadata)\n"
          "      CUDA attention/precision options use the shared SGLang pipeline\n",stderr);
}

static _Thread_local jmp_buf cli_parse_failure;
static _Thread_local const h3_cli_hooks *cli_hooks;
static _Noreturn void cli_bad_argument(void) { longjmp(cli_parse_failure, 1); }

static int parse_int(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(value, &end, 10);
    if (errno || end == value || *end || parsed < 0 || parsed > INT32_MAX) {
        fprintf(stderr, "h3cli: invalid %s: %s\n", label, value);
        cli_bad_argument();
    }
    return (int)parsed;
}

static float parse_strength(const char *value) {
    char *end = NULL;
    errno = 0;
    float parsed = strtof(value, &end);
    if (errno || end == value || *end || !isfinite(parsed) || parsed < 0 || parsed > 1) {
        fprintf(stderr, "h3cli: invalid bridge maximum strength (expected [0,1]): %s\n", value);
        cli_bad_argument();
    }
    return parsed;
}

static int frames_from_seconds(const char *value) {
    char *end = NULL;
    errno = 0;
    double seconds = strtod(value, &end);
    double frames = seconds * (double)H3_FPS;
    if (errno || !end || *end || !isfinite(seconds) || seconds <= 0.0 ||
        !isfinite(frames) || frames > (double)INT32_MAX) {
        fprintf(stderr, "h3cli: invalid seconds: %s\n", value);
        cli_bad_argument();
    }
    long long rounded = llround(frames);
    if (rounded < 1 || rounded > INT32_MAX) {
        fprintf(stderr, "h3cli: invalid seconds: %s\n", value);
        cli_bad_argument();
    }
    return (int)rounded;
}

static uint64_t parse_u64(const char *value, const char *label) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "h3cli: invalid %s: %s\n", label, value);
        cli_bad_argument();
    }
    return (uint64_t)parsed;
}

static h3_reference *append_reference(h3_reference references[12],
                                      size_t *count) {
    if (*count >= 12) {
        fprintf(stderr, "h3cli: Ref2VA supports at most 12 references\n");
        cli_bad_argument();
    }
    h3_reference *reference = &references[(*count)++];
    memset(reference, 0, sizeof(*reference));
    return reference;
}

static double gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void print_component(const char *label, const h3_component_info *item) {
    printf("  %-18s %2zu files  %4zu tensors  %7.3f GiB\n",
           label, item->files, item->tensors, gib(item->tensor_bytes));
}

static void print_info(const h3_ctx *ctx) {
    const h3_device_info *device = h3_device(ctx);
    const h3_model_info *model = h3_model(ctx);
    printf("h3cli %s (%s)\n", H3_VERSION, device->backend);
    printf("Device: %s (%s)\n", device->name, device->architecture);
    printf("  physical memory       %.1f GiB\n", gib(device->physical_memory));
    if (!strcmp(device->backend, "cuda")) {
        printf("  CUDA device           %d\n", device->device_index);
        printf("  compute capability    %d.%d\n", device->cuda_compute_major, device->cuda_compute_minor);
        printf("  VRAM total/free       %.1f / %.1f GiB\n", gib(device->device_memory), gib(device->free_device_memory));
        printf("  multiprocessors       %d\n", device->multiprocessor_count);
        printf("  CUDA runtime/driver   %d / %d\n", device->cuda_runtime_version, device->cuda_driver_version);
    } else {
    printf("  recommended GPU set   %.1f GiB\n", gib(device->recommended_working_set));
    printf("  max Metal buffer      %.1f GiB\n", gib(device->max_buffer_length));
    printf("  Apple GPU family      %d\n", device->apple_gpu_family);
    printf("  Metal 4               %s\n", device->metal4 ? "yes" : "no");
    printf("  unified memory        %s\n", device->unified_memory ? "yes" : "no");
    }
    printf("Native checkpoint inventory (header-only):\n");
    print_component("Qwen3-VL encoder", &model->text_encoder);
    print_component("FL2VA DiT", &model->fl2va_transformer);
    print_component("Ref2VA DiT", &model->ref2va_transformer);
    print_component("video VAE", &model->video_vae);
    print_component("audio VAE", &model->audio_vae);
}

typedef struct {
    h3_cli_progress_state progress;
    h3_terminal_protocol terminal;
    int display_failed;
    const char *frames_dir;
    int frame_write_failed;
} cli_state;

static int cli_progress(const char *phase, int completed, int total,
                        void *opaque) {
    cli_state *state = opaque;
    if(cli_hooks&&cli_hooks->progress&&cli_hooks->progress(phase,completed,total,cli_hooks->opaque))return 1;
    state->progress.phase_timing=h3_log_verbose();
    h3_cli_progress_update(&state->progress, stderr, phase, completed, total);
    return 0;
}

static void cli_log_line(void *opaque) {
    cli_state *state = opaque;
    h3_cli_progress_finish(&state->progress, stderr);
}

static int cli_frame(const h3_frame *frame, void *opaque) {
    cli_state *state = opaque;
    if(cli_hooks&&cli_hooks->frame&&cli_hooks->frame(frame,cli_hooks->opaque))return 1;
    int preview = frame->denoise_step >= 0;
    if (!preview && state->frames_dir && !state->frame_write_failed) {
        char path[1024];
        int length = snprintf(path, sizeof(path), "%s/frame-%04d.ppm",
                              state->frames_dir, frame->frame_index);
        FILE *output = length > 0 && (size_t)length < sizeof(path) ?
            fopen(path, "wb") : NULL;
        if (!output ||
            fprintf(output, "P6\n%d %d\n255\n", frame->width,
                    frame->height) < 0) {
            fprintf(stderr, "h3cli: cannot write frame %d to %s\n",
                    frame->frame_index, state->frames_dir);
            if (output) fclose(output);
            state->frame_write_failed = 1;
        } else {
            size_t row_bytes = (size_t)frame->width * 3;
            for (int row = 0; row < frame->height; row++) {
                if (fwrite(frame->rgb + (size_t)row * frame->stride, 1,
                           row_bytes, output) != row_bytes) {
                    state->frame_write_failed = 1;
                    break;
                }
            }
            if (fclose(output) != 0) state->frame_write_failed = 1;
            if (state->frame_write_failed)
                fprintf(stderr, "h3cli: incomplete frame %d in %s\n",
                        frame->frame_index, state->frames_dir);
        }
    }
    if (state->frame_write_failed) return 1;
    if (state->display_failed || state->terminal == H3_TERM_NONE) return 0;
    h3_cli_progress_finish(&state->progress, stderr);
    if (preview)
        fprintf(stderr,
                "h3cli: denoise preview %d/%d, video frame %d/%d via %s\n",
                frame->denoise_step + 1, frame->denoise_steps,
                frame->frame_index + 1, frame->frame_count,
                h3_terminal_protocol_name(state->terminal));
    else
        fprintf(stderr, "h3cli: frame %d/%d via %s\n", frame->frame_index + 1,
                frame->frame_count,
                h3_terminal_protocol_name(state->terminal));
    char error[256];
    if (!h3_terminal_display_rgb24(state->terminal, frame->rgb,
                                   frame->width, frame->height, frame->stride,
                                   error, sizeof(error))) {
        fprintf(stderr, "h3cli: terminal display disabled: %s\n", error);
        state->display_failed = 1;
    }
    return 0;
}

static _Thread_local const char **cli_lora_args;
/* CLI shorthand only: saved state and the engine retain the expanded controls.
 * Keep lossless and extra-high identical, as in SGLang's current H3 policy.
 * Native adaptive cache is CUDA-only; Metal high uses scheduled reuse instead. */
static int parse_quality(const char *value) {
    for(size_t i=0;i<sizeof(h3_quality_presets)/sizeof(h3_quality_presets[0]);i++)
        if(!strcmp(value,h3_quality_presets[i].name))return (int)i;
    fprintf(stderr,"h3cli: --quality must be lossless, extra-high, high, preview or fast-preview\n");
    cli_bad_argument();
}
static float parse_metal_float(const char *value, const char *label) {
    char *end=NULL;errno=0;float result=strtof(value,&end);
    if(errno||!end||end==value||*end||!isfinite(result)) {
        fprintf(stderr,"h3cli: invalid %s\n",label);cli_bad_argument();
    }
    return result;
}
static double model_preparation_seconds;
static int cli_quality_reported;
static int cli_execute(int argc, char **argv) {
    struct timespec operation_begin;
    clock_gettime(CLOCK_MONOTONIC, &operation_begin);
    int report_wall_time = 0;
    const char *model_dir = DEFAULT_MODEL_DIR,*upscale_path=NULL,*inspect_upscale=NULL;
    h3_upscale_options upscale=H3_UPSCALE_OPTIONS_DEFAULT;
    int upscale_given=0,upscale_import=0;
    h3_lora_options lora={0};
    cli_lora_args=calloc((size_t)argc,sizeof(*cli_lora_args));
    if(!cli_lora_args){fprintf(stderr,"h3cli: out of memory\n");return 1;}
    lora.adapters=cli_lora_args;
    const char *continue_from = NULL, *save_av_state = NULL, *decode_av_state = NULL;
    const char *decode_still = NULL, *image_vae = NULL;

    int context_given = 0;
    const char *prompt = NULL;
    const char *output = "outputs/h3cli.mp4";
    int output_given=0;
    h3_params params = H3_PARAMS_DEFAULT;
    h3_reference references[12];
    size_t reference_count = 0;
    /* Sample runners and tee forward pipes to a terminal. Keep carriage-return
     * updates through those pipes instead of printing a new row per update. */
    cli_state cli = {.progress = {.completed = -1, .total = -1, .terminal = 1}};
    h3_log_line_callback(cli_log_line, &cli);
    int show = 0;
    int profile = 0;
    int info = 0;
    int frames_given = 0;
    int seconds_given = 0;
    int option;
    int generation_given = 0, still_conflict_given = 0, quant_verify_given = 0;
    int quality = -1, steps_given = 0, reuse_given = 0, preview_vae_given = 0;
    while ((option = getopt_long(argc, argv, "d:p:o:h", h3_cli_getopt_options, NULL)) != -1) {
        if(option==OPT_CONTINUE_MODE || option==OPT_BRIDGE_STEPS || option==OPT_BRIDGE_STRENGTH || option==OPT_BRIDGE_PROFILE || option==OPT_KEEP_CONTINUATION_PREFIX)still_conflict_given=1;
        switch (option) {
            case OPT_UPSCALE_STATE: case OPT_UPSCALE_MODEL: case OPT_UPSCALE_STEPS: case OPT_UPSCALE_NOISE:
            case OPT_UPSCALE_SEED: case OPT_UPSCALE_IMPORT: case OPT_UPSCALE_INSPECT: case OPT_STATE_ONLY:
            case OPT_DECODE_STILL: case OPT_IMAGE_VAE: case 'd': case 'o': case 'h': case OPT_STOP_AFTER: case OPT_SAVE_SAMPLER:
            case OPT_RESUME_SAMPLER: case OPT_PREVIEW_STOP: case OPT_FRAMES_DIR:
            case OPT_SHOW: case OPT_ZOOM: case OPT_PROFILE: case OPT_VERBOSE: case OPT_INFO: case OPT_SAVE_AV_STATE:
            case OPT_CUDA_DEVICE: case OPT_CUDA_WEIGHT_MODE:
            case OPT_PREVIEW_VAE: case OPT_NO_PREVIEW_VAE: case OPT_PREVIEW_VAE_MODEL: case OPT_DECODE_AV_STATE:
            case OPT_OUTPUT_QUALITY: case OPT_FFMPEG_CRF: case OPT_LOSSLESS_VIDEO:
            case OPT_DENOISE_QUANT: case OPT_DENOISE_QUANT_CACHE: case OPT_DENOISE_QUANT_VERIFY:
            case OPT_ADAPTIVE_CACHE: case OPT_ADAPTIVE_THRESHOLD: case OPT_ADAPTIVE_MAX_HITS: case OPT_ADAPTIVE_BUDGET: case OPT_ADAPTIVE_WARMUP: case OPT_SUBBLOCK_WARMUP:
            case OPT_SUBBLOCK_SPARSITY: case OPT_CUDA_ATTENTION:
            case OPT_LORA: case OPT_LORA_CACHE: case OPT_LORA_MEMORY: break;
            default: generation_given = 1; break;
        }
        switch (option) {
            case OPT_LORA: cli_lora_args[lora.count++]=optarg; break;
            case OPT_LORA_CACHE: lora.cache_dir=optarg; break;
            case OPT_LORA_MEMORY: {
                char *end;errno=0;unsigned long long mib=strtoull(optarg,&end,10);
                int digits=*optarg!=0;for(const char *p=optarg;*p;p++)if(*p<'0'||*p>'9')digits=0;
                if(!digits||errno||*end||!mib||mib>SIZE_MAX/(1024*1024)){
                    fprintf(stderr,"h3cli: --lora-memory-mib requires a positive integer\n");return 2;
                }
                lora.memory_mib=(size_t)mib;break;
            }
            case OPT_QUALITY: quality = parse_quality(optarg); break;
            case OPT_OUTPUT_QUALITY:
                if(!h3_output_quality_parse(optarg,&params.output_encoding.quality)) {
                    fprintf(stderr,"h3cli: output-quality must be default, maximum, high, medium or low\n");return 2;
                }
                break;
            case OPT_FFMPEG_CRF:
                params.output_encoding.crf=parse_int(optarg,"ffmpeg-crf");params.output_encoding.crf_set=1;break;
            case OPT_LOSSLESS_VIDEO: params.output_encoding.lossless_video=1;break;
            case OPT_PREVIEW_VAE: params.preview_vae = 1; preview_vae_given = 1; break;
            case OPT_NO_PREVIEW_VAE: params.preview_vae = 0; preview_vae_given = 1; break;
            case OPT_BACKEND:
                if(!h3_backend_parse(optarg,&params.backend)) {
                    fprintf(stderr,"h3cli: backend must be mpsgraph or metal\n");return 2;
                }
                params.backend_set|=1;break;
            case OPT_METAL_ATTN_KERNEL:
                if(!strcmp(optarg,"steel"))params.metal_attention.candidate=0;
                else if(!strcmp(optarg,"steel-routed"))params.metal_attention.candidate=1;
                else {fprintf(stderr,"h3cli: Metal attention kernel must be steel or steel-routed\n");return 2;}
                params.backend_set|=8;break;
            case OPT_METAL_ATTN_DTYPE:
                if(!strcmp(optarg,"bf16"))params.metal_attention.precision=0;
                else if(!strcmp(optarg,"fp16"))params.metal_attention.precision=1;
                else {fprintf(stderr,"h3cli: Metal attention dtype must be bf16 or fp16\n");return 2;}
                params.backend_set|=8;break;
            case OPT_METAL_TIER:
                if(!strcmp(optarg,"diagnostic"))params.metal_attention.tier=0;
                else if(!strcmp(optarg,"reference"))params.metal_attention.tier=1;
                else if(!strcmp(optarg,"preview"))params.metal_attention.tier=2;
                else {fprintf(stderr,"h3cli: Metal tier must be diagnostic, reference or preview\n");return 2;}
                params.backend_set|=8;break;
            case OPT_METAL_LAYOUT:
                if(!strcmp(optarg,"adapter"))params.metal_attention.layout_fusion=0;
                else if(!strcmp(optarg,"fused"))params.metal_attention.layout_fusion=1;
                else {fprintf(stderr,"h3cli: Metal attention layout must be adapter or fused\n");return 2;}
                params.backend_set|=8;break;
            case OPT_METAL_Q8_KERNEL:
                if(!strcmp(optarg,"mpsgraph"))params.metal_attention.q8_kernel=0;
                else if(!strcmp(optarg,"simdgroup"))params.metal_attention.q8_kernel=1;
                else {fprintf(stderr,"h3cli: metal-q8-kernel must be mpsgraph or simdgroup\n");return 2;}
                params.backend_set|=8;break;
            case OPT_METAL_WEIGHT_FORMAT:
                if(!h3_weight_format_parse(optarg,&params.metal_attention.weight_format)) {
                    fprintf(stderr,"h3cli: metal-weight-format must be bf16 or q8\n");return 2;
                }
                params.backend_set|=8;break;
            case OPT_METAL_ANE:
                if(!strcmp(optarg,"off"))params.metal_attention.ane_mode=0;
                else if(!strcmp(optarg,"serial"))params.metal_attention.ane_mode=1;
                else if(!strcmp(optarg,"static"))params.metal_attention.ane_mode=2;
                else if(!strcmp(optarg,"dynamic"))params.metal_attention.ane_mode=3;
                else {fprintf(stderr,"h3cli: Metal ANE must be off, serial, static or dynamic\n");return 2;}
                params.backend_set|=8;break;
            case OPT_METAL_ANE_ROWS:
                params.metal_attention.ane_rows=parse_int(optarg,"Metal ANE rows");params.backend_set|=8;break;
            case OPT_METAL_ANE_CHUNK:
                params.metal_attention.ane_chunk=parse_int(optarg,"Metal ANE chunk");params.backend_set|=8;break;
            case OPT_SOL_Q_BLOCK:
                params.metal_attention.q_block=parse_int(optarg,"SOL Q block");params.cuda_sol_set|=H3_SOL_Q;break;
            case OPT_SOL_KV_BLOCK:
                params.metal_attention.kv_block=parse_int(optarg,"SOL K/V block");params.cuda_sol_set|=H3_SOL_KV;break;
            case OPT_SOL_TAU:
                params.metal_attention.tau=parse_metal_float(optarg,"SOL tau");params.cuda_sol_set|=H3_SOL_TAU;break;
            case OPT_SOL_DENSE_LAYERS:
                params.metal_attention.dense_layers=parse_int(optarg,"SOL dense layers");params.cuda_sol_set|=H3_SOL_LAYERS;break;
            case OPT_SOL_DENSE_STEPS:
                params.metal_attention.dense_steps=parse_int(optarg,"SOL dense steps");params.cuda_sol_set|=H3_SOL_STEPS;break;
            case OPT_SOL_DENSE_SIGMA:
                params.metal_attention.dense_sigma=parse_metal_float(optarg,"SOL dense sigma");params.cuda_sol_set|=H3_SOL_SIGMA;break;
            case OPT_SOL_LOCAL_RADIUS:
                params.metal_attention.local_radius=parse_int(optarg,"SOL local radius");params.cuda_sol_set|=H3_SOL_RADIUS;break;
            case OPT_SOL_MIN_EXACT:
                params.metal_attention.min_exact=parse_metal_float(optarg,"SOL minimum exact fraction");params.cuda_sol_set|=H3_SOL_MIN;break;
            case OPT_METAL_ATTENTION_MODE:
                if(!h3_attention_mode_parse(optarg,&params.attention_mode)) {
                    fprintf(stderr,"h3cli: --metal-attention must be dense or sol\n");return 2;
                }
                params.backend_set|=2;break;
            case OPT_SAVE_CONDITIONING: params.save_conditioning=optarg;break;
            case OPT_LOAD_CONDITIONING: params.load_conditioning=optarg;break;
            case OPT_CONDITIONING_SCHEDULE: params.conditioning_schedule=1;break;
            case OPT_PREVIEW_VAE_MODEL: params.preview_vae_model = optarg; break;
            case OPT_DECODE_STILL: decode_still=optarg; break;
            case OPT_IMAGE_VAE: image_vae=optarg; params.image_vae=optarg; break;
            case OPT_STILL: params.still=1; break;
            case OPT_SAVE_STILL: params.save_still_latent=optarg; break;
            case OPT_DECODE_AV_STATE: decode_av_state = optarg; break;
                        case OPT_ADAPTIVE_CACHE:
                if(!h3_adaptive_parse(optarg,&params.adaptive_cache)){fprintf(stderr,"h3cli: adaptive-cache must be off, conservative or aggressive\n");return 2;}
                params.adaptive_cache_set=1;break;
            case OPT_ADAPTIVE_THRESHOLD:
                if(!h3_adaptive_threshold_parse(optarg,&params.adaptive_cache_threshold)) {
                    fprintf(stderr,"h3cli: adaptive-cache-threshold must be a finite decimal FP32 value in [0,1]\n");return 2;
                }
                params.adaptive_cache_threshold_set=1;break;
            case OPT_ADAPTIVE_MAX_HITS:
                if(!h3_adaptive_max_hits_parse(optarg,&params.adaptive_cache_max_hits)) {
                    fprintf(stderr,"h3cli: adaptive-cache-max-hits must be a decimal integer in [1,16]\n");return 2;
                }
                params.adaptive_cache_max_hits_set=1;break;
            case OPT_ADAPTIVE_BUDGET:
                if(!h3_adaptive_mib_parse(optarg,&params.adaptive_cache_max_bytes)) {
                    fprintf(stderr,"h3cli: adaptive-cache-max-mib must be a positive decimal integer fitting platform size\n");return 2;
                }
                params.adaptive_cache_max_bytes_set=1;break;
            case OPT_ADAPTIVE_WARMUP:
                params.adaptive_cache_warmup=parse_int(optarg,"adaptive-cache-warmup");
                params.adaptive_cache_warmup_set=1;break;
            case OPT_SUBBLOCK_WARMUP:
                params.subblock_warmup=parse_int(optarg,"subblock-warmup");
                params.subblock_warmup_set=1;break;
            case OPT_SUBBLOCK_SPARSITY:
                params.subblock_sparsity=parse_metal_float(optarg,"SubBlock sparsity");
                params.subblock_sparsity_set=1;break;
            case OPT_CUDA_ATTENTION:
                if(!h3_attention_parse(optarg,&params.cuda_attention)) {
                    fprintf(stderr,"h3cli: cuda-attention must be default, sage2++, sage3, sol or subblock\n");return 2;
                }
                params.cuda_attention_set=1;break;
            case OPT_DENOISE_QUANT:
                if(!h3_quant_parse(optarg,&params.cuda_denoise_quant)) {
                    fprintf(stderr,"h3cli: cuda-denoise-quant must be off, fp8 or nvfp4\n");return 2;
                }
                params.cuda_denoise_quant_set=1;break;
            case OPT_DENOISE_QUANT_CACHE: params.cuda_denoise_quant_cache=optarg;break;
            case OPT_DENOISE_QUANT_VERIFY:
                if(setenv("H3_QUANT_VERIFY","1",1)){perror("h3cli: quantization verification option");return 1;}
                quant_verify_given=1;
                break;
            case OPT_CUDA_DEVICE: case OPT_CUDA_WEIGHT_MODE: {
                char detail[256];
                if (!h3_device_configure(option == OPT_CUDA_DEVICE ? optarg : NULL,
                        option == OPT_CUDA_WEIGHT_MODE ? optarg : NULL, detail, sizeof(detail))) {
                    fprintf(stderr, "h3cli: %s\n", detail); return 1;
                }
                break;
            }
            case OPT_MODELS_PATH: case OPT_DOWNLOAD_MODELS: case OPT_LIST_MODELS: case OPT_OFFLINE:
                break; /* Shared orchestration handles these before execution. */
            case 'd': model_dir = optarg; break;
            case 'p': prompt = optarg; break;
            case 'o': output = optarg; output_given=1; break;
            case 'h': usage(argv[0]); return 0;
            case OPT_WIDTH: params.width = parse_int(optarg, "width"); break;
            case OPT_HEIGHT: params.height = parse_int(optarg, "height"); break;
            case OPT_RENDER_WIDTH:
                params.render_width = parse_int(optarg, "render width");
                break;
            case OPT_RENDER_HEIGHT:
                params.render_height = parse_int(optarg, "render height");
                break;
            case OPT_FRAMES:
                params.frames = parse_int(optarg, "frames");
                frames_given = 1;
                break;
            case OPT_SECONDS:
                params.frames = frames_from_seconds(optarg);
                seconds_given = 1;
                break;
            case OPT_STOP_AFTER: params.stop_after_step=parse_int(optarg,"stop-after-step"); break;
            case OPT_SAVE_SAMPLER: params.save_sampler_state=optarg; break;
            case OPT_RESUME_SAMPLER: params.resume_sampler_state=optarg; break;
            case OPT_PREVIEW_STOP: params.preview_on_stop=1; break;
            case OPT_STEPS: params.steps = parse_int(optarg, "steps"); steps_given = 1; break;
            case OPT_REUSE:
                params.denoise_reuse = parse_int(optarg, "reuse");
                reuse_given = 1;
                break;
            case OPT_LAYERS:
                params.dit_layers = parse_int(optarg, "layers");
                break;
            case OPT_CORE_REUSE:
                params.core_reuse = parse_int(optarg, "core reuse");
                break;
            case OPT_CONTINUE_FROM: continue_from = optarg; break;
            case OPT_CONTINUE_MODE:
                if (!strcmp(optarg, "hard")) params.continuation_mode = H3_CONTINUE_HARD;
                else if (!strcmp(optarg, "bridge")) params.continuation_mode = H3_CONTINUE_BRIDGE;
                else { fprintf(stderr, "h3cli: invalid continuation mode: %s (expected hard or bridge)\n", optarg); return 2; }
                break;
            case OPT_BRIDGE_STEPS:
                if (!*optarg) { fprintf(stderr, "h3cli: bridge steps cannot be empty\n"); return 2; }
                params.bridge_video_steps = parse_int(optarg, "bridge steps"); break;
            case OPT_BRIDGE_STRENGTH: params.bridge_max_strength = parse_strength(optarg); break;
            case OPT_BRIDGE_PROFILE:
                if (!strcmp(optarg, "stepped")) params.bridge_profile = H3_BRIDGE_STEPPED;
                else if (!strcmp(optarg, "linear")) params.bridge_profile = H3_BRIDGE_LINEAR;
                else if (!strcmp(optarg, "ease-out")) params.bridge_profile = H3_BRIDGE_EASE_OUT;
                else { fprintf(stderr, "h3cli: invalid bridge profile: %s (expected stepped, linear or ease-out)\n", optarg); return 2; }
                break;
            case OPT_CONTINUE_CONTEXT:
                params.continuation_context_frames = parse_int(optarg,"continuation context");
                context_given = 1; break;
            case OPT_UPSCALE_STATE:upscale_path=optarg;break;
            case OPT_UPSCALE_INSPECT:inspect_upscale=optarg;break;
            case OPT_UPSCALE_MODEL:upscale.model_path=optarg;upscale_given=1;break;
            case OPT_UPSCALE_STEPS:upscale.refine_steps=parse_int(optarg,"upscale refinement steps");upscale_given=1;break;
            case OPT_UPSCALE_NOISE:upscale.sigma=parse_metal_float(optarg,"upscale sigma");upscale.sigma_set=1;upscale_given=1;break;
            case OPT_UPSCALE_SEED:upscale.seed=parse_u64(optarg,"upscale seed");upscale.seed_set=1;upscale_given=1;break;
            case OPT_UPSCALE_IMPORT:upscale_import=1;break;
            case OPT_SAVE_UPSCALE: params.save_upscale_state=optarg;break;
            case OPT_STATE_ONLY: params.state_only=1;break;
            case OPT_SAVE_AV_STATE: save_av_state = optarg; break;
            case OPT_KEEP_CONTINUATION_PREFIX: params.keep_continuation_prefix = 1; break;
            case OPT_TOKEN_REDUCTION: params.token_reduction = 1; break;
            case OPT_SSD_STREAMING: params.ssd_streaming = 1; break;
            case OPT_USE_INT8_ROW_FC2:
                params.use_int8_row_fc2 = 1;
                break;
            case OPT_USE_REFERENCE_ROPE:
                params.use_reference_rope = 1;
                break;
            case OPT_USE_SLOWER_BF16_MLP:
                params.use_slower_bf16_mlp = 1;
                break;
            case OPT_USE_SLOWER_BF16_QKV:
                params.use_slower_bf16_qkv = 1;
                break;
            case OPT_USE_SLOWER_BF16_ATTENTION_OUTPUT:
                params.use_slower_bf16_attention_output = 1;
                break;
            case OPT_USE_SLOWER_ROW_MAJOR_ATTENTION_OUTPUT:
                params.use_slower_row_major_attention_output = 1;
                break;
            case OPT_USE_SLOWER_UNFUSED_INT8_INPUTS:
                params.use_slower_unfused_int8_inputs = 1;
                break;
            case OPT_USE_SLOWER_UNFUSED_QKV_ROPE:
                params.use_slower_unfused_qkv_rope = 1;
                break;
            case OPT_USE_SLOWER_SCALAR_QKV_RMS:
                params.use_slower_scalar_qkv_rms = 1;
                break;
            case OPT_USE_SLOWER_UNCACHED_INT8_SCALES:
                params.use_slower_uncached_int8_scales = 1;
                break;
            case OPT_USE_SLOWER_DYNAMIC_FC1_K:
                params.use_slower_dynamic_fc1_k = 1;
                break;
            case OPT_USE_SLOWER_GROUPED_QUANTIZER:
                params.use_slower_grouped_quantizer = 1;
                break;
            case OPT_SEED:
                params.seed = parse_u64(optarg, "seed");
                break;
            case OPT_FIRST: params.first_frame = optarg; break;
            case OPT_LAST: params.last_frame = optarg; break;
            case OPT_REF_IMAGE: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_IMAGE;
                reference->path = optarg;
                break;
            }
            case OPT_REF_IMAGE_SIZE:
                if (!strcmp(optarg, "match"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_MATCH;
                else if (!strcmp(optarg, "max"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_MAX;
                else if (!strcmp(optarg, "high"))
                    params.reference_image_size = H3_REFERENCE_IMAGE_HIGH;
                else {
                    fprintf(stderr,
                        "h3cli: --ref-image-size must be match, high or max\n");
                    return 2;
                }
                break;
            case OPT_REF_VIDEO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO;
                reference->path = optarg;
                reference->include_embedded_audio = 1;
                break;
            }
            case OPT_REF_SILENT_VIDEO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO;
                reference->path = optarg;
                reference->include_embedded_audio = 0;
                break;
            }
            case OPT_REF_VIDEO_AUDIO: {
                if (optind >= argc) {
                    fprintf(stderr,
                        "h3cli: --ref-video-audio requires VIDEO and AUDIO\n");
                    return 2;
                }
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_VIDEO_AUDIO;
                reference->path = optarg;
                reference->audio_path = argv[optind++];
                break;
            }
            case OPT_REF_AUDIO: {
                h3_reference *reference = append_reference(
                    references, &reference_count);
                reference->kind = H3_REFERENCE_AUDIO;
                reference->path = optarg;
                break;
            }
            case OPT_FRAMES_DIR: cli.frames_dir = optarg; break;
            case OPT_SHOW: show = 1; break;
            case OPT_ZOOM:
                if (!h3_terminal_set_zoom(parse_int(optarg, "zoom"))) {
                    fprintf(stderr, "h3cli: --zoom must be at least 1\n");
                    return 2;
                }
                break;
            case OPT_PROFILE: profile = 1; break;
            case OPT_VERBOSE: break; /* Scoped by h3_cli_run, before validation. */
            case OPT_INFO: info = 1; break;
            default: usage(argv[0]); return 2;
        }
    }
    {
        char error[256];
        if(!h3_output_encoding_valid(&params.output_encoding,error,sizeof(error))) {
            fprintf(stderr,"h3cli: %s\n",error);return 2;
        }
        if(h3_output_encoding_selected(&params.output_encoding) &&
           (params.still||decode_still||info||inspect_upscale||params.state_only)) {
            fprintf(stderr,"h3cli: output encoding controls require video delivery\n");return 2;
        }
    }
    if(quality>=0) {
        if(params.resume_sampler_state||decode_av_state||decode_still||upscale_path||inspect_upscale||info) {
            fprintf(stderr,"h3cli: --quality requires fresh generation (including continuation); resume, decode, upscale and info modes use their own controls\n");return 2;
        }
        int reuse=h3_quality_presets[quality].reuse;
        int adaptive=h3_quality_presets[quality].adaptive;
#ifdef __APPLE__
        if(adaptive){reuse=2;adaptive=H3_ADAPTIVE_OFF;}
#endif
        if(!steps_given)params.steps=h3_quality_presets[quality].steps;
        if(!reuse_given)params.denoise_reuse=reuse;
        if(!preview_vae_given)params.preview_vae=h3_quality_presets[quality].preview;
        if(!params.adaptive_cache_set)params.adaptive_cache=adaptive;
        if(!cli_quality_reported++)H3_VERBOSE("h3cli: quality=%s steps=%d reuse=%d core-reuse=%d layers=%d adaptive-cache=%s cuda-attention=%s cuda-denoise-quant=%s preview-vae=%s\n",
            h3_quality_presets[quality].name,params.steps,params.denoise_reuse,params.core_reuse,params.dit_layers,
            h3_adaptive_name(params.adaptive_cache),h3_attention_name(params.cuda_attention),
            h3_quant_name(params.cuda_denoise_quant),params.preview_vae?"on":"off");
    }
    if(upscale_path||inspect_upscale) {
        char error[512]={0};
        if(optind!=argc||generation_given||info||decode_av_state||decode_still||
           params.resume_sampler_state||params.save_upscale_state||params.preview_vae||params.preview_vae_model||
           params.preview_on_stop||params.cuda_denoise_quant_set||params.cuda_denoise_quant_cache||quant_verify_given||params.cuda_attention_set||
           params.adaptive_cache_set||params.adaptive_cache_threshold_set||params.adaptive_cache_max_hits_set||params.adaptive_cache_max_bytes_set||params.subblock_sparsity_set||params.adaptive_cache_warmup_set||params.subblock_warmup_set||
           lora.count||lora.cache_dir||lora.memory_mib||(upscale_path&&inspect_upscale)||
           (inspect_upscale&&(upscale_given||params.state_only||save_av_state||output_given||params.save_sampler_state||params.stop_after_step>=0))) {
            fprintf(stderr,"h3cli: upscale mode restores prompt/geometry/conditioning; ordinary generation, approximation, resume and decode controls conflict\n");return 2;
        }
        if(upscale_path&&!upscale.model_path)upscale.model_path=DEFAULT_UPSCALE_MODEL;
        const char *input=upscale_path?upscale_path:inspect_upscale;
        upscale.stop_after_step=params.stop_after_step;upscale.save_sampler_state=params.save_sampler_state;upscale.state_only=params.state_only;
        upscale.delivery=(h3_decode_options){.output_path=params.state_only?NULL:output,.output_encoding=params.output_encoding,
            .on_frame=(show||cli.frames_dir)?cli_frame:NULL,.on_progress=cli_progress,.callback_opaque=&cli};
        int paused=upscale.stop_after_step>=0&&upscale.stop_after_step<upscale.refine_steps;
        if(!inspect_upscale&&(!upscale.model_path||
           (params.state_only&&!save_av_state&&!(paused&&params.save_sampler_state))||(save_av_state&&paused)||
           !h3_upscale_request_valid(&upscale,error,sizeof(error)))) {
            fprintf(stderr,"h3cli: upscale state-only needs --save-av-state (or a paused checkpoint); %s\n",error);return 2;
        }
        const char *paths[]={input,upscale.model_path,params.save_sampler_state,save_av_state,params.state_only?NULL:output};
        for(size_t i=0;i<5;i++)for(size_t j=i+1;j<5;j++)if(h3_upscale_paths_alias(paths[i],paths[j])) {
            fprintf(stderr,"h3cli: upscale paths must not alias source, weights or other outputs\n");return 2;
        }
        if(save_av_state) {
            char *side=NULL;if(asprintf(&side,"%s.presentation",save_av_state)<0)return 1;
            int collision=0;for(size_t i=0;i<5;i++)collision|=h3_upscale_paths_alias(side,paths[i]);free(side);
            if(collision){fprintf(stderr,"h3cli: upscale output aliases AV presentation\n");return 2;}
        }
        if(cli_hooks&&cli_hooks->validate_only)return 0;
        h3_upscale_source *source=upscale_import?h3_upscale_source_import_sampler(input,error,sizeof(error)):h3_upscale_source_load(input,error,sizeof(error));
        if(!source){fprintf(stderr,"h3cli: %s; expected a complete .h3up source (sampler import must be explicit)\n",error);return 1;}
        h3_upscale_plan *plan=h3_upscale_plan_create(source,error,sizeof(error));
        if(!plan){fprintf(stderr,"h3cli: %s\n",error);h3_upscale_source_free(source);return 1;}
        const h3_upscale_plan_info *pi=h3_upscale_plan_get_info(plan);
        if (inspect_upscale || h3_log_verbose()) fprintf(inspect_upscale?stdout:stderr,"h3cli: upscale source=%dx%d target=%dx%d frames=%d video_T=%d audio_T=%d rows=%zu profile=%d semantic_view=source seed=%" PRIu64 "\n",
            pi->source_width,pi->source_height,pi->width,pi->height,pi->frames,pi->video_t,pi->audio_t,pi->sequence_rows,pi->geometry_profile,source->state->params.seed);
        if(inspect_upscale) {
            printf("source_sha256 ");for(int i=0;i<32;i++)printf("%02x",source->state->loaded_hash[i]);
            printf("\naudio_sha256 ");for(int i=0;i<32;i++)printf("%02x",source->state->upscale.audio_hash[i]);
            printf("\nreferences=%zu keyframes=%u source_steps=%d network_reserve_bytes=%" PRIu64 " model_identity=local-metadata\n",
                source->state->reference_count,source->state->upscale.keyframe_count,source->state->total_steps,pi->network_reserve_bytes);
        }
        h3_upscale_plan_free(plan);
        if(inspect_upscale){h3_upscale_source_free(source);return 0;}
        if(cli.frames_dir&&mkdir(cli.frames_dir,0755)&&errno!=EEXIST){perror("frames directory");h3_upscale_source_free(source);return 1;}
        if (profile)
            setenv("H3_PROFILE", "1", 1);
        if (show)
            cli.terminal = h3_terminal_detect();
        h3_ctx *ctx=h3_load_dir(model_dir);
        if(!ctx){fprintf(stderr,"h3cli: %s\n",h3_last_error(NULL));h3_upscale_source_free(source);return 1;}
        h3_result *result=h3_upscale(ctx,source,&upscale);h3_upscale_source_free(source);
        h3_cli_progress_finish(&cli.progress,stderr);
        if(!result){fprintf(stderr,"h3cli: %s\n",h3_last_error(ctx));h3_free(ctx);return 1;}
        int ok=!save_av_state||h3_result_save_av_state(result,save_av_state,error,sizeof(error));
        if(!ok)fprintf(stderr,"h3cli: %s\n",error);
        else if(result->status==H3_RESULT_PAUSED)fprintf(stderr,"h3cli: paused upscale at %d/%d transitions\n",result->completed_steps,result->total_steps);
        else fprintf(stderr,"h3cli: upscale complete%s%s\n",params.state_only?"; state saved to ":"; wrote ",params.state_only?save_av_state:output);
        if (cli_hooks && cli_hooks->result)
            cli_hooks->result(result, cli_hooks->opaque);
        h3_result_free(result);
        h3_free(ctx);
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
        fprintf(stderr,"h3cli: upscale total wall time %.6f s\n",(double)(end.tv_sec-operation_begin.tv_sec)+(double)(end.tv_nsec-operation_begin.tv_nsec)/1e9+model_preparation_seconds);
        return !ok;
    }
    if(upscale_given||upscale_import) {fprintf(stderr,"h3cli: upscale controls require --upscale-state; resume restores them from its checkpoint\n");return 2;}
    if(params.state_only&&params.resume_sampler_state&&!save_av_state&&params.stop_after_step<0) {
        fprintf(stderr,"h3cli: state-only refinement resume requires --save-av-state or a paused checkpoint\n");return 2;
    }
    if(params.save_upscale_state||params.state_only) {
        char error[512];
        params.references=references;params.reference_count=reference_count;params.output_path=output;
        if(decode_av_state||decode_still||lora.count||continue_from||
            (params.state_only&&(show||cli.frames_dir))||
            !h3_upscale_options_valid(&params,error,sizeof(error))) {
            fprintf(stderr,"h3cli: invalid upscale source mode: %s\n",
                decode_av_state||decode_still||lora.count||continue_from?"incompatible decode/LoRA/continuation":
                params.state_only&&(show||cli.frames_dir)?"state-only does not deliver frames":error);return 2;
        }
        if(save_av_state&&h3_upscale_paths_alias(save_av_state,params.save_upscale_state)) {
            fprintf(stderr,"h3cli: AV output aliases upscale source\n");return 2;
        }
    }
    /* Resolve shared controls only after all options, so order is irrelevant. */
    if(params.cuda_sol_set) {
        if(params.cuda_attention==H3_ATTENTION_SOL || (params.resume_sampler_state&&!params.backend&&!params.backend_set)) {
            if(params.cuda_sol_set&H3_SOL_Q)params.cuda_sol.q_block=params.metal_attention.q_block;
            if(params.cuda_sol_set&H3_SOL_KV)params.cuda_sol.kv_block=params.metal_attention.kv_block;
            if(params.cuda_sol_set&H3_SOL_TAU)params.cuda_sol.tau=params.metal_attention.tau;
            if(params.cuda_sol_set&H3_SOL_LAYERS)params.cuda_sol.dense_layers=params.metal_attention.dense_layers;
            if(params.cuda_sol_set&H3_SOL_STEPS)params.cuda_sol.dense_steps=params.metal_attention.dense_steps;
            if(params.cuda_sol_set&H3_SOL_SIGMA)params.cuda_sol.dense_sigma=params.metal_attention.dense_sigma;
            if(params.cuda_sol_set&H3_SOL_RADIUS)params.cuda_sol.local_radius=params.metal_attention.local_radius;
            if(params.cuda_sol_set&H3_SOL_MIN)params.cuda_sol.min_exact=params.metal_attention.min_exact;
        } else {
            params.backend_set|=8;params.cuda_sol_set=0;
        }
    }
    if(quant_verify_given&&((!params.cuda_denoise_quant&&!params.resume_sampler_state)||decode_still||decode_av_state)) {
        fprintf(stderr,"h3cli: --cuda-denoise-quant-verify requires quantized generation or sampler resume\n");return 2;
    }
    if(!output_given && (decode_still || params.still))output="outputs/h3cli.png";
    if((decode_still || params.still) && !image_vae)image_vae=params.image_vae=DEFAULT_IMAGE_VAE;
    if((decode_still||decode_av_state)&&(params.adaptive_cache_max_bytes_set||params.adaptive_cache_threshold_set||params.adaptive_cache_max_hits_set)) {
        fprintf(stderr,"h3cli: decode-only does not use adaptive-cache controls\n");return 2;
    }
    if (decode_still) {
        if (!image_vae || !*image_vae || generation_given || info || decode_av_state ||
            params.preview_vae || params.preview_vae_model || save_av_state || params.resume_sampler_state ||
            params.save_sampler_state || params.stop_after_step>=0 || params.preview_on_stop ||
            params.cuda_attention_set || params.cuda_denoise_quant || params.cuda_denoise_quant_cache ||
            lora.count || lora.cache_dir || lora.memory_mib) {
            fprintf(stderr,"h3cli: --decode-still-latent accepts only image VAE, PNG output, display and profile options; --image-vae must be nonempty\n");return 2;
        }
        if(cli_hooks&&cli_hooks->validate_only)return 0;
        if(cli.frames_dir && mkdir(cli.frames_dir,0755) && errno!=EEXIST) {
            fprintf(stderr,"h3cli: cannot create frames directory: %s\n",strerror(errno));return 1;
        }
        if(profile)setenv("H3_PROFILE","1",1);
        if(show)cli.terminal=h3_terminal_detect();
        char error[512];h3_decode_options decode={.output_path=output,
            .on_frame=(show||cli.frames_dir)?cli_frame:NULL,.on_progress=cli_progress,.callback_opaque=&cli};
        h3_result *r=h3_decode_still_latent(decode_still,image_vae,&decode,error,sizeof(error));
        h3_cli_progress_finish(&cli.progress,stderr);
        if(!r){fprintf(stderr,"h3cli: %s\n",error);return 1;}
        if (cli_hooks && cli_hooks->result)
            cli_hooks->result(r, cli_hooks->opaque);
        h3_result_free(r);
        fprintf(stderr, "h3cli: wrote %s\n", output);
        return 0;
    }
    if(params.still) {
        if(still_conflict_given || seconds_given || (frames_given && params.frames!=1) || save_av_state || continue_from || context_given || decode_av_state || info) {
            fprintf(stderr,"h3cli: --still accepts one frame and cannot use duration, AV state or continuation options\n");return 2;
        }
        params.frames=1;params.output_path=output;params.references=references;params.reference_count=reference_count;
        char error[512];if(!h3_still_options(&params,error,sizeof(error))){fprintf(stderr,"h3cli: %s\n",error);return 2;}
    } else if(image_vae || params.save_still_latent){fprintf(stderr,"h3cli: image VAE and still latent output require --still\n");return 2;}
    char preview_error[512];
    if(!h3_backend_preflight(H3_BACKEND_SCOPE(&params),
        params.backend_set,preview_error,sizeof(preview_error)) ||
       (params.backend && !h3_sampler_checkpoint_options(&params,NULL,
            preview_error,sizeof(preview_error)))) {
        fprintf(stderr,"h3cli: %s\n",preview_error);return 2;
    }
    if ((params.save_conditioning && !*params.save_conditioning) ||
        (params.load_conditioning && !*params.load_conditioning) ||
        (params.conditioning_schedule && !params.save_conditioning && !params.load_conditioning) ||
        ((decode_av_state || params.resume_sampler_state) &&
         (params.save_conditioning || params.load_conditioning))) {
        fprintf(stderr,"h3cli: conditioning requires nonempty paths and a fresh generation; it cannot be combined with decode or sampler resume\n");return 2;
    }
    if(params.save_conditioning&&params.conditioning_schedule&&params.dit_layers!=50) {
        fprintf(stderr,"h3cli: schedule conditioning caches require all 50 blocks\n");return 2;
    }
    params.references=references;params.reference_count=reference_count;
    if(!h3_approximate_params_valid(&params,lora.count!=0,preview_error,sizeof(preview_error)) ||
       !h3_adaptive_preflight(params.adaptive_cache,preview_error,sizeof(preview_error)) ||
       !h3_cuda_sol_params_valid(&params,preview_error,sizeof(preview_error)) ||
       !h3_attention_preflight(params.cuda_attention,preview_error,sizeof(preview_error))) {
        fprintf(stderr,"h3cli: %s\n",preview_error);return 2;
    }
    if(decode_av_state&&params.backend_set) {
        fprintf(stderr,"h3cli: --backend/--metal-attention select denoising and cannot be used with decode-only mode\n");return 2;
    }
    if(decode_av_state&&params.cuda_attention_set) {
        fprintf(stderr,"h3cli: decode-only does not use cuda-attention\n");return 2;
    }
    if(decode_av_state&&(params.adaptive_cache_set||params.adaptive_cache_threshold_set||params.adaptive_cache_max_hits_set||params.adaptive_cache_max_bytes_set||params.subblock_sparsity_set||params.adaptive_cache_warmup_set||params.subblock_warmup_set)) {
        fprintf(stderr,"h3cli: decode-only does not use adaptive-cache, subblock-sparsity or warmup flags\n");return 2;
    }
    if (!h3_quant_options(params.cuda_denoise_quant,params.resume_sampler_state?NULL:params.cuda_denoise_quant_cache,preview_error,sizeof(preview_error)) ||
        !h3_quant_preflight(params.cuda_denoise_quant,preview_error,sizeof(preview_error))) {
        fprintf(stderr,"h3cli: %s\n",preview_error);return 2;
    }
    if(decode_av_state&&(params.cuda_denoise_quant||params.cuda_denoise_quant_cache)) {
        fprintf(stderr,"h3cli: decode-only does not use denoiser quantization options\n");return 2;
    }
    int defer_preview = cli_hooks && cli_hooks->validate_only && params.preview_vae &&
        access(h3_preview_vae_path(params.preview_vae_model), F_OK) != 0;
    if (!defer_preview && !h3_preview_vae_options(params.preview_vae,params.preview_vae_model,preview_error,sizeof(preview_error))) {
        fprintf(stderr,"h3cli: %s\n",preview_error); return 2;
    }
    if (decode_av_state) {
        if (lora.count || lora.cache_dir || lora.memory_mib || generation_given || info || save_av_state || params.resume_sampler_state || params.save_sampler_state ||
            params.stop_after_step>=0 || params.preview_on_stop) {
            fprintf(stderr,"h3cli: decode-only accepts output, decoder, profile, and display options; no prompt, generation settings or sampler states\n"); return 2;
        }
        if(cli_hooks&&cli_hooks->validate_only)return 0;
        if (cli.frames_dir && mkdir(cli.frames_dir,0755) && errno!=EEXIST) {
            fprintf(stderr,"h3cli: cannot create frames directory: %s\n",strerror(errno)); return 1;
        }
        if(profile)setenv("H3_PROFILE","1",1);
        if(show)cli.terminal=h3_terminal_detect();
        h3_decode_options decode={.output_path=output,.output_encoding=params.output_encoding,.preview_vae=params.preview_vae,
            .preview_vae_model=params.preview_vae_model,
            .on_frame=(show||cli.frames_dir)?cli_frame:NULL,.on_progress=cli_progress,.callback_opaque=&cli};
        h3_result *result=h3_decode_av_state(model_dir,decode_av_state,&decode,preview_error,sizeof(preview_error));
        h3_cli_progress_finish(&cli.progress,stderr);
        if(!result){fprintf(stderr,"h3cli: %s\n",preview_error);return 1;}
        if (cli_hooks && cli_hooks->result)
            cli_hooks->result(result, cli_hooks->opaque);
        h3_result_free(result);
        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed=(double)(end.tv_sec-operation_begin.tv_sec)+(double)(end.tv_nsec-operation_begin.tv_nsec)/1e9;
        fprintf(stderr,"h3cli: wrote %s\nh3cli: total wall time: %.2f s\n",output,elapsed+model_preparation_seconds);return 0;
    }
    if(!lora.count&&(lora.cache_dir||lora.memory_mib)){
        fprintf(stderr,"h3cli: --lora-cache/--lora-memory-mib require --lora\n");return 2;
    }
    int checkpoint = params.resume_sampler_state || params.save_sampler_state || params.stop_after_step >= 0 || params.preview_on_stop;
    if (params.resume_sampler_state && generation_given) {
        fprintf(stderr,"h3cli: resume checkpoint is authoritative; generation-changing arguments are not permitted\n"); return 2;
    }
    if (checkpoint && (info || (!prompt && !params.resume_sampler_state) ||
        (params.save_sampler_state && !*params.save_sampler_state) || (params.resume_sampler_state && !*params.resume_sampler_state) ||
        (params.stop_after_step>=0 && !params.save_sampler_state) ||
        (params.preview_on_stop && params.stop_after_step<0))) {
        fprintf(stderr,"h3cli: sampler state requires -p or --resume-sampler-state; stopping requires --save-sampler-state; preview requires --stop-after-step\n"); return 2;
    }
    if (checkpoint) {
        int steps=params.steps, next=0;
        if (params.resume_sampler_state) {
            char error[512]; h3_sampler_state *state=h3_sampler_state_load_with_budget(params.resume_sampler_state,params.adaptive_cache_max_bytes,params.adaptive_cache_max_bytes_set||params.adaptive_cache_max_bytes,error,sizeof(error));
            if (!state) { fprintf(stderr,"h3cli: %s\n",error); return 1; }
            if(((params.backend_set&1||params.backend)&&params.backend!=state->params.backend)||
               ((params.backend_set&2||params.attention_mode)&&params.attention_mode!=state->params.attention_mode)||
       ((params.backend_set&8)&&!h3_metal_options_equal(params.metal_attention,state->params.metal_attention))) {
                fprintf(stderr,"h3cli: resume backend/attention differs from checkpoint; restart denoising to change backend\n");
                h3_sampler_state_free(state);return 2;
            }
            if ((params.cuda_sol_set && (state->params.cuda_attention!=H3_ATTENTION_SOL ||
                 !h3_cuda_sol_options_match(params.cuda_sol,state->params.cuda_sol,params.cuda_sol_set))) ||
                ((params.cuda_attention_set || params.cuda_attention) &&
                 params.cuda_attention != state->params.cuda_attention)) {
                fprintf(stderr,"h3cli: resume attention differs from the checkpoint; restart denoising to change attention\n");
                h3_sampler_state_free(state);return 2;
            }
            if(((params.adaptive_cache_set||params.adaptive_cache)&&params.adaptive_cache!=state->params.adaptive_cache)||
               (params.subblock_sparsity_set&&params.subblock_sparsity!=state->params.subblock_sparsity)||
               !h3_warmups_match(&params,&state->params)||!h3_adaptive_controls_match(&params,&state->params)) {
                fprintf(stderr,"h3cli: adaptive/SubBlock policy differs from checkpoint\n");h3_sampler_state_free(state);return 2;
            }
            if (!h3_attention_preflight(state->params.cuda_attention,error,sizeof(error))) {
                fprintf(stderr,"h3cli: %s\n",error);h3_sampler_state_free(state);return 2;
            }
            steps=state->total_steps; next=state->next_step;
            h3_sampler_state_free(state);
        }
        if (params.stop_after_step>steps || (params.stop_after_step>=0 && params.stop_after_step<next)) {
            fprintf(stderr,"h3cli: stop-after-step must be an absolute boundary between next_step and the full schedule length\n"); return 2;
        }
        if (save_av_state && params.stop_after_step>=0 && params.stop_after_step<steps) {
            fprintf(stderr,"h3cli: a paused sampler has no complete .h3av; save it after final completion\n"); return 2;
        }
    }
    if (frames_given && seconds_given) {
        fprintf(stderr, "h3cli: --seconds and --frames are mutually exclusive\n");
        return 2;
    }
    if (!prompt && !params.resume_sampler_state && !info) {
        fprintf(stderr, "h3cli: generation requires -p/--prompt; use --info, "
                "--resume-sampler-state or a decode option for other operations\n");
        return 2;
    }
    if(prompt&&!params.resume_sampler_state&&!h3_test_evaluation_budget(
        params.stop_after_step>=0?params.stop_after_step:params.steps,preview_error,sizeof(preview_error))) {
        fprintf(stderr,"h3cli: %s\n",preview_error);return 2;
    }
    if (prompt && params.steps >= 2 && params.steps <= 7 &&
        params.denoise_reuse > 1) {
        fprintf(stderr,
            "h3cli: warning: --reuse with only %d denoising steps leaves very "
            "few fresh model evaluations\n", params.steps);
    }
    params.references = references;
    params.reference_count = reference_count;
    if (!(cli_hooks&&cli_hooks->validate_only) && cli.frames_dir && mkdir(cli.frames_dir, 0755) != 0 &&
        errno != EEXIST) {
        fprintf(stderr, "h3cli: cannot create frames directory %s: %s\n",
                cli.frames_dir, strerror(errno));
        return 1;
    }
    if (profile) setenv("H3_PROFILE", "1", 1);
    if (params.continuation_mode == H3_CONTINUE_BRIDGE && (!continue_from || !prompt)) {
        fprintf(stderr, "h3cli: bridge continuation requires --continue-from and -p\n");
        return 2;
    }
    if (((continue_from || save_av_state) && !prompt && !params.resume_sampler_state) ||
        ((context_given || params.keep_continuation_prefix) && !continue_from) ||
        (context_given && params.continuation_context_frames < 39)) {
        fprintf(stderr,"h3cli: state CLI options require -p; context/keep-prefix require --continue-from, context must be 39 + 51*k\n");
        return 2;
    }
    h3_av_state *continuation = NULL;
    if (continue_from) {
        char error[512];
        continuation = h3_av_state_load(continue_from,error,sizeof(error));
        if (!continuation) { fprintf(stderr,"h3cli: %s: %s\n",continue_from,error); return 1; }
        params.continuation = continuation;
        if (!(cli_hooks && cli_hooks->validate_only)) H3_VERBOSE("h3cli: continuation source: %s\n",continue_from);
    }
    if(cli_hooks&&cli_hooks->validate_only) {
        int ok=1;
        if(!params.resume_sampler_state&&!info) {
            ok=h3_request_params_valid(&params,NULL,preview_error,sizeof(preview_error))&&
               h3_approximate_params_valid(&params,lora.count!=0,preview_error,sizeof(preview_error));
            if(ok&&!params.still&&h3_align_frame_count(params.frames)<22){
                snprintf(preview_error,sizeof(preview_error),"generation needs at least 22 frames");ok=0;
            }
            if(!ok)fprintf(stderr,"h3cli: %s\n",preview_error);
        }
        h3_av_state_free(continuation);return ok?0:2;
    }
    h3_ctx *ctx = h3_load_dir_with_lora(model_dir,&lora);
    if (!ctx) {
        fprintf(stderr, "h3cli: %s\n", h3_last_error(NULL));
        h3_av_state_free(continuation);
        return 1;
    }
    if (info) {
        print_info(ctx);
        h3_lora_selection *selection=NULL;char error[512];
        if(h3_lora_selection_create(&lora,&selection,error,sizeof(error))) {
            h3_lora_selection_print(selection,stdout);h3_lora_selection_free(selection);
        }
    }
    if (prompt || params.resume_sampler_state) {
        params._arithmetic_recipe=h3_sglang_resolve(&params,h3_device(ctx)->backend,lora.count!=0);
        params.output_path = params.state_only?NULL:output;
        params.on_progress = cli_progress;
        params.callback_opaque = &cli;
        if (cli.frames_dir) params.on_frame = cli_frame;
        if (show&&cli_hooks) {params.on_frame=cli_frame;params.preview_denoise=!params.still;}
        if (show&&!cli_hooks) {
            cli.terminal = h3_terminal_detect();
            if (cli.terminal == H3_TERM_NONE) {
                fprintf(stderr, "h3cli: warning: --show needs Kitty, Ghostty, "
                        "iTerm2, WezTerm, or Konsole\n");
            } else {
                fprintf(stderr, "h3cli: graphical output uses %s\n",
                        h3_terminal_protocol_name(cli.terminal));
                params.on_frame = cli_frame;
                params.preview_denoise = !params.still;
            }
        }
        struct timespec generation_begin = {0};
        if (params._arithmetic_recipe) clock_gettime(CLOCK_MONOTONIC, &generation_begin);
        h3_result *result = h3_generate(ctx, prompt, &params);
        if (params._arithmetic_recipe && result && result->status == H3_RESULT_COMPLETE) {
            struct timespec generation_end;
            clock_gettime(CLOCK_MONOTONIC, &generation_end);
            double seconds = (double)(generation_end.tv_sec - generation_begin.tv_sec) +
                (double)(generation_end.tv_nsec - generation_begin.tv_nsec) / 1e9;
            /* Include lazy weight loading, conditioning and media delivery.
             * Subtracting their costs would conceal the cold pipeline cost. */
            H3_VERBOSE("h3cli: %s generation including lazy weights: %.6f s\n",
                (params.cuda_attention||params.cuda_denoise_quant)?"SGLang-base with explicit approximation":"reference",seconds);
        }
        if (!result) {
            h3_cli_progress_finish(&cli.progress, stderr);
            fprintf(stderr, "h3cli: %s\n", h3_last_error(ctx));
            h3_free(ctx);
            h3_av_state_free(continuation);
            return 1;
        }
        if (save_av_state && result->status == H3_RESULT_COMPLETE) {
            char error[512];
            if (!h3_result_save_av_state(result,save_av_state,error,sizeof(error))) {
                fprintf(stderr,"h3cli: %s\n",error);
                if (cli_hooks && cli_hooks->result)
                    cli_hooks->result(result, cli_hooks->opaque);
                h3_result_free(result);
                h3_free(ctx);
                h3_av_state_free(continuation);
                return 1;
            }
            fprintf(stderr,"h3cli: saved complete AV state: %s\n",save_av_state);
        }
        int wrote_media = !params.state_only && (result->status == H3_RESULT_COMPLETE || params.preview_on_stop);
        if (result->status == H3_RESULT_PAUSED) fprintf(stderr,"h3cli: paused successfully after %d/%d transitions\n",result->completed_steps,result->total_steps);
        if (cli_hooks && cli_hooks->result)
            cli_hooks->result(result, cli_hooks->opaque);
        h3_result_free(result);
        if (wrote_media && output && *output) {
            fprintf(stderr, "h3cli: wrote %s\n", output);
            report_wall_time = 1;
        }
        if (cli.frames_dir)
            fprintf(stderr, "h3cli: wrote frames to %s\n", cli.frames_dir);
    }
    h3_free(ctx);
    h3_av_state_free(continuation);
    if (report_wall_time) {
        struct timespec operation_end;
        clock_gettime(CLOCK_MONOTONIC, &operation_end);
        double elapsed = (double)(operation_end.tv_sec - operation_begin.tv_sec) +
            (double)(operation_end.tv_nsec - operation_begin.tv_nsec) / 1e9;
        fprintf(stderr, "h3cli: total wall time: %.2f s\n", elapsed+model_preparation_seconds);
    }
    return 0;
}


static int cli_parsed_run(int argc,char **argv,const h3_cli_hooks *hooks) {
    cli_hooks=hooks;cli_lora_args=NULL;
    optind=1;opterr=1;
#ifdef __APPLE__
    optreset=1;
#endif
    int status;
    if(setjmp(cli_parse_failure))status=2;
    else status=cli_execute(argc,argv);
    h3_log_line_callback(NULL, NULL);
    free(cli_lora_args);cli_lora_args=NULL;cli_hooks=NULL;
    return status;
}
static volatile sig_atomic_t download_cancelled;
typedef struct {
    const h3_cli_hooks *hooks;
    char phase[64];
    double last_elapsed;
} download_progress_state;

static void download_signal(int number) { (void)number; download_cancelled = 1; }
static int download_progress(const h3_model_progress *event, void *opaque) {
    download_progress_state *state = opaque;
    const h3_cli_hooks *hooks = state->hooks;
    if (download_cancelled) return 1;
    if (hooks && hooks->model_progress) return hooks->model_progress(event, hooks->opaque);
    if (!h3_log_verbose() && !strcmp(state->phase, event->phase) &&
        event->elapsed - state->last_elapsed < 5.0) return 0;
    snprintf(state->phase, sizeof(state->phase), "%s", event->phase);
    state->last_elapsed = event->elapsed;
    fprintf(stderr, "h3cli: %s %s; transferred %.1f / %.1f MiB, reused %.1f MiB (%.1fs)\n",
        event->phase, event->component, (double)event->completed/1048576.0,
        (double)event->total/1048576.0, (double)event->reused/1048576.0, event->elapsed);
    if (h3_log_verbose() && !strcmp(event->phase,"models_ready"))
        fprintf(stderr,"h3cli: model bytes downloaded=%" PRIu64 " reused=%" PRIu64 " required=%" PRIu64 "\n",
            event->completed,event->reused,event->total);
    return 0;
}
static int local_inputs(const h3_request *request, char *error, size_t size) {
    for (size_t i = 0; i < request->count; i++) {
        const h3_option_value *v = &request->values[i];
        if (!v->option->repeatable && h3_request_get(request, v->option->name) != v) continue;
        if (v->option->path != H3_PATH_MEDIA && v->option->path != H3_PATH_STATE && strcmp(v->option->name, "lora")) continue;
        char *first = strdup(v->value);
        if (!first) return 0;
        struct stat st;
        if (!strcmp(v->option->name, "lora") && stat(first, &st)) {
            char *colon = strrchr(first, ':'); if (colon) *colon = 0;
        }
        int ok = !stat(first, &st) && S_ISREG(st.st_mode);
        if (ok && v->second) ok = !stat(v->second, &st) && S_ISREG(st.st_mode);
        free(first);
        if (!ok) { snprintf(error, size, "--%s input is unavailable: %.400s; model downloads do not fetch user inputs", v->option->name,v->value); return 0; }
    }
    const h3_option_value *state=h3_request_get(request,"resume-sampler-state");
    if(state&&!h3_sampler_file_validate(state->value,0,error,size))return 0;
    state=h3_request_get(request,"upscale-state");
    if(state&&!h3_sampler_file_validate(state->value,!h3_request_get(request,"upscale-import-sampler"),error,size))return 0;
    state=h3_request_get(request,"decode-av-state");
    if(state){
        h3_av_state *av=h3_av_state_load(state->value,error,size);h3_presentation presentation={0};
        int ok=av&&h3_presentation_load(state->value,av,&presentation,error,size)==1;
        h3_av_state_free(av);if(!ok)return 0;
    }
    state=h3_request_get(request,"decode-still-latent");
    if(state){
        h3_still_latent latent={0};int ok=h3_still_latent_load(state->value,&latent,error,size);
        h3_still_latent_free(&latent);if(!ok)return 0;
    }
    return 1;
}
int h3_cli_run(int argc, char **argv, const h3_cli_hooks *hooks) {
    int previous_verbose = h3_log_exchange_verbose(-1);
    model_preparation_seconds=0;cli_quality_reported=0;
    h3_request request = {0}; h3_model_plan plan = {0};
    char error[1024] = {0}; char **effective = NULL;
    int count = 0, status = 2;
    if (!h3_request_parse_argv(&request, argc-1, argv+1, H3_SOURCE_NATIVE, error, sizeof(error))) {
        /* Retain native diagnostics for invalid CLI values. This path remains
         * validation-only and always fails the shared request contract. */
        h3_cli_hooks validation=hooks?*hooks:(h3_cli_hooks){0};validation.validate_only=1;
        if(cli_parsed_run(argc,argv,&validation))error[0]=0;
        status=2;goto done;
    }
    if (h3_request_get(&request, "verbose") || h3_request_get(&request, "profile"))
        h3_log_exchange_verbose(1);
    if (h3_request_get(&request, "help")) { usage(argv[0]); status = 0; goto done; }
    int prefetch=h3_request_get(&request,"download-models")!=NULL;
    int list=h3_request_get(&request,"list-models")!=NULL;
    if(!prefetch&&!list){
        /* Reject option conflicts before inspecting saved inputs or looking up
         * model availability. The native parser remains the admission policy. */
        h3_request validation_request={0};int copied=1;
        for(size_t i=0;copied&&i<request.count;i++){
            const h3_option_value *v=&request.values[i];
            if(!strcmp(v->option->name,"models-path")||!strcmp(v->option->name,"offline"))continue;
            copied=h3_request_add(&validation_request,v->option->name,v->value,v->second,v->source,error,sizeof(error));
        }
        effective=copied?h3_request_argv(&validation_request,&count):NULL;h3_request_free(&validation_request);
        if(!effective)goto done;
        h3_cli_hooks validation=hooks?*hooks:(h3_cli_hooks){0};validation.validate_only=1;
        status=cli_parsed_run(count,effective,&validation);
        h3_argv_free(effective);effective=NULL;
        if(status)goto done;
    }
    if(!h3_models_resolve(&request,&plan,error,sizeof(error))){status=2;goto done;}
    if(list){status=h3_models_list(&plan,stdout)?0:1;goto done;}
    if(!prefetch){
        if(!local_inputs(&request,error,sizeof(error))){status=2;goto done;}
        if(!h3_models_apply_paths(&request,&plan,error,sizeof(error))){status=2;goto done;}
        h3_request_remove(&request,"models-path");h3_request_remove(&request,"offline");
        effective=h3_request_argv(&request,&count);
        if(!effective){snprintf(error,sizeof(error),"cannot allocate effective CLI");status=2;goto done;}
        if(hooks&&hooks->validate_only){status=0;goto done;}
    }
    download_cancelled = 0;
    struct sigaction old_int, old_term, action = {0};
    action.sa_handler = download_signal; sigemptyset(&action.sa_mask);
    if (!hooks) { sigaction(SIGINT, &action, &old_int); sigaction(SIGTERM, &action, &old_term); }
    double preparing_began=h3_av_now();
    download_progress_state download_state = {.hooks = hooks};
    int ready = h3_models_prepare(&plan, download_progress, &download_state, error, sizeof(error));
    model_preparation_seconds=h3_av_now()-preparing_began;
    if(model_preparation_seconds>=.1)fprintf(stderr,"h3cli: model preparation %.3f s\n",model_preparation_seconds);
    if (!hooks) { sigaction(SIGINT, &old_int, NULL); sigaction(SIGTERM, &old_term, NULL); }
    if (!ready) { status = download_cancelled ? 130 : 1; goto done; }
    if (prefetch || (hooks && hooks->prepare_only)) { status = 0; goto done; }
    status = cli_parsed_run(count, effective, hooks);
done:
    if (*error) fprintf(stderr, "h3cli: %s\n", error);
    h3_log_exchange_verbose(previous_verbose);
    h3_argv_free(effective); h3_models_plan_free(&plan); h3_request_free(&request);
    return status;
}
int main(int argc,char **argv) {
    if(argc>1&&!strcmp(argv[1],"--server-worker"))return h3_server_worker_main(argc,argv);
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--server"))return h3_server_main(argc,argv);
        const h3_option_descriptor *d=NULL;
        if(!strncmp(argv[i],"--",2)) {if(strchr(argv[i],'='))continue;d=h3_option_find(argv[i]+2);}
        else if(argv[i][0]=='-'&&argv[i][1]&&!argv[i][2])d=h3_option_short(argv[i][1]);
        if(d)i+=d->arity;
    }
    return h3_cli_run(argc,argv,NULL);
}
