.DEFAULT_GOAL := all
# All test recipes and child harnesses inherit the six-evaluation ceiling.
# Pure host schedule/serialization fixtures may describe longer schedules;
# executing a denoiser under this environment is guarded in the library.
export H3_TEST_MAX_EVALUATIONS := 6
# Ordinary build/test invocations never acquire model weights.
export H3_OFFLINE := 1
PLATFORM := $(shell uname -s)
CC := clang
AR := ar
SHA256 := shasum -a 256
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate
LDLIBS := $(FRAMEWORKS) -licucore -lm

LIB_C := src/engine.c src/weights/lora.c src/weights/lora_json.c src/metal/shader.c src/memory.c src/host.c src/media/preview.c src/sampling/bridge.c src/sampling/av_state.c src/sampling/sampler_state.c src/sampling/sampler_file.c src/weights/safetensors.c src/weights/weights.c src/conditioning/text_encoder.c \
	src/denoise/dit_schedule.c src/denoise/dit.c src/testing/test_teacher.c

LIB_C += src/vae/tiny_vae.c src/vae/video_vae.c src/vae/video_encoder.c src/vae/video_posterior.c src/vae/audio_vae.c src/media/ffmpeg.c src/media/refvideo.c \
	src/cli/terminal.c src/conditioning/vision_encoder.c src/conditioning/multimodal.c
COMMON_C := src/vae/image_vae.c $(LIB_C) src/media/delivery.c src/media/presentation.c src/media/decode.c src/platform.c src/cuda/cuda_policy.c src/weights/residency.c src/sglang/sglang.c src/sglang/sglang_media.c src/weights/quant.c src/weights/quant_cache.c src/weights/q8.c src/denoise/attention.c src/backend.c src/conditioning/conditioning.c src/upscale/upscale_state.c src/upscale/upscale_network.c src/upscale/upscale_plan.c src/upscale/upscale_refine.c src/upscale/upscale_runtime.c src/profile.c src/runtime/runtime.c src/denoise/sol.c src/cuda/cuda_sol_policy.c src/denoise/adaptive_cache.c src/denoise/subblock.c src/denoise/approximate.c

COMMON_C += src/log.c

.PHONY: test-weight-residency
bin/weight_residency_test: tests/weight_residency.c src/weights/residency.c src/weights/residency.h | bin
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. tests/weight_residency.c src/weights/residency.c -o $@
test-weight-residency: bin/weight_residency_test
	./bin/weight_residency_test
METAL_ATTN_SOURCES := $(shell find third_party/mlx-attention -type f | sort) scripts/embed_metal_attention.py src/metal/native_attention.metal src/metal/routed_attention.metal src/metal/sol.metal src/metal/fp16_attention.metal src/upscale/upscale.metal src/upscale/upscale_metal.inc src/metal/native_diagnostics.metal src/metal/native_attention_host.inc src/metal/fp16_attention_host.inc src/metal/native_diagnostics_host.inc src/metal/metal_vae_host.inc
ifeq ($(PLATFORM),Darwin)
PLATFORM_C := src/media/resize_metal.c
ifeq ($(PACKAGE_RUNTIME),1)
CPPFLAGS += -DH3_PACKAGE_RUNTIME
PACKAGE_M := src/runtime/macos.m src/runtime/runtime_macos.m
endif
LIB_M := $(PACKAGE_M) src/metal/metal.m src/metal/gpu.m src/conditioning/tokenizer.m src/metal/tiny_vae_metal.m
LIB_M += src/metal/ane.m src/metal/ane_split.m
LIB_CXX := src/metal/ane_graph.cpp third_party/vpipe-ane/ane-emitter.cpp
LDLIBS += -framework CoreML -framework CoreVideo -lc++
src/metal/native_attention.inc: $(METAL_ATTN_SOURCES)
	python3 scripts/embed_metal_attention.py $@
src/metal/gpu.o: src/metal/native_attention.inc
else ifeq ($(PLATFORM),Linux)
CC := gcc
SHA256 := sha256sum
CFLAGS := $(filter-out -D_DARWIN_C_SOURCE,$(CFLAGS)) -D_GNU_SOURCE
CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
CUDA_ARCH ?= auto
CUDA_GENCODE := $(shell scripts/cuda_arch.sh '$(CUDA_ARCH)' '$(NVCC)')
ifneq ($(findstring ERROR,$(CUDA_GENCODE)),)
$(error $(CUDA_GENCODE))
endif
CPPFLAGS += -I$(CUDA_PATH)/include
ifeq ($(PACKAGE_RUNTIME),1)
CPPFLAGS += -DH3_PACKAGE_RUNTIME
endif
LDLIBS := -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 -lcudart -lcublas -lcublasLt -lstdc++ -licuuc -ljson-c -lpthread -lm -ldl
# Optional CPU SHA acceleration for compatibility/container digests.
# Portable hashing remains the default execution path and the build fallback.
CUDA_OPENSSL ?= $(shell pkg-config --exists libcrypto 2>/dev/null && echo 1 || echo 0)
ifeq ($(CUDA_OPENSSL),1)
CPPFLAGS += -DH3_USE_OPENSSL $(shell pkg-config --cflags libcrypto)
LDLIBS += $(shell pkg-config --libs libcrypto)
endif
PLATFORM_C := src/digest.c src/cuda/cuda_probe.c src/media/resize_portable.c src/conditioning/tokenizer_portable.c
LIB_CUDA := src/cuda/gpu_cuda.cu src/cuda/tiny_vae_cuda.cu
LIB_CXX += src/sglang/sglang_rng.cpp
# Ordinary CUDA video requires both libraries. Minimal builds may exercise
# shared low-level primitives but reject video generation; no legacy fallback.
CUDA_SGLANG ?= 1
CUDA_CUDNN ?= 1
ifeq ($(CUDA_SGLANG),1)
ifeq ($(SGLANG_CUTLASS_PATH),)
$(error CUDA_SGLANG=1 requires SGLANG_CUTLASS_PATH at the pinned PyTorch CUTLASS revision; see third_party/flash-attention/README.md)
endif
LIB_CUDA += src/cuda/cuda_sglang_flash.cu
CPPFLAGS += -DH3_CUDA_USE_SGLANG_FLASH
SGLANG_FLASH_SOURCES := $(shell find third_party/flash-attention -type f | sort)
# Pinned PyTorch math flags, isolated from explicit approximate kernels. Approximate division causes
# BF16 tie differences that compound through the complete text encoder.
SGLANG_FLASH_FLAGS := -O3 -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda
endif
ifeq ($(CUDA_SOL),1)
LIB_CUDA += src/cuda/cuda_sol.cu
CPPFLAGS += -DH3_CUDA_USE_SOL
endif
ifeq ($(CUDA_SUBBLOCK),1)
LIB_CUDA += src/cuda/cuda_subblock.cu
CPPFLAGS += -DH3_CUDA_USE_SUBBLOCK
endif
ifeq ($(CUDA_CUDNN),1)
ifeq ($(CUDNN_FRONTEND_PATH),)
$(error CUDA_CUDNN=1 requires CUDNN_FRONTEND_PATH pointing to cudnn-frontend/include)
endif
LIB_CUDA += src/cuda/cuda_cudnn.cu
CPPFLAGS += -DH3_CUDA_USE_CUDNN -DCUDNN_FRONTEND_SKIP_JSON_LIB -I$(CUDNN_FRONTEND_PATH)
LDLIBS += -lcudnn
endif
ifeq ($(CUDA_SAGE),1)
ifeq ($(SAGE_CUTLASS_PATH),)
$(error CUDA_SAGE=1 requires SAGE_CUTLASS_PATH at pinned CUTLASS v4.2.1)
endif
LIB_CUDA += src/cuda/cuda_sage.cu src/cuda/cuda_sage2.cu src/cuda/cuda_sage3.cu
CPPFLAGS += -DH3_CUDA_USE_SAGE
LDLIBS += -lcuda
SAGE_SOURCES := $(shell find third_party/sageattention -type f | sort)
SAGE_NVCCFLAGS := -gencode=arch=compute_120a,code=sm_120a
SAGE2_NVCCFLAGS := --use_fast_math --fmad=true
SAGE3_NVCCFLAGS := --use_fast_math --fmad=true --expt-relaxed-constexpr --expt-extended-lambda -DQBLKSIZE=128 -DKBLKSIZE=128 -DCTA256 -DDQINRMEM -DEXECMODE=0 -DNDEBUG
endif
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo --fmad=false
else
$(error Unsupported platform $(PLATFORM); expected Darwin or Linux)
endif
LIB_C := $(COMMON_C) $(PLATFORM_C)
LIB_OBJ := $(LIB_C:.c=.o) $(LIB_M:.m=.o) $(LIB_CUDA:.cu=.o) $(LIB_CXX:.cpp=.o)
SERVER_C := src/server/resources.c src/server/bundle.c src/server/json.c src/server/sglang.c src/server/server.c src/server/http.c src/server/queue.c src/server/assets.c src/server/worker.c
MODEL_C := src/models/catalog.c src/models/common.c src/models/resolve.c src/models/transfer.c src/models/store.c
CLI_C := src/h3cli.c src/cli/cli_progress.c src/cli/options.c src/cli/arguments.c src/request.c $(MODEL_C) $(SERVER_C)
CLI_OBJ := $(CLI_C:.c=.o) third_party/civetweb/src/civetweb.o
CLI_LDLIBS := -lsqlite3 -lcurl -lpthread
src/models/catalog.inc: src/models/catalog.json scripts/embed_model_catalog.py
	python3 scripts/embed_model_catalog.py
src/models/catalog.o: src/models/catalog.inc
third_party/civetweb/src/civetweb.o: CFLAGS := -std=c11 -O2 -MMD -MP -D_DARWIN_C_SOURCE -D_GNU_SOURCE -DNO_SSL -DNO_CGI -DNO_FILES -DNO_CACHING -DUSE_IPV6 -Ithird_party/civetweb/include
UPSCALE_SOURCES := src/upscale/upscale.metal src/upscale/upscale_metal.inc src/upscale/upscale_schema.inc

# Source identity includes headers, shader and compiler options, even for a
# dirty working tree or a source archive without .git metadata.
SOURCE_FILES := $(sort $(filter-out src/metal/native_attention.inc,$(shell find src -type f)))
SOURCE_CODE := $(filter %.c %.m %.h %.cu %.cuh %.cpp %.metal %.inc %.def,$(SOURCE_FILES))
SOURCE_HEADERS := $(filter %.h %.cuh,$(SOURCE_CODE))
H3_BUILD_HASH := $(shell (cat $(LIB_C) $(LIB_M) $(LIB_CUDA) $(LIB_CXX) $(wildcard third_party/vpipe-ane/*.h) $(SOURCE_HEADERS) $(SAGE_SOURCES) $(SGLANG_FLASH_SOURCES) $(METAL_ATTN_SOURCES) $(UPSCALE_SOURCES) src/metal/shaders.metal src/metal/q8_host.inc src/vae/image_vae_schema.inc; echo '$(CPPFLAGS) $(CFLAGS) $(OBJCFLAGS) $(NVCCFLAGS) $(CUDA_GENCODE) $(SAGE_NVCCFLAGS) $(SAGE2_NVCCFLAGS) $(SAGE3_NVCCFLAGS) $(SGLANG_FLASH_FLAGS)') 2>/dev/null | $(SHA256) | cut -d' ' -f1)
H3_GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
H3_PORTABLE_BUILD_HASH := $(shell cat $(SOURCE_CODE) $(sort $(wildcard third_party/vpipe-ane/*.cpp third_party/vpipe-ane/*.h)) $(METAL_ATTN_SOURCES) $(UPSCALE_SOURCES) 2>/dev/null | $(SHA256) | cut -d' ' -f1)
src/sampling/sampler_state.o: override CFLAGS += -DH3_GIT_COMMIT='"$(H3_GIT_COMMIT)"' -DH3_BUILD_ID='"$(H3_BUILD_HASH)"' -DH3_PORTABLE_BUILD_ID='"$(H3_PORTABLE_BUILD_HASH)"'
src/sampling/sampler_state.o: $(SOURCE_CODE) $(wildcard third_party/vpipe-ane/*.cpp third_party/vpipe-ane/*.h) Makefile $(SAGE_SOURCES) $(METAL_ATTN_SOURCES)
src/sampling/sampler_state.o: $(SGLANG_FLASH_SOURCES)
src/sampling/sampler_state.o: $(UPSCALE_SOURCES)
src/cuda/cuda_sglang_flash.o: src/cuda/cuda_sglang_flash.cu $(SGLANG_FLASH_SOURCES)
	python3 scripts/verify_sglang_dependencies.py '$(SGLANG_CUTLASS_PATH)'
	$(NVCC) $(CPPFLAGS) $(SGLANG_FLASH_FLAGS) $(CUDA_GENCODE) -I. -I$(SGLANG_CUTLASS_PATH)/include -MMD -MP -c $< -o $@

.PHONY: all test test-sampler test-sampler-gpu test-sampler-sanitize test-continuation-sanitize test-bridge-sanitize test-bridge-quality test-bridge-gpu-sanitize clean

bin bin/sanitizers/lora-runtime:
	mkdir -p $@

all: bin/h3cli bin/libh3.a

bin/adaptive_policy_tests: tests/test_adaptive_policy.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/adaptive_probe: tests/adaptive_probe.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/adaptive_latent: tests/adaptive_latent.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/adaptive_continuation_context: tests/adaptive_continuation_context.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/subblock_native: tests/subblock_native.o src/cuda/cuda_subblock.o src/denoise/subblock.o | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/subblock_context: tests/subblock_context.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/approximate_decode: tests/approximate_decode.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-adaptive-policy
test-adaptive-policy: bin/adaptive_policy_tests bin/h3cli
	./bin/adaptive_policy_tests
	python3 tests/test_adaptive_cli.py

%.o: %.cpp
	$(if $(filter Darwin,$(PLATFORM)),clang++,$(CXX)) $(CPPFLAGS) $(filter-out -std=c11,$(CFLAGS)) -std=c++20 -I. -c $< -o $@

bin/ane_probe: tests/ane_probe.o src/metal/ane.o src/metal/ane_graph.o third_party/vpipe-ane/ane-emitter.o | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/ane_tests: tests/ane_split.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Recipe 1's multiply/add and rank order must survive caller build overrides.
src/weights/lora.o src/weights/lora_json.o: override CFLAGS += -fno-fast-math -ffp-contract=off
src/sglang/sglang.o src/sglang/sglang_rng.o: override CFLAGS += -fno-fast-math -ffp-contract=off

tests/lora_runtime_test.o: src/weights/lora.c src/weights/lora.h src/weights/lora_json.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DH3_LORA_TESTING -fno-fast-math -ffp-contract=off -I. -c $< -o $@

bin/lora_host: tests/lora_host.o tests/lora_runtime_test.o src/weights/lora_json.o src/weights/safetensors.o src/memory.o $(if $(filter Linux,$(PLATFORM)),src/digest.o src/weights/quant.o) | bin
	$(CC) $(LDFLAGS) -o $@ $^ -lm $(if $(filter 1,$(CUDA_OPENSSL)),-lcrypto)

bin/lora_runtime_generate: tests/lora_runtime_generate.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-lora-runtime test-lora-runtime-sanitize
test-lora-runtime: bin/lora_host
	python3 tests/test_lora_runtime.py

test-lora-runtime-sanitize: | bin/sanitizers/lora-runtime
	$(CC) $(CPPFLAGS) -I. -std=c11 -O1 -g -D_GNU_SOURCE -D_DARWIN_C_SOURCE -fno-fast-math -ffp-contract=off -DH3_LORA_TESTING -fsanitize=address,undefined -fno-omit-frame-pointer tests/lora_host.c src/weights/lora.c src/weights/lora_json.c src/weights/safetensors.c src/memory.c $(if $(filter Linux,$(PLATFORM)),src/digest.c src/weights/quant.c) -lm $(if $(filter 1,$(CUDA_OPENSSL)),-lcrypto) -o bin/sanitizers/lora-runtime/lora_host_sanitize
	H3_LORA_HOST=bin/sanitizers/lora-runtime/lora_host_sanitize python3 tests/test_lora_runtime.py

bin/quant_native: tests/quant_native.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/quant_projection: tests/quant_projection.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/quant_workflow: tests/quant_workflow.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/quant_prepare: src/weights/quant_prepare.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/quant_cache_tests: tests/test_quant_cache.o src/weights/quant_cache.o src/weights/quant.o $(if $(filter Linux,$(PLATFORM)),src/digest.o) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(if $(filter 1,$(CUDA_OPENSSL)),-lcrypto)

.PHONY: test-quant test-quant-cache
test-quant-cache: bin/quant_cache_tests
	./bin/quant_cache_tests

test-quant: test-sampler test-preview-vae test-quant-cache
	python3 tests/test_quant.py

bin/preview_vae_decode: tests/preview_vae_decode.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/preview_vae_tests: tests/test_preview_vae.o $(if $(filter Darwin,$(PLATFORM)),tests/preview_vae_memory.o) $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/preview_vae_generate: tests/preview_vae_generate.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-preview-vae
test-preview-vae: bin/h3cli bin/preview_vae_tests
	./bin/preview_vae_tests host
	python3 tests/test_preview_vae.py

.PHONY: test-cli
test-cli: bin/h3cli
	python3 tests/test_cli.py
	python3 tests/test_quality_cli.py

test: test-cli test-preview-vae test-quant

.PHONY: test-refvideo test-refvideo-sanitize
test-refvideo: bin/refvideo_tests bin/refvideo_layout_tests bin/h3cli bin/sampler_tests
	./bin/refvideo_tests
	./bin/refvideo_layout_tests
	python3 tests/test_refvideo.py

test-refvideo-sanitize: bin/libh3.a
	mkdir -p bin/sanitizers/refvideo-validation
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		tests/test_refvideo.c src/media/refvideo.c src/media/ffmpeg.c src/engine.c bin/libh3.a $(LDLIBS) \
		-o bin/sanitizers/refvideo-validation/refvideo_tests
	./bin/sanitizers/refvideo-validation/refvideo_tests
	H3_REFVIDEO_TEST_BINARY=bin/sanitizers/refvideo-validation/refvideo_tests python3 tests/test_refvideo.py
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		tests/test_refvideo_layout.c src/engine.c src/host.c bin/libh3.a $(LDLIBS) \
		-o bin/sanitizers/refvideo-validation/refvideo_layout_tests
	./bin/sanitizers/refvideo-validation/refvideo_layout_tests

bin/refvideo_tests: tests/test_refvideo.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-video-posterior test-video-posterior-sanitize
test-video-posterior: bin/video_posterior_tests
	./bin/video_posterior_tests

bin/video_posterior_tests: tests/test_video_posterior.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/refvideo_encoder_test: tests/refvideo_encoder.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/refvideo_layout_tests: tests/test_refvideo_layout.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/refvideo_dit_test: tests/refvideo_dit.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test-video-posterior-sanitize: bin/libh3.a
	mkdir -p bin/sanitizers/refvideo-encoder-validation
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		tests/test_video_posterior.c src/vae/video_posterior.c src/vae/video_encoder.c bin/libh3.a $(LDLIBS) \
		-o bin/sanitizers/refvideo-encoder-validation/video_posterior_tests
	./bin/sanitizers/refvideo-encoder-validation/video_posterior_tests
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		tests/refvideo_encoder.c src/vae/video_posterior.c src/vae/video_encoder.c bin/libh3.a $(LDLIBS) \
		-o bin/sanitizers/refvideo-encoder-validation/refvideo_encoder_test

bin/h3cli: $(CLI_OBJ) $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS) $(CLI_LDLIBS)

bin/libh3.a: $(LIB_OBJ) | bin
	rm -f $@
	$(AR) rcs $@ $^

bin/conditioning_tests: tests/test_conditioning.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/reference_image_tests: tests/test_reference_image.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
.PHONY: test-reference-image
test-reference-image: bin/h3cli bin/reference_image_tests
	./bin/reference_image_tests
	python3 tests/test_reference_image_cli.py

bin/conditioning_context: tests/conditioning_context.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-metal-native-host
bin/metal_attention: tests/metal_attention.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/metal_layout: tests/metal_layout.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/metal_fp16: tests/metal_fp16.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/metal_diagnostics: tests/metal_diagnostics.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/metal_sol: tests/metal_sol.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/sol_layout_tests: tests/test_sol_layout.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test-metal-native-host: bin/h3cli bin/conditioning_tests bin/sol_layout_tests bin/sampler_tests
	H3_TEST_MAX_EVALUATIONS=6 ./bin/conditioning_tests
	./bin/sol_layout_tests
	./bin/sampler_tests
	python3 tests/test_conditioning_file.py
	python3 tests/test_metal_native_cli.py
	python3 tests/test_metal_native_compare.py
	python3 tests/test_metal_fp16_records.py
	python3 tests/test_metal_teacher.py
	python3 tests/test_metal_reference_records.py
	python3 tests/test_metal_sol_records.py

.PHONY: test-metal-native-sanitize
test-metal-native-sanitize: $(LIB_OBJ)
	mkdir -p bin/sanitizers/metal-native
	$(CC) $(CPPFLAGS) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_conditioning.c src/conditioning/conditioning.c src/backend.c src/sampling/sampler_file.c $(filter-out src/conditioning/conditioning.o src/backend.o src/sampling/sampler_file.o,$(LIB_OBJ)) $(LDLIBS) -o bin/sanitizers/metal-native/conditioning
	./bin/sanitizers/metal-native/conditioning
	H3_CONDITIONING_TEST_BINARY=bin/sanitizers/metal-native/conditioning python3 tests/test_conditioning_file.py
	$(CC) $(CPPFLAGS) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_sol_layout.c src/denoise/sol.c src/backend.c $(filter-out src/denoise/sol.o src/backend.o,$(LIB_OBJ)) $(LDLIBS) -o bin/sanitizers/metal-native/sol-layout
	./bin/sanitizers/metal-native/sol-layout

test-sampler: bin/h3cli bin/sampler_tests
	./bin/sampler_tests
	python3 tests/test_sampler_file.py
	python3 tests/sampler_cli.py

.PHONY: test-upscale-state
test-upscale-state: test-sampler
	python3 tests/test_upscale_file.py
	python3 tests/test_upscale_cli.py

bin/upscale_network: tests/upscale_network.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/cuda_reference_policy_tests: tests/test_cuda_reference_policy.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-cuda-reference-policy
test-cuda-reference-policy: test-sampler bin/cuda_reference_policy_tests
	./bin/cuda_reference_policy_tests

test-sampler-gpu: bin/sampler_gpu_tests
	./bin/sampler_gpu_tests

test-sampler-sanitize: bin/libh3.a
	mkdir -p bin/sanitizers/resume-validation
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		tests/test_sampler.c src/sampling/sampler_state.c src/sampling/sampler_file.c src/sampling/av_state.c src/host.c \
		bin/libh3.a $(LDLIBS) -o bin/sanitizers/resume-validation/sampler_tests
	./bin/sanitizers/resume-validation/sampler_tests
	H3_SAMPLER_TEST_BINARY=bin/sanitizers/resume-validation/sampler_tests python3 tests/test_sampler_file.py
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		tests/test_adaptive_policy.c src/denoise/adaptive_cache.c src/denoise/approximate.c \
		bin/libh3.a $(LDLIBS) -o bin/sanitizers/resume-validation/adaptive_policy_tests
	./bin/sanitizers/resume-validation/adaptive_policy_tests

bin/sampler_generate: tests/sampler_generate.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/sampler_tests: tests/test_sampler.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/sampler_gpu_tests: tests/test_sampler_gpu.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/tests: tests/test_h3.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/continuation_tests: tests/test_continuation.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/bridge_tests: tests/test_bridge.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/bridge_gpu_tests: tests/test_bridge_gpu.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

BRIDGE_PYTHON ?= python3
test-bridge-quality:
	$(BRIDGE_PYTHON) tests/test_bridge_quality.py

test-bridge-gpu-sanitize: bin/libh3.a
	mkdir -p bin/sanitizers/bridge-quality
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		-fobjc-arc tests/test_bridge_gpu.c src/sampling/bridge.c src/denoise/dit.c src/metal/gpu.m \
		bin/libh3.a $(LDLIBS) -o bin/sanitizers/bridge-quality/bridge_gpu_tests
	./bin/sanitizers/bridge-quality/bridge_gpu_tests

bin/real_bridge_schedule_test: tests/test_real_bridge_schedule.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

test-bridge-sanitize: bin/libh3.a
	mkdir -p bin/sanitizers/bridge-validation
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		tests/test_bridge.c src/sampling/bridge.c src/sampling/av_state.c src/host.c src/denoise/dit_schedule.c src/engine.c \
		bin/libh3.a $(LDLIBS) -o bin/sanitizers/bridge-validation/bridge_tests
	./bin/sanitizers/bridge-validation/bridge_tests

bin/continuation_generate: tests/continuation_generate.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/upscale_lifecycle: tests/upscale_lifecycle.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/upscale_probe: tests/upscale_probe.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/upscale_campaign: tests/upscale_campaign.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Instrument the new host/state/schedule paths and their callers. The archive
# supplies unchanged platform code; this host suite does not initialize Metal.
test-continuation-sanitize: bin/libh3.a
	mkdir -p bin/sanitizers/continuation-validation
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		tests/test_continuation.c src/sampling/av_state.c src/host.c src/denoise/dit_schedule.c src/engine.c src/denoise/dit.c \
		bin/libh3.a $(LDLIBS) -o bin/sanitizers/continuation-validation/continuation_tests
	./bin/sanitizers/continuation-validation/continuation_tests


bin/bf16_tests: tests/test_bf16.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/tokenizer_tests: tests/test_tokenizer.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/tokenizer_conformance_tests: tests/test_tokenizer_conformance.o tests/test_tokenizer_multimodal.o $(filter src/conditioning/tokenizer.o src/conditioning/tokenizer_portable.o,$(LIB_OBJ)) src/conditioning/multimodal.o | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

ifeq ($(PLATFORM),Darwin)
bin/tokenizer_dump: tests/tokenizer_dump.o src/conditioning/tokenizer.o | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
endif

.PHONY: test-tokenizer test-tokenizer-sanitize
test-tokenizer: bin/tokenizer_conformance_tests bin/tokenizer_tests bin/tokenizer_dump
	./bin/tokenizer_conformance_tests
	@found=0; for mode in FL2VA Ref2VA; do \
		if test -f models/MiniMax-H3/$$mode/tokenizer/tokenizer.json; then \
			found=1; ./bin/tokenizer_tests models/MiniMax-H3/$$mode/tokenizer/tokenizer.json || exit 1; \
			python3 tests/tokenizer_reference.py models/MiniMax-H3/$$mode/tokenizer --golden-only --native ./bin/tokenizer_dump || exit 1; \
		fi; \
	done; \
	if test $$found = 0; then echo "skip: released Qwen vocabulary is not installed (H3 conformance tested above)"; fi
	python3 tests/test_tokenizer_diagnostics.py

test-tokenizer-sanitize:
	mkdir -p bin/sanitizers/tokenfix-validation
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		$(if $(filter Darwin,$(PLATFORM)),-fobjc-arc) tests/test_tokenizer_conformance.c tests/test_tokenizer_multimodal.c \
		$(if $(filter Darwin,$(PLATFORM)),src/conditioning/tokenizer.m,src/conditioning/tokenizer_portable.c) src/conditioning/multimodal.c $(LDLIBS) -o bin/sanitizers/tokenfix-validation/tokenizer_tests
	./bin/sanitizers/tokenfix-validation/tokenizer_tests


bin/audio_gpu_tests: tests/test_audio_gpu.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/audio_long_test: tests/test_audio_long.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)



bin/av_mux_test: tests/test_av_mux.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)









bin/semantic_dit_test: tests/test_semantic_dit.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/dit_bench: tests/bench_dit.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/dit_bench_864: tests/bench_dit_864.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

tests/bench_dit_864.o: tests/bench_dit.c
	$(CC) $(CFLAGS) -I. -DH3_BENCH_LATENT_H=30 \
		-DH3_BENCH_LATENT_W=54 -c $< -o $@


bin/semantic_vae_test: tests/test_semantic_vae.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

ifeq ($(PLATFORM),Darwin)
test: test-progress test-bugfix1 test-preview test-qwen-scaling test-video-vae-tiles test-tokenizer bin/memory_tests bin/memory_gpu_tests bin/refvideo_layout_tests bin/video_posterior_tests bin/refvideo_tests bin/sampler_tests bin/sampler_gpu_tests bin/tests bin/continuation_tests bin/bridge_tests bin/bridge_gpu_tests bin/bf16_tests bin/audio_gpu_tests bin/av_mux_test
	./bin/memory_tests
	./bin/memory_gpu_tests
	./bin/video_posterior_tests
	./bin/sampler_gpu_tests
	./bin/tests
	./bin/continuation_tests
	./bin/bridge_tests
	./bin/bridge_gpu_tests
	./bin/bf16_tests
	./bin/audio_gpu_tests
	./bin/av_mux_test
	python3 tests/continuation_oracle.py

else
test: cuda-host-test cuda-test cuda-tokenizer-test
endif

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c $< -o $@

%.o: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_GENCODE) -I. -MMD -MP -c $< -o $@

ifeq ($(PLATFORM),Linux)
.PHONY: cuda-build-config-force
.cuda-build-config: cuda-build-config-force
	@printf '%s\n' '$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_GENCODE)' > $@.tmp
	@cmp -s $@.tmp $@ && rm $@.tmp || mv $@.tmp $@
$(LIB_OBJ) $(CLI_OBJ) src/cuda/gpu_cuda.fat.o: .cuda-build-config
$(patsubst %.c,%.o,$(wildcard tests/*.c)) $(patsubst %.cu,%.o,$(wildcard tests/*.cu)): .cuda-build-config
endif

%.o: %.m
	$(CC) $(CPPFLAGS) $(OBJCFLAGS) -I. -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c $< -o $@

.PHONY: cuda-fat
cuda-fat:
	$(MAKE) CUDA_ARCH=fat bin/h3cli-cuda-fat

bin/h3cli-cuda-fat: $(CLI_OBJ) $(filter-out src/cuda/gpu_cuda.o,$(LIB_OBJ)) src/cuda/gpu_cuda.fat.o | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS) $(CLI_LDLIBS)

src/cuda/gpu_cuda.fat.o: src/cuda/gpu_cuda.cu $(SOURCE_HEADERS) scripts/cuda_arch.sh
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(shell scripts/cuda_arch.sh fat '$(NVCC)') -I. -MMD -MP -c $< -o $@

-include $(filter %.d,$(SOURCE_FILES)) $(wildcard tests/*.d third_party/vpipe-ane/*.d third_party/civetweb/src/*.d)

clean:
	rm -rf bin
	rm -f bin/libh3.a .cuda-build-config .cuda-build-config.tmp src/metal/native_attention.inc
	rm -f *.o *.d
	find src tests third_party/vpipe-ane third_party/civetweb -type f \( -name '*.o' -o -name '*.d' \) -delete

.PHONY: test-memory test-memory-gpu test-memory-models test-memory-sanitize
bin/memory_tests: tests/test_memory.o src/memory.o | bin
	$(CC) $(LDFLAGS) -o $@ $^ -lm
bin/memory_gpu_tests: tests/test_memory_gpu.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/metal_lifetime: tests/metal_lifetime.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/metal_cache_lifetime: tests/metal_cache_lifetime.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
test-memory: bin/memory_tests
	./bin/memory_tests
	python3 tests/memory_decoder.py
test-memory-gpu: bin/memory_gpu_tests
	./bin/memory_gpu_tests
test-memory-models: bin/libh3.a
	python3 tests/memory_cancellation.py
test-memory-sanitize:
	H3_TEST_LDLIBS="$(LDFLAGS) $(LDLIBS)" python3 tests/memory_decoder.py --sanitize

.PHONY: test-video-vae-tiles test-video-vae-tiles-sanitize
# These test translation units include the private decoder implementation.
bin/video_vae_tile_tests: tests/test_video_vae_tiles.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/tilefix_decode: tests/tilefix_decode.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
test-video-vae-tiles: bin/video_vae_tile_tests bin/tilefix_decode
	./bin/video_vae_tile_tests
	python3 tests/test_tilefix_policy.py
test-video-vae-tiles-sanitize: bin/libh3.a
	mkdir -p bin/sanitizers/tilefix-validation
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		tests/test_video_vae_tiles.c bin/libh3.a $(LDLIBS) -o bin/sanitizers/tilefix-validation/policy
	./bin/sanitizers/tilefix-validation/policy

bin/qwen_scaling_tests: tests/test_qwen_scaling.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
.PHONY: test-qwen-scaling
test-qwen-scaling: bin/qwen_scaling_tests
	./bin/qwen_scaling_tests
bin/qwen_scaling_bench: tests/scalingfix_bench.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-preview test-preview-sanitize
bin/preview_tests: tests/test_preview.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
test-preview: bin/preview_tests bin/preview_cache_tests bin/libh3.a
	./bin/preview_tests
	./bin/preview_cache_tests
	python3 tests/denoise_sampler.py

test-preview-sanitize: bin/libh3.a
	mkdir -p bin/sanitizers/denoise-validation
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		-fobjc-arc tests/test_preview.c src/media/preview.c src/metal/gpu.m bin/libh3.a $(LDLIBS) \
		-o bin/sanitizers/denoise-validation/preview
	./bin/sanitizers/denoise-validation/preview
	$(CC) -I. -std=c11 -D_DARWIN_C_SOURCE -g -O1 -fsanitize=address,undefined \
		tests/test_preview_cache.c bin/libh3.a $(LDLIBS) \
		-o bin/sanitizers/denoise-validation/cache
	./bin/sanitizers/denoise-validation/cache
	python3 tests/denoise_sampler.py --sanitize

bin/preview_generate: tests/denoise_generate.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/preview_cache_tests: tests/test_preview_cache.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/preview_decode: tests/denoise_decode.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/bugfix1_probe: tests/bugfix1_probe.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-bugfix1
test-bugfix1: bin/bugfix1_probe bin/video_vae_tile_tests bin/tilefix_decode
	python3 tests/test_bugfix1.py

.PHONY: test-bugfix1-sanitize
test-bugfix1-sanitize: bin/libh3.a
	mkdir -p bin/sanitizers/bugfix1-validation
	$(CC) -I. $(CFLAGS) -O1 -g -fsanitize=address,undefined -fobjc-arc \
		tests/bugfix1_probe.c src/metal/shader.c src/weights/safetensors.c src/weights/weights.c src/metal/gpu.m bin/libh3.a $(LDLIBS) \
		-o bin/sanitizers/bugfix1-validation/probe
	H3_BUGFIX1_PROBE=bin/sanitizers/bugfix1-validation/probe python3 tests/test_bugfix1.py

.PHONY: test-progress
bin/progress_tests: tests/test_progress.o src/cli/cli_progress.o | bin
	$(CC) $(LDFLAGS) -o $@ $^
bin/log_tests: tests/test_log.o src/log.o | bin
	$(CC) $(LDFLAGS) -o $@ $^
test-progress: bin/progress_tests bin/log_tests
	./bin/progress_tests
	./bin/log_tests
	python3 tests/test_attention_log.py

bin/progress_models_test: tests/progress_models.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
.PHONY: test-progress-models
test-progress-models: bin/progress_models_test
	./bin/progress_models_test

bin/progress_vae_test: tests/progress_vae.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
.PHONY: test-progress-vae
test-progress-vae: bin/progress_vae_test
	mkdir -p outputs/vae-progress-validation/decoder
	./bin/progress_vae_test outputs/vae-progress-validation/decoder

# Optional offline LoRA tooling has its own isolated Python dependencies.
LORA_PYTHON ?= lora/.venv/bin/python
.PHONY: test-lora test-lora-models
test-lora:
	$(LORA_PYTHON) -m unittest discover -s lora/tests -v

test-lora-models: all bin/memory_tests bin/qwen_scaling_tests
	$(LORA_PYTHON) lora/tests/integration.py

bin/gpu_contract_test: tests/test_gpu_contract.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
.PHONY: test-gpu-contract cuda-test
test-gpu-contract: bin/gpu_contract_test
	python3 tests/check_gpu_contract.py
	./bin/gpu_contract_test
cuda-test: test-gpu-contract bin/cuda_runtime_test bin/bf16_tests bin/sampler_gpu_tests bin/bridge_gpu_tests bin/audio_gpu_tests bin/qwen_scaling_tests bin/preview_tests bin/preview_cache_tests
	mkdir -p outputs/cuda-validation
	./bin/cuda_runtime_test
	./bin/bf16_tests
	./bin/sampler_gpu_tests
	./bin/bridge_gpu_tests
	./bin/audio_gpu_tests
	./bin/qwen_scaling_tests
	./bin/preview_tests
	./bin/preview_cache_tests

ifeq ($(PLATFORM),Linux)
bin/tokenizer_dump: tests/tokenizer_dump_portable.o src/conditioning/tokenizer_portable.o | bin
	$(CC) $(LDFLAGS) -o $@ $^ -licuuc -ljson-c
endif

bin/cuda_resume_test: tests/cuda_resume.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/cuda_references_test: tests/cuda_references.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/cuda_runtime_test: tests/test_cuda_runtime.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/weight_residency_context: tests/weight_residency_context.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/cuda_memory_test: tests/cuda_memory.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
.PHONY: cuda-memory-test
cuda-memory-test: bin/cuda_memory_test
	./bin/cuda_memory_test
bin/cuda_conv_test: tests/cuda_conv.o tests/cuda_operator_reference.o $(filter-out src/cuda/gpu_cuda.o,$(LIB_OBJ)) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
.PHONY: cuda-conv-test
cuda-conv-test: bin/cuda_conv_test
	./bin/cuda_conv_test
bin/cuda_attention_test: tests/cuda_attention.o tests/cuda_operator_reference.o $(filter-out src/cuda/gpu_cuda.o,$(LIB_OBJ)) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
tests/cuda_operator_reference.o: tests/cuda_operator_reference.cu src/cuda/gpu_cuda.cu
bin/cuda_sglang_replay: tests/cuda_sglang_replay.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/cuda_sglang_time: tests/cuda_sglang_time.o $(filter-out src/denoise/dit_schedule.o,$(LIB_OBJ)) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/cuda_sglang_vae: tests/cuda_sglang_vae.o $(filter-out src/vae/video_vae.o,$(LIB_OBJ)) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/cuda_sglang_vision: tests/cuda_sglang_vision.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/cuda_sglang_audio: tests/cuda_sglang_audio.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/cuda_sglang_encoder: tests/cuda_sglang_encoder.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/cuda_sglang_media_input: tests/cuda_sglang_media_input.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/cuda_sglang_encoder_recovery: tests/cuda_sglang_encoder_recovery.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
tests/cuda_sglang_vae.o: tests/fast_vae_bench.c src/vae/video_vae.c src/sglang/sglang.h
tests/cuda_sglang_time.o: src/denoise/dit_schedule.c src/sglang/sglang.h
bin/cuda_sglang_isolation: tests/cuda_sglang_isolation.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/cuda_sglang_patch_test: tests/cuda_sglang_patch.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
.PHONY: test-cuda-sglang-patch
test-cuda-sglang-patch: bin/cuda_sglang_patch_test
	./bin/cuda_sglang_patch_test
.PHONY: cuda-attention-test
cuda-attention-test: bin/cuda_attention_test
	./bin/cuda_attention_test
ifeq ($(PLATFORM),Linux)
cuda-test: cuda-conv-test cuda-attention-test cuda-sm90-test cuda-sm120-test
bin/cuda_sm90_test bin/cuda_sm120_test: .cuda-build-config | bin
endif
bin/cuda_sm90_test: tests/cuda_fixed128.cu src/cuda/cuda_attention.cuh src/cuda/cuda_dispatch.h | bin
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_GENCODE) -DH3_TEST_TARGET_SM=90 -I. tests/cuda_fixed128.cu -o $@
bin/cuda_sm120_test: tests/cuda_fixed128.cu src/cuda/cuda_attention.cuh src/cuda/cuda_dispatch.h | bin
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_GENCODE) -DH3_TEST_TARGET_SM=120 -I. tests/cuda_fixed128.cu -o $@
.PHONY: cuda-sm90-test cuda-sm120-test
cuda-sm90-test: bin/cuda_sm90_test
	./bin/cuda_sm90_test
cuda-sm120-test: bin/cuda_sm120_test
	./bin/cuda_sm120_test
.PHONY: cuda-smoke cuda-features
cuda-smoke: bin/h3cli
	python3 tests/cuda_integration.py smoke
cuda-features: bin/h3cli bin/cuda_resume_test bin/continuation_generate
	python3 tests/cuda_integration.py features

.PHONY: cuda-host-test
cuda-host-test: test-current-host bin/tests bin/continuation_tests bin/bridge_tests bin/memory_tests bin/progress_tests
	./bin/tests
	./bin/continuation_tests
	./bin/bridge_tests
	./bin/memory_tests
	./bin/progress_tests
	python3 tests/continuation_oracle.py

.PHONY: cuda-tokenizer-test
cuda-tokenizer-test: test-tokenizer

# Sage instruction families are specialized independently of the generic fat binary.
src/cuda/cuda_sage2.o: src/cuda/cuda_sage2.cu src/cuda/cuda_sage_internal.cuh $(SAGE_SOURCES)
	$(NVCC) $(CPPFLAGS) $(filter-out --fmad=%,$(NVCCFLAGS)) $(SAGE_NVCCFLAGS) $(SAGE2_NVCCFLAGS) -MMD -MP -I. -c $< -o $@
src/cuda/cuda_sage3.o: src/cuda/cuda_sage3.cu src/cuda/cuda_sage_internal.cuh $(SAGE_SOURCES)
	$(NVCC) $(CPPFLAGS) $(filter-out --fmad=%,$(NVCCFLAGS)) $(SAGE_NVCCFLAGS) $(SAGE3_NVCCFLAGS) -I$(SAGE_CUTLASS_PATH)/include -I$(SAGE_CUTLASS_PATH)/tools/util/include -MMD -MP -I. -c $< -o $@

bin/attention_native: tests/attention_native.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/attention_failures: tests/attention_failures.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/attention_context: tests/attention_context.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-attention-host
test-attention-host: test-sampler test-preview-vae
	python3 tests/test_attention.py

.PHONY: sage-dependencies
sage-dependencies:
	python3 scripts/verify_sage_dependencies.py '$(SAGE_CUTLASS_PATH)'
ifeq ($(CUDA_SAGE),1)
src/cuda/cuda_sage2.o src/cuda/cuda_sage3.o: | sage-dependencies
endif

# M6 Q8 host packing, descriptor/linear correctness and bounded microbenchmark.
bin/metal_q8: tests/metal_q8.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/still_probe: tests/still_probe.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/still_tests: tests/test_still.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
.PHONY: test-still-host test-still-gpu test-still-sanitize
test-still-host: bin/h3cli bin/still_tests bin/still_probe bin/still_layout_fixture
	python3 tests/still_host.py ./bin/still_tests
	python3 tests/test_still_cli.py
	python3 tests/test_still_metadata.py
	python3 tests/test_still_layout.py
test-still-gpu: bin/still_tests
	./bin/still_tests gpu
test-still-sanitize: bin/libh3.a
	mkdir -p bin/sanitizers/single-still
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -g -fsanitize=address,undefined -I. tests/test_still.c bin/libh3.a $(LDLIBS) -o bin/sanitizers/single-still/still
	python3 tests/still_host.py ./bin/sanitizers/single-still/still

bin/still_generate: tests/still_generate.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean-still:
	rm -f bin/still_probe bin/still_generate bin/still_tests bin/still_controls bin/still_layout_fixture bin/still_lifecycle
.PHONY: clean-still
clean: clean-still

bin/still_controls: tests/still_controls.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/still_layout_fixture: tests/still_layout_fixture.o src/host.o | bin
	$(CC) $(LDFLAGS) -o $@ $^ -lm

bin/still_lifecycle: tests/still_lifecycle.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/cuda_sol_policy_tests: tests/test_cuda_sol_policy.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/cuda_sol_native: tests/cuda_sol_native.o src/cuda/cuda_sol.o src/cuda/cuda_sol_policy.o | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-cuda-sol-host
test-cuda-sol-host: test-sampler test-preview-vae bin/cuda_sol_policy_tests bin/sol_layout_tests
	./bin/cuda_sol_policy_tests
	./bin/sol_layout_tests
	python3 tests/test_cuda_sol_cli.py

bin/full_vae_tests: tests/test_full_vae.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
.PHONY: test-full-vae
test-full-vae: bin/h3cli bin/full_vae_tests bin/video_vae_tile_tests bin/video_posterior_tests bin/tilefix_decode
	./bin/full_vae_tests
	./bin/video_vae_tile_tests
	./bin/video_posterior_tests
	python3 tests/test_tilefix_policy.py
bin/fast_vae_encoder: tests/fast_vae_encoder.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/fast_vae_batch: tests/fast_vae_batch.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS) -lpthread
bin/fast_vae_fixture: tests/fast_vae_fixture.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/fast_vae_lifetime: tests/fast_vae_lifetime.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/fast_vae_session: tests/fast_vae_session.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/fast_vae_delivery: tests/fast_vae_delivery.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/fast_vae_resume: tests/fast_vae_resume.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
bin/fast_vae_tiny_bench: tests/fast_vae_tiny_bench.o bin/libh3.a | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
clean-fast-vae-tiny:
	rm -f bin/fast_vae_tiny_bench
.PHONY: clean-fast-vae-tiny
clean: clean-fast-vae-tiny

bin/cuda_sglang_soundtrack_input: tests/cuda_sglang_soundtrack_input.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/cuda_sglang_audio_encode: tests/cuda_sglang_audio_encode.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/cuda_sglang_session: tests/cuda_sglang_session.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/cuda_sglang_lifetime: tests/cuda_sglang_lifetime.o $(filter-out src/vae/video_vae.o,$(LIB_OBJ)) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/cuda_reference_probe: tests/cuda_reference_probe.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
tests/cuda_reference_probe.o: tests/cuda_sglang_isolation.c
.PHONY: test-cuda-reference-regression
test-cuda-reference-regression:
	python3 tests/cuda_reference_regression.py --source . --out "$${H3_REFERENCE_REGRESSION_OUT:?set a fresh output directory}"

bin/cuda_policy_context: tests/cuda_policy_context.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
tests/cuda_policy_context.o: tests/cuda_sglang_isolation.c

bin/cuda_shared_exact: tests/cuda_shared_exact.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
tests/cuda_shared_exact.o: tests/cuda_sglang_isolation.c

# Mutable exact-substitution probe includes backend internals for fault injection;
# it links all other production objects, never a second backend implementation.
bin/cuda_reference_exact: tests/cuda_reference_exact.o $(filter-out src/cuda/gpu_cuda.o,$(LIB_OBJ)) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
tests/cuda_reference_exact.o: tests/cuda_reference_exact.cu src/cuda/gpu_cuda.cu src/cuda/cuda_sglang_vae.cuh
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(CUDA_GENCODE) -I. -MMD -MP -c $< -o $@

bin/cuda_single_matrix: tests/cuda_single_matrix.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/cuda_sol_context: tests/cuda_sol_context.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Functional image-capacity checks, independent of the recorded CUDA gate.
bin/reference_image_gpu: tests/reference_image_gpu.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/reference_image_vision: tests/reference_image_vision.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/cuda_multi_reference_host: tests/cuda_multi_reference_host.o tests/test_tokenizer_multimodal.o src/host.o src/media/refvideo.o src/media/ffmpeg.o src/memory.o src/conditioning/multimodal.o $(filter src/conditioning/tokenizer.o src/conditioning/tokenizer_portable.o,$(LIB_OBJ)) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Complete current host/API coverage, in addition to platform operator tests.
# Historical CUDA numerical baselines have been retired; the recorded SGLang
# regression is the sole numerical gate and runs separately in an isolated build.
.PHONY: test-current-host test-setup test-linux-build-host
test-linux-build-host:
	python3 tests/test_linux_build.py
	python3 tests/test_cuda_reference_gate.py
test-setup:
	python3 tests/setup_scripts.py

test-current-host: test-sampler test-refvideo test-reference-image test-upscale-state test-cuda-reference-policy test-weight-residency test-adaptive-policy test-still-host test-full-vae test-attention-host test-cuda-sol-host test-lora-runtime test-setup test-server-contract test-server-dependency test-linux-build-host bin/conditioning_tests
	./bin/conditioning_tests
	python3 tests/test_conditioning_file.py

test: test-current-host

.PHONY: test-output-encoding
bin/output_encoding: tests/output_encoding.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)
test-output-encoding: bin/output_encoding bin/h3cli
	python3 tests/output_encoding.py
test: test-output-encoding

# Requires the qualified optional CUDA kernels and a freshly generated AV state.
# Kept separate because these probes load real models and exercise GPU lifetimes.
.PHONY: test-current-cuda
test-current-cuda: all
	python3 tests/current_cuda_suite.py --out "$${H3_CUDA_CURRENT_OUT:?set a fresh output directory}" --state "$${H3_CUDA_CURRENT_STATE:?set a current CUDA AV state}" --cow-directory "$${H3_CUDA_COW_TMP:?set a scratch directory on a CoW filesystem}"

SERVER_CONTRACT_C := tests/server_contract.c src/server/sglang.c src/server/json.c src/request.c src/cli/options.c src/cli/arguments.c src/weights/lora_json.c
bin/server_contract: $(SERVER_CONTRACT_C) src/cli/options.def | bin
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. $(SERVER_CONTRACT_C) -lm -o $@
# Deliberately separate executable: no fault injection in production workers.
src/server/worker.test.o: src/server/worker.c tests/server_fake_worker.inc
	$(CC) $(CPPFLAGS) $(CFLAGS) -DH3_SERVER_TESTING -I. -c $< -o $@
src/server/queue.test.o: src/server/queue.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -DH3_SERVER_TESTING -I. -c $< -o $@
src/server/assets.test.o: src/server/assets.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -DH3_SERVER_TESTING -I. -c $< -o $@
bin/h3cli-server-test: $(filter-out src/server/worker.o src/server/queue.o src/server/assets.o,$(CLI_OBJ)) src/server/worker.test.o src/server/queue.test.o src/server/assets.test.o $(LIB_OBJ) | bin
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS) $(CLI_LDLIBS)
.PHONY: test-server-contract test-server-http test-server-sanitize test-server-dependency
test-server-dependency:
	python3 tests/server_dependency.py
bin/server_artifacts: tests/server_artifacts.c bin/libh3.a | bin
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. $< bin/libh3.a $(LDLIBS) -o $@
test-server-contract: bin/server_contract bin/h3cli
	python3 tests/server_contract.py
test-server-http: bin/h3cli-server-test
	python3 tests/server_http.py
test-server-sanitize: | bin
	$(CC) $(CPPFLAGS) -std=c11 -D_DARWIN_C_SOURCE -D_GNU_SOURCE -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -I. $(SERVER_CONTRACT_C) -lm -o bin/server_contract_sanitize
	H3_SERVER_CONTRACT_BINARY=bin/server_contract_sanitize python3 tests/server_contract.py

SERVER_SANITIZE_OBJ := $(patsubst %.c,bin/sanitizers/server/%.o,$(CLI_C)) bin/sanitizers/server/third_party/civetweb/src/civetweb.o
bin/sanitizers/server/%.o: %.c src/cli/options.def
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) -std=c11 -D_DARWIN_C_SOURCE -D_GNU_SOURCE -DH3_SERVER_TESTING -DNO_SSL -DNO_CGI -DNO_FILES -DNO_CACHING -DUSE_IPV6 -Ithird_party/civetweb/include -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -I. -c $< -o $@
bin/h3cli-server-sanitize: $(SERVER_SANITIZE_OBJ) $(LIB_OBJ)
	$(CC) $(LDFLAGS) -fsanitize=address,undefined -o $@ $^ $(LDLIBS) $(CLI_LDLIBS)
.PHONY: test-server-http-sanitize
test-server-http-sanitize: bin/h3cli-server-sanitize
	H3_SERVER_HTTP_BINARY=bin/h3cli-server-sanitize python3 tests/server_http.py

# Test-only HTTP fixture entry point; production embeds immutable HTTPS sources.
bin/model_downloads: tests/model_downloads.c $(MODEL_C) src/server/json.c src/weights/lora_json.c src/request.c src/cli/options.c src/cli/arguments.c bin/libh3.a | bin
	$(CC) $(CPPFLAGS) $(CFLAGS) -DH3_MODEL_TESTING -I. $(filter %.c,$^) bin/libh3.a $(LDFLAGS) $(LDLIBS) -lcurl -lpthread -o $@
.PHONY: test-model-downloads
test-model-downloads: bin/model_downloads bin/h3cli bin/sampler_tests
	python3 scripts/embed_model_catalog.py --check
	python3 tests/model_downloads.py
test-current-host: test-model-downloads

ifeq ($(PLATFORM),Darwin)
.PHONY: test-macos-package
test-macos-package:
	python3 tests/test_macos_bundle.py
endif
