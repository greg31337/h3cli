#!/bin/bash
# Executed inside the locked x86-64 builder. All inputs are fetched separately.
set -euo pipefail
prefix=/opt/h3deps
build=/build/dependencies
cache=/inputs
jobs=${H3_BUILD_JOBS:-8}
mkdir -p "$prefix" "$build"
cd "$build"
extract() { mkdir -p "$2"; tar -xf "$cache/$1" --strip-components=1 -C "$2"; }
extract icu4c-74_2-src.tgz icu
(cd icu/source; ./configure --prefix="$prefix" --disable-tests --disable-samples; make -j"$jobs"; make install)
extract libjpeg-turbo-2.1.5.tar.gz jpeg
cmake -S jpeg -B jpeg-build -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR="$prefix/lib" -DCMAKE_BUILD_TYPE=Release -DENABLE_SHARED=TRUE -DENABLE_STATIC=FALSE
cmake --build jpeg-build -j"$jobs"
cmake --install jpeg-build
extract ffmpeg-9.0.2.tar.xz ffmpeg
(cd ffmpeg; ./configure --prefix="$prefix" --disable-debug --disable-doc --disable-autodetect --enable-gpl --enable-libx264 --enable-zlib --enable-static --disable-shared; make -j"$jobs"; make install)
# Historical decoding is a validation-only dependency.
extract ffmpeg-6.1.1.tar.xz reference-ffmpeg
(cd reference-ffmpeg; ./configure --prefix="$prefix/reference-media" --disable-debug --disable-doc --disable-autodetect --enable-gpl --enable-libx264 --enable-zlib --enable-static --disable-shared; make -j"$jobs"; make install)
extract cudnn-frontend.tar.gz cudnn-frontend
extract sglang-cutlass.tar.gz sglang-cutlass
extract sage-cutlass.tar.gz sage-cutlass
python3 -m venv "$prefix/venv"
"$prefix/venv/bin/python" -m pip install --no-index --find-links="$cache/wheels" --no-deps --require-hashes -r /source/scripts/requirements-sglang-runtime.txt -r /source/scripts/requirements-reference-media.txt
site=$("$prefix/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
ln -sfn libcudnn.so.9 "$site/nvidia/cudnn/lib/libcudnn.so"
ln -sfn libcublas.so.13 "$site/nvidia/cu13/lib/libcublas.so"
ln -sfn libcublasLt.so.13 "$site/nvidia/cu13/lib/libcublasLt.so"
cat > "$prefix/environment.sh" <<EOF
export CUDA_PATH=/usr/local/cuda-13.0
export NVCC=/usr/local/cuda-13.0/bin/nvcc
export PATH="$prefix/bin:/usr/local/cuda-13.0/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_ARCH=fat CUDA_SGLANG=1 CUDA_CUDNN=1 CUDA_SAGE=1 CUDA_SOL=1 CUDA_SUBBLOCK=1 CUDA_OPENSSL=1
export CUDNN_FRONTEND_PATH="$build/cudnn-frontend/include"
export SGLANG_CUTLASS_PATH="$build/sglang-cutlass"
export SAGE_CUTLASS_PATH="$build/sage-cutlass"
export CPATH="$prefix/include:$site/nvidia/cudnn/include"
export LIBRARY_PATH="$prefix/lib:$prefix/lib64:$site/nvidia/cudnn/lib:$site/nvidia/cu13/lib:/usr/local/cuda-13.0/lib64/stubs"
export LD_LIBRARY_PATH="$prefix/lib:$prefix/lib64:$site/nvidia/cudnn/lib:$site/nvidia/cu13/lib:/usr/local/cuda-13.0/lib64"
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig:$prefix/lib64/pkgconfig"
export H3_SGLANG_CUBLAS_LIBRARY="$site/nvidia/cu13/lib/libcublas.so.13"
export H3_SGLANG_CUDNN_LIBRARY="$site/nvidia/cudnn/lib/libcudnn.so.9"
export H3_SGLANG_JPEG_LIBRARY="$prefix/lib/libturbojpeg.so.0"
export H3_SGLANG_INPUT_FFMPEG="$prefix/bin/ffmpeg"
export H3_FFPROBE="$prefix/bin/ffprobe"
export H3_FFMPEG="$prefix/bin/ffmpeg"
EOF
