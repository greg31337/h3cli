# Source materials and component terms

Distribute `h3cli-linux-x86_64-sources.tar.gz` and its checksum beside the
matching `h3cli-linux-x86_64` executable. The source archive is a companion
release asset; users need only the executable to run h3cli. Nothing in this
build process publishes either asset. The release page must link both files.

`share/licenses/redistribution.json` in the extracted runtime identifies every
shipped binary, its source package or upstream source, and its retained license
notice. It also inventories the static launcher's dependencies. The project MIT
license applies to h3cli; dependency licenses remain in force. In particular,
FFmpeg is built with GPL components and libx264 and is GPL-2.0-or-later. h3cli
invokes it as a separate process. NVIDIA runtime libraries retain their NVIDIA
terms and are copied without modifying their bytes. No CUDA driver is included.
This software contains source code provided by NVIDIA Corporation.

The source archive contains:

- `h3cli/`: the exact checkout used for this release, including local changes,
  vendored code/notices and complete build, packaging and relinking scripts.
- Pinned upstream archives for FFmpeg 9.0.2, ICU, TurboJPEG, CUTLASS and the cuDNN
  frontend. Their hashes are in `h3cli/scripts/linux/lock.json`.
- `ubuntu/`: exact `.dsc`, upstream archives, Ubuntu/Debian patch archives and
  available detached signatures for every packaged distro library and the static
  launcher dependencies. The lock includes source versions, download URLs and
  hashes. `.dsc` SHA-256 lists bind each original archive and patch archive.
- `configuration/`: generated media configuration alongside the authoritative
  configure/CMake/compiler commands in `h3cli/scripts/linux/`.
- `redistribution.json` and `manifest.json`: component mapping and per-file hashes.

From `h3cli/` on Linux x86-64, install Python 3.10+, GNU coreutils, skopeo and
umoci, then run `scripts/build_linux.sh`. This fetches checksum-pinned build
inputs and produces the executable, runtime, sources and separate validation
kit. Use `--fetch-only --cache PATH` once and then `--offline --cache PATH`
with fresh `--work-dir` and `--output` directories for offline rebuilds. Models
and a GPU are unnecessary for compilation. See `docs/build/linux-distribution.md`
for full build and runtime instructions.

For distro libraries, extract the matching `.dsc` with `dpkg-source -x` and
follow its `debian/rules` and `debian/control` build dependencies in the locked
Ubuntu 22.04 environment. Both upstream source and packaging patches are
included. The FFmpeg command is in `build_dependencies.sh`; no upstream FFmpeg
source is patched. Packaging sets relative ELF paths on open-source binaries
using `package.py`; it leaves NVIDIA libraries unchanged. The static launcher
compile/link command is in `build_release.sh`, and its full source and libc,
libgcc, OpenSSL, json-c and zlib sources are supplied.

You may rebuild h3cli/its launcher and relink with modified open-source
libraries under their applicable licenses. Repackage the resulting runtime with
the supplied build scripts to regenerate the integrity manifests and launcher
payload. The package checks accidental corruption; it is not signed or locked
to a publisher key. NVIDIA component terms apply separately to their binaries.

The old FFmpeg 4.2.2 encoder and 6.1.1 decoder/probe are exclusively private
regression assets in `linux-validation/`. Do not upload that validation kit as
part of the public release: this source archive does not purport to provide
corresponding source for its historical prebuilt encoder.

License references: [FFmpeg](https://ffmpeg.org/legal.html),
[GPLv2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html),
[LGPLv2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html),
[CUDA](https://docs.nvidia.com/cuda/archive/13.0.3/eula/index.html), and
[cuDNN](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html).
The copies supplied with the pinned components are authoritative for this build.
