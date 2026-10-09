# Build and check the Linux download

Use the clean clone from the [release guide](README.md). Build and test on the
qualified Linux x86-64 PRO 5000 to avoid moving private runtimes and tool caches
between machines. No signing key is needed on this machine.

## Configure once

Replace the repository, version, full commit and model path:

```sh
./release/linux/step1-prerequisites.sh --install
./release/configure.sh --repo OWNER/REPO --tag v0.1.0 \
  --commit REPLACE_WITH_FULL_COMMIT_ID --models-path /absolute/path/to/models
```

The main model defaults to `MODELS/MiniMaxH3`, with an existing `MiniMax-H3`
fallback. Add `--model /custom/main-model` if necessary. Set `--jobs 16` for a
larger builder, or reduce the default eight jobs if memory is limited.

## Run the whole Linux sequence

```sh
./release/linux/run.sh --install
```

`--install` permits the prerequisites step to install the small Ubuntu/Debian
host helpers with `apt-get`, using sudo when needed. Omit it on an already
prepared host. It does not install a CUDA toolkit on the host. Step 1 is a shell
bootstrap and can install Python before configuration; the later runner requires
Python 3.10+. Install Git first if needed to obtain the initial checkout.

To compile without GPU qualification, stop after step 2:

```sh
./release/linux/run.sh --through 2
```

Use these scripts individually to resume or inspect a particular step:

| Step | Script | What it does |
| --- | --- | --- |
| 1 | [step1-prerequisites.sh](../../release/linux/step1-prerequisites.sh) | Checks Linux x86-64 and build helpers; `--install` installs missing prerequisites |
| 2 | [step2-build.sh](../../release/linux/step2-build.sh) | Fetches pinned dependencies, builds offline, checks commit and artifact hashes |
| 3 | [step3-models.sh](../../release/linux/step3-models.sh) | Verifies/provisions all main and auxiliary model groups |
| 4 | [step4-test.sh](../../release/linux/step4-test.sh) | Runs parity, three-userland checks, runtime checks and small feature/server renders |
| 5 | [step5-export.sh](../../release/linux/step5-export.sh) | Creates the checked private handoff for the signing Mac |

For example:

```sh
./release/linux/step4-test.sh
./release/linux/step5-export.sh
```

Build work and caches go under `outputs/linux-build/`. The portable output is
`bin/linux-RUN/`, evidence is under `outputs/release/RUN/`, and the handoff is
`bin/linux-handoff-RUN/`. `RUN` defaults to `v0.1.0-01` in this example.
Budget roughly 30 GiB per clean build, plus cached inputs, models and test media.

## What the tests check

Step 4 uses the final standalone executable and its matching private probes;
it does not rebuild inference. It first verifies the unpacked runtime, then
runs the complete **204-output SGLang golden gate**. It repeats that gate in
Ubuntu 22.04, Ubuntu 24.04 and Debian 12 environments. All four distro checks
(including Ubuntu 22.04 unpacked mode) must pass with 204 outputs each.

The historical gate uses its private FFmpeg 4.2.2 encoder and 6.1.1 decoder.
Current-feature and runtime tests use production FFmpeg/FFprobe 9.0.2. The runner
never changes goldens, adds tolerances or skips failed cases.

The feature matrix tests mostly 256×256, two-step clips, including references,
stills, saved state, continuation, bridge, upscale and server behavior. The
server check downloads a fresh preview VAE and needs network access. Retain the
JSON reports. The run's `features/review.html` is available for optional inspection.
These userlands share the PRO 5000's driver and kernel; they do not establish
physical qualification for every compiled GPU target.

For the separately configured native development toolchain, the source-suite
command from the contributor guide is also wrapped:

```sh
./release/linux/test-source.sh --environment outputs/setup/linux-env.sh
```

Only use a trusted environment file produced by the normal source setup process.
This runs `make all test test-server-http`; it is separate from the packaged
204-output gate and does not replace any feature-specific checks required by
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## Signing and handoff

Step 5 exports only the executable, its checksum, the matching dependency source
archive, its checksum, `build.json`, and `qualification.json`. The qualification
record contains the actual test results and the artifact hashes. The signing
Mac verifies that record before staging files for publication.

Do not upload the whole build or handoff directory. The Mac publication steps
select the public assets explicitly. They sign the checksum list with Minisign,
which authenticates the Linux download without changing its tested bytes.
The source archive must accompany it; see the
[source-material instructions](../../scripts/linux/SOURCE-MATERIALS.md).

The runtime needs glibc 2.35+, AVX2/FMA and NVIDIA driver 580.126.20+ with support
for the GPU. It supplies its own CUDA libraries and production media tools.
End users do not need a toolkit, Python or system FFmpeg.

If PRoot/ptrace is blocked, use a permitted host. If a pinned download hash or a
golden comparison fails, preserve its logs and investigate; do not change the
lock or goldens to obtain a pass. More background is in the
[portable Linux guide](../build/linux-distribution.md).
