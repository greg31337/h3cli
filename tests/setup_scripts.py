#!/usr/bin/env python3
"""Exercise setup scripts with fake installers; never installs host packages.

Run: python3 tests/setup_scripts.py
Only fixed OS/install paths are relocated in temporary script copies. Commands
that can install or download anything are replaced with logged test doubles.
"""

import os
os.environ["H3_OFFLINE"] = "1"  # Tests use provisioned weights; never fetch implicitly.
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOCK = r'''
import json, os, pathlib, shutil, sys
name, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
with open(os.environ['MOCK_LOG'], 'a') as f:
    f.write(json.dumps([name, *args]) + '\n')
base = pathlib.Path(os.environ['MOCK_ROOT'])
if os.environ.get('MOCK_FAIL') == name:
    sys.exit(19)
if name == 'uname':
    print(os.environ.get('MOCK_OS', 'Linux') if args == ['-s'] else
          os.environ.get('MOCK_ARCH', 'x86_64'))
elif name == 'id':
    print(os.environ.get('MOCK_UID', '0'))
elif name == 'sw_vers':
    print(os.environ.get('MOCK_MACOS', '26.0'))
elif name == 'xcode-select':
    if args == ['-p'] and os.environ.get('MOCK_NO_CLT'):
        sys.exit(1)
    print('/Library/Developer/CommandLineTools')
elif name == 'xcrun':
    print(os.environ.get('MOCK_SDK', '26.0') if '--show-sdk-version' in args else 'clang')
elif name == 'brew':
    if args == ['shellenv']:
        print('export H3_MOCK_BREW_ACTIVE=1')
elif name == 'nvidia-smi':
    if os.environ.get('MOCK_NO_GPU'):
        sys.exit(1)
    if '--query-gpu=compute_cap' in args:
        print(os.environ.get('MOCK_CC', '9.0'))
    elif '--query-gpu=driver_version' in args:
        print(os.environ.get('MOCK_DRIVER', '580.173.02'))
    else:
        print('NVIDIA GPU')
elif name == 'nvcc':
    if args == ['--list-gpu-code']:
        print(os.environ.get('MOCK_GPU_CODE', 'sm_86\nsm_89\nsm_90\nsm_100\nsm_120'))
    elif '-c' in args:
        output = pathlib.Path(args[args.index('-o') + 1])
        output.write_text('mock CUDA object\n')
        if '-MMD' in args:
            output.with_suffix('.d').write_text(
                f'{output}: src/cuda/gpu_cuda.cu src/cuda/cuda_cudnn.h\n')
    else:
        print('CUDA 13.0, V13.0.88')
elif name == 'apt-cache' and os.environ.get('MOCK_CUDA_REPO'):
    print('500 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64 / Packages')
elif name == 'apt-get' and 'cuda-toolkit-13-0=13.0.3-1' in args:
    (base / 'cuda-13.0/version.json').write_text('{"cuda":{"version":"13.0.3"}}')
elif name == 'systemd-detect-virt':
    sys.exit(0 if os.environ.get('MOCK_CONTAINER') else 1)
elif name == 'sudo':
    os.execvp(args[0], args)
elif name == 'curl':
    target = pathlib.Path(args[args.index('-o') + 1])
    if any('Homebrew' in arg for arg in args):
        target.write_text('#!/bin/bash\ncp "$MOCK_ROOT/bin/mock" "$MOCK_ROOT/homebrew/bin/brew"\n')
    else:
        target.write_bytes(b'fake keyring')
elif name == 'git':
    if args[0] == 'clone':
        dest = pathlib.Path(args[-1])
        (dest / 'include').mkdir(parents=True)
        (dest / 'include/cudnn_frontend.h').touch()
    elif args[0] == 'init':
        (pathlib.Path(args[-1]) / 'include').mkdir(parents=True)
    elif 'rev-parse' in args:
        if 'sglang-cutlass' in args[1]:
            print('different' if os.environ.get('MOCK_WRONG_CUTLASS') else 'da5e086dab31d63815acafdac9a9c5893b1c69e2')
        else:
            print('different' if os.environ.get('MOCK_WRONG_TAG') and args[-1] == 'HEAD' else 'v1.11.0-commit')
    elif 'status' in args and os.environ.get('MOCK_DIRTY'):
        print(' M include/cudnn_frontend.h')
elif name in ('python3', 'python'):
    if args and args[0].endswith('/setup_ffmpeg.py'):
        if os.environ.get('MOCK_MEDIA_FAIL'): sys.exit(19)
        dest = pathlib.Path(args[args.index('--prefix')+1]) if '--prefix' in args else base/'outputs/setup/ffmpeg'
        (dest/'bin').mkdir(parents=True, exist_ok=True)
        for tool in ('ffmpeg', 'ffprobe'):
            path = dest/'bin'/tool
            if not path.exists(): path.symlink_to(base/'bin/mock')
    elif args[:2] == ['-m', 'venv']:
        dest = pathlib.Path(args[-1])
        (dest / 'bin').mkdir(parents=True)
        (dest / 'bin/python').symlink_to(base / 'bin/mock')
        for part in ('nvidia/cudnn/lib', 'nvidia/cudnn/include', 'nvidia/cu13/lib'):
            (dest / 'lib/python3.12/site-packages' / part).mkdir(parents=True)
    elif args[:2] == ['-m', 'pip'] and os.environ.get('MOCK_PIP_FAIL'):
        sys.exit(19)
    elif args and args[0] == '-c' and 'sysconfig' in args[1]:
        print(pathlib.Path(sys.argv[0]).parent.parent / 'lib/python3.12/site-packages')
    elif args and args[0] == '-c' and 'json.load' in args[1]:
        assert json.loads(pathlib.Path(args[2]).read_text())['cuda']['version'] == '13.0.3'
    elif args and args[0].endswith('verify_cuda_runtime.py') and os.environ.get('MOCK_RUNTIME_FAIL'):
        sys.exit(19)
elif name == 'make' and '-C' in args:
    dest = pathlib.Path(args[args.index('-C') + 1]) / 'bin/h3cli'
    if not dest.exists(): dest.symlink_to(base / 'bin/mock')
'''


class SetupScripts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='h3-setup-tests-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "checkout with spaces ' $literal"
        self.bin = self.root / 'bin'
        self.bin.mkdir(parents=True)
        self.log = self.root / 'commands.jsonl'
        mock = self.bin / 'mock'
        mock.write_text(f'#!{sys.executable}\n' + MOCK)
        mock.chmod(0o755)
        for command in ('uname', 'id', 'sw_vers', 'xcode-select', 'xcrun',
                        'brew', 'nvidia-smi', 'systemd-detect-virt', 'sudo',
                        'curl', 'git', 'apt-get', 'apt-cache', 'dpkg', 'ubuntu-drivers',
                        'gcc', 'make', 'pkg-config', 'ffmpeg', 'ffprobe', 'python3'):
            (self.bin / command).symlink_to(mock)
        self.cuda = self.root / 'cuda-13.0'
        (self.cuda / 'bin').mkdir(parents=True)
        (self.cuda / 'bin/nvcc').symlink_to(mock)
        (self.cuda / 'version.json').write_text('{"cuda":{"version":"13.0.3"}}')
        (self.root / 'homebrew/bin').mkdir(parents=True)
        self.os_release = self.root / 'os-release'
        self.os_release.write_text('ID=ubuntu\nVERSION_ID=24.04\n')
        scripts = self.root / 'scripts'
        scripts.mkdir()
        for platform in ('macos', 'linux'):
            source = (ROOT / f'scripts/setup_{platform}.sh').read_text()
            # Relocate absolute paths, including the container markers, so
            # these tests are safe on either macOS or Linux and need no root.
            for old, new in {
                '/opt/homebrew/bin/brew': str(self.root / 'homebrew/bin/brew'),
                '/usr/local/cuda-13.0': str(self.cuda),
                '/etc/os-release': str(self.os_release),
                '/.dockerenv': str(self.root / '.dockerenv'),
                '/run/.containerenv': str(self.root / '.containerenv'),
            }.items():
                # Original constants are unquoted; quote relocated paths.
                source = source.replace(old, "'" + new.replace("'", "'\\''") + "'")
            (scripts / f'setup_{platform}.sh').write_text(source)
        (scripts / 'cuda_arch.sh').write_text((ROOT / 'scripts/cuda_arch.sh').read_text())
        self.env = dict(os.environ, PATH=f'{self.bin}:/usr/bin:/bin',
                        MOCK_ROOT=str(self.root), MOCK_LOG=str(self.log))
        # Do not inherit shell hooks or CUDA overrides from the developer.
        for key in ('BASH_ENV', 'ENV', 'CUDA_PATH', 'CUDA_ARCH', 'CUDA_CUDNN',
                    'CUDNN_FRONTEND_PATH', 'H3_BUILD_JOBS'):
            self.env.pop(key, None)

    def run_setup(self, platform='linux', *args, **overrides):
        env = dict(self.env)
        if platform == 'macos':
            env.update(MOCK_OS='Darwin', MOCK_ARCH='arm64', MOCK_UID='501')
        env.update(overrides)
        return subprocess.run(['/bin/bash', str(self.root / f'scripts/setup_{platform}.sh'), *args],
                              env=env, text=True, capture_output=True, timeout=30)

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def source_env(self, platform, variables):
        path = self.root / f'outputs/setup/{platform}-env.sh'
        result = subprocess.run(['/bin/bash', '-c',
                                 '. "$1"; shift; for name; do printf "%s\n" "${!name}"; done',
                                 'test', str(path), *variables], env=self.env,
                                text=True, capture_output=True, check=True)
        return result.stdout.splitlines()

    def test_help_dry_runs_and_invalid_options_do_not_change_host(self):
        for platform in ('macos', 'linux'):
            self.assert_ok(self.run_setup(platform, '--help'))
            self.assert_ok(self.run_setup(platform, '--dry-run'))
            self.assertEqual(self.run_setup(platform, '--bogus').returncode, 2)
        self.assert_ok(self.run_setup('linux', '--dry-run', '--with-cudnn', '--install-driver'))
        self.assertEqual(self.commands(), [])
        self.assertFalse((self.root / 'outputs').exists())

    def test_wrong_platform_or_distro_stops_before_installation(self):
        self.assertNotEqual(self.run_setup('macos', MOCK_ARCH='x86_64').returncode, 0)
        self.assertNotEqual(self.run_setup('linux', MOCK_ARCH='aarch64').returncode, 0)
        self.os_release.write_text('ID=ubuntu\nVERSION_ID=22.04\n')
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Requires Ubuntu 24.04', result.stderr)
        self.assertFalse(any(c[0] in ('apt-get', 'brew', 'curl') for c in self.commands()))

    def test_mac_sdk_os_and_root_guards(self):
        for settings in ({'MOCK_MACOS': '15.0'}, {'MOCK_SDK': '15.0'}, {'MOCK_UID': '0'}):
            with self.subTest(settings=settings):
                self.assertNotEqual(self.run_setup('macos', **settings).returncode, 0)
        self.assertFalse(any(c[0] == 'brew' for c in self.commands()))

    def test_mac_missing_clt_requests_installer_and_rerun(self):
        result = self.run_setup('macos', MOCK_NO_CLT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(['xcode-select', '--install'], self.commands())
        self.assertIn('rerun', result.stderr)
        self.assertFalse(any(c[0] in ('brew', 'curl') for c in self.commands()))

    def test_mac_existing_brew_and_environment(self):
        self.assert_ok(self.run_setup('macos'))
        self.assertIn(['brew', 'install', 'x264', 'pkg-config', 'python'], self.commands())
        self.assertFalse(any(c[0] == 'curl' for c in self.commands()))
        self.assertEqual(self.source_env('macos', ['H3_MOCK_BREW_ACTIVE']), ['1'])

    def test_mac_installs_missing_brew(self):
        (self.bin / 'brew').unlink()
        self.assert_ok(self.run_setup('macos'))
        self.assertEqual(self.source_env('macos', ['H3_MOCK_BREW_ACTIVE']), ['1'])
        self.assertTrue(any(c[0] == 'curl' for c in self.commands()))

    def test_mac_brew_failure_stops_setup(self):
        result = self.run_setup('macos', MOCK_FAIL='brew')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'outputs/setup/macos-env.sh').exists())
        self.assertFalse(any(c[:2] == ['brew', 'install'] for c in self.commands()))

    def test_linux_base_preserves_driver_and_uses_root_without_sudo(self):
        self.assert_ok(self.run_setup('linux', '--install-driver'))
        self.assertFalse(any(c[0] in ('sudo', 'ubuntu-drivers') for c in self.commands()))
        self.assertEqual(self.source_env('linux', ['CUDA_PATH', 'CUDA_ARCH', 'CUDA_CUDNN']),
                         [str(self.cuda), 'auto', '1'])
        self.assertTrue(any(c[0] == 'make' and 'all' in c for c in self.commands()))
        self.assertTrue(any(c[0] == 'h3cli' and '--help' in c for c in self.commands()))

    def test_linux_cudnn_sudo_environment_and_repeat_run(self):
        for _ in range(2):
            self.assert_ok(self.run_setup('linux', '--with-cudnn', MOCK_UID='1000'))
        self.assertTrue(any(c[:2] == ['sudo', 'apt-get'] for c in self.commands()))
        self.assertEqual(sum(c[:2] == ['git', 'clone'] for c in self.commands()), 1)
        self.assertEqual(self.source_env('linux', ['CUDA_CUDNN', 'CUDNN_FRONTEND_PATH']),
                         ['1', str(self.root / 'outputs/setup/cudnn-frontend/include')])
        self.assert_ok(self.run_setup())
        self.assertEqual(self.source_env('linux', ['CUDA_CUDNN', 'CUDA_SGLANG']), ['1', '1'])

    def test_linux_cudnn_rejects_modified_or_wrong_checkout(self):
        self.assert_ok(self.run_setup('linux', '--with-cudnn'))
        for overrides in ({'MOCK_DIRTY': '1'}, {'MOCK_WRONG_TAG': '1'}):
            result = self.run_setup('linux', '--with-cudnn', **overrides)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('unmodified v1.11.0', result.stderr)

    def test_linux_compile_without_gpu_or_driver_installation(self):
        self.assert_ok(self.run_setup('linux', MOCK_NO_GPU='1', MOCK_CONTAINER='1'))
        self.assertEqual(self.source_env('linux', ['CUDA_ARCH']), ['fat'])
        self.assertFalse(any(c[0] == 'ubuntu-drivers' for c in self.commands()))

    def test_linux_driver_install_on_bare_host(self):
        result = self.run_setup('linux', '--install-driver', MOCK_NO_GPU='1')
        self.assert_ok(result)
        self.assertIn(['ubuntu-drivers', 'install'], self.commands())
        self.assertIn('reboot', result.stdout.lower())

    def test_linux_keeps_existing_cuda_repository(self):
        result = self.run_setup('linux', '--with-cudnn', MOCK_CUDA_REPO='1')
        self.assert_ok(result)
        self.assertIn('Keeping the existing NVIDIA CUDA package repository', result.stdout)
        self.assertFalse(any(c[0] in ('curl', 'dpkg') for c in self.commands()))
        self.assertTrue(any('cuda-toolkit-13-0=13.0.3-1' in c for c in self.commands()))
        self.assertFalse(any(any('libcudnn9-' in arg for arg in c) for c in self.commands()))

    def test_linux_b200_selects_native_architecture(self):
        self.assert_ok(self.run_setup('linux', MOCK_CC='10.0'))
        self.assertEqual(self.source_env('linux', ['CUDA_ARCH']), ['auto'])

    def test_linux_keeps_complete_provider_toolkit(self):
        for relative in ('include/cuda_runtime.h', 'include/cublasLt.h',
                         'lib64/libcudart.so', 'lib64/libcublas.so', 'lib64/libcublasLt.so'):
            target = self.cuda / relative
            target.parent.mkdir(exist_ok=True)
            target.touch()
        result = self.run_setup('linux', MOCK_CUDA_REPO='1')
        self.assert_ok(result)
        self.assertIn('Keeping the existing complete CUDA 13.0 Update 3 toolkit', result.stdout)
        self.assertFalse(any('cuda-toolkit-13-0=13.0.3-1' in c for c in self.commands()))

    def test_linux_old_driver_rejected_before_installation(self):
        result = self.run_setup('linux', MOCK_DRIVER='570.133.20')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CUDA 13 runtime', result.stderr)
        self.assertFalse(any(c[0] == 'apt-get' for c in self.commands()))

    def test_linux_earlier_toolkit_update_is_upgraded_even_with_matching_nvcc(self):
        for name in ('include/cuda_runtime.h', 'include/cublasLt.h',
                     'lib64/libcudart.so', 'lib64/libcublas.so', 'lib64/libcublasLt.so'):
            target = self.cuda / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        (self.cuda / 'version.json').write_text('{"cuda":{"version":"13.0.1"}}')
        self.assert_ok(self.run_setup('linux', MOCK_CUDA_REPO='1'))
        self.assertTrue(any('cuda-toolkit-13-0=13.0.3-1' in c for c in self.commands()))

    def test_linux_dependencies_only_and_runtime_paths(self):
        self.assert_ok(self.run_setup('linux', '--no-build'))
        self.assertFalse(any(c[0] == 'make' for c in self.commands()))
        variables = ['SGLANG_CUTLASS_PATH', 'H3_SGLANG_CUBLAS_LIBRARY',
                     'H3_SGLANG_CUDNN_LIBRARY', 'H3_FFMPEG', 'CPATH', 'LD_LIBRARY_PATH']
        values = self.source_env('linux', variables)
        self.assertTrue(values[0].endswith('/outputs/setup/sglang-cutlass'))
        self.assertIn('/sglang-runtime/', values[1])
        self.assertTrue(values[1].endswith('/nvidia/cu13/lib/libcublas.so.13'))
        self.assertTrue(values[2].endswith('/nvidia/cudnn/lib/libcudnn.so.9'))
        self.assertTrue(values[3].endswith('/outputs/setup/ffmpeg/bin/ffmpeg'))
        self.assertIn('/sglang-runtime/', values[4])
        self.assertIn('/sglang-runtime/lib/python3.12/site-packages/nvidia/cudnn/lib', values[5])
        self.assertNotIn('/cuda-runtime/', ':'.join(values))
        installs = [c for c in self.commands() if c[1:3] == ['-m', 'pip']]
        self.assertEqual(len(installs), 1)
        for command in installs:
            self.assertIn('--require-hashes', command)
            self.assertIn('--no-deps', command)

    def test_linux_cutlass_mismatch_stops_before_build(self):
        result = self.run_setup('linux', MOCK_WRONG_CUTLASS='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CUTLASS', result.stderr)
        self.assertFalse(any(c[0] == 'make' for c in self.commands()))

    def test_linux_runtime_install_and_validation_failures_stop_build(self):
        for settings in ({'MOCK_PIP_FAIL': '1'}, {'MOCK_RUNTIME_FAIL': '1'}, {'MOCK_MEDIA_FAIL': '1'}):
            with self.subTest(settings=settings):
                result = self.run_setup('linux', **settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / 'outputs/setup/linux-env.sh').exists())
                self.assertFalse(any(c[0] == 'make' for c in self.commands()))

    def test_linux_jobs_and_build_failure(self):
        self.assertNotEqual(self.run_setup('linux', H3_BUILD_JOBS='0').returncode, 0)
        result = self.run_setup('linux', H3_BUILD_JOBS='3', MOCK_FAIL='make')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'outputs/setup/linux-env.sh').exists())
        self.assertTrue(any(c[0] == 'make' and '-j3' in c for c in self.commands()))

    def test_linux_unsupported_gpu_diagnostic(self):
        result = self.run_setup('linux', MOCK_CC='10.3')
        self.assert_ok(result)
        self.assertIn('visible NVIDIA GPU', result.stdout)
        self.assertIn('not qualified', result.stdout)
        self.assertNotIn('No working visible', result.stdout)
        self.assertEqual(self.source_env('linux', ['CUDA_ARCH']), ['fat'])

    def run_arch(self, arch, **overrides):
        return subprocess.run(['/bin/sh', str(ROOT / 'scripts/cuda_arch.sh'), arch,
                               str(self.cuda / 'bin/nvcc')], text=True, capture_output=True,
                              env=dict(self.env, **overrides), timeout=10)

    def test_arch_native_auto_and_fat_flags(self):
        for sm in ('86', '89', '90', '100', '120'):
            with self.subTest(sm=sm):
                expected = f'-gencode=arch=compute_{sm},code=sm_{sm} -gencode=arch=compute_{sm},code=compute_{sm}'
                explicit = self.run_arch(sm)
                automatic = self.run_arch('auto', MOCK_CC=f'{int(sm)//10}.{int(sm)%10}')
                self.assert_ok(explicit)
                self.assert_ok(automatic)
                self.assertEqual(explicit.stdout.strip(), expected)
                self.assertEqual(automatic.stdout.strip(), expected)
        result = self.run_arch('fat')
        self.assert_ok(result)
        self.assertEqual(result.stdout.split(),
                         [f'-gencode=arch=compute_{sm},code=sm_{sm}' for sm in ('86', '89', '90', '100', '120')]
                         + ['-gencode=arch=compute_86,code=compute_86'])

    def test_arch_rejects_missing_gpu_toolkit_target_and_invalid_input(self):
        for arch, overrides, message in (
                ('auto', {'MOCK_NO_GPU': '1'}, 'cannot detect'),
                ('100', {'MOCK_GPU_CODE': 'sm_90'}, 'cannot build SM100'),
                ('fat', {'MOCK_GPU_CODE': 'sm_86\nsm_89\nsm_90\nsm_120'}, 'cannot build SM100'),
                ('90', {'MOCK_FAIL': 'nvcc'}, 'CUDA compiler not found'),
                ('103', {}, 'unknown CUDA_ARCH'),
                ('auto', {'MOCK_CC': '10.3'}, 'unknown CUDA_ARCH')):
            with self.subTest(arch=arch, overrides=overrides):
                result = self.run_arch(arch, **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout)

    def test_fat_object_rebuilds_for_flags_headers_and_architecture_helper(self):
        # Exercise the real Make dependency graph with a fake compiler. This
        # catches stale fat objects without needing CUDA or installing tools.
        (self.root / 'src/cuda').mkdir(parents=True)
        for name in ('src/cuda/gpu_cuda.cu', 'src/cuda/cuda_dispatch.h', 'src/gpu.h', 'src/cuda/cuda_cudnn.h'):
            (self.root / name).touch()
        helper = self.root / 'scripts/cuda_arch.sh'
        helper.chmod(0o755)
        (self.bin / 'nvcc').symlink_to(self.bin / 'mock')
        sha = 'shasum -a 256' if sys.platform == 'darwin' else 'sha256sum'

        def build(flags=''):
            result = subprocess.run(['/usr/bin/make', '-f', str(ROOT / 'Makefile'),
                                     'PLATFORM=Linux', 'CUDA_ARCH=90', 'NVCC=nvcc',
                                     'CUDA_OPENSSL=0', 'CUDA_SGLANG=0', 'CUDA_CUDNN=0',
                                     f'SHA256={sha}', f'CPPFLAGS={flags}',
                                     'src/cuda/gpu_cuda.fat.o'], cwd=self.root, env=self.env,
                                    text=True, capture_output=True, timeout=15)
            self.assert_ok(result)
            return [c for c in self.commands() if c[0] == 'nvcc' and '-c' in c]

        commands = build()
        self.assertEqual(len(commands), 1)
        self.assertIn('-gencode=arch=compute_100,code=sm_100', commands[-1])
        # macOS's older Make can compare timestamps at second granularity.
        # Age the first build coherently so a fast mock compiler does not hide
        # the subsequent configuration change within the same clock tick.
        old = (self.root / 'src/cuda/gpu_cuda.fat.o').stat().st_mtime - 5
        for path in (*self.root.glob('src/**/*.cu'), *self.root.glob('src/**/*.h'), helper,
                     self.root / '.cuda-build-config', self.root / 'src/cuda/gpu_cuda.fat.o'):
            os.utime(path, (old, old))
        self.assertEqual(len(build()), 1)
        self.assertEqual(len(build('-DH3_CUDA_USE_CUDNN')), 2)
        header = self.root / 'src/cuda/cuda_cudnn.h'
        future = (self.root / 'src/cuda/gpu_cuda.fat.o').stat().st_mtime + 2
        os.utime(header, (future, future))
        self.assertEqual(len(build('-DH3_CUDA_USE_CUDNN')), 3)
        os.utime(header, (1, 1))
        future = (self.root / 'src/cuda/gpu_cuda.fat.o').stat().st_mtime + 2
        os.utime(helper, (future, future))
        self.assertEqual(len(build('-DH3_CUDA_USE_CUDNN')), 4)

    def test_linux_refuses_driver_install_in_container(self):
        result = self.run_setup('linux', '--install-driver', MOCK_NO_GPU='1', MOCK_CONTAINER='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('container host', result.stderr)
        self.assertFalse(any(c[0] in ('apt-get', 'ubuntu-drivers') for c in self.commands()))

    def test_download_package_and_verification_failures_stop_setup(self):
        for command in ('curl', 'apt-get', 'nvcc', 'python3'):
            with self.subTest(command=command):
                result = self.run_setup('linux', MOCK_FAIL=command)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / 'outputs/setup/linux-env.sh').exists())
        (self.bin / 'brew').unlink()
        self.assertNotEqual(self.run_setup('macos', MOCK_FAIL='curl').returncode, 0)
        self.assertFalse((self.root / 'outputs/setup/macos-env.sh').exists())


if __name__ == '__main__':
    unittest.main()
