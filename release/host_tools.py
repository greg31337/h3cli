"""Select installed host helpers without changing the qualified toolchain lock."""
import os
from pathlib import Path
import subprocess
import sys

from common import read


def macos_environment(source, checkout, environment=None):
    env = dict(os.environ if environment is None else environment)
    lock = Path(source)/'scripts/macos/lock.json'
    expected = read(lock)['toolchain']['pkg_config']
    path_directories = [Path(p or '.').absolute() for p in env.get('PATH', os.defpath).split(os.pathsep)]
    directories = [*path_directories, Path(checkout)/'outputs/setup/tools/bin', Path(source)/'outputs/setup/tools/bin']
    found = []
    for directory in dict.fromkeys(directories):
        tool = directory/'pkg-config'
        if not tool.is_file() or not os.access(tool, os.X_OK):
            continue
        try:
            result = subprocess.run([str(tool), '--version'], env=env, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            continue
        version = result.stdout.strip()
        if result.returncode or version != expected:
            found.append(str(tool)+' ('+(version if not result.returncode else 'failed')+')')
            continue
        prefix = [str(directory)]
        # Older selected snapshots unconditionally put Python's directory first.
        # Homebrew's interpreter-specific bin avoids promoting its shared bin,
        # which can contain a different pkg-config. Keep virtual environments.
        python_directory = Path(sys.executable).parent
        if (python_directory/'python3').is_file() and not (python_directory/'pkg-config').exists():
            prefix.insert(0, str(python_directory))
        env['PATH'] = os.pathsep.join(dict.fromkeys([*prefix, *map(str, path_directories)]))
        print('macOS pkg-config '+expected+': '+str(tool), flush=True)
        return env
    detail = '; '.join(found) or 'none installed'
    raise ValueError('macOS release requires pkg-config '+expected+' from '+str(lock)+
                     '; found '+detail+'. Put the matching version on PATH or install it under '+
                     str(Path(checkout)/'outputs/setup/tools/bin')+'. The toolchain lock is unchanged.')
