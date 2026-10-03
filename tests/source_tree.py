"""Copy buildable validation sources and locate files in archived layouts."""
from pathlib import Path
import re
import shutil
import subprocess

from cuda_reference_regression import source_files


def copy_source_tree(root, destination):
    root, destination = Path(root), Path(destination)
    for name in source_files(root):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)


def build_target(root, name):
    """Current builds use bin/; archived comparisons retain their own Makefile."""
    root = Path(root)
    target = 'bin/' + name
    makefile = root / 'Makefile'
    if not makefile.exists() or re.search(r'^' + re.escape(target) + r'[: ]',
                                         makefile.read_text(), re.M):
        return target
    return name if name.startswith('h3cli') or name == 'libh3.a' else 'h3_' + name


def build_path(root, name):
    return Path(root) / build_target(root, name)


def source_path(root, name):
    """Historical comparisons also read checkouts from before source relocation."""
    root = Path(root)
    candidates = [root / 'src' / name, *sorted((root / 'src').rglob(name)), root / ('h3_' + name)]
    if name == 'engine.c':
        candidates.append(root / 'h3.c')
    return next((path for path in candidates if path.is_file()), candidates[0])


def historical_source(root, revision_path):
    """Adapt only include paths when compiling an archived source with today's headers."""
    root = Path(root)
    source = subprocess.check_output(['git', 'show', revision_path], cwd=root).decode()
    def include(match):
        name = Path(match[2]).name.removeprefix('h3_')
        path = source_path(root, name).relative_to(root)
        return match[1] + str(path) + '"' if (root / path).is_file() else match[0]
    return re.sub(r'(#\s*include\s*")([^"]+)"', include, source).encode()


if __name__ == '__main__':
    import sys
    sys.stdout.buffer.write(historical_source(Path(__file__).resolve().parents[1], sys.argv[1]))
