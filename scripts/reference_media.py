"""Private media profile used only by the immutable SGLang regression gate."""
import hashlib
import json
from pathlib import Path
import subprocess

VERSIONS = {'H3_FFMPEG': '4.2.2', 'H3_SGLANG_INPUT_FFMPEG': '6.1.1',
            'H3_FFPROBE': '6.1.1'}
PATHS = {'H3_FFMPEG': 'output/ffmpeg', 'H3_SGLANG_INPUT_FFMPEG': 'input/ffmpeg',
         'H3_FFPROBE': 'input/ffprobe'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_profile(root):
    root = Path(root).resolve()
    files = {}
    for name, relative in PATHS.items():
        path = root / relative
        first = subprocess.check_output([str(path), '-version'], text=True).splitlines()[0]
        if first.split()[2].split('-')[0] != VERSIONS[name]:
            raise ValueError('Unexpected regression media version: ' + first)
        files[name] = dict(path=relative, sha256=sha(path), size=path.stat().st_size,
                           version=VERSIONS[name])
    profile = root / 'profile.json'
    profile.write_text(json.dumps(dict(schema=1, purpose='sglang-regression', files=files),
                                  sort_keys=True, indent=2) + '\n')
    return profile


def read_profile(profile, expected=None):
    profile = Path(profile).resolve(strict=True)
    digest = sha(profile)
    if expected is not None and digest != expected:
        raise ValueError('Changed regression media profile')
    record = json.loads(profile.read_text())
    if record.get('schema') != 1 or record.get('purpose') != 'sglang-regression' or set(record.get('files', {})) != set(VERSIONS):
        raise ValueError('Invalid regression media profile')
    env = {'H3_TEST_REFERENCE_MEDIA': str(profile), 'H3_TEST_REFERENCE_MEDIA_SHA256': digest}
    identities = {str(profile): digest}
    for name, item in record['files'].items():
        relative = Path(item['path'])
        path = profile.parent / relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(profile.parent):
            raise ValueError('Unsafe regression media path')
        if item.get('version') != VERSIONS[name] or not path.is_file() or path.stat().st_size != item['size'] or sha(path) != item['sha256']:
            raise ValueError('Missing or changed regression media: ' + name)
        env[name] = str(path)
        identities[str(path)] = item['sha256']
    return env, identities
