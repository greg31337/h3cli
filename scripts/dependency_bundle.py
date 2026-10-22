"""Immutable, checksum-pinned development dependencies shared by releases.

The small reference JSON is the trust anchor. It pins an archive and a separate
qualification report; neither a mutable URL nor a cache readiness marker is a
substitute for checking the pinned bytes and their complete inventory.
"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import sys
import urllib.parse
import urllib.request
from contextlib import contextmanager

PLATFORMS = ('linux-x86_64', 'macos-arm64')
SCHEMA = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b'\n')


def recipe(source, platform):
    source = Path(source)
    if platform not in PLATFORMS:
        raise ValueError('Unsupported dependency platform: ' + platform)
    common = ['scripts/dependency_bundle.py', 'scripts/build_process.py', 'scripts/dependencies.py']
    if platform == 'linux-x86_64':
        lock = read(source/'scripts/linux/lock.json')
        # Distro test images do not change the dependency build. Source/package
        # digests and the builder image completely identify the compiler/sysroot.
        lock['images'] = {'builder': lock['images']['builder']}
        names = common + ['scripts/linux/dependencies.py', 'scripts/linux/build.py', 'scripts/linux/package.py',
                          'scripts/linux/environment.py', 'scripts/linux/build_dependencies.sh',
                          'scripts/requirements-sglang-runtime.txt', 'scripts/requirements-reference-media.txt']
    else:
        lock = read(source/'scripts/macos/lock.json')
        lock['ffmpeg'] = read(source/'scripts/linux/lock.json')['sources']['ffmpeg-9.0.2.tar.xz']
        names = common + ['scripts/macos/dependencies.py', 'scripts/macos/build.py', 'scripts/macos/package.py']
    row = dict(schema=SCHEMA, platform=platform, lock=lock,
               recipes={name: sha(source/name) for name in names})
    return row


def adapter(source, platform):
    directory = Path(source)/'scripts'/('linux' if platform == 'linux-x86_64' else 'macos')
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('h3_dependencies_'+platform.replace('-', '_'), directory/'dependencies.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def recipe_id(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def member_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name or '\\' in name:
        raise ValueError('Unsafe dependency member: ' + name)
    return p


def inventory(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        name = path.relative_to(root).as_posix()
        member_name(name)
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError('Dependency payload must contain regular files/directories: ' + name)
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o7000:
            raise ValueError('Special permission bits in dependency payload: ' + name)
        if path.is_file():
            result[name] = dict(sha256=sha(path), size=path.stat().st_size, mode=mode)
    return result


def verify(root, expected_recipe=None):
    root = Path(root)
    if root.is_symlink() or (root/'manifest.json').is_symlink() or (root/'payload').is_symlink():
        raise ValueError('Symlink in dependency cache')
    manifest = read(root/'manifest.json')
    if manifest.get('schema') != SCHEMA or manifest.get('platform') not in PLATFORMS:
        raise ValueError('Unsupported dependency manifest')
    if manifest.get('recipe_id') != recipe_id(manifest['recipe']):
        raise ValueError('Changed dependency recipe')
    if expected_recipe is not None and manifest['recipe'] != expected_recipe:
        raise ValueError('Incompatible dependency bundle; build and qualify the current recipe')
    if inventory(root/'payload') != manifest['files']:
        raise ValueError('Changed or incomplete dependency payload')
    if set(p.name for p in root.iterdir()) != {'payload', 'manifest.json'}:
        raise ValueError('Unrecorded dependency cache content')
    return manifest


def pack(payload, archive, recipe_value, audit):
    """Create a deterministic regular-file archive; OCI blobs hold Linux links."""
    payload, archive = Path(payload), Path(archive)
    if archive.exists():
        raise ValueError('Dependency archive already exists: ' + str(archive))
    manifest = dict(schema=SCHEMA, platform=recipe_value['platform'], recipe=recipe_value,
                    recipe_id=recipe_id(recipe_value), files=inventory(payload), audit=audit)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dependency-manifest-', dir=archive.parent) as tmp:
        metadata = Path(tmp)/'manifest.json';write(metadata, manifest)
        partial = archive.with_name(archive.name+'.partial')
        try:
            with tarfile.open(partial, 'w', format=tarfile.PAX_FORMAT) as tar:
                paths = [(metadata, 'manifest.json'), (payload, 'payload')]
                paths.extend((p, 'payload/'+p.relative_to(payload).as_posix()) for p in sorted(payload.rglob('*')))
                for path, name in paths:
                    if path.is_symlink():
                        raise ValueError('Symlink in dependency archive')
                    info = tar.gettarinfo(str(path), name)
                    info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
                    info.pax_headers = {}
                    if path.is_file():
                        with path.open('rb') as file:tar.addfile(info, file)
                    else:tar.addfile(info)
            partial.rename(archive)
        finally:
            partial.unlink(missing_ok=True)
    return manifest


def extract(archive, destination):
    """No links, duplicate paths, special files, or writes through ancestors."""
    destination = Path(destination)
    destination.mkdir()
    seen = set()
    with tarfile.open(archive, 'r:') as tar:
        for item in tar:
            member_name(item.name)
            if item.name in seen or not (item.isdir() or item.isreg()) or item.mode & 0o7000:
                raise ValueError('Unsafe or duplicate dependency archive entry: ' + item.name)
            seen.add(item.name)
            target = destination/item.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.isdir():target.mkdir(exist_ok=True)
            else:
                with tar.extractfile(item) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            target.chmod(item.mode & 0o777)


def reference(path):
    """Resolve relative locations before embedding a pin into release config."""
    path = Path(path).expanduser().resolve()
    row = read(path)
    validate_reference(row)
    for key in ('archive', 'qualification'):
        if key not in row:continue
        location = row[key]['url']
        if not urllib.parse.urlparse(location).scheme:
            row[key]['url'] = str((path.parent/location).resolve())
    return row


def validate_reference(row):
    if row.get('schema') != SCHEMA or row.get('platform') not in PLATFORMS:
        raise ValueError('Invalid dependency reference')
    if len(row.get('recipe_id', '')) != 64:
        raise ValueError('Invalid dependency recipe identity')
    for key in ('archive', 'qualification'):
        if key not in row and key == 'qualification':continue
        item = row.get(key, {})
        digest = item.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid dependency digest: ' + key)
        if type(item.get('size')) is not int or item['size'] <= 0 or not item.get('url'):
            raise ValueError('Invalid dependency location/size: ' + key)
        parsed = urllib.parse.urlparse(item['url'])
        if parsed.scheme not in ('', 'file', 'https') or parsed.username or parsed.password:
            raise ValueError('Dependency locations must be local paths or credential-free HTTPS URLs')


@contextmanager
def cache_lock(cache, identity):
    cache = Path(cache);cache.mkdir(parents=True, exist_ok=True)
    lock = cache/(identity+'.lock')
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield


def acquire(item, cache, offline):
    """Only qualified references supply trusted digests; never trust cache flags."""
    cache = Path(cache);cache.mkdir(parents=True, exist_ok=True)
    path = cache/item['sha256']
    if path.is_symlink():raise ValueError('Symlink in dependency download cache')
    if path.exists():
        if not path.is_file() or path.stat().st_size != item['size'] or sha(path) != item['sha256']:
            raise ValueError('Changed cached dependency download: ' + str(path))
        return path
    location = item['url'];parsed = urllib.parse.urlparse(location)
    if offline and parsed.scheme == 'https':
        raise ValueError('Missing offline dependency artifact: ' + item['sha256'])
    temporary = path.with_name(path.name+'.partial')
    if temporary.is_symlink():raise ValueError('Symlink in dependency staging path')
    try:
        with (urllib.request.urlopen(location, timeout=120) if parsed.scheme == 'https'
              else Path(urllib.request.url2pathname(parsed.path) if parsed.scheme == 'file' else location).open('rb')) as src:
            with temporary.open('wb') as dst:
                copied = 0
                while chunk := src.read(1 << 20):
                    copied += len(chunk)
                    if copied > item['size']:raise ValueError('Dependency download exceeds pinned size')
                    dst.write(chunk)
                dst.flush();os.fsync(dst.fileno())
        if temporary.stat().st_size != item['size'] or sha(temporary) != item['sha256']:
            raise ValueError('Dependency download checksum/size mismatch: ' + location)
        temporary.replace(path)
    except OSError as exc:
        raise OSError('Dependency download failed: '+location+': '+str(exc)) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load(pin, cache, expected_recipe, offline=False, candidate=False):
    pin = reference(pin) if isinstance(pin, (str, Path)) else pin
    validate_reference(pin)
    if pin['recipe_id'] != recipe_id(expected_recipe) or pin['platform'] != expected_recipe['platform']:
        raise ValueError('Incompatible dependency bundle; create and qualify the current recipe')
    identity = pin['archive']['sha256'];cache = Path(cache)
    with cache_lock(cache, identity):
        if not candidate:
            if 'qualification' not in pin:
                raise ValueError('Dependency bundle is unqualified; run scripts/dependencies.py qualify')
            report = read(acquire(pin['qualification'], cache/'downloads', offline))
            required = {'media', 'relocation', 'artifact', 'features'}
            if pin['platform'] == 'linux-x86_64':required |= {'parity', 'distros', 'runtime'}
            checks = report.get('checks', {})
            if (report.get('passed') is not True or report.get('archive_sha256') != identity or
                report.get('recipe_id') != pin['recipe_id'] or report.get('platform') != pin['platform'] or
                not report.get('source_sha256') or not report.get('reference_commit') or
                not required <= checks.keys() or any(checks[k].get('passed') is not True for k in required)):
                raise ValueError('Invalid or mismatched dependency qualification')
            if pin['platform'] == 'linux-x86_64' and (checks['parity'].get('outputs') != 204 or
                    checks['distros'].get('outputs') != [204]*4):
                raise ValueError('Incomplete dependency CUDA qualification')
        archive = acquire(pin['archive'], cache/'downloads', offline)
        root = cache/'bundles'/identity
        root.parent.mkdir(exist_ok=True)
        if not root.exists():
            staging = Path(tempfile.mkdtemp(prefix=identity+'.partial-', dir=root.parent))
            try:
                extract(archive, staging/'bundle')
                verify(staging/'bundle', expected_recipe)
                (staging/'bundle').rename(root)
            finally:
                shutil.rmtree(staging)
        # Bind the extracted inventory to the pinned archive, including warm
        # caches: an attacker cannot bless edited files by editing manifest.json.
        with tarfile.open(archive, 'r:') as tar:
            item = tar.getmember('manifest.json')
            if not item.isreg() or item.size > 32*1024*1024:
                raise ValueError('Invalid dependency manifest member')
            original = tar.extractfile(item).read()
        if root.is_symlink() or (root/'manifest.json').is_symlink() or (root/'manifest.json').read_bytes() != original:
            raise ValueError('Changed cached dependency manifest')
        manifest = verify(root, expected_recipe)
        return root/'payload', dict(platform=pin['platform'], recipe_id=pin['recipe_id'],
            archive_sha256=identity, qualification_sha256=pin.get('qualification', {}).get('sha256'),
            manifest_sha256=sha(root/'manifest.json'), mode='candidate' if candidate else 'qualified')


def artifact(path):
    path = Path(path)
    return dict(url=path.name, sha256=sha(path), size=path.stat().st_size)
