"""Package and restore this project's large files as GitHub Release assets.

Uses only the Python standard library. No credentials are stored in the manifest.
"""
import argparse
import bisect
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.github-transfer'
MANIFEST = ROOT / 'github-assets.json'
REPO = 'suan1111111/promptinjection-attack'
TAG = 'project-assets-20261112'
SKIP = {'.git', '.venv', '.venv-ocr', 'node_modules', '__pycache__',
        '.downloads', '.pip-cache', '.pnpm-home', '.pnpm-store', '.npm-cache',
        '.hf-cache', '.modelscope-cache', '.cache', '.github-transfer',
        '.run', '.tmp', 'tmp', 'temp', 'dist', 'build', '.idea', '.vscode'}
CODE = {'.py', '.js', '.jsx', '.ts', '.tsx', '.sh', '.ps1', '.bat', '.cmd',
        '.html', '.css', '.scss', '.md', '.rst', '.txt', '.toml', '.yaml', '.yml'}
BINARY = {'.safetensors', '.bin', '.onnx', '.pt', '.pth', '.h5', '.gguf',
          '.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.ico', '.tif', '.tiff',
          '.mp3', '.wav', '.mp4', '.webm', '.flac', '.ogg', '.m4a', '.pdf',
          '.zip', '.gz', '.7z', '.tar', '.rar', '.parquet', '.arrow', '.npy',
          '.npz', '.pkl', '.pickle', '.db', '.sqlite', '.sqlite3', '.csv', '.tsv',
          '.exe', '.dll', '.pyd', '.so', '.whl', '.woff', '.woff2', '.ttf'}


def inventory():
    source, assets = [], []
    for directory, dirs, files in os.walk(ROOT):
        rel_dir = Path(directory).relative_to(ROOT)
        dirs[:] = sorted(d for d in dirs if d not in SKIP
                         and not d.startswith(('.chart-data-', 'tmp'))
                         and not (rel_dir == Path('tools') and d in {'node', 'poppler', 'ffmpeg'}))
        for name in sorted(files):
            p = Path(directory) / name
            rel = p.relative_to(ROOT)
            if p.is_symlink():
                raise RuntimeError(f'Symlink needs explicit handling: {rel}')
            if (name == '.DS_Store' or name == 'Thumbs.db' or (name.endswith(('.log', '.pyc', '.pyo', '.lock', '.incomplete', '.db-shm', '.db-wal')) and name not in {'yarn.lock', 'uv.lock', 'poetry.lock'})
                    or name in {'.env', '.env.local'} or (name.startswith('.env.') and name != '.env.example')):
                continue
            # Locally downloaded Git LFS attributes must not activate LFS billing.
            if name == '.gitattributes' and rel.parts[0] in {'models', 'data'}:
                continue
            # Keep dependency locks even though generated cache lock files are excluded.
            is_asset = (rel.parts[0] == 'models' or p.suffix.lower() in BINARY
                        or p.stat().st_size >= 10 * 1024**2
                        or (rel.parts[0] in {'data', 'output', 'outputs', 'test-data'}
                            and p.suffix.lower() not in CODE))
            (assets if is_asset else source).append(rel.as_posix())
    return source, assets


class SplitWriter:
    def __init__(self, limit):
        self.limit, self.parts, self.stream, self.size, self.hash = limit, [], None, 0, None

    def finish_part(self):
        if self.stream:
            self.stream.close()
            self.parts.append({'name': self.path.name, 'size': self.size, 'sha256': self.hash.hexdigest()})
            print(f'Packed {self.path.name}: {self.size} bytes', flush=True)
            self.stream = None

    def write(self, data):
        original = len(data)
        while data:
            if self.stream is None:
                self.path = WORK / f'project-assets.tar.part{len(self.parts):03d}'
                self.stream = self.path.open('wb')
                self.size, self.hash = 0, hashlib.sha256()
            take = min(len(data), self.limit - self.size)
            block = data[:take]
            self.stream.write(block)
            self.hash.update(block)
            self.size += take
            data = data[take:]
            if self.size == self.limit:
                self.finish_part()
        return original


def pack():
    WORK.mkdir(exist_ok=True)
    if MANIFEST.exists() or list(WORK.glob('project-assets.tar.part*')):
        raise RuntimeError('An asset package already exists; preserve it or choose a fresh workspace before repacking.')
    source, assets = inventory()
    total = sum((ROOT / x).stat().st_size for x in assets)
    (WORK / 'source-files.json').write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
    (WORK / 'asset-files.json').write_text(json.dumps(assets, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Packing {len(assets)} assets ({total / 1024**3:.2f} GiB); {len(source)} source files', flush=True)
    writer = SplitWriter(1024**3)
    with tarfile.open(fileobj=writer, mode='w|', dereference=True) as tar:
        for name in assets:
            tar.add(ROOT / name, arcname=name, recursive=False)
    writer.finish_part()
    manifest = {'repository': REPO, 'tag': TAG, 'format': 'split-tar-v1',
                'asset_count': len(assets), 'original_bytes': total, 'parts': writer.parts}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {MANIFEST.name}', flush=True)


def credentials():
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                            text=True, capture_output=True, timeout=90,
                            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'Never'})
    values = dict(x.split('=', 1) for x in result.stdout.splitlines() if '=' in x)
    if not values.get('password'):
        raise RuntimeError('GitHub login missing. Use git credential-manager github login --browser')
    return values['password']


def request(url, token=None, method='GET', data=None, headers=None, timeout=60):
    h = {'User-Agent': 'PrivaSee-project-sync', 'Accept': 'application/vnd.github+json'}
    if token:
        h['Authorization'] = 'Bearer ' + token
    h.update(headers or {})
    opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
              if os.environ.get('GITHUB_SYNC_DIRECT') == '1' else urllib.request.build_opener())
    return opener.open(urllib.request.Request(url, data=data, headers=h, method=method), timeout=timeout)


def part_groups(manifest):
    return [manifest['parts']] + [x['parts'] for x in manifest.get('overlays', [])]


class UploadReader:
    def __init__(self, stream, name, total):
        self.stream, self.name, self.total = stream, name, total
        self.sent, self.last = 0, time.monotonic()

    def read(self, size=-1):
        block = self.stream.read(1024**2)
        self.sent += len(block)
        if time.monotonic() - self.last >= 25:
            print(f'{self.name}: {self.sent / 1024**2:.0f}/{self.total / 1024**2:.0f} MiB sent', flush=True)
            self.last = time.monotonic()
        return block


def upload():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    token = credentials()
    base = f'https://api.github.com/repos/{manifest["repository"]}'
    try:
        with request(f'{base}/releases/tags/{manifest["tag"]}', token) as r:
            release = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        body = json.dumps({'tag_name': manifest['tag'], 'target_commitish': 'main',
                           'name': 'Complete project assets (models, datasets and results)',
                           'body': 'Download using: python tools/github_assets.py restore\n'
                                   'All parts are required. SHA-256 checksums are in github-assets.json.',
                           'draft': True, 'prerelease': False}).encode()
        with request(f'{base}/releases', token, 'POST', body, {'Content-Type': 'application/json'}) as r:
            release = json.load(r)
    with request(f'{base}/releases/{release["id"]}/assets?per_page=100', token) as r:
        existing = {x['name']: x for x in json.load(r)}
    # Validate release uploads with the small database overlay before the model volumes.
    groups = part_groups(manifest)
    for part in [p for group in groups[1:] + groups[:1] for p in group]:
        old = existing.get(part['name'])
        if old:
            if old.get('size') == part['size'] and old.get('digest') == 'sha256:' + part['sha256']:
                print('Already uploaded and verified: ' + part['name'], flush=True)
                continue
            if release.get('draft') and old.get('state') != 'uploaded':
                with request(f'{base}/releases/assets/{old["id"]}', token, 'DELETE'):
                    pass
            else:
                raise RuntimeError('Existing release asset differs: ' + part['name'])
        path = WORK / part['name']
        if digest(path) != part['sha256']:
            raise RuntimeError('Local package checksum mismatch: ' + part['name'])
        url = release['upload_url'].split('{')[0] + '?name=' + urllib.parse.quote(part['name'])
        print('Uploading ' + part['name'], flush=True)
        with path.open('rb') as stream:
            with request(url, token, 'POST', UploadReader(stream, part['name'], part['size']), {'Content-Type': 'application/octet-stream',
                         'Content-Length': str(part['size'])}, timeout=1800) as r:
                item = json.load(r)
        if item.get('size') != part['size'] or item.get('digest') != 'sha256:' + part['sha256']:
            raise RuntimeError('Uploaded asset failed checksum verification: ' + part['name'])
        print('Verified ' + part['name'], flush=True)
    body = json.dumps({'draft': False}).encode()
    with request(f'{base}/releases/{release["id"]}', token, 'PATCH', body,
                 {'Content-Type': 'application/json'}) as r:
        print('Published ' + json.load(r)['html_url'], flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


class JoinedReader:
    def __init__(self, paths):
        self.paths, self.stream = iter(paths), None

    def read(self, size):
        blocks = []
        remaining = size
        while remaining:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.stream = path.open('rb')
            block = self.stream.read(remaining)
            if not block:
                self.stream.close()
                self.stream = None
            else:
                blocks.append(block)
                remaining -= len(block)
        return b''.join(blocks)

    def close(self):
        if self.stream:
            self.stream.close()


class SeekableParts(io.RawIOBase):
    """Inspect tar headers without reading through every model weight."""
    def __init__(self, paths):
        self.paths = list(paths)
        self.starts = [0]
        for path in self.paths:
            self.starts.append(self.starts[-1] + path.stat().st_size)
        self.position = 0
        self.stream = None
        self.index = -1

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.starts[-1] if whence == 2 else 0)
        if position < 0:
            raise ValueError('Negative seek')
        self.position = position
        return position

    def read(self, size=-1):
        remaining = min(self.starts[-1] - self.position, size if size >= 0 else self.starts[-1])
        blocks = []
        while remaining > 0:
            index = bisect.bisect_right(self.starts, self.position) - 1
            if self.index != index:
                if self.stream:
                    self.stream.close()
                self.stream = self.paths[index].open('rb')
                self.index = index
            self.stream.seek(self.position - self.starts[index])
            block = self.stream.read(min(remaining, self.starts[index + 1] - self.position))
            if not block:
                raise EOFError('Truncated split archive')
            self.position += len(block)
            remaining -= len(block)
            blocks.append(block)
        return b''.join(blocks)

    def close(self):
        if self.stream:
            self.stream.close()
        super().close()


def restore(overwrite=False):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    WORK.mkdir(exist_ok=True)
    paths = []
    for part in [p for group in part_groups(manifest) for p in group]:
        path = WORK / part['name']
        if not path.is_file() or path.stat().st_size != part['size'] or digest(path) != part['sha256']:
            url = f'https://github.com/{manifest["repository"]}/releases/download/{manifest["tag"]}/{part["name"]}'
            print('Downloading ' + part['name'], flush=True)
            partial = path.with_suffix(path.suffix + '.downloading')
            with request(url, timeout=1800) as response, partial.open('wb') as target:
                while True:
                    block = response.read(8 * 1024**2)
                    if not block:
                        break
                    target.write(block)
            if partial.stat().st_size != part['size'] or digest(partial) != part['sha256']:
                raise RuntimeError('Download checksum mismatch: ' + part['name'])
            partial.replace(path)
        paths.append(path)
    count = 0
    restored = set()
    for group in part_groups(manifest):
        stream = JoinedReader([WORK / p['name'] for p in group])
        try:
            with tarfile.open(fileobj=stream, mode='r|') as tar:
                for member in tar:
                    target = (ROOT / member.name).resolve()
                    if not target.is_relative_to(ROOT) or not (member.isfile() or member.islnk()):
                        raise RuntimeError('Unsafe archive entry: ' + member.name)
                    if target.exists() and not overwrite and target not in restored:
                        raise RuntimeError(f'Existing file would be overwritten: {member.name}; use --overwrite deliberately.')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + '.restoring')
                    if member.islnk():
                        linked = (ROOT / member.linkname).resolve()
                        if not linked.is_relative_to(ROOT) or linked not in restored:
                            raise RuntimeError('Unsafe archive hard link: ' + member.name)
                        shutil.copyfile(linked, temporary)
                    else:
                        with tar.extractfile(member) as source, temporary.open('wb') as out:
                            for block in iter(lambda: source.read(8 * 1024**2), b''):
                                out.write(block)
                    temporary.replace(target)
                    restored.add(target)
                    count += 1
        finally:
            stream.close()
    print(f'Restored {count} files. Original paths preserved.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'pack', 'upload', 'restore'])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.command == 'inventory':
        source, assets = inventory()
        print(json.dumps({'source_files': len(source), 'asset_files': len(assets),
                          'asset_GiB': round(sum((ROOT / x).stat().st_size for x in assets) / 1024**3, 3)}, indent=2))
    elif args.command == 'pack':
        pack()
    elif args.command == 'upload':
        upload()
    else:
        restore(args.overwrite)
