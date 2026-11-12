"""Download the Qwen3 privacy model into the project; existing files are reused."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MODELSCOPE_CACHE', str(ROOT / '.modelscope-cache'))


def main():
    from modelscope.hub.snapshot_download import snapshot_download
    path = snapshot_download(
        'Qwen/Qwen3-4B-Instruct-2507',
        local_dir=str(ROOT / 'models' / 'qwen3-4b-instruct-2507'),
        ignore_file_pattern=['*.bin', '*.pt', '*.gguf'],
    )
    print(f'Privacy model ready: {path}')


if __name__ == '__main__':
    main()
