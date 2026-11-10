"""Download any missing models into ./models (stdlib only, so it runs before heavy imports)."""
import sys
import urllib.request
from pathlib import Path

import os

MODELS_DIR = Path(os.environ.get('CHANGELING_MODELS_DIR') or os.environ.get('ROOP_MODELS_DIR') or Path(__file__).resolve().parent.parent / 'models')

MODELS = {
    'inswapper_128.onnx': 'https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx',
    'GFPGANv1.4.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
}


def download(url, dest):
    tmp = dest.with_name(dest.name + '.part')
    print(f'Downloading {dest.name} ...')

    def report(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(f'\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB')
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=report)
    print()
    tmp.replace(dest)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        path = MODELS_DIR / name
        if not path.exists():
            download(url, path)


if __name__ == '__main__':
    main()
