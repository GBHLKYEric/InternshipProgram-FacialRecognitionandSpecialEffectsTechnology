"""Download versioned, SHA256-verified assets from their official repositories."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def verify(path, expected):
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected


def fetch(with_3d=False):
    registry = json.loads((ROOT / 'models/registry.json').read_text(encoding='utf-8'))
    for item in registry['assets']:
        if item['group'] == '3d' and not with_3d:
            continue
        path = ROOT / item['path']
        if not verify(path, item['sha256']):
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(item['url'], timeout=180) as src, temporary.open('wb') as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
            if not verify(temporary, item['sha256']):
                temporary.unlink(missing_ok=True)
                raise ValueError(f"SHA256 mismatch: {item['path']}")
            temporary.replace(path)
        print(f"Verified {item['path']}")
    import cv2
    img = cv2.imread(str(ROOT / 'assets/astronaut.png'))
    if img is None or not cv2.imwrite(str(ROOT / 'assets/sample.jpg'), img):
        raise RuntimeError('Could not create sample.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-3d', action='store_true', help='Also download 3DDFA model assets; review models/README.md first.')
    fetch(parser.parse_args().with_3d)
