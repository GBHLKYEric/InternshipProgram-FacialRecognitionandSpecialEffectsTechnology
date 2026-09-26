"""Copy Git-visible project files to a user-selected directory, then verify SHA256.

No deletion, environment copying, model redistribution, or Git metadata copying.
Run from anywhere: python scripts/copy_source_delivery.py --destination PATH.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def copy_delivery(destination, report):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination).resolve()
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError('Delivery directory must be separate from the project')
    files = subprocess.check_output(
        ['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']
    ).decode('utf-8').split('\0')
    inventory = []
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(filter(None, files))):
        source = (root / name).resolve()
        target = (destination / name).resolve()
        if root not in source.parents or destination not in target.parents:
            raise ValueError(f'Unsafe path: {name}')
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Expected ordinary source file: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Copy verification failed: {name}')
        inventory.append({'path': name, 'bytes': target.stat().st_size, 'sha256': digest})
    result = {'created_at': datetime.now(timezone.utc).isoformat(),
              'destination': str(destination), 'file_count': len(inventory),
              'bytes': sum(item['bytes'] for item in inventory),
              'all_copied_files_sha256_verified': True,
              'scope': 'Git-visible source, configuration, documentation and text evidence. '
                       'Ignored datasets, weights, virtual environments and private camera images are excluded.',
              'files': inventory}
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    (destination / 'SOURCE_DELIVERY_MANIFEST.json').write_text(payload, encoding='utf-8')
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    Path(report).write_text(payload, encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    copy_delivery(args.destination, args.report)
