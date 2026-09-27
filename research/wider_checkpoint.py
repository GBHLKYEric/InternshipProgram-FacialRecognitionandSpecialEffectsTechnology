"""Inspect a locally produced WIDER checkpoint in its MMDetection environment.

PyTorch checkpoint deserialization can execute Python. Run this only on the
checkpoint produced by this local experiment, never an untrusted supplied file.
The main application venv intentionally does not contain MMEngine; this tool
runs in the existing MMDetection container and performs no training or inference.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main(args):
    import torch
    import mmengine
    import mmdet

    torch.set_num_threads(1)
    path = Path(args.checkpoint)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    meta = checkpoint['meta']
    config = meta.get('cfg') or meta.get('config')
    if not config:
        raise ValueError('Checkpoint does not contain its effective configuration')
    saved = Path(args.config)
    result = {
        'checkpoint': str(path),
        'epoch': meta.get('epoch'),
        'iter': meta.get('iter'),
        'meta_keys': sorted(meta),
        'checkpoint_config_sha256': hashlib.sha256(config.encode()).hexdigest(),
        'saved_config_sha256': hashlib.sha256(saved.read_bytes()).hexdigest(),
        'checkpoint_config_equals_saved_config': config == saved.read_text(),
        'inspection_runtime': {'torch': torch.__version__, 'mmengine': mmengine.__version__, 'mmdet': mmdet.__version__},
    }
    if result['iter'] != 6440 or not result['checkpoint_config_equals_saved_config']:
        raise ValueError('The full-run final checkpoint does not match its expected iteration/configuration')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/wider-mmdet-full-stable/iter_6440.pth')
    parser.add_argument('--config', default='runs/wider-mmdet-full-stable/effective-config.py')
    parser.add_argument('--output', default='reports/wider-checkpoint-metadata.json')
    main(parser.parse_args())
