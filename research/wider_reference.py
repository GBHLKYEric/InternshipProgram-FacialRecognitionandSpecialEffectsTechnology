"""Independently cross-check complete WIDER predictions with OpenCV's evaluator.

Only the reviewed evaluation functions are called, never its prediction loop
that inserts fallback boxes. Source is pinned by SHA256 and kept with local
research data. OpenCV Zoo upstream code is under Apache-2.0; see its LICENSE.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import types
import urllib.request

from research.wider_eval import load_ground_truth, read_prediction

REFERENCE_COMMIT = 'b1b4df493f77451851ae03dd4c8fa374f3ec3ea6'
SOURCE = ('https://raw.githubusercontent.com/opencv/opencv_zoo/'
          +REFERENCE_COMMIT+'/tools/eval/datasets/widerface.py')
SHA256 = '94461193a8a263af97cf34a0d7f9263edab10dd45e042d0b79bd1d219a0c6c5f'


def run(args):
    source = Path(args.root)/'wider-full/reference/opencv-widerface.py'
    if not source.exists():
        content = urllib.request.urlopen(SOURCE, timeout=30).read(128*1024)
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError('Upstream reference changed; review the new source instead of silently accepting it')
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError('Pinned reference SHA256 mismatch')
    module = types.ModuleType('opencv_wider_reference')
    exec(compile(content, str(source), 'exec'), module.__dict__)
    items, _ = load_ground_truth(args.root)
    prediction = {}
    for event, name, _, _ in items:
        prediction.setdefault(event, {})[name] = read_prediction(Path(args.prediction_dir)/event/(name+'.txt'))
    start = time.perf_counter()
    reference = module.evaluation(prediction, str(Path(args.root)/'eval_tools/ground_truth'))
    report = json.loads(Path(args.project_report).read_text())
    project = [report['metrics'][name]['ap'] for name in ['easy', 'medium', 'hard']]
    errors = [abs(a-b) for a, b in zip(reference, project)]
    result = {'source': SOURCE, 'reference_commit': REFERENCE_COMMIT,
              'reference_source_sha256': SHA256, 'reference_AP': reference,
              'project_AP': project, 'absolute_differences': errors,
              'all_differences_below_1e-12': all(value < 1e-12 for value in errors),
              'seconds': time.perf_counter()-start,
              'scope': 'Same complete real 3226-image predictions; reference functions, no fabricated empty-image detections'}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['all_differences_below_1e-12']:
        raise AssertionError('Independent AP comparison differs; inspect before claiming equivalence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data')
    parser.add_argument('--prediction-dir', default='runs/wider-yunet-full/predictions')
    parser.add_argument('--project-report', default='reports/wider-yunet-full.json')
    parser.add_argument('--output', default='reports/wider-evaluation-crosscheck.json')
    run(parser.parse_args())
