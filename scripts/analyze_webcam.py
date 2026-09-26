"""Validate an exported real-camera timing report and plot its measured samples."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    samples = report['samples']
    if not samples or report['status'] != 'completed' or report['seconds'] < 20:
        raise ValueError('A complete 20-second camera measurement is required')
    if report.get('frontend', '').startswith('native'):
        return analyze_desktop(path, report)
    values = sorted(sample['capture_to_decode_ms'] for sample in samples)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Invalid per-frame timing')
    checks = {
        'frame_count': len(samples) == report['frames'],
        'face_frames': sum(sample['faces'] > 0 for sample in samples) == report['frames_with_face'],
        'visible_frames': sum(sample['visible'] for sample in samples) == report['frames_visible'],
        'fps': math.isclose(len(samples)/report['seconds'], report['observed_fps'], rel_tol=1e-8),
        'median_nearest_rank': math.isclose(values[math.ceil(.5*len(values))-1], report['capture_to_decode_median_ms']),
        'p95_nearest_rank': math.isclose(values[math.ceil(.95*len(values))-1], report['capture_to_decode_p95_ms']),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    ax.plot(range(1, len(samples)+1), [s['capture_to_decode_ms'] for s in samples],
            color='#27735a', linewidth=1.4, label='Capture to result decode')
    no_face = [(i+1, s['capture_to_decode_ms']) for i, s in enumerate(samples) if s['faces'] == 0]
    if no_face:
        ax.scatter(*zip(*no_face), s=12, color='#b36435', label='No face detected')
    ax.axhline(report['capture_to_decode_p95_ms'], color='#71807c', linestyle='--', label='P95')
    ax.set(xlabel='Processed frame index', ylabel='Measured latency (ms)',
           title=f'Real webcam: {report["frames"]} frames / {report["seconds"]:.2f}s / {report["observed_fps"]:.2f} FPS')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower right')
    target = path.with_name('webcam-latency.svg')
    fig.savefig(target)
    plt.close(fig)
    output = {'checks': checks, 'source': path.name, 'plot': target.name,
              'face_presence_fraction': report['frames_with_face']/report['frames'],
              'interpretation': 'Face presence is not annotated detection accuracy; latency excludes sensor age and physical display.'}
    path.with_name('webcam-validation.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(output, indent=2))


def analyze_desktop(path, report):
    samples = report['samples']
    values = [sample['latency_ms'] for sample in samples]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Invalid native per-frame timing')
    checks = {
        'frame_count': len(samples) == report['frames_rendered'],
        'face_frames': sum(s['faces'] > 0 for s in samples) == report['frames_with_face'],
        'fps': math.isclose(len(samples)/report['seconds'], report['render_submission_fps']),
        'median': math.isclose(float(np.median(values)), report['latency_median_ms']),
        'p95_higher': math.isclose(float(np.percentile(values, 95, method='higher')), report['latency_p95_ms']),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    groups = {}
    for name, present in [('face_present', True), ('no_face_detected', False)]:
        subset = [s['latency_ms'] for s in samples if (s['faces'] > 0) == present]
        groups[name] = {'frames': len(subset), 'median_ms': float(np.median(subset)) if subset else None,
                        'p95_higher_ms': float(np.percentile(subset, 95, method='higher')) if subset else None}
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    ax.plot(range(1, len(samples)+1), values, color='#27735a', linewidth=1, label='Read to Tk render submission')
    empty = [(i+1, s['latency_ms']) for i, s in enumerate(samples) if not s['faces']]
    if empty: ax.scatter(*zip(*empty), color='#bd783f', s=12, label='No face detected')
    ax.axhline(report['latency_p95_ms'], color='#71807c', linestyle='--', label='P95')
    ax.set(xlabel='Rendered frame index', ylabel='Measured latency (ms)',
           title=f'Native desktop: {len(samples)} frames / {report["seconds"]:.2f}s / {report["render_submission_fps"]:.2f} FPS')
    ax.spines[['top', 'right']].set_visible(False); ax.legend()
    prefix = path.with_suffix('')
    for suffix in ('.svg', '.png'): fig.savefig(str(prefix)+'-latency'+suffix, dpi=150)
    plt.close(fig)
    output = {'source': path.name, 'checks': checks, 'groups': groups,
              'interpretation': 'Timing conditioned on detector output is descriptive only; no annotated detection accuracy. Face/no-face composition and concurrent load differ between runs.'}
    path.with_name(path.stem+'-validation.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('reports/webcam-chrome.json'))
    analyze(parser.parse_args().report)
