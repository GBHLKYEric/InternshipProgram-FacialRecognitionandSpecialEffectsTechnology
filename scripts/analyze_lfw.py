"""Measure the local LFW identity distribution; never invent demographic labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'data/lfw/lfw')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/lfw-distribution')
    args = parser.parse_args()
    paths = sorted(args.root.glob('*/*.jpg'))
    if not paths:
        parser.error('No LFW images found; prepare the dataset using docs/sources.md.')
    counts = Counter(path.parent.name for path in paths)
    values = list(counts.values())
    ranges = [(1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, max(values))]
    bins = {f'{lo}-{hi}' if lo != hi else str(lo): sum(lo <= n <= hi for n in values)
            for lo, hi in ranges}
    report = {
        'source': 'Counts computed from local LFW image paths, not preset dataset statistics.',
        'images': len(paths), 'identities': len(counts),
        'images_per_identity_min': min(values), 'images_per_identity_max': max(values),
        'images_per_identity_median': statistics.median(values),
        'single_image_identities': sum(n == 1 for n in values),
        'identity_histogram': bins,
        'limitations': 'Counts do not establish consent, demographic balance, label correctness, or recognition performance.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
    bars = ax.bar(list(bins), list(bins.values()), color='#27735a')
    ax.bar_label(bars, padding=3)
    ax.set(xlabel='Images per identity (inclusive)', ylabel='Number of identities',
           title=f'Local LFW distribution: {len(paths):,} images / {len(counts):,} identities')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(bins.values())*1.15)
    fig.savefig(args.output.with_suffix('.svg'))
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
